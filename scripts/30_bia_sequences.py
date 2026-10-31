
#!/usr/bin/env python3
"""Create time-safe wallet action histories for Q2 GRU."""

import argparse
import json
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STEPS = 16
DIM = 5

RULE = """
    p_market_before BETWEEN 0 AND 1
    AND price_age_seconds > 0
    AND price_age_seconds <= 43200
    AND pre_close_1h = TRUE
    AND y_yes IN (0, 1)
    AND outcome_seq IN (1, 2)
"""

SCHEMA = pa.schema([
    pa.field("obs_id", pa.int64()),
    pa.field("market_type", pa.string()),
    pa.field("condition_id", pa.string()),
    pa.field("cluster_id", pa.string()),
    pa.field("category", pa.string()),
    pa.field("y_yes", pa.int8()),
    pa.field("p_market_before", pa.float32()),
    pa.field("price_age_hours", pa.float32()),
    pa.field("direction", pa.int8()),
    pa.field("log_trade_size", pa.float32()),
    pa.field("n_past_sampled_markets", pa.int32()),
    pa.field("n_past_resolved_markets", pa.int32()),
    pa.field("historical_shrunk_skill", pa.float32()),
    pa.field("history_count", pa.int8()),
    pa.field(
        "history",
        pa.list_(pa.float32(), STEPS * DIM),
    ),
])


def log(msg):
    print(msg, flush=True)


def flush(split, buffers, writers, out):
    rows = buffers[split]
    if not rows["obs_id"]:
        return

    cols = {}

    for field in SCHEMA:
        key = field.name

        if key == "history":
            flat = np.asarray(
                rows[key], dtype=np.float32
            ).reshape(-1)

            cols[key] = (
                pa.FixedSizeListArray.from_arrays(
                    pa.array(flat, type=pa.float32()),
                    STEPS * DIM,
                )
            )
        else:
            cols[key] = pa.array(
                rows[key], type=field.type
            )

    table = pa.Table.from_pydict(
        cols, schema=SCHEMA
    )

    if split not in writers:
        writers[split] = pq.ParquetWriter(
            out / f"{split}.parquet",
            SCHEMA,
            compression="zstd",
        )

    writers[split].write_table(table)

    buffers[split] = {
        f.name: [] for f in SCHEMA
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument(
        "--grace-days", type=int, default=7
    )
    ap.add_argument(
        "--memory-gb", type=int, default=12
    )
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--batch-size", type=int, default=16000
    )
    args = ap.parse_args()

    root = (
        Path(args.drive_root).expanduser()
        / "data"
    )

    alignment = root / "probability_alignment"

    source = (
        root / "research_results"
        / f"p1a_bia_features_{args.grace_days}d"
    )

    out = (
        root / "research_results"
        / f"p1a_bia_sequences_{args.grace_days}d"
    )

    out.mkdir(parents=True, exist_ok=True)

    if len(list(
        alignment.glob("wallet_market_*.parquet")
    )) != 15:
        raise RuntimeError(
            "Expected 15 alignment partitions"
        )

    splits = ("train", "validation", "final_test")

    for split in splits:
        if not (source / f"{split}.parquet").exists():
            raise FileNotFoundError(
                source / f"{split}.parquet"
            )

    with tempfile.TemporaryDirectory(
        prefix="bia_seq_", dir="/content"
    ) as td:

        tmp = Path(td)
        (tmp / "spill").mkdir()

        ordered = tmp / "ordered_events.parquet"

        con = duckdb.connect(
            str(tmp / "seq.duckdb")
        )
        con.execute(
            f"SET memory_limit='{args.memory_gb}GB'"
        )
        con.execute(f"SET threads={args.threads}")
        con.execute(
            f"SET temp_directory='{tmp / 'spill'}'"
        )

        target_sql = " UNION ALL ".join(
            f"""
            SELECT *, '{split}' AS target_split
            FROM read_parquet(
                '{source / f"{split}.parquet"}'
            )
            """
            for split in splits
        )

        con.execute(
            f"CREATE VIEW targets AS {target_sql}"
        )

        expected = dict(con.execute("""
            SELECT target_split, COUNT(*)
            FROM targets
            GROUP BY 1
        """).fetchall())

        log(f"[1/4] Targets: {expected}")

        pattern = (
            alignment / "wallet_market_*.parquet"
        )

        log("[2/4] Sorting historical actions...")

        con.execute(f"""
            COPY (
                WITH first_actions AS (
                    SELECT
                        LOWER(wallet) AS wallet,
                        market_type,
                        condition_id,

                        CAST(trade_ts AS BIGINT)
                            AS trade_ts,

                        CASE
                            WHEN outcome_seq = 1 THEN 1
                            ELSE -1
                        END AS direction,

                        LN(
                            1 + GREATEST(
                                0,
                                COALESCE(usdc_amount, 0)
                            )
                        ) AS log_trade_size,

                        CAST(p_market_before AS DOUBLE)
                            AS p_market_before,

                        CAST(
                            price_age_seconds AS DOUBLE
                        ) / 3600.0
                            AS price_age_hours,

                        ROW_NUMBER() OVER (
                            PARTITION BY
                                LOWER(wallet),
                                market_type,
                                condition_id
                            ORDER BY trade_ts, outcome_seq
                        ) AS rn

                    FROM read_parquet('{pattern}')
                    WHERE {RULE}
                )

                SELECT
                    e.wallet,
                    e.market_type,
                    e.condition_id,
                    e.trade_ts,
                    e.direction,
                    e.log_trade_size,
                    e.p_market_before,
                    e.price_age_hours,

                    t.obs_id,
                    t.target_split,
                    t.cluster_id,
                    t.category,
                    t.y_yes,
                    t.n_past_sampled_markets,
                    t.n_past_resolved_markets,
                    t.historical_shrunk_skill

                FROM first_actions e

                LEFT JOIN targets t
                  ON e.wallet = LOWER(t.wallet)
                 AND e.market_type = t.market_type
                 AND e.condition_id = t.condition_id
                 AND e.trade_ts =
                     CAST(t.trade_ts AS BIGINT)

                WHERE e.rn = 1

                ORDER BY
                    e.wallet,
                    e.trade_ts,
                    e.market_type,
                    e.condition_id

            )
            TO '{ordered}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        con.close()

        log("[3/4] Constructing historical sequences...")

        parquet = pq.ParquetFile(ordered)

        writers = {}

        buffers = {
            s: {f.name: [] for f in SCHEMA}
            for s in splits
        }

        counts = {s: 0 for s in splits}
        history_1 = {s: 0 for s in splits}
        history_5 = {s: 0 for s in splits}

        current_wallet = None
        current_ts = None

        previous = deque(maxlen=STEPS)
        simultaneous = []

        events_seen = 0

        for batch in parquet.iter_batches(
            batch_size=args.batch_size
        ):
            data = batch.to_pydict()

            for i in range(len(data["wallet"])):
                wallet = data["wallet"][i]
                ts = int(data["trade_ts"][i])

                if wallet != current_wallet:
                    current_wallet = wallet
                    current_ts = None
                    previous.clear()
                    simultaneous = []

                if (
                    current_ts is not None
                    and ts != current_ts
                ):
                    previous.extend(simultaneous)
                    simultaneous = []

                current_ts = ts
                split = data["target_split"][i]

                if split is not None:
                    hist = list(previous)
                    n = len(hist)

                    sequence = np.zeros(
                        (STEPS, DIM),
                        dtype=np.float32,
                    )

                    start = STEPS - n

                    for j, past in enumerate(hist):
                        (
                            old_ts,
                            side,
                            amount,
                            market_p,
                            age_h,
                        ) = past

                        sequence[start + j] = [
                            side,
                            amount,
                            market_p,
                            age_h,
                            np.log1p(
                                max(0, ts - old_ts)
                                / 86400.0
                            ),
                        ]

                    row = buffers[split]

                    for key in row:
                        if key == "history":
                            value = sequence.reshape(-1)

                        elif key == "history_count":
                            value = n

                        elif key in (
                            "p_market_before",
                            "price_age_hours",
                            "log_trade_size",
                        ):
                            value = data[key][i]

                        else:
                            value = data[key][i]

                        row[key].append(value)

                    counts[split] += 1
                    history_1[split] += int(n >= 1)
                    history_5[split] += int(n >= 5)

                    if (
                        len(row["obs_id"])
                        >= args.batch_size
                    ):
                        flush(
                            split, buffers,
                            writers, out,
                        )

                simultaneous.append((
                    ts,
                    data["direction"][i],
                    data["log_trade_size"][i],
                    data["p_market_before"][i],
                    data["price_age_hours"][i],
                ))

                events_seen += 1

            if events_seen % 1_000_000 < args.batch_size:
                log(
                    f"      Processed: "
                    f"{events_seen:,} actions"
                )

        for split in splits:
            flush(split, buffers, writers, out)

        for writer in writers.values():
            writer.close()

        if counts != expected:
            raise RuntimeError(
                "Target coverage mismatch: "
                f"{counts} vs {expected}"
            )

        report = {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
            ).strip(),
            "sequence_length": STEPS,
            "sequence_features": [
                "direction",
                "log_trade_size",
                "p_market_before",
                "price_age_hours",
                "log1p_days_before_current",
            ],
            "rule": (
                "Historical action trade_ts "
                "strictly earlier than target trade_ts"
            ),
            "historical_outcomes_used": False,
            "actions_scanned": events_seen,
            "split_rows": counts,
            "history_ge_1": history_1,
            "history_ge_5": history_5,
        }

        (out / "SEQUENCE_MANIFEST.json").write_text(
            json.dumps(report, indent=2)
        )

        log(f"[4/4] Output rows: {counts}")
        log(f"[DONE] {out}")


if __name__ == "__main__":
    main()
