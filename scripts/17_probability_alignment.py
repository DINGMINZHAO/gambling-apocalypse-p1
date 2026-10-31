import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb

START = date(2025, 1, 1)
LAST = date(2026, 3, 1)


def next_month(d):
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def epoch(d):
    return int(datetime(
        d.year, d.month, d.day,
        tzinfo=timezone.utc
    ).timestamp())


def save_json(path, obj):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--max-output-gb", type=float, default=1.0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo, text=True
    ).strip()

    data = Path(args.drive_root) / "data"
    trades = data / "research_raw" / "trades"
    candidates = data / "market_probabilities" / "candidates.parquet"
    prices = data / "market_probabilities" / "prices_*.parquet"
    registry = data / "research_index" / "market_registry.parquet"

    out = data / "probability_alignment"
    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out / "ALIGNMENT_MANIFEST.json"

    source_manifest = json.loads((
        data / "market_probabilities"
        / "PROBABILITY_RUN_MANIFEST.json"
    ).read_text())

    if source_manifest["next_index"] != source_manifest["candidates"]:
        raise RuntimeError("Probability acquisition is incomplete")

    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest["alignment_commit"] != commit:
            raise RuntimeError(
                "Resume using original alignment Git commit"
            )
    else:
        manifest = {
            "alignment_commit": commit,
            "trade_commit": json.loads((
                data / "research_raw" / "RUN_MANIFEST.json"
            ).read_text())["git_commit"],
            "probability_commit": source_manifest["git_commit"],
            "method": "First taker BUY per wallet-market-month",
            "price_rule": "last completed bucket strictly before trade",
            "months": {},
        }
        save_json(manifest_file, manifest)

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=2")
    con.execute("SET temp_directory='/content/align_duckdb_tmp'")

    con.execute(f"""
        CREATE VIEW probability AS
        SELECT market_type, condition_id,
               available_ts, p_yes
        FROM read_parquet('{prices}')
        WHERE p_yes BETWEEN 0 AND 1
    """)

    month = START

    while month <= LAST:
        end = next_month(month)
        label = month.strftime("%Y_%m")
        target = out / f"wallet_market_{label}.parquet"

        if (
            label in manifest["months"]
            and target.exists()
            and target.stat().st_size ==
                manifest["months"][label]["bytes"]
        ):
            print(f"[SKIP] {label}", flush=True)
            month = end
            continue

        files = []
        d = month
        while d <= end:
            fname = d.strftime("%Y_%m_%d") + ".parquet"
            for kind in ("standard", "multi"):
                path = trades / kind / fname
                if path.exists():
                    files.append(str(path))
            d += timedelta(days=1)

        if not files:
            raise RuntimeError(f"No trade files for {label}")

        file_sql = "[" + ",".join(
            "'" + f.replace("'", "''") + "'"
            for f in files
        ) + "]"

        print(f"[START] {label}", flush=True)

        with tempfile.TemporaryDirectory(dir="/content") as td:
            local = Path(td) / "aligned.parquet"

            con.execute(f"""
            COPY (
                WITH eligible AS (
                    SELECT
                        t.market_type,
                        t.condition_id,
                        LOWER(t.taker) AS wallet,
                        t.block_timestamp AS trade_ts,
                        t.outcome_seq,
                        t.D AS direction,
                        t.price AS trade_price,
                        t.usdc_amount,
                        c.token_id AS yes_token_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY
                                t.market_type,
                                t.condition_id,
                                LOWER(t.taker)
                            ORDER BY
                                t.block_timestamp,
                                t.usdc_amount DESC,
                                t.asset_id
                        ) AS rn
                    FROM read_parquet(
                        {file_sql}, union_by_name=true
                    ) t
                    JOIN read_parquet('{candidates}') c
                      ON t.market_type = c.market_type
                     AND t.condition_id = c.condition_id
                    WHERE t.block_timestamp >= {epoch(month)}
                      AND t.block_timestamp < {epoch(end)}
                      AND t.taker_direction = 'BUY'
                      AND t.outcome_seq IN (1, 2)
                      AND t.price BETWEEN 0.01 AND 0.99
                      AND t.taker IS NOT NULL
                ),
                first_buy AS (
                    SELECT *
                    FROM eligible
                    WHERE rn = 1
                ),
                enriched AS (
                    SELECT
                        f.*,
                        r.category,
                        r.neg_risk_market_id,
                        r.close_at,
                        CASE
                            WHEN LOWER(TRIM(
                                r.winning_outcome_label
                            )) = 'yes' THEN 1
                            WHEN LOWER(TRIM(
                                r.winning_outcome_label
                            )) = 'no' THEN 0
                            ELSE NULL
                        END AS y_yes
                    FROM first_buy f
                    LEFT JOIN read_parquet('{registry}') r
                      ON f.market_type = r.market_type
                     AND f.condition_id = r.condition_id
                     AND f.yes_token_id = r.token_id
                )
                SELECT
                    e.market_type,
                    e.condition_id,
                    e.wallet,
                    e.trade_ts,
                    e.outcome_seq,
                    e.direction,
                    e.trade_price,
                    e.usdc_amount,
                    e.category,
                    e.neg_risk_market_id,
                    e.y_yes,
                    e.close_at,
                    p.p_yes AS p_market_before,
                    p.available_ts AS price_available_ts,
                    e.trade_ts - p.available_ts
                        AS price_age_seconds,
                    CASE
                        WHEN e.close_at IS NOT NULL
                        THEN e.trade_ts <
                             EPOCH(e.close_at) - 3600
                        ELSE NULL
                    END AS pre_close_1h
                FROM enriched e
                ASOF LEFT JOIN probability p
                  ON e.market_type = p.market_type
                 AND e.condition_id = p.condition_id
                 AND e.trade_ts > p.available_ts
            )
            TO '{local}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

            q = con.execute(f"""
                SELECT
                    COUNT(*) AS observations,
                    COUNT_IF(p_market_before IS NOT NULL)
                        AS matched,
                    COUNT_IF(price_age_seconds <= 3600)
                        AS within_1h,
                    COUNT_IF(price_age_seconds <= 10800)
                        AS within_3h,
                    COUNT_IF(
                        price_age_seconds <= 10800
                        AND pre_close_1h = TRUE
                        AND y_yes IS NOT NULL
                    ) AS candidate_valid,
                    COUNT(DISTINCT condition_id) AS markets,
                    COUNT(DISTINCT wallet) AS wallets
                FROM read_parquet('{local}')
            """)

            stats = dict(zip(
                [x[0] for x in q.description],
                q.fetchone()
            ))

            partial = target.with_suffix(".partial")
            shutil.copyfile(local, partial)
            if partial.stat().st_size != local.stat().st_size:
                raise IOError("Incomplete Google Drive copy")
            partial.replace(target)

        manifest["months"][label] = {
            **stats,
            "bytes": target.stat().st_size
        }
        save_json(manifest_file, manifest)

        print(
            f"[DONE] {label} | "
            f"Observations={stats['observations']:,} | "
            f"Matched={stats['matched']:,} | "
            f"Within 3h={stats['within_3h']:,} | "
            f"Candidate valid={stats['candidate_valid']:,}",
            flush=True
        )

        total_bytes = sum(
            v["bytes"] for v in manifest["months"].values()
        )
        if total_bytes >= args.max_output_gb * 1024**3:
            print("OUTPUT BUDGET REACHED — checkpoint saved")
            break

        month = end

    print("\n=== ALIGNMENT SUMMARY ===")
    totals = {}
    for key in (
        "observations", "matched",
        "within_1h", "within_3h",
        "candidate_valid"
    ):
        totals[key] = sum(
            v[key] for v in manifest["months"].values()
        )
        print(f"{key}: {totals[key]:,}")

    print("Months completed:", len(manifest["months"]))
    print("Manifest:", manifest_file)
    print("STATUS: ALIGNMENT ONLY — not final skill inference")
    con.close()


if __name__ == "__main__":
    main()
