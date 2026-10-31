
#!/usr/bin/env python3
"""Part I-A Q2: point-in-time behavioral features."""

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb


RULE = """
    p_market_before BETWEEN 0 AND 1
    AND price_age_seconds > 0
    AND price_age_seconds <= 43200
    AND pre_close_1h = TRUE
    AND y_yes IN (0, 1)
    AND outcome_seq IN (1, 2)
"""


def epoch(value):
    return int(
        datetime.strptime(value, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


def log(message):
    print(message, flush=True)


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--grace-days", type=int, default=7)
    ap.add_argument(
        "--evaluation-cutoff", default="2026-10-31"
    )
    ap.add_argument(
        "--shrinkage-k", type=float, default=20.0
    )
    ap.add_argument("--memory-gb", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--temp-root", default="/content")
    args = ap.parse_args()

    if args.grace_days < 0 or args.shrinkage_k <= 0:
        raise ValueError("Invalid grace-days or shrinkage-k")

    training_end = epoch("2025-10-01")
    validation_end = epoch("2026-01-01")
    evaluation_end = epoch(args.evaluation_cutoff)

    if evaluation_end <= epoch("2026-03-31"):
        raise ValueError("Evaluation cutoff too early")

    root = Path(args.drive_root).expanduser()
    data = root / "data"

    files = sorted(
        (data / "probability_alignment").glob(
            "wallet_market_*.parquet"
        )
    )
    registry = (
        data / "research_index" / "market_registry.parquet"
    )

    if len(files) != 15 or not registry.exists():
        raise RuntimeError(
            "Missing alignment partitions or market registry"
        )

    out = (
        data / "research_results"
        / f"p1a_bia_features_{args.grace_days}d"
    )
    out.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parents[1]
    git_commit = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
        ).strip()
        if (repo / ".git").exists()
        else "unknown"
    )

    log(f"[START] Q2 features | commit={git_commit[:12]}")

    scratch_base = Path(args.temp_root)
    if not scratch_base.exists():
        scratch_base = Path("/tmp")

    with tempfile.TemporaryDirectory(
        prefix="p1a_bia_",
        dir=scratch_base,
    ) as tmp:

        scratch = Path(tmp)
        (scratch / "spill").mkdir()

        con = duckdb.connect(
            str(scratch / "work.duckdb")
        )
        con.execute(
            f"SET memory_limit='{args.memory_gb}GB'"
        )
        con.execute(f"SET threads={args.threads}")
        con.execute(
            f"SET temp_directory='{scratch / 'spill'}'"
        )

        pattern = str(
            data / "probability_alignment"
            / "wallet_market_*.parquet"
        )

        # ------------------------------------------
        # 1. First valid BUY per wallet-market
        # ------------------------------------------

        log("[1/6] Selecting first qualifying BUY...")

        con.execute(f"""
            CREATE TABLE entries AS
            SELECT * EXCLUDE(rn)
            FROM (
                SELECT
                    LOWER(wallet) AS wallet,
                    market_type,
                    condition_id,
                    CAST(trade_ts AS BIGINT) AS trade_ts,
                    CAST(outcome_seq AS INTEGER) AS outcome_seq,
                    CAST(y_yes AS INTEGER) AS y_yes,
                    CAST(p_market_before AS DOUBLE)
                        AS p_market_before,
                    CAST(price_age_seconds AS DOUBLE)
                        AS price_age_seconds,
                    CAST(usdc_amount AS DOUBLE)
                        AS usdc_amount,
                    CAST(trade_price AS DOUBLE)
                        AS trade_price,
                    category,
                    EPOCH(close_at) AS close_ts,

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
            WHERE rn = 1
        """)

        n_entries = scalar(
            con, "SELECT COUNT(*) FROM entries"
        )
        log(f"      Wallet-market pairs: {n_entries:,}")

        # ------------------------------------------
        # 2. Market outcome availability
        # ------------------------------------------

        log("[2/6] Matching resolution timestamps...")

        con.execute(f"""
            CREATE TABLE market_meta AS
            WITH parsed AS (
                SELECT
                    market_type,
                    condition_id,
                    EPOCH(close_at) AS close_ts,

                    CASE
                        WHEN TRY_CAST(
                            resolved_at AS DOUBLE
                        ) BETWEEN 946684800 AND 4102444800
                        THEN TRY_CAST(
                            resolved_at AS DOUBLE
                        )

                        WHEN TRY_CAST(
                            resolved_at AS DOUBLE
                        ) BETWEEN 946684800000
                          AND 4102444800000
                        THEN TRY_CAST(
                            resolved_at AS DOUBLE
                        ) / 1000.0

                        ELSE EPOCH(
                            TRY_CAST(
                                resolved_at AS TIMESTAMPTZ
                            )
                        )
                    END AS resolved_ts,

                    neg_risk_market_id

                FROM read_parquet('{registry}')
                WHERE outcome_seq = 1
            )

            SELECT
                market_type,
                condition_id,

                COALESCE(
                    MIN(resolved_ts),
                    MIN(close_ts)
                    + {args.grace_days} * 86400
                ) AS label_available_ts,

                COUNT(DISTINCT close_ts)
                    AS close_versions,

                COUNT(DISTINCT resolved_ts)
                    AS resolution_versions,

                ANY_VALUE(
                    NULLIF(
                        TRIM(
                            CAST(
                                neg_risk_market_id
                                AS VARCHAR
                            )
                        ),
                        ''
                    )
                ) AS event_id

            FROM parsed
            GROUP BY 1, 2
        """)

        con.execute("""
            CREATE TABLE observations AS
            SELECT
                ROW_NUMBER() OVER () AS obs_id,
                x.*
            FROM (
                SELECT
                    e.wallet,
                    e.market_type,
                    e.condition_id,
                    e.trade_ts,
                    e.outcome_seq,
                    e.y_yes,
                    e.p_market_before,
                    e.price_age_seconds,
                    e.usdc_amount,
                    e.trade_price,
                    e.category,
                    e.close_ts,
                    m.label_available_ts,

                    COALESCE(
                        'event:' || m.event_id,
                        'market:' || e.market_type
                        || ':' || e.condition_id
                    ) AS cluster_id,

                    CASE
                        WHEN e.outcome_seq = 1 THEN 1
                        ELSE -1
                    END AS direction,

                    CASE
                        WHEN e.outcome_seq = 1
                        THEN e.y_yes - e.p_market_before
                        ELSE e.p_market_before - e.y_yes
                    END AS score

                FROM entries e
                JOIN market_meta m
                    USING (market_type, condition_id)

                WHERE m.label_available_ts > e.trade_ts
                  AND m.resolution_versions <= 1
                  AND m.close_versions = 1
            ) x
        """)

        n_obs = scalar(
            con, "SELECT COUNT(*) FROM observations"
        )
        log(
            f"      Valid observations: {n_obs:,} "
            f"(excluded {n_entries - n_obs:,})"
        )

        # ------------------------------------------
        # 3. Prior participation experience
        # ------------------------------------------

        log("[3/6] Computing wallet experience...")

        con.execute("""
            CREATE TABLE experience AS
            SELECT
                obs_id,

                COALESCE(
                    COUNT(*) OVER (
                        PARTITION BY wallet
                        ORDER BY trade_ts
                        RANGE BETWEEN
                            UNBOUNDED PRECEDING
                            AND 1 PRECEDING
                    ),
                    0
                ) AS n_past_sampled_markets

            FROM observations
        """)

        # ------------------------------------------
        # 4. Point-in-time historical skill
        # ------------------------------------------

        log("[4/6] Computing time-safe wallet skill...")

        # Query events occur at trade_ts.
        # Historical scores enter only at settlement.
        # At equal timestamps, queries precede
        # settlement events.

        con.execute("""
            CREATE TABLE ledger AS

            SELECT
                wallet,
                trade_ts AS ts,
                0 AS kind,
                obs_id,
                CAST(0 AS DOUBLE) AS contribution
            FROM observations

            UNION ALL

            SELECT
                wallet,
                CAST(label_available_ts AS BIGINT) AS ts,
                1 AS kind,
                CAST(NULL AS BIGINT) AS obs_id,
                CAST(score AS DOUBLE) AS contribution
            FROM observations
        """)

        con.execute("""
            CREATE TABLE historical AS
            SELECT
                obs_id,
                n_past_resolved_markets,
                historical_score_sum
            FROM (
                SELECT
                    obs_id,
                    kind,

                    SUM(
                        CASE WHEN kind = 1 THEN 1
                        ELSE 0 END
                    ) OVER w AS n_past_resolved_markets,

                    SUM(contribution)
                        OVER w AS historical_score_sum

                FROM ledger

                WINDOW w AS (
                    PARTITION BY wallet
                    ORDER BY ts, kind
                    ROWS BETWEEN
                        UNBOUNDED PRECEDING
                        AND CURRENT ROW
                )
            )
            WHERE kind = 0
        """)

        if scalar(
            con, "SELECT COUNT(*) FROM historical"
        ) != n_obs:
            raise RuntimeError(
                "Missing historical feature snapshots"
            )

        # ------------------------------------------
        # 5. Assemble modeling features
        # ------------------------------------------

        log("[5/6] Assembling modeling dataset...")

        con.execute(f"""
            CREATE TABLE features AS
            SELECT
                o.obs_id,
                o.wallet,
                o.market_type,
                o.condition_id,
                o.cluster_id,
                o.trade_ts,
                o.label_available_ts,
                o.y_yes,
                o.p_market_before,
                o.outcome_seq,
                o.direction,
                o.category,
                o.trade_price,

                o.price_age_seconds / 3600.0
                    AS price_age_hours,

                GREATEST(
                    0.0,
                    (o.close_ts - o.trade_ts) / 3600.0
                ) AS hours_to_close,

                LN(
                    1.0 + GREATEST(
                        0.0,
                        COALESCE(o.usdc_amount, 0.0)
                    )
                ) AS log_trade_size,

                COALESCE(
                    e.n_past_sampled_markets, 0
                ) AS n_past_sampled_markets,

                COALESCE(
                    h.n_past_resolved_markets, 0
                ) AS n_past_resolved_markets,

                CASE
                    WHEN h.n_past_resolved_markets > 0
                    THEN h.historical_score_sum
                         / h.n_past_resolved_markets
                    ELSE 0.0
                END AS historical_raw_skill,

                h.historical_score_sum /
                    (
                        h.n_past_resolved_markets
                        + {args.shrinkage_k}
                    ) AS historical_shrunk_skill

            FROM observations o
            JOIN historical h USING (obs_id)
            JOIN experience e USING (obs_id)
        """)

        if scalar(
            con, "SELECT COUNT(*) FROM features"
        ) != n_obs:
            raise RuntimeError(
                "Feature join duplicated or dropped rows"
            )

        invalid = scalar(con, """
            SELECT COUNT(*)
            FROM features
            WHERE n_past_resolved_markets
                > n_past_sampled_markets
        """)

        if invalid:
            raise RuntimeError(
                "Historical feature look-ahead check failed"
            )

        # Only labels known by the relevant cutoff
        # enter each modeling split.

        conditions = {
            "train": (
                f"trade_ts < {training_end} "
                f"AND label_available_ts < {training_end}"
            ),
            "validation": (
                f"trade_ts >= {training_end} "
                f"AND trade_ts < {validation_end} "
                f"AND label_available_ts < {validation_end}"
            ),
            "final_test": (
                f"trade_ts >= {validation_end} "
                f"AND label_available_ts < {evaluation_end}"
            ),
        }

        counts = {}

        for name, where in conditions.items():
            path = out / f"{name}.parquet"

            con.execute(f"""
                COPY (
                    SELECT *
                    FROM features
                    WHERE {where}
                )
                TO '{path}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

            counts[name] = scalar(
                con,
                f"SELECT COUNT(*) FROM features WHERE {where}",
            )

            log(
                f"      {name}: {counts[name]:,} "
                f"-> {path.name}"
            )

        # ------------------------------------------
        # 6. Manifest and diagnostics
        # ------------------------------------------

        log("[6/6] Saving diagnostics...")

        diagnostics = con.execute("""
            SELECT
                COUNT(*) AS all_first_entries,
                COUNT(DISTINCT wallet) AS wallets,

                COUNT(
                    DISTINCT market_type || ':' || condition_id
                ) AS markets,

                COUNT(*) FILTER (
                    WHERE n_past_resolved_markets > 0
                ) AS with_history,

                AVG(n_past_resolved_markets)
                    AS mean_resolved_history,

                AVG(historical_shrunk_skill)
                    AS mean_shrunk_skill

            FROM features
        """).fetchone()

        report = {
            "git_commit": git_commit,
            "unit": (
                "first valid alignment BUY "
                "per wallet-market"
            ),
            "time_split": {
                "train_before": "2025-10-01",
                "validation": (
                    "2025-10-01 to 2025-12-31"
                ),
                "final_test": (
                    "2026-01-01 to 2026-03-31"
                ),
                "evaluation_cutoff": args.evaluation_cutoff,
            },
            "label_availability": (
                "resolved_at else close_at + grace_days"
            ),
            "grace_days": args.grace_days,
            "shrinkage_k": args.shrinkage_k,
            "shrinkage_definition": (
                "score_sum / (n_resolved + k); "
                "fixed-prior proxy, not fitted EB"
            ),
            "sample_limitations": [
                "History covers eligible aligned markets only",
                "Market probability may lag by up to 12h",
                "Resolution availability uses timestamp proxies",
            ],
            "first_qualifying_pairs": n_entries,
            "temporally_valid_pairs": n_obs,
            "split_rows": counts,
            "diagnostics": dict(zip(
                [
                    "all_first_entries",
                    "wallets",
                    "markets",
                    "with_history",
                    "mean_resolved_history",
                    "mean_shrunk_skill",
                ],
                diagnostics,
            )),
        }

        (out / "BIA_FEATURES_MANIFEST.json").write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
        )

        log(f"[DONE] {out}")
        log("[NEXT] Behavioral prediction models")
        con.close()


if __name__ == "__main__":
    main()
