import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--drive-root", required=True)
args = ap.parse_args()

data = Path(args.drive_root) / "data"
source = data / "probability_alignment"
files = sorted(source.glob("wallet_market_*.parquet"))

if len(files) != 15:
    raise RuntimeError(
        f"Expected 15 monthly files, found {len(files)}"
    )

out = data / "research_results"
out.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")
con.execute("SET threads=2")

# One common research sample, maximum price age 12h.
RULE = """
    p_market_before BETWEEN 0 AND 1
    AND price_age_seconds > 0
    AND price_age_seconds <= 43200
    AND pre_close_1h = TRUE
    AND y_yes IN (0, 1)
    AND outcome_seq IN (1, 2)
"""

RULE_3H = RULE.replace(
    "price_age_seconds <= 43200",
    "price_age_seconds <= 10800"
)

# Compare the execution price with the corresponding
# YES or NO market-implied probability.
SIDE_PRICE = """
    CASE
        WHEN outcome_seq = 1 THEN p_market_before
        WHEN outcome_seq = 2 THEN 1 - p_market_before
    END
"""

GAP = f"ABS(trade_price - ({SIDE_PRICE}))"


def query(sql):
    result = con.execute(sql)
    names = [x[0] for x in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


print("=== 12H SAMPLE VALIDATION ===", flush=True)

monthly = []

for index, path in enumerate(files, 1):
    month = path.stem.replace("wallet_market_", "")
    print(f"[START {index}/15] {month}", flush=True)

    row = query(f"""
        SELECT
            COUNT(*) AS observations,
            COUNT_IF(p_market_before IS NOT NULL)
                AS probability_matched,
            COUNT_IF(
                price_age_seconds > 0
                AND price_age_seconds <= 43200
            ) AS within_12h,
            COUNT_IF({RULE_3H}) AS valid_3h,
            COUNT_IF({RULE}) AS valid_12h,
            ROUND(AVG(
                CASE WHEN {RULE} THEN {GAP} END
            ), 4) AS mean_abs_gap_12h,
            ROUND(APPROX_QUANTILE(
                CASE WHEN {RULE} THEN {GAP} END,
                0.5
            ), 4) AS median_abs_gap_12h
        FROM read_parquet('{path}')
    """)[0]

    monthly.append({"month": month, **row})

    print(
        f"[DONE] {month} | "
        f"Matched={row['probability_matched']:,} | "
        f"Valid 12h={row['valid_12h']:,} | "
        f"Median gap={row['median_abs_gap_12h']}",
        flush=True
    )

pattern = source / "wallet_market_*.parquet"

print("\n[START] Overall wallet/market coverage", flush=True)

coverage = query(f"""
    SELECT
        COUNT(DISTINCT wallet) AS wallets,
        COUNT(DISTINCT
            market_type || ':' || condition_id
        ) AS markets
    FROM read_parquet('{pattern}')
    WHERE {RULE}
""")[0]

print("[DONE] Coverage", coverage, flush=True)

print("\n[START] Probability freshness comparison", flush=True)

freshness = query(f"""
    SELECT
        CASE
            WHEN price_age_seconds <= 10800 THEN '0-3h'
            WHEN price_age_seconds <= 21600 THEN '3-6h'
            ELSE '6-12h'
        END AS age_group,
        COUNT(*) AS observations,
        ROUND(AVG({GAP}), 4) AS mean_abs_gap,
        ROUND(APPROX_QUANTILE({GAP}, 0.5), 4)
            AS median_abs_gap
    FROM read_parquet('{pattern}')
    WHERE {RULE}
    GROUP BY 1
    ORDER BY 1
""")

for r in freshness:
    print(r, flush=True)

print("\n[START] Top market categories", flush=True)

categories = query(f"""
    SELECT
        COALESCE(category, 'Unknown') AS category,
        COUNT(*) AS observations,
        COUNT(DISTINCT condition_id) AS markets,
        ROUND(AVG({GAP}), 4) AS mean_abs_gap
    FROM read_parquet('{pattern}')
    WHERE {RULE}
    GROUP BY 1
    ORDER BY observations DESC
    LIMIT 12
""")

for r in categories:
    print(r, flush=True)

totals = {
    key: sum(row[key] for row in monthly)
    for key in (
        "observations",
        "probability_matched",
        "within_12h",
        "valid_3h",
        "valid_12h",
    )
}

report = {
    "generated_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "sample_rule": "12h + valid label + preclose 1h",
    "totals": totals,
    "coverage": coverage,
    "freshness": freshness,
    "categories": categories,
    "monthly": monthly,
    "notes": [
        "No additional raw data extraction",
        "First taker BUY per wallet-market-month",
        "Preclose is a practical proxy, not verified outcome-publication time",
        "Execution prices are not independent market benchmarks",
        "Candidate markets are acquisition-prioritized"
    ],
}

dest = out / "P1_SAMPLE_12H_REPORT.json"
dest.write_text(
    json.dumps(report, indent=2, ensure_ascii=False)
)

print("\n=== FINAL 12H SAMPLE SUMMARY ===", flush=True)
for k, v in totals.items():
    print(f"{k}: {v:,}", flush=True)

print("distinct_wallets:", coverage["wallets"])
print("distinct_markets:", coverage["markets"])
print("Report:", dest)
print("STATUS: COMPLETE", flush=True)

con.close()
