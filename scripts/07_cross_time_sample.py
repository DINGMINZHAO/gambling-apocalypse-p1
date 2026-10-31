from pathlib import Path
import duckdb

OUT = Path("data/processed/cross_time")
OUT.mkdir(parents=True, exist_ok=True)

BASE = (
    "hf://datasets/TimeSeventeen/"
    "Polymarket-v1/daily_aligned"
)

DATES = [
    "2025_01_15",
    "2025_03_15",
    "2025_05_15",
    "2025_07_15",
    "2025_09_15",
    "2025_11_15",
]

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")
con.execute("SET enable_progress_bar=true")
con.execute("LOAD httpfs")

print("\n=== CROSS-TIME EXTRACTION ===")

files = []

for day in DATES:
    source = f"{BASE}/{day}.parquet"
    output = OUT / f"{day}.parquet"

    print(f"\nProcessing {day}...", flush=True)

    try:
        con.execute(f"""
        COPY (
            SELECT
                '{day}' AS sample_day,
                LOWER(taker) AS wallet,
                LOWER(maker) AS maker,
                condition_id,
                block_timestamp,
                price,
                p_event,
                D AS direction,
                taker_direction,
                usdc_amount,
                category,
                category_refined,
                outcome_label,
                winning_outcome_label,
                resolution_status,
                close_at,
                resolved_at,
                market_slug,
                (
                    close_at IS NOT NULL
                    AND block_timestamp < epoch(close_at)
                ) AS pre_close,
                (
                    resolved_at IS NOT NULL
                    AND block_timestamp < epoch(resolved_at)
                ) AS pre_resolution
            FROM read_parquet('{source}')
            WHERE resolution_status = 'resolved'
              AND taker IS NOT NULL
              AND condition_id IS NOT NULL
        )
        TO '{output}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        files.append(str(output))

        row = con.execute(f"""
            SELECT
                COUNT(*),
                COUNT(DISTINCT wallet),
                COUNT(DISTINCT condition_id),
                COUNT_IF(pre_close)
            FROM read_parquet('{output}')
        """).fetchone()

        print(
            f"Fills: {row[0]:,} | "
            f"Wallets: {row[1]:,} | "
            f"Markets: {row[2]:,} | "
            f"Pre-close: {row[3]:,}"
        )

    except Exception as exc:
        print(f"Failed on {day}: {str(exc)[:400]}")

if not files:
    raise RuntimeError("No cross-time files retrieved")

file_list = "[" + ",".join(repr(f) for f in files) + "]"
con.execute(
    f"CREATE VIEW trades AS SELECT * FROM read_parquet({file_list})"
)

print("\n=== CROSS-TIME SUMMARY ===")

query = """
WITH wallet_activity AS (
    SELECT
        wallet,
        COUNT(DISTINCT sample_day) AS active_days,
        COUNT(DISTINCT condition_id) AS markets
    FROM trades
    WHERE pre_close = TRUE
    GROUP BY wallet
)
SELECT
    COUNT(*) AS wallets,
    COUNT_IF(active_days >= 2) AS multi_day_wallets,
    COUNT_IF(active_days >= 3) AS three_day_wallets,
    COUNT_IF(markets >= 5) AS five_market_wallets,
    COUNT_IF(markets >= 20) AS twenty_market_wallets
FROM wallet_activity
"""

result = con.execute(query)
names = [x[0] for x in result.description]
values = result.fetchone()

for name, value in zip(names, values):
    print(f"{name}: {value:,}")

print("\n=== MARKET CATEGORIES ===")

for category, n in con.execute("""
    SELECT category_refined, COUNT(DISTINCT condition_id)
    FROM trades
    WHERE pre_close = TRUE
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 10
""").fetchall():
    print(f"{str(category):25} {n:,}")

print("\n=== TIME QUALITY ===")

quality = con.execute("""
    SELECT
        COUNT(*) AS fills,
        COUNT_IF(pre_close) AS pre_close,
        COUNT_IF(pre_resolution) AS pre_resolution,
        COUNT_IF(resolved_at IS NULL) AS missing_resolution_time,
        COUNT_IF(wallet = maker) AS same_address_fills
    FROM trades
""").fetchone()

for name, value in zip(
    ["Total fills", "Pre-close",
     "Pre-resolution", "Missing resolution time",
     "Same-address fills"],
    quality
):
    print(f"{name}: {value:,}")

print("\nCompleted.")
con.close()
