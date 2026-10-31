from pathlib import Path
import duckdb

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

TRADES = "data/processed/pilot_maker_trades.parquet"
MAPPING = "data/processed/pilot_token_market_map.parquet"
RESULT = "data/processed/pilot_labeled_trades.parquet"

BASE = "hf://datasets/moose-code/polymarket-onchain-v1"
CONDITIONS = f"{BASE}/condition.parquet"
MARKET_DATA = f"{BASE}/market_data.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")
con.execute("LOAD httpfs")

print("\n=== JOINING RESOLUTION DATA ===")

con.execute(f"""
COPY (
    WITH condition_labels AS (
        SELECT
            id AS condition_id,
            CASE
                WHEN array_length(positionIds) = 2
                 AND array_length(payoutNumerators) = 2
                 AND payoutDenominator <> '0'
                 AND payoutNumerators[1] = payoutDenominator
                 AND payoutNumerators[2] = '0'
                THEN positionIds[1]

                WHEN array_length(positionIds) = 2
                 AND array_length(payoutNumerators) = 2
                 AND payoutDenominator <> '0'
                 AND payoutNumerators[2] = payoutDenominator
                 AND payoutNumerators[1] = '0'
                THEN positionIds[2]

                ELSE NULL
            END AS winning_token_id
        FROM read_parquet('{CONDITIONS}')
    )
    SELECT
        t.*,
        m.condition_id,
        CASE
            WHEN c.winning_token_id IS NULL
            THEN NULL
            ELSE t.token_id = c.winning_token_id
        END AS token_won
    FROM read_parquet('{TRADES}') t
    LEFT JOIN read_parquet('{MAPPING}') m
        ON t.token_id = m.token_id
    LEFT JOIN condition_labels c
        ON m.condition_id = c.condition_id
)
TO '{RESULT}'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n=== RESOLUTION SUMMARY ===")

summary = con.execute(f"""
SELECT
    COUNT(*) AS fills,
    COUNT(DISTINCT wallet) AS wallets,
    COUNT(DISTINCT condition_id) AS markets,
    COUNT(DISTINCT CASE
        WHEN token_won IS NOT NULL THEN condition_id
    END) AS resolved_markets,
    COUNT_IF(token_won IS NOT NULL) AS labeled_fills,
    COUNT_IF(token_won = TRUE) AS winning_token_fills,
    COUNT_IF(token_won = FALSE) AS losing_token_fills
FROM read_parquet('{RESULT}')
""")

values = summary.fetchone()
names = [c[0] for c in summary.description]

for name, value in zip(names, values):
    print(f"{name:25}: {value:,}")

print("\n=== REPEAT WALLET COVERAGE ===")

repeat = con.execute(f"""
WITH wallet_markets AS (
    SELECT
        wallet,
        COUNT(DISTINCT condition_id) AS n_markets
    FROM read_parquet('{RESULT}')
    WHERE token_won IS NOT NULL
    GROUP BY wallet
)
SELECT
    COUNT(*) AS wallets,
    COUNT_IF(n_markets >= 2) AS wallets_2plus,
    COUNT_IF(n_markets >= 5) AS wallets_5plus,
    COUNT_IF(n_markets >= 10) AS wallets_10plus
FROM wallet_markets
""").fetchone()

for name, value in zip(
    ["Labeled wallets", "2+ markets",
     "5+ markets", "10+ markets"],
    repeat
):
    print(f"{name:25}: {value:,}")

print("\n=== MARKET METADATA SCHEMA ===")

try:
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{MARKET_DATA}')"
    ).fetchall()

    for field in schema:
        print(f"{field[0]:30} {field[1]}")
except Exception as exc:
    print("Metadata inspection failed:", str(exc)[:300])

print("\nSaved:", RESULT)
print("File size MB:", round(
    Path(RESULT).stat().st_size / 1024**2, 2
))

con.close()
print("\nCompleted.")
