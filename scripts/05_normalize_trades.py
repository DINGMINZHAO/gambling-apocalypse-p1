from pathlib import Path
import duckdb

RAW = "data/raw/onchain_sample_2025.parquet"
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

TRADES = str(OUT / "pilot_maker_trades.parquet")
MAPPING = str(OUT / "pilot_token_market_map.parquet")

CONDITIONS = (
    "hf://datasets/moose-code/"
    "polymarket-onchain-v1/condition.parquet"
)

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")
con.execute("SET enable_progress_bar=true")
con.execute("LOAD httpfs")

# 1. Normalize maker-side fills
print("\n=== NORMALIZING TRADES ===")

con.execute(f"""
COPY (
    WITH base AS (
        SELECT
            id AS fill_id,
            LOWER(maker) AS wallet,
            transactionHash AS tx_hash,
            CAST(timestamp AS BIGINT) AS timestamp,
            CASE
                WHEN makerAssetId = '0'
                THEN 'BUY'
                ELSE 'SELL'
            END AS side,
            CASE
                WHEN makerAssetId = '0'
                THEN takerAssetId
                ELSE makerAssetId
            END AS token_id,
            CASE
                WHEN makerAssetId = '0'
                THEN TRY_CAST(makerAmountFilled AS DOUBLE)
                ELSE TRY_CAST(takerAmountFilled AS DOUBLE)
            END / 1000000.0 AS usdc,
            CASE
                WHEN makerAssetId = '0'
                THEN TRY_CAST(takerAmountFilled AS DOUBLE)
                ELSE TRY_CAST(makerAmountFilled AS DOUBLE)
            END / 1000000.0 AS shares
        FROM read_parquet('{RAW}')
        WHERE makerAssetId = '0'
           OR takerAssetId = '0'
    )
    SELECT
        *,
        usdc / NULLIF(shares, 0) AS price
    FROM base
    WHERE shares > 0
      AND usdc > 0
)
TO '{TRADES}'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

# 2. Match token IDs to condition IDs
print("\n=== MATCHING MARKETS ===")

con.execute(f"""
COPY (
    WITH tokens AS (
        SELECT DISTINCT token_id
        FROM read_parquet('{TRADES}')
    ),
    conditions AS (
        SELECT
            id AS condition_id,
            UNNEST(positionIds) AS token_id
        FROM read_parquet('{CONDITIONS}')
    )
    SELECT DISTINCT
        t.token_id,
        c.condition_id
    FROM tokens t
    JOIN conditions c USING (token_id)
)
TO '{MAPPING}'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

# 3. Print summary
print("\n=== NORMALIZED DATA SUMMARY ===")

result = con.execute(f"""
SELECT
    COUNT(*) AS fills,
    COUNT(DISTINCT wallet) AS wallets,
    COUNT(DISTINCT token_id) AS tokens,
    COUNT_IF(side = 'BUY') AS buys,
    COUNT_IF(side = 'SELL') AS sells,
    COUNT_IF(price > 0 AND price <= 1) AS valid_prices
FROM read_parquet('{TRADES}')
""").fetchone()

for k, v in zip(
    ["Fills", "Maker wallets", "Tokens",
     "BUY fills", "SELL fills", "Prices in (0,1]"],
    result
):
    print(f"{k}: {v:,}")

result = con.execute(f"""
SELECT
    COUNT(DISTINCT token_id),
    COUNT(DISTINCT condition_id)
FROM read_parquet('{MAPPING}')
""").fetchone()

print("\nMatched tokens:", result[0])
print("Matched markets:", result[1])

print("\n=== FIRST NORMALIZED TRADE ===")

cursor = con.execute(f"""
SELECT wallet, side, token_id, usdc, shares, price
FROM read_parquet('{TRADES}')
LIMIT 1
""")

row = cursor.fetchone()
for (name, *_), value in zip(cursor.description, row):
    print(f"{name}: {value}")

print("\nSaved:")
print(TRADES)
print(MAPPING)

con.close()
