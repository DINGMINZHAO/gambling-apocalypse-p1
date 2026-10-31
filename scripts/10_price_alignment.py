import json
import time
from pathlib import Path

import duckdb
import requests

SOURCE = "data/processed/pilot_labeled_trades.parquet"
OUT = Path("data/processed/price_audit")
OUT.mkdir(parents=True, exist_ok=True)

HISTORY = OUT / "historical_prices.jsonl"
MATCHED = OUT / "matched_trades.parquet"

API = "https://data-api.polymarket.com/v2/prices-history"

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")

# Select tokens across different activity levels.
tokens = con.execute(f"""
WITH activity AS (
    SELECT
        condition_id,
        token_id,
        COUNT(*) AS n
    FROM read_parquet('{SOURCE}')
    WHERE token_won IS NOT NULL
      AND price BETWEEN 0.05 AND 0.95
    GROUP BY 1, 2
    HAVING COUNT(*) >= 30
),
one_per_market AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY condition_id
            ORDER BY n DESC, token_id
        ) AS rn
    FROM activity
),
tiered AS (
    SELECT *,
        NTILE(3) OVER (ORDER BY n) AS tier
    FROM one_per_market
    WHERE rn = 1
)
SELECT token_id, n
FROM tiered
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY tier
    ORDER BY HASH(token_id)
) <= 8
""").fetchall()

print("=== DOWNLOADING PRICE HISTORY ===", flush=True)

session = requests.Session()
successful = 0
points_saved = 0

def request(params):
    for attempt in range(4):
        try:
            r = session.get(API, params=params, timeout=45)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 12))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError("Request failed")

with HISTORY.open("w") as f:
    for idx, (token_id, n) in enumerate(tokens, 1):
        try:
            params = {
                "token_id": token_id,
                "interval": "max",
                "bucket_seconds": 10800,
                "limit": 10000,
            }

            rows = 0

            for page in range(20):
                response = request(params)

                for p in response.get("data", []):
                    width = int(p.get("resolution_seconds") or 0)
                    if width != 10800:
                        continue

                    price = float(p["price"])
                    if not 0 <= price <= 1:
                        continue

                    f.write(json.dumps({
                        "token_id": token_id,
                        "available_ts": int(p["timestamp"]) + width,
                        "market_price": price,
                    }) + "\n")
                    rows += 1

                cursor = (
                    response.get("pagination") or {}
                ).get("next_cursor")

                if not cursor:
                    break

                params = {"cursor": cursor}

            if rows:
                successful += 1
                points_saved += rows

            print(
                f"[{idx}/{len(tokens)}] "
                f"Fills={n} | Price points={rows}",
                flush=True
            )

        except Exception as e:
            print(
                f"[{idx}] Failed: {str(e)[:150]}",
                flush=True
            )

        time.sleep(0.2)

if not points_saved:
    raise RuntimeError("No historical price points retrieved")

print("\n=== MATCHING PRE-TRADE PRICES ===", flush=True)

con.execute(f"""
COPY (
    WITH trade_sample AS (
        SELECT
            wallet,
            condition_id,
            token_id,
            timestamp,
            side,
            price AS trade_price,
            token_won
        FROM read_parquet('{SOURCE}')
        WHERE token_id IN (
            SELECT DISTINCT token_id
            FROM read_json_auto('{HISTORY}')
        )
    ),
    prices AS (
        SELECT
            token_id,
            CAST(available_ts AS BIGINT) AS available_ts,
            market_price
        FROM read_json_auto('{HISTORY}')
    )
    SELECT
        t.*,
        p.market_price AS p_before,
        p.available_ts AS price_available_ts,
        t.timestamp - p.available_ts AS price_age_seconds
    FROM trade_sample t
    ASOF LEFT JOIN prices p
        ON t.token_id = p.token_id
       AND t.timestamp > p.available_ts
)
TO '{MATCHED}'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

result = con.execute(f"""
SELECT
    COUNT(*) AS fills,
    COUNT(p_before) AS matched,
    COUNT_IF(price_age_seconds <= 3600) AS within_1h,
    COUNT_IF(price_age_seconds <= 10800) AS within_3h,
    COUNT_IF(price_age_seconds <= 43200) AS within_12h,
    ROUND(MEDIAN(price_age_seconds) / 3600, 2)
        AS median_age_hours,
    ROUND(AVG(ABS(trade_price - p_before)), 4)
        AS mean_abs_price_difference
FROM read_parquet('{MATCHED}')
""")

print("\n=== PRICE QUALITY SUMMARY ===")

for name, value in zip(
    [d[0] for d in result.description],
    result.fetchone()
):
    print(f"{name}: {value}")

print("\nTokens tested:", len(tokens))
print("Tokens with price history:", successful)
print("Historical price points:", points_saved)
print("Saved:", MATCHED)

con.close()
