import json
import time
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import requests

SOURCE = "data/processed/pilot_labeled_trades.parquet"
OUTPUT = Path("data/raw/market_probability_audit.json")
API = "https://data-api.polymarket.com/v2/prices-history"

con = duckdb.connect()

# Select active tokens from distinct markets.
markets = con.execute(f"""
WITH activity AS (
    SELECT
        condition_id,
        token_id,
        COUNT(*) AS fills,
        CAST(MEDIAN(CAST(timestamp AS BIGINT)) AS BIGINT)
            AS trade_ts
    FROM read_parquet('{SOURCE}')
    WHERE token_won IS NOT NULL
      AND price BETWEEN 0.05 AND 0.95
    GROUP BY 1, 2
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY condition_id
            ORDER BY fills DESC
        ) AS rn
    FROM activity
)
SELECT condition_id, token_id, fills, trade_ts
FROM ranked
WHERE rn = 1
ORDER BY fills DESC
LIMIT 4
""").fetchall()
con.close()


def utc(ts):
    return datetime.fromtimestamp(
        ts, timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def fetch(params):
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, timeout=35)
            if r.status_code in (429, 503):
                time.sleep(float(
                    r.headers.get("Retry-After") or 2 ** attempt
                ))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == 2:
                print("API error:", str(exc)[:250])
                return None
            time.sleep(2 ** attempt)
    return None


def point_info(p, cutoff):
    if not p:
        return None

    ts = int(p["timestamp"])
    width = int(p.get("resolution_seconds") or 0)

    # Exclude zero-width terminal settlement points.
    # Aggregate buckets must finish before the cutoff.
    safe = width > 0 and ts + width <= cutoff

    return {
        "timestamp": ts,
        "price": p.get("price"),
        "resolution_seconds": width,
        "bucket_completed_before_cutoff": safe,
        "age_seconds": cutoff - (ts + width)
            if safe else None,
    }


results = []

print("\n=== HISTORICAL PROBABILITY AUDIT ===")

for idx, (condition, token, fills, trade_ts) in enumerate(
    markets, 1
):
    cutoff = trade_ts - 60

    print(f"\n[{idx}/{len(markets)}]")
    print("Market:", condition[:22] + "...")
    print("Token:", token[:24] + "...")
    print("Sample fills:", fills)
    print("Trade time:", utc(trade_ts))

    # Point-in-time query
    asof = fetch({
        "token_id": token,
        "as_of": cutoff,
    })

    asof_points = (
        asof.get("data") or []
        if isinstance(asof, dict) else []
    )
    asof_info = point_info(
        asof_points[-1] if asof_points else None,
        cutoff,
    )

    # Historical 3-hour buckets
    history = fetch({
        "token_id": token,
        "start": cutoff - 2 * 86400,
        "end": cutoff + 1,
        "bucket_seconds": 10800,
        "limit": 300,
    })

    points = (
        history.get("data") or []
        if isinstance(history, dict) else []
    )

    usable = []
    for p in points:
        info = point_info(p, cutoff)
        if info and info["bucket_completed_before_cutoff"]:
            usable.append(info)

    usable.sort(key=lambda x: x["timestamp"])
    last_safe = usable[-1] if usable else None

    print("As-of response:", asof_info)
    print("Historical points:", len(points))
    print("Completed pre-trade buckets:", len(usable))
    print("Last safe historical point:", last_safe)

    results.append({
        "condition_id": condition,
        "token_id": token,
        "trade_ts": trade_ts,
        "cutoff": cutoff,
        "asof": asof_info,
        "history_points": len(points),
        "safe_points": len(usable),
        "last_safe_point": last_safe,
    })

    time.sleep(0.3)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print("\n=== SUMMARY ===")
print("Markets tested:", len(results))
print(
    "Markets with safe historical points:",
    sum(x["last_safe_point"] is not None for x in results),
)
print("Saved:", OUTPUT)
