import json
from pathlib import Path
from datetime import datetime, timezone
import requests

root = Path("data/raw")
market = json.loads((root / "sample_market.json").read_text())
resolution = json.loads((root / "sample_resolution.json").read_text())

condition = market["conditionId"]
end_date = datetime.fromisoformat(
    market["endDate"].replace("Z", "+00:00")
).timestamp()

resolution_update = int(
    resolution["data"][0]["last_update_timestamp"]
)

r = requests.get(
    "https://data-api.polymarket.com/v2/trades",
    params={
        "condition": condition,
        "limit": 1000,
        "taker_only": "false",
    },
    timeout=60,
)
r.raise_for_status()
result = r.json()
trades = result["data"]

(root / "sample_trades_1000.json").write_text(
    json.dumps(result), encoding="utf-8"
)

times = [int(t["timestamp"]) for t in trades]
wallets = {t["proxy_wallet"] for t in trades}

def utc(t):
    return datetime.fromtimestamp(
        t, timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

print("=== TEMPORAL COVERAGE ===")
print("Trades:", len(trades))
print("Unique wallets:", len(wallets))
print("Earliest trade:", utc(min(times)))
print("Latest trade:", utc(max(times)))
print("Market end date:", utc(end_date))
print("Resolution last update:", utc(resolution_update))

print("\n=== TIMING CHECK ===")
print("Before market end:", sum(t < end_date for t in times))
print("Before resolution update:", sum(
    t < resolution_update for t in times
))
print("Before both:", sum(
    t < min(end_date, resolution_update) for t in times
))

print("\n=== PAGINATION ===")
print("Has more:", result["pagination"]["has_more"])
print("Next cursor available:", bool(
    result["pagination"].get("next_cursor")
))
