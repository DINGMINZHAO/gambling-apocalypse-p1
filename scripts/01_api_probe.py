import json
from pathlib import Path

import requests

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

DATA = "https://data-api.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def fetch(base, endpoint, **params):
    r = requests.get(
        base + endpoint,
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def unpack(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return []
    return x or []


def save(name, obj):
    (OUT / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# 1. Find a resolved binary market
markets = fetch(
    GAMMA, "/markets",
    closed="true",
    order="volumeNum",
    ascending="false",
    limit=100,
)

candidates = []
for m in markets:
    try:
        prices = [float(p) for p in unpack(m.get("outcomePrices"))]
        if sorted(prices) == [0.0, 1.0] and m.get("conditionId"):
            candidates.append(m)
    except (TypeError, ValueError):
        continue

if not candidates:
    raise RuntimeError("No resolved binary market candidate found")

market = next(
    (m for m in candidates if (m.get("endDate") or "") >= "2025-01-01"),
    candidates[0],
)

condition = market["conditionId"]
tokens = unpack(market.get("clobTokenIds"))

save("sample_market.json", market)

print("\n=== MARKET ===")
print("Question:", market.get("question"))
print("Condition:", condition)
print("Outcomes:", unpack(market.get("outcomes")))
print("Final prices:", unpack(market.get("outcomePrices")))
print("End date:", market.get("endDate"))


# 2. Fetch trades, including both maker and taker
response = fetch(
    DATA, "/v2/trades",
    condition=condition,
    limit=100,
    taker_only="false",
)
trades = response.get("data", [])

save("sample_trades.json", response)

print("\n=== TRADES ===")
print("Rows:", len(trades))
print("Unique wallets:", len({
    t.get("proxy_wallet") for t in trades
    if t.get("proxy_wallet")
}))

if trades:
    print("Fields:", list(trades[0].keys()))
    print("First trade:", {
        k: trades[0].get(k)
        for k in [
            "proxy_wallet", "side", "outcome",
            "price", "size", "timestamp",
            "condition_id", "token_id",
        ]
    })


# 3. Check official resolution state
resolution = fetch(
    DATA, "/v2/resolutions",
    condition=condition,
)
save("sample_resolution.json", resolution)

print("\n=== RESOLUTION ===")
print(json.dumps(resolution.get("data"), indent=2)[:1500])


# 4. Check historical price availability
if tokens:
    history = fetch(
        CLOB, "/prices-history",
        market=tokens[0],
        interval="max",
        fidelity=1440,
    )
    save("sample_price_history.json", history)

    print("\n=== PRICE HISTORY ===")
    print("Points:", len(history.get("history", [])))
    print("First points:", history.get("history", [])[:3])

print("\nProbe completed. Raw samples saved in data/raw/")
