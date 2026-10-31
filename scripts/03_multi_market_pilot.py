import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

import requests

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"

TARGET_MARKETS = 8
MAX_DISCOVERY_PAGES = 10
MAX_TRADE_PAGES = 3

session = requests.Session()


def fetch(base, endpoint, **params):
    for attempt in range(4):
        try:
            r = session.get(
                base + endpoint,
                params=params,
                timeout=40
            )
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("API request failed")


def unpack(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return value or []


def volume(m):
    return float(m.get("volumeNum") or m.get("volume") or 0)


def save_json(name, obj):
    (OUT / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ==========================================
# 1. Market Discovery
# ==========================================

print("\n=== MARKET DISCOVERY ===")

stats = Counter()
candidates = []
seen = set()

for page in range(MAX_DISCOVERY_PAGES):
    markets = fetch(
        GAMMA,
        "/markets",
        closed="true",
        order="id",
        ascending="false",
        limit=100,
        offset=page * 100
    )

    if not markets:
        break

    for m in markets:
        stats["total"] += 1

        outcomes = unpack(m.get("outcomes"))
        tokens = unpack(m.get("clobTokenIds"))
        condition = m.get("conditionId")

        if len(outcomes) == 2:
            stats["binary"] += 1
        else:
            continue

        if len(tokens) == 2:
            stats["with_tokens"] += 1

        if 5000 <= volume(m) <= 100000:
            stats["moderate_volume"] += 1

        if not condition or condition in seen:
            continue

        seen.add(condition)
        candidates.append(m)

    print(
        f"Scanned: {stats['total']} | "
        f"Binary: {stats['binary']} | "
        f"Candidates: {len(candidates)}"
    )

    if len(candidates) >= 100:
        break

print("\n=== DISCOVERY DIAGNOSTICS ===")
print("Total markets:", stats["total"])
print("Binary markets:", stats["binary"])
print("Binary with token IDs:", stats["with_tokens"])
print("Binary with moderate volume:", stats["moderate_volume"])

if not candidates:
    print("Example raw markets:")
    for m in markets[:3]:
        print({
            "question": m.get("question"),
            "outcomes": m.get("outcomes"),
            "volumeNum": m.get("volumeNum"),
            "closed": m.get("closed"),
        })
    raise SystemExit("No binary markets found.")


# ==========================================
# 2. Select verified resolved markets
# ==========================================

# Prefer moderate volume and available price tokens
candidates.sort(
    key=lambda m: (
        not (5000 <= volume(m) <= 100000),
        len(unpack(m.get("clobTokenIds"))) != 2,
        abs(volume(m) - 30000)
    )
)

selected = []
resolutions = []
seen_events = set()

print("\n=== RESOLUTION VERIFICATION ===")

for m in candidates:
    if len(selected) >= TARGET_MARKETS:
        break

    condition = m["conditionId"]

    events = m.get("events") or []
    event_id = (
        events[0].get("id")
        if events and isinstance(events[0], dict)
        else None
    )
    event_key = (
        event_id
        or m.get("eventSlug")
        or m.get("slug")
        or condition
    )

    if event_key in seen_events:
        continue

    try:
        result = fetch(
            DATA,
            "/v2/resolutions",
            condition=condition
        )
    except requests.RequestException:
        continue

    records = result.get("data") or []
    resolved = [
        x for x in records
        if x.get("status") == "resolved"
    ]

    if not resolved:
        continue

    selected.append(m)
    resolutions.extend(resolved)
    seen_events.add(event_key)

    print(
        f"[{len(selected)}] "
        f"{m.get('question', '')[:75]} "
        f"| Volume: ${volume(m):,.0f}"
    )

save_json("pilot_markets.json", selected)
save_json("pilot_resolutions.json", resolutions)

if not selected:
    raise SystemExit(
        "No verified resolved markets found. "
        "Review discovery diagnostics."
    )


# ==========================================
# 3. Download trades
# ==========================================

print("\n=== DOWNLOADING TRADES ===")

all_trades = []
wallet_markets = defaultdict(set)

for index, market in enumerate(selected, 1):
    condition = market["conditionId"]
    cursor = None
    rows = []

    end_date = market.get("endDate")
    end_ts = (
        datetime.fromisoformat(
            end_date.replace("Z", "+00:00")
        ).timestamp()
        if end_date else None
    )

    for page in range(MAX_TRADE_PAGES):
        params = {
            "condition": condition,
            "limit": 1000,
            "taker_only": "false"
        }

        if cursor:
            params["cursor"] = cursor

        response = fetch(
            DATA,
            "/v2/trades",
            **params
        )

        batch = response.get("data") or []
        rows.extend(batch)

        pagination = response.get("pagination") or {}
        next_cursor = pagination.get("next_cursor")

        if (
            not batch
            or not pagination.get("has_more")
            or not next_cursor
            or next_cursor == cursor
        ):
            break

        cursor = next_cursor

    before_end = 0

    for t in rows:
        wallet = t.get("proxy_wallet")
        ts = int(t["timestamp"])

        t["sample_market_id"] = condition
        t["before_scheduled_end"] = (
            ts < end_ts if end_ts is not None else None
        )

        if t["before_scheduled_end"] is True:
            before_end += 1

        if wallet:
            wallet_markets[wallet].add(condition)

    all_trades.extend(rows)

    print(
        f"[{index}/{len(selected)}] "
        f"Trades: {len(rows):,} | "
        f"Before scheduled end: {before_end:,}"
    )

with (OUT / "pilot_trades.jsonl").open(
    "w", encoding="utf-8"
) as f:
    for t in all_trades:
        f.write(json.dumps(t) + "\n")


# ==========================================
# 4. Summary
# ==========================================

repeat_wallets = sum(
    len(markets) >= 2
    for markets in wallet_markets.values()
)

print("\n=== PILOT SUMMARY ===")
print("Markets:", len(selected))
print("Trades:", len(all_trades))
print("Unique wallets:", len(wallet_markets))
print("Wallets in 2+ markets:", repeat_wallets)
print(
    "Trades before scheduled end:",
    sum(
        t["before_scheduled_end"] is True
        for t in all_trades
    )
)

print("\nSaved:")
print("data/raw/pilot_markets.json")
print("data/raw/pilot_resolutions.json")
print("data/raw/pilot_trades.jsonl")

print("\nPilot completed.")
