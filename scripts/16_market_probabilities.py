import argparse
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import requests

API = "https://data-api.polymarket.com/v2/prices-history"

START_TS = int(datetime(
    2025, 1, 1, tzinfo=timezone.utc
).timestamp())

END_TS = int(datetime(
    2026, 4, 1, tzinfo=timezone.utc
).timestamp())

SCHEMA = pa.schema([
    ("condition_id", pa.string()),
    ("token_id", pa.string()),
    ("market_type", pa.string()),
    ("bucket_start", pa.int64()),
    ("available_ts", pa.int64()),
    ("p_yes", pa.float32()),
    ("resolution_seconds", pa.int32()),
])

thread_local = threading.local()
rate_lock = threading.Lock()
last_request = [0.0]


def fixed_rate():
    # Shared request limiter, approximately 6 requests/sec.
    with rate_lock:
        delay = last_request[0] + 0.16 - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        last_request[0] = time.monotonic()


def fetch_market(row):
    condition, token, market_type, month, fills = row

    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()

    session = thread_local.session

    params = {
        "token_id": token,
        "interval": "max",
        "bucket_seconds": 10800,
        "limit": 10000,
    }

    saved = []
    seen = set()
    requests_count = 0

    for _ in range(30):
        response = None

        for attempt in range(7):
            fixed_rate()

            try:
                r = session.get(
                    API, params=params, timeout=60
                )
                requests_count += 1

                if r.status_code in (
                    429, 500, 502, 503, 504
                ):
                    wait = r.headers.get("Retry-After")

                    try:
                        wait = min(60, float(wait))
                    except (TypeError, ValueError):
                        wait = min(30, 2 ** attempt)

                    time.sleep(wait)
                    continue

                if r.status_code in (400, 404):
                    return (
                        saved,
                        requests_count,
                        f"HTTP_{r.status_code}",
                    )

                r.raise_for_status()
                response = r.json()
                break

            except (requests.RequestException, ValueError):
                time.sleep(min(30, 2 ** attempt))

        if response is None:
            raise RuntimeError(
                f"API retries exhausted: {condition}"
            )

        for p in response.get("data") or []:
            try:
                width = int(
                    p.get("resolution_seconds") or 0
                )
                ts = int(p["timestamp"])
                price = float(p["price"])
            except (TypeError, ValueError, KeyError):
                continue

            # Historical 3-hour buckets only.
            # Exclude terminal zero-width observations.
            if width != 10800 or not (0 <= price <= 1):
                continue

            available = ts + width

            if START_TS - 10800 <= available <= END_TS:
                saved.append((
                    condition,
                    token,
                    market_type,
                    ts,
                    available,
                    price,
                    width,
                ))

        pagination = response.get("pagination") or {}
        cursor = pagination.get("next_cursor")

        if not pagination.get("has_more") or not cursor:
            return (
                saved,
                requests_count,
                "OK" if saved else "NO_3H_POINTS",
            )

        if cursor in seen:
            raise RuntimeError(
                f"Repeated pagination cursor: {condition}"
            )

        seen.add(cursor)
        params = {"cursor": cursor}

    raise RuntimeError(
        f"Excessive pagination: {condition}"
    )


def atomic_json(path, obj):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2)
    )
    os.replace(tmp, path)


def copy_atomic(local, target):
    tmp = target.with_name(target.name + ".partial")
    shutil.copyfile(local, tmp)

    if tmp.stat().st_size != local.stat().st_size:
        raise IOError("Incomplete Drive write")

    os.replace(tmp, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument(
        "--per-month-type", type=int, default=800
    )
    parser.add_argument(
        "--max-output-gb", type=float, default=1.5
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()

    data = Path(args.drive_root) / "data"
    trades = (
        data / "research_raw" / "trades" / "*" / "*.parquet"
    )
    registry = (
        data / "research_index" / "market_registry.parquet"
    )

    out = data / "market_probabilities"
    out.mkdir(parents=True, exist_ok=True)

    activity = out / "market_activity.parquet"
    candidates_path = out / "candidates.parquet"
    manifest_file = out / "PROBABILITY_RUN_MANIFEST.json"

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=2")
    con.execute(
        "SET temp_directory='/content/prob_duckdb_tmp'"
    )

    # Compute market activity once.
    # No duplicate raw trade table is written.
    if not activity.exists():
        print(
            "=== BUILDING MARKET ACTIVITY ===",
            flush=True
        )

        con.execute(f"""
        COPY (
            SELECT
                market_type,
                condition_id,
                COUNT(*) AS fills,
                MIN(block_timestamp) AS first_ts,
                MAX(block_timestamp) AS last_ts
            FROM read_parquet(
                '{trades}', union_by_name=true
            )
            WHERE condition_id IS NOT NULL
            GROUP BY 1, 2
        )
        TO '{activity}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        print("Market activity saved.", flush=True)

    # Select active markets with a known final outcome.
    # These filters prioritize API acquisition only.
    if not candidates_path.exists():
        print(
            "=== SELECTING PROBABILITY CANDIDATES ===",
            flush=True
        )

        con.execute(f"""
        COPY (
            WITH eligible AS (
                SELECT
                    a.market_type,
                    a.condition_id,
                    r.token_id,
                    a.fills,
                    a.first_ts,
                    STRFTIME(
                        TO_TIMESTAMP(a.first_ts),
                        '%Y-%m'
                    ) AS month,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            a.market_type,
                            a.condition_id
                        ORDER BY r.token_id
                    ) AS one_token
                FROM read_parquet('{activity}') a
                JOIN read_parquet('{registry}') r
                    ON a.market_type = r.market_type
                   AND a.condition_id = r.condition_id
                WHERE r.outcome_seq = 1
                  AND r.winning_outcome_label IS NOT NULL
                  AND a.fills >= 50
                  AND a.last_ts - a.first_ts >= 21600
            ),
            ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY month, market_type
                        ORDER BY fills DESC, condition_id
                    ) AS rank_in_group
                FROM eligible
                WHERE one_token = 1
            )
            SELECT
                condition_id,
                token_id,
                market_type,
                month,
                fills
            FROM ranked
            WHERE rank_in_group <= {args.per_month_type}
            ORDER BY
                month, market_type,
                fills DESC, condition_id
        )
        TO '{candidates_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    candidates = con.execute(f"""
        SELECT
            condition_id, token_id,
            market_type, month, fills
        FROM read_parquet('{candidates_path}')
        ORDER BY
            month, market_type,
            fills DESC, condition_id
    """).fetchall()

    print(
        f"Probability candidates: {len(candidates):,}",
        flush=True
    )

    if not candidates:
        raise RuntimeError(
            "No eligible markets. Review registry mapping."
        )

    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())

        if (
            manifest["git_commit"] != commit
            or manifest["candidates"] != len(candidates)
        ):
            raise RuntimeError(
                "Run manifest version mismatch. "
                "Resume using the original code version."
            )

    else:
        manifest = {
            "git_commit": commit,
            "candidates": len(candidates),
            "next_index": 0,
            "requested": 0,
            "markets_with_prices": 0,
            "price_points": 0,
            "api_requests": 0,
            "empty_or_invalid": 0,
            "shards": [],
            "source": API,
            "bucket_seconds": 10800,
        }
        atomic_json(manifest_file, manifest)

    batch_size = 50

    with ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(
            manifest["next_index"],
            len(candidates),
            batch_size
        ):
            segment = candidates[start:start + batch_size]
            outcomes = list(
                pool.map(fetch_market, segment)
            )

            records = []
            requests_count = 0
            n_market = 0
            n_empty = 0

            for rows, reqs, status in outcomes:
                records.extend(rows)
                requests_count += reqs
                n_market += int(bool(rows))
                n_empty += int(not rows)

            table = pa.Table.from_pylist(
                [
                    dict(zip(SCHEMA.names, r))
                    for r in records
                ],
                schema=SCHEMA,
            )

            target = out / (
                f"prices_{start:06d}_"
                f"{start + len(segment):06d}.parquet"
            )

            with tempfile.TemporaryDirectory(
                dir="/content"
            ) as td:
                local = Path(td) / "prices.parquet"
                pq.write_table(
                    table, local, compression="zstd"
                )
                copy_atomic(local, target)

            manifest["requested"] += len(segment)
            manifest["markets_with_prices"] += n_market
            manifest["empty_or_invalid"] += n_empty
            manifest["price_points"] += len(records)
            manifest["api_requests"] += requests_count
            manifest["next_index"] = start + len(segment)

            manifest["shards"].append({
                "file": target.name,
                "markets": len(segment),
                "points": len(records),
                "bytes": target.stat().st_size,
            })

            atomic_json(manifest_file, manifest)

            used = sum(
                x["bytes"] for x in manifest["shards"]
            ) / 1024**3

            print(
                f"[SAVED] "
                f"{manifest['next_index']:,}/"
                f"{len(candidates):,} markets | "
                f"Price points: "
                f"{manifest['price_points']:,} | "
                f"{used:.3f} GB",
                flush=True
            )

            if used >= args.max_output_gb:
                print(
                    "STOP: Probability storage budget reached.",
                    flush=True
                )
                break

    print("\n=== PROBABILITY ACQUISITION ===")

    for key in (
        "requested",
        "markets_with_prices",
        "empty_or_invalid",
        "price_points",
        "api_requests",
        "next_index",
    ):
        print(f"{key}: {manifest[key]}")

    print("Output:", out)
    print("Manifest:", manifest_file)
    con.close()


if __name__ == "__main__":
    main()
