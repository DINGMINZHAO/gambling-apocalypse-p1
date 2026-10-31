import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
from huggingface_hub import HfApi

DATASET = "TimeSeventeen/Polymarket-v1"
LAYERS = {
    "standard": "daily_aligned",
    "multi": "daily_aligned_multi",
}
START = date(2025, 1, 1)
END = date(2026, 3, 31)


def commit_hash():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def save_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False)
    )
    os.replace(tmp, path)


def utc_seconds(d):
    return int(datetime(
        d.year, d.month, d.day, tzinfo=timezone.utc
    ).timestamp())


def day_range(start, end):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def copy_to_drive(local, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")

    shutil.copyfile(local, temporary)

    if temporary.stat().st_size != local.stat().st_size:
        raise IOError("Google Drive copy size mismatch")

    os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    args = parser.parse_args()

    drive = Path("/content/drive/MyDrive")
    if not drive.is_dir():
        raise RuntimeError("Mount Google Drive first.")

    root = Path(args.drive_root)
    root.mkdir(parents=True, exist_ok=True)

    output = root / "data" / "research_raw"
    output.mkdir(parents=True, exist_ok=True)

    manifest_file = output / "RUN_MANIFEST.json"
    commit = commit_hash()
    api = HfApi()

    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())

        if manifest["git_commit"] != commit:
            raise RuntimeError(
                "Git commit mismatch. Resume with commit: "
                + manifest["git_commit"]
            )

        revision = manifest["dataset_revision"]

        if (
            manifest["window_start"] != START.isoformat()
            or manifest["window_end"] != END.isoformat()
        ):
            raise RuntimeError("Research window mismatch")

    else:
        revision = api.repo_info(
            repo_id=DATASET,
            repo_type="dataset"
        ).sha

        manifest = {
            "project": "Gambling Apocalypse Part I",
            "git_commit": commit,
            "dataset": DATASET,
            "dataset_revision": revision,
            "window_start": START.isoformat(),
            "window_end": END.isoformat(),
            "source_timezone": "UTC+08:00",
            "research_timezone": "UTC",
            "schema_version": "research_raw_v1",
            "partitions": {},
        }

        save_json(manifest_file, manifest)

    print("Git commit:", commit, flush=True)
    print("Dataset revision:", revision, flush=True)

    source_files = set(api.list_repo_files(
        repo_id=DATASET,
        repo_type="dataset",
        revision=revision
    ))

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=2")
    con.execute(
        "SET temp_directory='/content/duckdb_tmp'"
    )
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    start_ts = utc_seconds(START)
    end_ts = utc_seconds(END + timedelta(days=1))

    # Source files use UTC+8 dates.
    # Include the extra final boundary date.
    last_source_day = END + timedelta(days=1)

    failures = 0

    for day in day_range(START, last_source_day):
        label = day.strftime("%Y_%m_%d")

        for kind, folder in LAYERS.items():
            key = f"{kind}/{label}"
            source_path = f"{folder}/{label}.parquet"

            fact_path = (
                output / "trades" / kind / f"{label}.parquet"
            )
            meta_path = (
                output / "market_metadata" / kind
                / f"{label}.parquet"
            )

            old = manifest["partitions"].get(key, {})

            if (
                old.get("status") == "completed"
                and fact_path.exists()
                and meta_path.exists()
                and fact_path.stat().st_size == old["fact_bytes"]
                and meta_path.stat().st_size == old["meta_bytes"]
            ):
                continue

            if source_path not in source_files:
                manifest["partitions"][key] = {
                    "status": "source_missing",
                    "source": source_path,
                }
                save_json(manifest_file, manifest)
                continue

            source = (
                f"hf://datasets/{DATASET}@{revision}/"
                f"{source_path}"
            )

            print(f"[START] {key}", flush=True)
            error = None

            for attempt in range(3):
                try:
                    with tempfile.TemporaryDirectory(
                        dir="/content"
                    ) as tmp:
                        tmp = Path(tmp)

                        stage = tmp / "stage.parquet"
                        facts = tmp / "trades.parquet"
                        metadata = tmp / "metadata.parquet"

                        if kind == "standard":
                            parent_id = (
                                "CAST(NULL AS VARCHAR)"
                            )
                        else:
                            parent_id = (
                                "CAST(neg_risk_market_id "
                                "AS VARCHAR)"
                            )

                        # One remote scan, retaining the
                        # fields required for later research.
                        con.execute(f"""
                        COPY (
                            SELECT
                                block_timestamp,
                                LOWER(maker) AS maker,
                                LOWER(taker) AS taker,
                                taker_direction,
                                asset_id,
                                condition_id,
                                outcome_seq,
                                price,
                                p_event,
                                D,
                                usdc_amount,
                                fee_usdc,
                                category,
                                category_refined,
                                outcome_label,
                                winning_outcome_label,
                                resolution_status,
                                opens_at,
                                close_at,
                                resolved_at,
                                market_slug,
                                taker_base_fee,
                                maker_base_fee,
                                {parent_id}
                                    AS neg_risk_market_id,
                                '{kind}' AS market_type
                            FROM read_parquet('{source}')
                            WHERE block_timestamp >= {start_ts}
                              AND block_timestamp < {end_ts}
                        )
                        TO '{stage}'
                        (FORMAT PARQUET, COMPRESSION ZSTD)
                        """)

                        # Transaction-level behavior.
                        # Preserve BOTH maker and taker.
                        con.execute(f"""
                        COPY (
                            SELECT
                                market_type,
                                block_timestamp,
                                maker,
                                taker,
                                taker_direction,
                                condition_id,
                                asset_id,
                                outcome_seq,
                                price,
                                p_event,
                                D,
                                usdc_amount,
                                fee_usdc
                            FROM read_parquet('{stage}')
                        )
                        TO '{facts}'
                        (FORMAT PARQUET, COMPRESSION ZSTD)
                        """)

                        # Repeated market metadata stored
                        # separately instead of every fill.
                        # Outcome-level metadata is retained.
                        con.execute(f"""
                        COPY (
                            SELECT DISTINCT
                                market_type,
                                condition_id,
                                asset_id,
                                outcome_seq,
                                neg_risk_market_id,
                                market_slug,
                                category,
                                category_refined,
                                outcome_label,
                                winning_outcome_label,
                                resolution_status,
                                opens_at,
                                close_at,
                                resolved_at,
                                taker_base_fee,
                                maker_base_fee
                            FROM read_parquet('{stage}')
                        )
                        TO '{metadata}'
                        (FORMAT PARQUET, COMPRESSION ZSTD)
                        """)

                        rows = con.execute(f"""
                            SELECT COUNT(*)
                            FROM read_parquet('{facts}')
                        """).fetchone()[0]

                        copy_to_drive(facts, fact_path)
                        copy_to_drive(metadata, meta_path)

                    manifest["partitions"][key] = {
                        "status": "completed",
                        "source": source_path,
                        "rows": rows,
                        "fact_bytes": fact_path.stat().st_size,
                        "meta_bytes": meta_path.stat().st_size,
                        "completed_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    }

                    save_json(manifest_file, manifest)

                    print(
                        f"[DONE] {key} | "
                        f"{rows:,} fills | "
                        f"{(fact_path.stat().st_size + meta_path.stat().st_size) / 1048576:.1f} MB",
                        flush=True
                    )

                    error = None
                    break

                except Exception as exc:
                    error = str(exc)
                    print(
                        f"[RETRY {attempt+1}/3] "
                        f"{key}: {error[:200]}",
                        flush=True
                    )
                    time.sleep(min(2 ** attempt, 10))

            if error:
                failures += 1
                manifest["partitions"][key] = {
                    "status": "failed",
                    "error": error[:500],
                    "source": source_path,
                }
                save_json(manifest_file, manifest)

                # Stop repeated systemic failures
                # instead of wasting cloud resources.
                if failures >= 8:
                    print("Stopping: too many failures")
                    break

        if failures >= 8:
            break

    partitions = manifest["partitions"]

    completed = [
        x for x in partitions.values()
        if x["status"] == "completed"
    ]

    missing = [
        k for k, x in partitions.items()
        if x["status"] == "source_missing"
    ]

    failed = [
        k for k, x in partitions.items()
        if x["status"] == "failed"
    ]

    expected_days = (
        last_source_day - START
    ).days + 1

    expected_parts = expected_days * len(LAYERS)

    print("\n=== EXTRACTION SUMMARY ===")
    print("Git commit:", commit)
    print("Dataset revision:", revision)
    print("Expected partitions:", expected_parts)
    print("Completed partitions:", len(completed))
    print("Missing source partitions:", len(missing))
    print("Failed partitions:", len(failed))
    print(
        "Total fills:",
        f"{sum(x['rows'] for x in completed):,}"
    )
    print(
        "Storage GB:",
        round(
            sum(
                x["fact_bytes"] + x["meta_bytes"]
                for x in completed
            ) / 1024**3,
            2
        )
    )
    print("Manifest:", manifest_file)

    if missing:
        print("Missing source examples:", missing[:10])

    if failed:
        print("Failed examples:", failed[:10])

    if len(completed) == expected_parts:
        print("STATUS: COMPLETE")
    else:
        print("STATUS: INCOMPLETE - review missing/failed days")

    con.close()

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
