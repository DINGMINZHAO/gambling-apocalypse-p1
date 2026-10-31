import argparse
import json
import subprocess
from pathlib import Path

import duckdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    args = parser.parse_args()

    root = Path(args.drive_root) / "data"
    raw = root / "research_raw"
    source = raw / "market_metadata" / "*" / "*.parquet"
    manifest_path = raw / "RUN_MANIFEST.json"

    manifest = json.loads(manifest_path.read_text())
    completed = sum(
        p["status"] == "completed"
        for p in manifest["partitions"].values()
    )

    if completed != 912:
        raise RuntimeError(
            f"Expected 912 partitions; found {completed}"
        )

    out = root / "research_index"
    out.mkdir(parents=True, exist_ok=True)
    registry = out / "market_registry.parquet"

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=2")

    print("=== BUILDING MARKET REGISTRY ===", flush=True)

    con.execute(f"""
    COPY (
        SELECT
            market_type,
            condition_id,
            asset_id AS token_id,
            ANY_VALUE(outcome_seq) AS outcome_seq,
            ANY_VALUE(outcome_label) AS outcome_label,
            ANY_VALUE(winning_outcome_label)
                AS winning_outcome_label,
            ANY_VALUE(category_refined) AS category,
            ANY_VALUE(market_slug) AS market_slug,
            ANY_VALUE(neg_risk_market_id)
                AS neg_risk_market_id,
            ANY_VALUE(close_at) AS close_at,
            ANY_VALUE(resolved_at) AS resolved_at,
            COUNT(DISTINCT COALESCE(
                winning_outcome_label, '__NULL__'
            )) AS winner_versions,
            COUNT(DISTINCT COALESCE(
                category_refined, '__NULL__'
            )) AS category_versions,
            COUNT(DISTINCT COALESCE(
                CAST(close_at AS VARCHAR), '__NULL__'
            )) AS close_versions
        FROM read_parquet(
            '{source}', union_by_name=true
        )
        WHERE condition_id IS NOT NULL
          AND asset_id IS NOT NULL
        GROUP BY market_type, condition_id, asset_id
    )
    TO '{registry}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    summary = con.execute(f"""
        SELECT
            COUNT(*) AS outcome_tokens,
            COUNT(DISTINCT condition_id) AS markets,
            COUNT(DISTINCT neg_risk_market_id)
                AS multi_event_groups,
            COUNT_IF(winning_outcome_label IS NULL)
                AS tokens_missing_winner,
            COUNT_IF(winner_versions > 1)
                AS winner_conflicts,
            COUNT_IF(category_versions > 1)
                AS category_conflicts,
            COUNT_IF(close_versions > 1)
                AS close_time_conflicts
        FROM read_parquet('{registry}')
    """)

    names = [d[0] for d in summary.description]
    values = summary.fetchone()
    report = dict(zip(names, values))

    print("\n=== MARKET COVERAGE ===")
    for k, v in report.items():
        print(f"{k}: {v:,}")

    print("\n=== MARKET TYPES ===")
    for kind, markets, tokens in con.execute(f"""
        SELECT market_type,
               COUNT(DISTINCT condition_id),
               COUNT(*)
        FROM read_parquet('{registry}')
        GROUP BY 1 ORDER BY 1
    """).fetchall():
        print(f"{kind}: {markets:,} markets, {tokens:,} tokens")

    print("\n=== MARKET CATEGORIES ===")
    rows = con.execute(f"""
        SELECT category,
               COUNT(DISTINCT condition_id) AS markets
        FROM read_parquet('{registry}')
        GROUP BY 1
        ORDER BY 2 DESC
    """).fetchall()

    for category, markets in rows:
        print(f"{category}: {markets:,}")

    repo = Path(__file__).resolve().parents[1]
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo, text=True
    ).strip()

    report["source_git_commit"] = manifest["git_commit"]
    report["source_dataset_revision"] = (
        manifest["dataset_revision"]
    )
    report["registry_code_commit"] = code_commit
    report["categories"] = [
        {"category": c, "markets": n}
        for c, n in rows
    ]

    (out / "MARKET_REGISTRY_SUMMARY.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print("\nSaved:", registry)
    print("Completed.")
    con.close()

if __name__ == "__main__":
    main()
