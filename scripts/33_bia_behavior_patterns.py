
#!/usr/bin/env python3
"""Step 33: Interpretable Behavioral Intelligence.

Exploratory subgroup analysis of C1 vs C0.
No new model training or historical data reconstruction.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EPS = 1e-5
STEPS, DIM = 16, 5

FEATURES = {
    "history_length": ["0", "1-4", "5-9", "10-16"],
    "recency": ["<=7d", "7-30d", ">30d"],
    "stake_shift": [
        "Lower stake", "Similar stake", "Higher stake"
    ],
    "odds_shift": [
        "Lower chosen odds", "Similar chosen odds",
        "Higher chosen odds"
    ],
    "side_consistency": [
        "Same side", "Changed side"
    ],
}

COLORS = {
    "validation": "#FFBE7A",
    "final_test": "#82B0D2",
}

INPUT_COLS = [
    "obs_id", "market_type", "condition_id",
    "cluster_id", "history_count", "direction",
    "log_trade_size", "p_market_before",
    "y_yes", "history",
]


def log(msg):
    print(msg, flush=True)


def cache_one(src, target, expected):
    if not target.exists() or (
        target.stat().st_size != src.stat().st_size
    ):
        log(f"[CACHE] Copy {src.name} to runtime")
        shutil.copy2(src, target)

    rows = pq.ParquetFile(target).metadata.num_rows

    if rows != expected:
        raise RuntimeError(
            f"{target}: {rows:,} != {expected:,}"
        )

    return target


def features_for_batch(m, h):
    """Derive point-in-time behavioral patterns."""

    n = m["history_count"].to_numpy(np.int64)
    side = m["direction"].to_numpy(np.float32)
    size = m["log_trade_size"].to_numpy(np.float32)
    p = m["p_market_before"].to_numpy(np.float32)

    mask = (
        np.arange(STEPS)[None, :]
        >= STEPS - n[:, None]
    )

    denom = np.maximum(n, 1)

    # Chosen-token market probability:
    # BUY YES => p_yes; BUY NO => 1 - p_yes.
    hist_odds = np.where(
        h[:, :, 0] > 0,
        h[:, :, 2],
        1 - h[:, :, 2],
    )

    current_odds = np.where(
        side > 0, p, 1 - p
    )

    odds_mean = (
        np.where(mask, hist_odds, 0).sum(axis=1)
        / denom
    )

    size_mean = (
        np.where(mask, h[:, :, 1], 0).sum(axis=1)
        / denom
    )

    size_delta = size - size_mean
    odds_delta = current_odds - odds_mean

    # Most recent event is right-aligned.
    recency_days = np.expm1(h[:, -1, 4])

    changed_side = side != h[:, -1, 0]

    length_bin = np.select(
        [n == 0, n < 5, n < 10],
        ["0", "1-4", "5-9"],
        default="10-16",
    )

    recency_bin = np.select(
        [
            recency_days <= 7,
            recency_days <= 30,
        ],
        ["<=7d", "7-30d"],
        default=">30d",
    )

    stake_bin = np.select(
        [
            size_delta < -np.log(2),
            size_delta > np.log(2),
        ],
        ["Lower stake", "Higher stake"],
        default="Similar stake",
    )

    odds_bin = np.select(
        [
            odds_delta < -0.20,
            odds_delta > 0.20,
        ],
        [
            "Lower chosen odds",
            "Higher chosen odds",
        ],
        default="Similar chosen odds",
    )

    side_bin = np.where(
        changed_side,
        "Changed side",
        "Same side",
    )

    # Require >=5 historical events for mechanisms.
    return {
        "history_length": length_bin,
        "recency": np.where(
            n >= 5, recency_bin, "Not eligible"
        ),
        "stake_shift": np.where(
            n >= 5, stake_bin, "Not eligible"
        ),
        "odds_shift": np.where(
            n >= 5, odds_bin, "Not eligible"
        ),
        "side_consistency": np.where(
            n >= 5, side_bin, "Not eligible"
        ),
    }


def build_analysis_rows(
    seq_path, prediction_path, dest, batch_size
):
    pred = pd.read_parquet(
        prediction_path,
        columns=["obs_id", "p_C0", "p_C1"],
    )

    parquet = pq.ParquetFile(seq_path)

    if parquet.metadata.num_rows != len(pred):
        raise RuntimeError(
            "Sequence/prediction row count mismatch"
        )

    ids = pred.obs_id.to_numpy(np.int64)

    p0 = np.clip(
        pred.p_C0.to_numpy(np.float64),
        EPS, 1 - EPS,
    )
    p1 = np.clip(
        pred.p_C1.to_numpy(np.float64),
        EPS, 1 - EPS,
    )

    writer = None
    position = 0

    try:
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=INPUT_COLS,
        ):
            N = batch.num_rows

            meta = (
                pa.Table.from_batches([batch])
                .drop(["history"])
                .to_pandas()
            )

            hist = batch.column(
                batch.schema.get_field_index("history")
            )

            if (
                not pa.types.is_fixed_size_list(hist.type)
                or hist.type.list_size != 80
            ):
                raise RuntimeError(
                    f"Unexpected history type: {hist.type}"
                )

            h = (
                hist.values.to_numpy(
                    zero_copy_only=False
                )
                .reshape(N, STEPS, DIM)
            )

            if not np.array_equal(
                ids[position:position + N],
                meta.obs_id.to_numpy(np.int64),
            ):
                raise RuntimeError(
                    f"obs_id mismatch at {position:,}"
                )

            y = meta.y_yes.to_numpy(np.float64)

            q0 = p0[position:position + N]
            q1 = p1[position:position + N]

            patterns = features_for_batch(meta, h)

            b0 = (y - q0) ** 2
            b1 = (y - q1) ** 2

            l0 = (
                -y * np.log(q0)
                - (1-y) * np.log1p(-q0)
            )
            l1 = (
                -y * np.log(q1)
                - (1-y) * np.log1p(-q1)
            )

            columns = {
                "market_type": (
                    meta.market_type.fillna("unknown")
                    .astype(str).to_numpy()
                ),
                "condition_id": (
                    meta.condition_id.fillna("unknown")
                    .astype(str).to_numpy()
                ),
                "cluster_id": (
                    meta.cluster_id.fillna("unknown")
                    .astype(str).to_numpy()
                ),
                "history_count": (
                    meta.history_count.to_numpy(np.int8)
                ),
                "b0": b0.astype(np.float32),
                "b1": b1.astype(np.float32),
                "l0": l0.astype(np.float32),
                "l1": l1.astype(np.float32),
                **patterns,
            }

            table = pa.Table.from_pandas(
                pd.DataFrame(columns),
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    dest, table.schema,
                    compression="zstd",
                )

            writer.write_table(table)
            position += N

            if position % 500000 < N:
                log(
                    f"    Engineered {position:,}"
                    f" / {len(pred):,} rows"
                )

    finally:
        if writer is not None:
            writer.close()

    if position != len(pred):
        raise RuntimeError("Incomplete sequence processing")

    log(f"[ROWS] {dest.name}: {position:,}")


def market_summary(con, path, feature, value):
    sql = f"""
        SELECT
            market_type,
            condition_id,
            MIN(cluster_id) AS cluster_id,
            COUNT(*) AS n_rows,
            AVG(b0) AS b0,
            AVG(b1) AS b1,
            AVG(l0) AS l0,
            AVG(l1) AS l1
        FROM read_parquet(?)
        WHERE {feature} = ?
        GROUP BY 1, 2
    """

    result = con.execute(
        sql, [str(path), value]
    ).df()

    if result.empty:
        raise RuntimeError(
            f"Empty subgroup: {feature} / {value}"
        )

    return result


def cluster_ci(markets, prefix, nboot, seed):
    delta = (
        markets[f"{prefix}0"].to_numpy()
        - markets[f"{prefix}1"].to_numpy()
    )

    agg = (
        pd.DataFrame({
            "cluster": markets.cluster_id,
            "delta": delta,
        })
        .groupby("cluster", sort=False)
        .delta.agg(["sum", "size"])
    )

    sums = agg["sum"].to_numpy()
    sizes = agg["size"].to_numpy()

    rng = np.random.default_rng(seed)
    draws = np.empty(nboot)

    for start in range(0, nboot, 40):
        k = min(40, nboot - start)

        idx = rng.integers(
            0, len(sums),
            size=(k, len(sums)),
        )

        draws[start:start+k] = (
            sums[idx].sum(axis=1)
            / sizes[idx].sum(axis=1)
        )

    ci = np.quantile(draws, [.025, .975])

    return float(delta.mean()), ci, len(sums)


def make_figures(summary, out):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    mechanisms = [
        "recency",
        "stake_shift",
        "odds_shift",
        "side_consistency",
    ]

    fig, axs = plt.subplots(
        2, 2, figsize=(13, 8.4)
    )

    for ax, feature in zip(
        axs.ravel(), mechanisms
    ):
        bins = FEATURES[feature]

        for offset, split in (
            (-0.12, "validation"),
            (0.12, "final_test"),
        ):
            t = summary[
                (summary.feature == feature)
                & (summary.split == split)
                & (summary.metric == "brier")
            ].set_index("group").reindex(bins)

            x = np.arange(len(bins)) + offset
            y = t.delta.to_numpy() * 1e4

            lo = np.maximum(
                (t.delta - t.ci_low).to_numpy()
                * 1e4, 0,
            )
            hi = np.maximum(
                (t.ci_high - t.delta).to_numpy()
                * 1e4, 0,
            )

            ax.errorbar(
                x, y,
                yerr=[lo, hi],
                fmt="o",
                lw=1.6,
                capsize=3,
                color=COLORS[split],
                label=split,
            )

        ax.axhline(
            0, color="#999999",
            ls="--", lw=1,
        )

        ax.set_xticks(
            range(len(bins)),
            bins,
            rotation=15,
            ha="right",
        )

        ax.set_title(
            feature.replace("_", " ").title()
        )
        ax.set_ylabel(
            "C1 vs C0: Brier improvement x 10,000"
        )
        ax.grid(axis="y", alpha=0.12)

    axs[0, 0].legend(frameon=False)

    fig.suptitle(
        "Where Does Wallet History Help?"
        " | History >= 5",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        out / "01_behavior_heterogeneity.png",
        dpi=200, bbox_inches="tight",
    )

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--grace-days", type=int, default=7)
    ap.add_argument(
        "--cache-root", default="/content/bia33_cache"
    )
    ap.add_argument("--batch-size", type=int, default=32000)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--memory-gb", type=int, default=4)

    args = ap.parse_args()

    base = (
        Path(args.drive_root)
        / "data" / "research_results"
    )

    seq_dir = (
        base / f"p1a_bia_sequences_{args.grace_days}d"
    )
    pred_dir = (
        base / f"p1a_bia_history_control_{args.grace_days}d"
    )
    out = (
        base / f"p1a_bia_behavior_patterns_{args.grace_days}d"
    )

    out.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache_root)
    cache.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (seq_dir / "SEQUENCE_MANIFEST.json").read_text()
    )

    control = json.loads(
        (pred_dir / "CONTROL_MANIFEST.json").read_text()
    )

    assert control["rows"] == manifest["split_rows"]

    log(
        "[START] Step 33 | Existing ensemble predictions"
    )

    con = duckdb.connect()
    con.execute(
        f"SET memory_limit='{args.memory_gb}GB'"
    )
    con.execute("SET threads=4")

    summaries = []

    for si, split in enumerate(
        ("validation", "final_test")
    ):
        n = int(manifest["split_rows"][split])

        seq = cache_one(
            seq_dir / f"{split}.parquet",
            cache / f"{split}.parquet",
            n,
        )

        pred = (
            pred_dir
            / f"{split}_ensemble_predictions.parquet"
        )

        if not pred.exists():
            raise FileNotFoundError(pred)

        rows_path = cache / f"analysis_{split}.parquet"

        log(
            f"[1/2] {split}: deriving behavior patterns..."
        )

        build_analysis_rows(
            seq, pred, rows_path, args.batch_size
        )

        log(
            f"[2/2] {split}: Market-equal "
            "evaluation and bootstrap..."
        )

        for fi, (feature, groups) in enumerate(
            FEATURES.items()
        ):
            for gi, group in enumerate(groups):
                markets = market_summary(
                    con, rows_path, feature, group
                )

                for mi, metric in enumerate(
                    ("brier", "logloss")
                ):
                    point, ci, nclusters = cluster_ci(
                        markets,
                        "b" if metric == "brier" else "l",
                        args.bootstrap,
                        4200 + si*1000 + fi*100
                        + gi*10 + mi,
                    )

                    summaries.append({
                        "split": split,
                        "feature": feature,
                        "group": group,
                        "metric": metric,
                        "delta": point,
                        "ci_low": float(ci[0]),
                        "ci_high": float(ci[1]),
                        "rows": int(markets.n_rows.sum()),
                        "markets": len(markets),
                        "clusters": nclusters,
                    })

                if feature != "history_length":
                    log(
                        f"    {feature:18s} "
                        f"{group:20s} | "
                        f"n={markets.n_rows.sum():,} | "
                        f"Gain x1e4="
                        f"{summaries[-2]['delta']*1e4:+.3f}"
                    )

    result = pd.DataFrame(summaries)

    result.to_csv(
        out / "behavior_subgroup_scores.csv",
        index=False,
    )

    make_figures(result, out)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()

    report = {
        "git_commit": commit,
        "sequence_commit": manifest["git_commit"],
        "controlled_experiment_commit": (
            control["git_commit"]
        ),
        "source": (
            "Step 30 sequences + "
            "Step 32 ensemble predictions"
        ),
        "sample": (
            "History >=5 for behavioral patterns; "
            "all rows for history length"
        ),
        "fixed_bins": FEATURES,
        "evaluation": (
            "Market-equal Brier / Log Loss; "
            "event-cluster bootstrap within groups"
        ),
        "interpretation": (
            "Exploratory subgroup associations, "
            "not causal feature attribution"
        ),
        "limitations": [
            "YES/NO side across unrelated markets "
            "does not imply comparable beliefs",
            "Chosen-token odds approximated using "
            "historical YES probability and direction",
            "Up to 16 qualifying buys, "
            "not full wallet histories",
            "Multiple comparisons and previously "
            "inspected final test",
        ],
    }

    (out / "BEHAVIOR_MANIFEST.json").write_text(
        json.dumps(report, indent=2)
    )

    print(
        "\n=== BEHAVIOR SUBGROUPS: "
        "BRIER GAIN x10,000 ===",
        flush=True,
    )

    t = result[
        (result.metric == "brier")
        & (result.feature != "history_length")
    ].copy()

    for c in ("delta", "ci_low", "ci_high"):
        t[c] = (t[c] * 10000).round(3)

    print(
        t[[
            "split", "feature", "group",
            "delta", "ci_low", "ci_high",
            "rows", "markets",
        ]].to_string(index=False),
        flush=True,
    )

    con.close()
    log(f"[DONE] {out}")


if __name__ == "__main__":
    main()
