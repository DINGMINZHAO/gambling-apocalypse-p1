
#!/usr/bin/env python3
"""Q2: Cluster-robust BIA evidence and wallet heterogeneity."""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, ORANGE, CORAL = "#8ECFC9", "#FFBE7A", "#FA7F6F"
BLUE, PURPLE, GRAY = "#82B0D2", "#BEB8DC", "#999999"
INK = "#344054"

MODELS = [
    "M0_raw_market",
    "M1_calibrated_market",
    "M2_anonymous_behavior",
    "M3_wallet_intelligence",
]

PAIRS = [
    ("M1 vs M0", 1, 0),
    ("M2 vs M1", 2, 1),
    ("M3 vs M2", 3, 2),
]


def log(message):
    print(message, flush=True)


def bootstrap(df, values, times=800, seed=42):
    """Event-cluster bootstrap with equal market weight."""
    tmp = pd.DataFrame({
        "cluster": df.cluster_id,
        "delta": np.asarray(values),
    })

    g = tmp.groupby("cluster", sort=False).agg(
        sums=("delta", "sum"),
        markets=("delta", "size"),
    )

    sums = g.sums.to_numpy()
    sizes = g.markets.to_numpy()

    k = len(g)
    rng = np.random.default_rng(seed)
    draws = []

    for start in range(0, times, 100):
        batch = min(100, times - start)
        ids = rng.integers(
            0, k, size=(batch, k)
        )
        draws.extend(
            (
                sums[ids].sum(axis=1)
                / sizes[ids].sum(axis=1)
            ).tolist()
        )

    low, high = np.quantile(draws, [.025, .975])

    return (
        float(np.mean(values)),
        float(low),
        float(high),
        k,
    )


def market_losses(con, path):
    expressions = [
        "market_type",
        "condition_id",
        "MIN(cluster_id) AS cluster_id",
        "COUNT(*) AS n_predictions",
    ]

    for name in MODELS:
        p = (
            "GREATEST(0.00001, "
            f"LEAST(0.99999, p_{name}))"
        )

        expressions.append(
            f"AVG(POWER({p}-y_yes,2)) AS b_{name}"
        )
        expressions.append(
            f"AVG(-y_yes*LN({p})"
            f"-(1-y_yes)*LN(1-{p})) AS l_{name}"
        )

    sql = (
        "SELECT " + ", ".join(expressions)
        + f" FROM read_parquet('{path}')"
        + " GROUP BY market_type, condition_id"
    )

    return con.execute(sql).df()


def subgroup(con, path, mode):
    if mode == "experience":
        group = """
            CASE
                WHEN n_past_resolved_markets = 0
                    THEN '0'
                WHEN n_past_resolved_markets < 5
                    THEN '1-4'
                WHEN n_past_resolved_markets < 20
                    THEN '5-19'
                ELSE '20+'
            END
        """
        where = ""

    else:
        group = """
            (CASE
                WHEN historical_shrunk_skill < 0
                    THEN 'Past-negative'
                ELSE 'Past-nonnegative'
             END)
             || ' / ' ||
            (CASE
                WHEN direction = 1
                    THEN 'BUY YES'
                ELSE 'BUY NO'
             END)
        """
        where = "WHERE n_past_resolved_markets >= 5"

    sql = f"""
        WITH obs AS (
            SELECT
                market_type,
                condition_id,
                cluster_id,
                {group} AS grp,
                POWER(
                    p_M2_anonymous_behavior-y_yes, 2
                )
                - POWER(
                    p_M3_wallet_intelligence-y_yes, 2
                ) AS delta
            FROM read_parquet('{path}')
            {where}
        )

        SELECT
            market_type,
            condition_id,
            MIN(cluster_id) AS cluster_id,
            grp,
            COUNT(*) AS n_predictions,
            AVG(delta) AS delta
        FROM obs
        GROUP BY 1, 2, 4
    """

    return con.execute(sql).df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument(
        "--grace-days", type=int, default=7
    )
    ap.add_argument(
        "--bootstrap", type=int, default=800
    )
    args = ap.parse_args()

    source = (
        Path(args.drive_root)
        / "data" / "research_results"
        / f"p1a_bia_models_{args.grace_days}d"
    )

    out = source / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=2")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.labelcolor": INK,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    records = []
    subgroup_records = []

    for split in ("validation", "final_test"):
        path = source / f"{split}_predictions.parquet"

        if not path.exists():
            raise FileNotFoundError(path)

        log(f"[1/3] {split}: Market-level losses...")

        markets = market_losses(con, path)

        log(
            f"      Markets: {len(markets):,} | "
            f"Clusters: {markets.cluster_id.nunique():,}"
        )

        for j, (name, new, old) in enumerate(PAIRS):
            for metric in ("b", "l"):
                delta = (
                    markets[f"{metric}_{MODELS[old]}"]
                    - markets[f"{metric}_{MODELS[new]}"]
                )

                mean, low, high, clusters = bootstrap(
                    markets,
                    delta,
                    args.bootstrap,
                    42 + j * 17 + int(metric == "l"),
                )

                records.append({
                    "split": split,
                    "comparison": name,
                    "metric": (
                        "Brier" if metric == "b"
                        else "LogLoss"
                    ),
                    "delta": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "markets": len(markets),
                    "clusters": clusters,
                })

        log(f"[2/3] {split}: Wallet subgroups...")

        for mode in ("experience", "skill_x_side"):
            frame = subgroup(con, path, mode)

            for j, (name, group) in enumerate(
                frame.groupby("grp")
            ):
                mean, low, high, clusters = bootstrap(
                    group,
                    group.delta,
                    args.bootstrap,
                    1000 + j,
                )

                subgroup_records.append({
                    "split": split,
                    "grouping": mode,
                    "group": name,
                    "delta": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "markets": len(group),
                    "clusters": clusters,
                    "predictions": int(
                        group.n_predictions.sum()
                    ),
                })

    results = pd.DataFrame(records)
    groups = pd.DataFrame(subgroup_records)

    results.to_csv(
        out / "cluster_bootstrap_results.csv",
        index=False,
    )
    groups.to_csv(
        out / "wallet_subgroup_results.csv",
        index=False,
    )

    log("[3/3] Generating research figures...")

    # Figure 1: Incremental Brier Score and confidence intervals

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(PAIRS))

    for j, (split, color) in enumerate([
        ("validation", ORANGE),
        ("final_test", BLUE),
    ]):
        rows = results[
            (results.split == split)
            & (results.metric == "Brier")
        ]

        rows = rows.set_index("comparison").loc[
            [p[0] for p in PAIRS]
        ]

        xx = x + (j - .5) * .28
        y = rows.delta.to_numpy() * 10000

        low = np.maximum(
            0,
            (rows.delta - rows.ci_low).to_numpy()
            * 10000,
        )
        high = np.maximum(
            0,
            (rows.ci_high - rows.delta).to_numpy()
            * 10000,
        )

        ax.errorbar(
            xx, y,
            yerr=[low, high],
            fmt="o",
            capsize=4,
            color=color,
            lw=2,
            markersize=8,
            label=split,
        )

    ax.axhline(0, color=GRAY, ls="--")
    ax.set_xticks(x, [p[0] for p in PAIRS])
    ax.set_ylabel("Brier improvement × 10,000")
    ax.set_title(
        "Incremental Predictive Value | Event-Cluster 95% CI"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.13)

    fig.savefig(
        out / "01_incremental_brier_ci.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2: Experience heterogeneity

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    categories = ["0", "1-4", "5-19", "20+"]
    x = np.arange(4)

    for j, (split, color) in enumerate([
        ("validation", ORANGE),
        ("final_test", BLUE),
    ]):
        t = groups[
            (groups.split == split)
            & (groups.grouping == "experience")
        ]

        values = (
            t.set_index("group")
            .reindex(categories)
            .delta.to_numpy()
            * 10000
        )

        ax.bar(
            x + (j - .5) * .35,
            values,
            width=.33,
            color=color,
            label=split,
        )

    ax.axhline(0, color=GRAY, ls="--")
    ax.set_xticks(x, categories)
    ax.set_xlabel("Previously Resolved Markets")
    ax.set_ylabel(
        "M3 vs M2 Brier Improvement × 10,000"
    )
    ax.set_title(
        "Does Experience Change Wallet Information Value?"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.13)

    fig.savefig(
        out / "02_information_by_experience.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 3: Past-negative vs past-nonnegative wallets

    t = groups[
        (groups.split == "final_test")
        & (groups.grouping == "skill_x_side")
    ].sort_values("group")

    fig, ax = plt.subplots(figsize=(9, 5.1))

    colors = [
        CORAL if "Past-negative" in name else TEAL
        for name in t.group
    ]

    ax.barh(
        t.group,
        t.delta * 10000,
        color=colors,
    )
    ax.axvline(0, color=GRAY, ls="--")

    ax.set_xlabel(
        "M3 vs M2 Brier Improvement × 10,000"
    )
    ax.set_title(
        "Past-Negative Wallets: Exploratory Diagnostic"
    )
    ax.grid(axis="x", alpha=.13)

    fig.savefig(
        out / "03_past_skill_and_direction.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)

    (out / "BIA_ANALYSIS_MANIFEST.json").write_text(
        json.dumps({
            "bootstrap_resamples": args.bootstrap,
            "bootstrap_unit": (
                "Event cluster; equal market weight"
            ),
            "primary_comparison": "M3 vs M2",
            "status": "Exploratory",
        }, indent=2)
    )

    log("\n=== EVENT-CLUSTER BOOTSTRAP ===")

    for r in records:
        log(
            f"{r['split']:<11} "
            f"{r['metric']:<7} "
            f"{r['comparison']:<9} "
            f"Delta={r['delta']:+.7f} "
            f"CI=[{r['ci_low']:+.7f}, "
            f"{r['ci_high']:+.7f}]"
        )

    con.close()
    log(f"[DONE] {out}")


if __name__ == "__main__":
    main()
