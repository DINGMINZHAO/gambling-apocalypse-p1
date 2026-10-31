
#!/usr/bin/env python3
"""Q2: Market probability freshness robustness."""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-5

FAMILIES = {
    "logistic": {
        "folder": "p1a_bia_models_{days}d",
        "models": [
            "M0_raw_market",
            "M1_calibrated_market",
            "M2_anonymous_behavior",
            "M3_wallet_intelligence",
        ],
        "pairs": [
            ("M1_calibrated_market", "M0_raw_market"),
            ("M2_anonymous_behavior", "M1_calibrated_market"),
            ("M3_wallet_intelligence", "M2_anonymous_behavior"),
        ],
    },
    "lightgbm": {
        "folder": "p1a_bia_lightgbm_{days}d",
        "models": [
            "L1_market",
            "L2_action",
            "L3_experience",
            "L4_skill",
        ],
        "pairs": [
            ("L2_action", "L1_market"),
            ("L3_experience", "L2_action"),
            ("L4_skill", "L3_experience"),
            ("L4_skill", "L2_action"),
        ],
    },
}


def run_one(con, path, models, max_age):
    cols = [
        "market_type", "condition_id",
        "cluster_id", "y_yes",
        "COUNT(*) AS n_rows",
    ]

    for name in models:
        p = (
            f"LEAST(1-{EPS}, "
            f"GREATEST({EPS}, p_{name}))"
        )

        cols.extend([
            f"AVG(POWER({p}-y_yes, 2)) AS b_{name}",
            f"AVG(-y_yes*LN({p})"
            f"-(1-y_yes)*LN(1-{p})) AS l_{name}",
        ])

    sql = f"""
        SELECT {', '.join(cols)}
        FROM read_parquet(?)
        WHERE price_age_hours > 0
          AND price_age_hours <= ?
        GROUP BY
            market_type, condition_id,
            cluster_id, y_yes
    """

    return con.execute(
        sql, [str(path), float(max_age)]
    ).df()


def cluster_ci(g, delta, n_boot, seed):
    groups = pd.DataFrame({
        "cluster": g.cluster_id,
        "delta": delta,
    }).groupby(
        "cluster", sort=False
    ).delta.agg(["sum", "size"])

    sums = groups["sum"].to_numpy()
    sizes = groups["size"].to_numpy()

    rng = np.random.default_rng(seed)
    k = len(groups)
    draws = np.empty(n_boot)

    for start in range(0, n_boot, 50):
        end = min(n_boot, start + 50)
        ids = rng.integers(
            0, k, size=(end - start, k)
        )

        draws[start:end] = (
            sums[ids].sum(axis=1)
            / sizes[ids].sum(axis=1)
        )

    ci = np.quantile(draws, [.025, .975])
    return ci, k


def make_figure(results, output):
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.3)
    )

    selected = [
        ("L2_action vs L1_market", "Anonymous behavior"),
        ("L4_skill vs L3_experience", "Historical skill"),
    ]

    for ax, (comparison, title) in zip(
        axes, selected
    ):
        for split, color in [
            ("validation", "#FFBE7A"),
            ("final_test", "#82B0D2"),
        ]:
            t = results[
                (results.family == "lightgbm")
                & (results.metric == "Brier")
                & (results.comparison == comparison)
                & (results.split == split)
            ].sort_values("max_age_hours")

            if t.empty:
                continue

            x = t.max_age_hours.to_numpy()
            y = t.delta.to_numpy() * 10000

            ax.plot(
                x, y, "o-",
                color=color, lw=2, label=split
            )
            ax.fill_between(
                x,
                t.ci_low.to_numpy() * 10000,
                t.ci_high.to_numpy() * 10000,
                color=color, alpha=.16,
            )

        ax.axhline(
            0, color="#999999", ls="--"
        )
        ax.set_xticks(
            [1, 3, 12], ["1h", "3h", "12h"]
        )
        ax.set_title(title)
        ax.set_xlabel("Maximum probability age")
        ax.grid(axis="y", alpha=.13)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(
        "Brier improvement × 10,000"
    )
    axes[1].legend(frameon=False)

    fig.suptitle(
        "Behavioral Information vs Price Freshness"
    )
    fig.tight_layout()

    fig.savefig(
        output, dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--grace-days", type=int, default=7)
    ap.add_argument("--bootstrap", type=int, default=400)
    args = ap.parse_args()

    base = (
        Path(args.drive_root)
        / "data" / "research_results"
    )

    out = (
        base / f"p1a_bia_freshness_{args.grace_days}d"
    )
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=2")

    scores = []
    gains = []

    print(
        "[START] BIA Price Freshness Sensitivity",
        flush=True
    )

    for family, spec in FAMILIES.items():
        folder = base / spec["folder"].format(
            days=args.grace_days
        )

        for split in ("validation", "final_test"):
            path = (
                folder / f"{split}_predictions.parquet"
            )

            if not path.exists():
                raise FileNotFoundError(path)

            for hours in (12, 3, 1):
                print(
                    f"[CHECK] {family} | {split} | "
                    f"Price age <= {hours}h",
                    flush=True,
                )

                g = run_one(
                    con, path, spec["models"], hours
                )

                if g.empty:
                    print("[WARNING] No observations")
                    continue

                sizes = g.groupby(
                    ["market_type", "condition_id"],
                    dropna=False,
                ).size()

                if sizes.max() > 1:
                    raise RuntimeError(
                        "Inconsistent market metadata "
                        "or resolution labels"
                    )

                n_rows = int(g.n_rows.sum())
                n_markets = len(g)

                print(
                    f"        {n_rows:,} predictions | "
                    f"{n_markets:,} markets",
                    flush=True,
                )

                for name in spec["models"]:
                    scores.append({
                        "family": family,
                        "split": split,
                        "max_age_hours": hours,
                        "model": name,
                        "predictions": n_rows,
                        "markets": n_markets,
                        "brier": float(
                            g[f"b_{name}"].mean()
                        ),
                        "logloss": float(
                            g[f"l_{name}"].mean()
                        ),
                    })

                for j, (new, old) in enumerate(
                    spec["pairs"]
                ):
                    for metric, prefix in (
                        ("Brier", "b"),
                        ("LogLoss", "l"),
                    ):
                        delta = (
                            g[f"{prefix}_{old}"]
                            - g[f"{prefix}_{new}"]
                        ).to_numpy()

                        ci, clusters = cluster_ci(
                            g,
                            delta,
                            args.bootstrap,
                            17 + j * 17 + hours * 7
                            + (metric == "LogLoss"),
                        )

                        result = {
                            "family": family,
                            "split": split,
                            "max_age_hours": hours,
                            "comparison": (
                                f"{new} vs {old}"
                            ),
                            "metric": metric,
                            "delta": float(delta.mean()),
                            "ci_low": float(ci[0]),
                            "ci_high": float(ci[1]),
                            "predictions": n_rows,
                            "markets": n_markets,
                            "clusters": clusters,
                        }

                        gains.append(result)

                        print(
                            f"        {metric}: "
                            f"{new} vs {old} | "
                            f"Delta={result['delta']:+.7f} "
                            f"CI=[{ci[0]:+.7f}, "
                            f"{ci[1]:+.7f}]",
                            flush=True,
                        )

    pd.DataFrame(scores).to_csv(
        out / "freshness_scores.csv",
        index=False,
    )

    gains_df = pd.DataFrame(gains)
    gains_df.to_csv(
        out / "freshness_incremental.csv",
        index=False,
    )

    make_figure(
        gains_df,
        out / "01_freshness_sensitivity.png",
    )

    con.close()
    print(f"[DONE] {out}", flush=True)


if __name__ == "__main__":
    main()
