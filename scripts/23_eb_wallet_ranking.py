
#!/usr/bin/env python3
"""Part I-A: Empirical Bayes wallet ranking."""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Research palette
TEAL, ORANGE, CORAL = "#8ECFC9", "#FFBE7A", "#FA7F6F"
BLUE, PURPLE, BEIGE, GRAY = (
    "#82B0D2", "#BEB8DC", "#E7DAD2", "#999999"
)
INK = "#344054"

THRESHOLDS = [5, 10, 20, 50]


def log(msg):
    print(msg, flush=True)


def fit_eb(x, n):
    """
    Normal-Normal Empirical Bayes.

    x_i | theta_i ~ Normal(theta_i, sigma2 / n_i)
    theta_i ~ Normal(mu, tau2)
    """
    z = np.asarray(x, dtype=float) * 100
    n = np.asarray(n, dtype=float)

    v = max(float(np.var(z)), 1e-3)

    initial = [
        float(z.mean()),
        np.log(v * 0.4),
        np.log(v * 0.6 * np.median(n)),
    ]

    def nll(params):
        mu, log_tau, log_sigma = params
        variance = (
            np.exp(log_tau)
            + np.exp(log_sigma) / n
        )
        return 0.5 * np.sum(
            np.log(variance)
            + (z - mu) ** 2 / variance
        )

    fit = minimize(
        nll,
        initial,
        method="L-BFGS-B",
        bounds=[
            (-100, 100),
            (-18, 20),
            (-18, 20),
        ],
    )

    if not fit.success:
        raise RuntimeError(
            f"EB optimization failed: {fit.message}"
        )

    mu, log_tau, log_sigma = fit.x

    return {
        "mu": float(mu / 100),
        "tau2": float(np.exp(log_tau) / 10000),
        "sigma2": float(np.exp(log_sigma) / 10000),
    }


def select_ends(frame, column):
    ordered = frame.sort_values(
        [column, "wallet"],
        ascending=[False, True],
    )
    k = max(1, len(ordered) // 10)

    return ordered.head(k), ordered.tail(k)


def evaluate(frame, score_col):
    top, bottom = select_ends(frame, score_col)

    return {
        "top_future": float(top.future_skill.mean()),
        "bottom_future": float(bottom.future_skill.mean()),
        "gap": float(
            top.future_skill.mean()
            - bottom.future_skill.mean()
        ),
        "median_top_markets": float(
            top.n_train.median()
        ),
        "top_count": len(top),
    }


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRAY,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": BEIGE,
        "grid.alpha": 0.7,
        "figure.facecolor": "white",
    })


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(
        folder / name,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    log(f"[FIGURE] {name}")


def make_figures(w, metrics, parts, model, out):
    style()

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # 01. Raw score vs EB score
    fig, ax = plt.subplots(figsize=(8, 6.1))

    groups = [
        ("5-9", w.n_train.between(5, 9), BLUE),
        ("10-49", w.n_train.between(10, 49), TEAL),
        ("50+", w.n_train >= 50, CORAL),
    ]

    for band, condition, color in groups:
        z = w.loc[condition]
        ax.scatter(
            z.train_skill,
            z.eb_score,
            s=7,
            alpha=0.18,
            c=color,
            edgecolors="none",
            label=f"{band} markets (n={len(z):,})",
        )

    ax.plot(
        [-1, 1], [-1, 1],
        "--", color=GRAY, lw=1,
    )
    ax.axhline(
        model["mu"],
        color=ORANGE,
        lw=1,
        ls=":",
    )

    ax.set(
        xlabel="Raw historical score",
        ylabel="Empirical Bayes score",
        title="01  Small samples shrink toward the mean",
    )
    ax.legend(
        frameon=False,
        markerscale=2,
        fontsize=8,
    )
    ax.grid(alpha=0.3)
    save(fig, figures, "01_raw_vs_eb.png")

    # 02. Reliability weight by historical markets
    counts = np.arange(5, 151)

    weights = model["tau2"] / (
        model["tau2"] + model["sigma2"] / counts
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        counts, weights,
        color=PURPLE,
        lw=2.5,
    )

    points = np.array([5, 10, 20, 50, 100])
    point_weights = model["tau2"] / (
        model["tau2"] + model["sigma2"] / points
    )

    ax.scatter(
        points,
        point_weights,
        color=CORAL,
        zorder=3,
    )

    ax.set(
        xlabel="Historical markets per wallet",
        ylabel="Historical score weight",
        ylim=(0, 1.03),
        title="02  Reliability weights by market coverage",
    )
    ax.grid(axis="y")
    save(fig, figures, "02_shrinkage_weight.png")

    # 03. Future performance of Top 10%
    t = metrics.pivot(
        index="min_train",
        columns="method",
        values="future_top10_mean",
    ).reindex(THRESHOLDS)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    pos = np.arange(len(THRESHOLDS))

    ax.plot(
        pos, t["Raw"],
        "o-",
        color=CORAL,
        lw=2.2,
        label="Raw ranking",
    )
    ax.plot(
        pos, t["EB"],
        "s-",
        color=TEAL,
        lw=2.2,
        label="EB ranking",
    )

    n = w.n_train.to_numpy()

    ax.set_xticks(
        pos,
        [
            f">={k}\nn={(n >= k).sum():,}"
            for k in THRESHOLDS
        ],
    )

    ax.set(
        xlabel="Minimum historical market coverage",
        ylabel="Top 10% future mean score",
        title="03  Does EB select better future performers?",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y")
    save(fig, figures, "03_oos_top10_comparison.png")

    # 04. Ranking turnover
    names = ["Raw-only", "Both", "EB-only"]

    values = [
        parts[name].future_skill.mean()
        if len(parts[name]) else np.nan
        for name in names
    ]
    counts = [len(parts[name]) for name in names]

    fig, ax = plt.subplots(figsize=(9, 5.2))

    bars = ax.bar(
        names,
        values,
        color=[CORAL, BLUE, TEAL],
        width=0.55,
    )

    for bar, count in zip(bars, counts):
        if np.isfinite(bar.get_height()):
            ax.annotate(
                f"n={count:,}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                ),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    ax.axhline(0, color=GRAY, ls="--")
    ax.set(
        ylabel="Mean future wallet score",
        title="04  Who does EB replace in the Top 10%?",
    )
    ax.grid(axis="y")
    save(fig, figures, "04_ranking_turnover_future.png")

    # 05. EB minus Raw future performance
    gain = (t["EB"] - t["Raw"]).to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(
        pos,
        gain,
        color=[
            TEAL if value >= 0 else CORAL
            for value in gain
        ],
        width=0.64,
    )

    ax.axhline(0, color=GRAY, ls="--")
    ax.set_xticks(
        pos,
        [str(k) + "+" for k in THRESHOLDS],
    )

    ax.set(
        xlabel="Minimum historical market coverage",
        ylabel="EB minus Raw future Top 10% score",
        title="05  Out-of-sample selection improvement",
    )
    ax.grid(axis="y")
    save(fig, figures, "05_oos_eb_minus_raw.png")

    # 06. Future performance across ranking deciles
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    for col, color, name in [
        ("train_skill", CORAL, "Raw"),
        ("eb_score", TEAL, "EB"),
    ]:
        ranked = w.sort_values(
            [col, "wallet"],
            ascending=[True, True],
        ).copy()

        ranked["group"] = (
            np.floor(
                np.arange(len(ranked))
                * 10 / len(ranked)
            ).astype(int) + 1
        )

        means = ranked.groupby(
            "group"
        ).future_skill.mean()

        ax.plot(
            means.index,
            means.values,
            marker="o",
            lw=2,
            color=color,
            label=name,
        )

    ax.axhline(0, color=GRAY, ls="--")
    ax.set_xticks(
        range(1, 11),
        [f"D{i}" for i in range(1, 11)],
    )

    ax.set(
        xlabel="Historical rank decile (D10 highest)",
        ylabel="Mean future wallet score",
        title="06  Future performance by historical rank",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y")
    save(fig, figures, "06_future_decile_profile.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--grace-days", type=int, default=7)
    args = ap.parse_args()

    base = (
        Path(args.drive_root).expanduser()
        / "data" / "research_results"
    )

    source = (
        base
        / f"p1a_skill_persistence_proxy_{args.grace_days}d"
        / "wallet_oos_scores.parquet"
    )

    if not source.exists():
        raise FileNotFoundError(str(source))

    out = (
        base
        / f"p1a_empirical_bayes_{args.grace_days}d"
    )
    out.mkdir(parents=True, exist_ok=True)

    log("=== PART I-A: EMPIRICAL BAYES RANKING ===")

    # 1. Existing wallet OOS scores
    log("[1/5] Loading existing wallet OOS scores")

    columns = [
        "wallet",
        "n_train",
        "train_skill",
        "n_future",
        "future_skill",
    ]

    w = pd.read_parquet(
        source, columns=columns
    ).dropna().copy()

    w["wallet"] = w["wallet"].str.lower()

    if len(w) < 100 or w.wallet.duplicated().any():
        raise RuntimeError("Invalid wallet cohort")

    if (
        (w.n_train < 5).any()
        or (w.n_future < 5).any()
    ):
        raise RuntimeError(
            "Expected existing 5/5 wallet cohort"
        )

    log(f"[CHECK] Wallets: {len(w):,}")

    # 2. Fit EB using historical data only
    log("[2/5] Fitting Normal-Normal Empirical Bayes")

    model = fit_eb(
        w.train_skill.to_numpy(),
        w.n_train.to_numpy(),
    )

    w["reliability_weight"] = (
        model["tau2"] /
        (
            model["tau2"]
            + model["sigma2"] / w.n_train
        )
    )

    w["eb_score"] = (
        model["mu"]
        + w.reliability_weight
        * (w.train_skill - model["mu"])
    )

    w["shrinkage_magnitude"] = (
        w.eb_score - w.train_skill
    ).abs()

    log(f"Prior mean: {model['mu']:+.5f}")
    log(
        "Between-wallet SD: "
        f"{np.sqrt(model['tau2']):.5f}"
    )
    log(
        "Noise SD: "
        f"{np.sqrt(model['sigma2']):.5f}"
    )

    if model["tau2"] < 1e-8:
        log(
            "[WARNING] Very small between-wallet variance"
        )

    # 3. Compare raw ranking and EB ranking
    log("[3/5] Evaluating historical Top 10% rankings")

    rows = []

    for threshold in THRESHOLDS:
        subset = w.loc[w.n_train >= threshold]

        if len(subset) < 100:
            continue

        for column, name in [
            ("train_skill", "Raw"),
            ("eb_score", "EB"),
        ]:
            result = evaluate(subset, column)

            rows.append({
                "min_train": threshold,
                "method": name,
                "wallets": int(len(subset)),
                "future_top10_mean": result["top_future"],
                "future_bottom10_mean": result["bottom_future"],
                "top_minus_bottom": result["gap"],
                "top_train_markets_median": (
                    result["median_top_markets"]
                ),
                "top_count": result["top_count"],
            })

    metrics = pd.DataFrame(rows)

    raw_top, _ = select_ends(w, "train_skill")
    eb_top, _ = select_ends(w, "eb_score")

    raw_set = set(raw_top.wallet)
    eb_set = set(eb_top.wallet)

    parts = {
        "Raw-only": w.loc[
            w.wallet.isin(raw_set - eb_set)
        ],
        "Both": w.loc[
            w.wallet.isin(raw_set & eb_set)
        ],
        "EB-only": w.loc[
            w.wallet.isin(eb_set - raw_set)
        ],
    }

    overlap = len(raw_set & eb_set) / len(raw_set)

    log(f"[CHECK] Top 10% ranking overlap: {overlap:.1%}")

    # 4. Figures
    log("[4/5] Generating six figures")
    make_figures(w, metrics, parts, model, out)

    # 5. Save outputs
    log("[5/5] Saving ranking tables and report")

    ranked = w.sort_values(
        ["eb_score", "wallet"],
        ascending=[False, True],
    )

    ranked.to_parquet(
        out / "wallet_eb_rankings.parquet",
        index=False,
    )

    metrics.to_csv(
        out / "ranking_oos_comparison.csv",
        index=False,
    )

    ranked.head(100).to_csv(
        out / "top100_eb_candidates.csv",
        index=False,
    )

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"

    report = {
        "git_commit": commit,
        "grace_days": args.grace_days,
        "wallets": len(w),
        "prior_mean": model["mu"],
        "between_wallet_sd": float(
            np.sqrt(model["tau2"])
        ),
        "score_noise_sd": float(
            np.sqrt(model["sigma2"])
        ),
        "top10_overlap_fraction": float(overlap),
        "median_abs_shrinkage": float(
            w.shrinkage_magnitude.median()
        ),
        "mean_abs_shrinkage": float(
            w.shrinkage_magnitude.mean()
        ),
        "mean_future_raw_top10": float(
            raw_top.future_skill.mean()
        ),
        "mean_future_eb_top10": float(
            eb_top.future_skill.mean()
        ),
        "eb_minus_raw_future_top10": float(
            eb_top.future_skill.mean()
            - raw_top.future_skill.mean()
        ),
        "limitations": [
            "Approximate Normal-Normal EB model.",
            "Markets may share correlated outcomes.",
            "Fit uses OOS-active cohort, selected ex-post.",
            "Future scores not used directly in fitting.",
            "Result availability may use close-time proxy.",
            "No profitability or wallet-level significance claim.",
        ],
    }

    (
        out / "EB_RANKING_SUMMARY.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    log("\n=== FINAL RESULTS ===")

    for key, value in report.items():
        if key != "limitations":
            log(f"{key}: {value}")

    log(
        "\n" + metrics.to_string(
            index=False,
            float_format=lambda z: f"{z:.5f}",
        )
    )

    log(f"\n=== COMPLETE: {out} ===")


if __name__ == "__main__":
    main()
