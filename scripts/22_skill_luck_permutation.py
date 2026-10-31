
#!/usr/bin/env python3
"""Part I-A: coverage-stratified wallet score permutation."""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Research color palette
TEAL, ORANGE, CORAL = "#8ECFC9", "#FFBE7A", "#FA7F6F"
BLUE, PURPLE, BEIGE, GRAY = (
    "#82B0D2", "#BEB8DC", "#E7DAD2", "#999999"
)
INK = "#344054"

BANDS = ["5-9", "10-19", "20-49", "50+"]
THRESHOLDS = [5, 10, 20, 50]


def log(message):
    print(message, flush=True)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIGURE] {path.name}")


def rank_gap(scores, future, indices):
    """Future performance: historical top 10% minus bottom 10%."""
    ranked = indices[
        np.argsort(scores[indices], kind="stable")
    ]
    k = max(1, len(ranked) // 10)
    return float(
        future[ranked[-k:]].mean()
        - future[ranked[:k]].mean()
    )


def corr(x, y):
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den else float("nan")


def pvalue(observed, null):
    center = float(null.mean())
    count = np.count_nonzero(
        np.abs(null - center) >= abs(observed - center)
    )
    return float((count + 1) / (len(null) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    if args.permutations < 100:
        raise ValueError("Use at least 100 permutations")

    root = (
        Path(args.drive_root).expanduser()
        / "data" / "research_results"
    )

    source = (
        root / "p1a_skill_persistence_proxy_7d"
        / "wallet_oos_scores.parquet"
    )

    if not source.exists():
        raise FileNotFoundError(f"Missing source: {source}")

    out = root / "p1a_score_permutation"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "unknown"

    log("=== PART I-A: SKILL OR LUCK ===")

    # 1. Load existing wallet OOS results
    log("[1/5] Loading existing OOS wallet scores ...")

    w = pd.read_parquet(
        source,
        columns=[
            "wallet", "n_train", "n_future",
            "train_skill", "future_skill",
        ],
    ).dropna().reset_index(drop=True)

    if len(w) < 100:
        raise RuntimeError("Insufficient wallet sample")

    if (w.n_train.lt(5) | w.n_future.lt(5)).any():
        raise RuntimeError("Expected the existing 5/5 cohort")

    # Stratify by both historical and future coverage
    w["train_band"] = pd.cut(
        w.n_train,
        bins=[4, 9, 19, 49, np.inf],
        labels=BANDS,
    )

    w["future_band"] = pd.cut(
        w.n_future,
        bins=[4, 9, 19, 49, np.inf],
        labels=BANDS,
    )

    strata = [
        group.index.to_numpy()
        for _, group in w.groupby(
            ["train_band", "future_band"],
            observed=True,
        )
    ]

    band_indices = [
        np.flatnonzero(
            (w.train_band == band).to_numpy()
        )
        for band in BANDS
    ]

    cumulative = [
        np.flatnonzero(
            (w.n_train >= threshold).to_numpy()
        )
        for threshold in THRESHOLDS
    ]

    x = w.train_skill.to_numpy(dtype=np.float64)
    y = w.future_skill.to_numpy(dtype=np.float64)

    # Remove coverage-stratum means before correlation
    group_means = np.zeros(len(w))
    y_centered = y.copy()

    for ix in strata:
        group_means[ix] = x[ix].mean()
        y_centered[ix] -= y[ix].mean()

    x_centered = x - group_means

    norm = (
        np.linalg.norm(x_centered)
        * np.linalg.norm(y_centered)
    )

    if norm == 0:
        raise RuntimeError("Zero variance in score data")

    log(f"Wallets: {len(w):,}")
    log(f"Coverage strata: {len(strata)}")

    # 2. Actual statistics
    log("[2/5] Computing observed statistics ...")

    observed_r = corr(x_centered, y_centered)

    observed_gaps = np.array([
        rank_gap(x, y, ix)
        if len(ix) >= 30 else np.nan
        for ix in band_indices
    ])

    observed_corrs = np.array([
        corr(x_centered[ix], y_centered[ix])
        if len(ix) >= 30 else np.nan
        for ix in cumulative
    ])

    top_means = []
    bottom_means = []

    for ix in band_indices:
        if len(ix) < 30:
            top_means.append(np.nan)
            bottom_means.append(np.nan)
            continue

        ranked = ix[
            np.argsort(x[ix], kind="stable")
        ]
        k = max(1, len(ranked) // 10)

        top_means.append(float(y[ranked[-k:]].mean()))
        bottom_means.append(float(y[ranked[:k]].mean()))

    # 3. Permutation simulation
    log("[3/5] Running stratified permutations ...")

    rng = np.random.default_rng(args.seed)

    null_r = np.empty(args.permutations)

    null_gaps = np.full(
        (args.permutations, len(BANDS)), np.nan
    )

    null_corrs = np.full(
        (args.permutations, len(THRESHOLDS)), np.nan
    )

    for b in range(args.permutations):
        xp_centered = x_centered.copy()

        for ix in strata:
            if len(ix) > 1:
                xp_centered[ix] = (
                    x_centered[ix][rng.permutation(len(ix))]
                )

        xp = xp_centered + group_means

        null_r[b] = float(
            np.dot(xp_centered, y_centered) / norm
        )

        for j, ix in enumerate(band_indices):
            if len(ix) >= 30:
                null_gaps[b, j] = rank_gap(xp, y, ix)

        for j, ix in enumerate(cumulative):
            if len(ix) >= 30:
                null_corrs[b, j] = corr(
                    xp_centered[ix],
                    y_centered[ix],
                )

        if (
            (b + 1) % 100 == 0
            or b + 1 == args.permutations
        ):
            log(
                f"[PERMUTATION] "
                f"{b + 1:,}/{args.permutations:,}"
            )

    # 4. Tabular results
    log("[4/5] Summarizing results and figures ...")

    rows = []

    for j, band in enumerate(BANDS):
        ix = band_indices[j]

        if len(ix) < 30:
            continue

        ref = null_gaps[:, j]

        rows.append({
            "train_band": band,
            "wallets": int(len(ix)),
            "future_top10_mean": top_means[j],
            "future_bottom10_mean": bottom_means[j],
            "observed_top_minus_bottom": float(
                observed_gaps[j]
            ),
            "null_mean": float(ref.mean()),
            "null_p025": float(np.quantile(ref, 0.025)),
            "null_p975": float(np.quantile(ref, 0.975)),
            "permutation_p_two_sided": pvalue(
                observed_gaps[j], ref
            ),
        })

    table = pd.DataFrame(rows)

    table.to_csv(
        out / "experience_band_results.csv",
        index=False,
    )

    correlation_table = pd.DataFrame({
        "min_train_markets": THRESHOLDS,
        "n_wallets": [len(ix) for ix in cumulative],
        "adjusted_correlation": observed_corrs,
        "null_p025": np.quantile(
            null_corrs, 0.025, axis=0
        ),
        "null_p975": np.quantile(
            null_corrs, 0.975, axis=0
        ),
        "permutation_p_two_sided": [
            pvalue(observed_corrs[j], null_corrs[:, j])
            for j in range(len(THRESHOLDS))
        ],
    })

    correlation_table.to_csv(
        out / "coverage_correlation_results.csv",
        index=False,
    )

    # Figure style
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
        "grid.alpha": 0.45,
        "grid.color": BEIGE,
        "figure.facecolor": "white",
    })

    # Figure 01: Adjusted correlation vs null
    fig, ax = plt.subplots(figsize=(9, 5.3))

    ax.hist(
        null_r, bins=45,
        color=BLUE,
        edgecolor="white",
        linewidth=0.6,
    )

    ax.axvline(
        observed_r,
        color=CORAL, lw=2.6,
        label=f"Observed {observed_r:+.4f}",
    )

    ax.axvline(
        float(null_r.mean()),
        color=GRAY, ls="--", lw=1.5,
        label="Permutation center",
    )

    ax.set(
        title="01  Is historical skill predictive beyond luck?",
        xlabel="Coverage-adjusted train/future correlation",
        ylabel="Permutation frequency",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y")

    save(fig, figs / "01_adjusted_correlation_null.png")

    # Figure 02: Historical winners vs losers
    fig, ax = plt.subplots(figsize=(9, 5.3))

    positions = np.arange(len(table))

    ax.plot(
        positions,
        table.future_top10_mean,
        "o-", color=TEAL, lw=2.2,
        label="Historical top 10%",
    )

    ax.plot(
        positions,
        table.future_bottom10_mean,
        "s-", color=CORAL, lw=2.2,
        label="Historical bottom 10%",
    )

    ax.axhline(0, color=GRAY, ls="--")

    ax.set_xticks(
        positions,
        [
            f"{r.train_band}\nn={r.wallets:,}"
            for r in table.itertuples()
        ],
    )

    ax.set(
        title="02  Future performance by historical experience",
        xlabel="Historical market coverage",
        ylabel="Mean future wallet score",
    )

    ax.legend(frameon=False)
    ax.grid(axis="y")

    save(fig, figs / "02_top_bottom_by_experience.png")

    # Figure 03: Score gaps vs random distribution
    fig, ax = plt.subplots(figsize=(9, 5.3))

    low = table.null_p025.to_numpy()
    high = table.null_p975.to_numpy()
    center = table.null_mean.to_numpy()

    ax.errorbar(
        positions,
        center,
        yerr=[center - low, high - center],
        fmt="o",
        capsize=5,
        color=BLUE,
        label="Permutation 95% range",
    )

    ax.scatter(
        positions,
        table.observed_top_minus_bottom,
        s=90,
        marker="D",
        color=CORAL,
        zorder=4,
        label="Observed gap",
    )

    ax.axhline(0, color=GRAY, ls="--")
    ax.set_xticks(positions, table.train_band)

    ax.set(
        title="03  Winners vs losers: signal or random ranking?",
        xlabel="Historical market coverage",
        ylabel="Future top 10% - bottom 10% score",
    )

    ax.legend(frameon=False)
    ax.grid(axis="y")

    save(fig, figs / "03_experience_gap_vs_null.png")

    # Figure 04: Increasing experience threshold
    fig, ax = plt.subplots(figsize=(9, 5.3))

    xp = np.arange(len(THRESHOLDS))

    low = np.quantile(null_corrs, 0.025, axis=0)
    high = np.quantile(null_corrs, 0.975, axis=0)

    ax.fill_between(
        xp, low, high,
        color=PURPLE,
        alpha=0.6,
        label="Permutation 95% range",
    )

    ax.plot(
        xp, observed_corrs,
        "o-", color=CORAL, lw=2.2,
        label="Observed adjusted correlation",
    )

    ax.axhline(0, color=GRAY, ls="--")

    ax.set_xticks(
        xp,
        [
            f">={t}\nn={len(ix):,}"
            for t, ix in zip(THRESHOLDS, cumulative)
        ],
    )

    ax.set(
        title="04  Persistence vs minimum historical experience",
        xlabel="Minimum historical markets",
        ylabel="Coverage-adjusted score correlation",
    )

    ax.legend(frameon=False)
    ax.grid(axis="y")

    save(fig, figs / "04_adjusted_corr_by_coverage.png")

    # Figure 05: High-experience wallets
    fig, ax = plt.subplots(figsize=(9, 5.3))

    last = null_gaps[:, -1]

    ax.hist(
        last, bins=40,
        color=PURPLE,
        edgecolor="white",
        linewidth=0.6,
    )

    ax.axvline(
        observed_gaps[-1],
        color=CORAL, lw=2.4,
        label=f"Observed 50+ gap: {observed_gaps[-1]:+.4f}",
    )

    ax.axvline(
        float(last.mean()),
        color=GRAY, ls="--",
        label="Permutation center",
    )

    ax.set(
        title="05  Experienced wallets: null distribution",
        xlabel="Future score gap (historical top - bottom 10%)",
        ylabel="Permutation frequency",
    )

    ax.legend(frameon=False)
    ax.grid(axis="y")

    save(fig, figs / "05_experienced_wallet_null.png")

    # 5. Final report
    log("[5/5] Saving research report ...")

    report = {
        "git_commit": commit,
        "source": str(source),
        "wallets": int(len(w)),
        "permutations": args.permutations,
        "seed": args.seed,
        "n_coverage_strata": len(strata),
        "observed_coverage_adjusted_correlation": (
            observed_r
        ),
        "permutation_correlation_mean": float(
            null_r.mean()
        ),
        "permutation_correlation_ci95": [
            float(v) for v in
            np.quantile(null_r, [0.025, 0.975])
        ],
        "permutation_p_two_sided": pvalue(
            observed_r, null_r
        ),
        "experience_groups": rows,
        "limitations": [
            "Exploratory wallet-level permutation.",
            "Not a market-cluster outcome simulation.",
            "Stratified by historical and future coverage.",
            "Future participation selected ex-post.",
            "Uses previous 7-day label-timing proxy.",
            "Multiple experience groups are exploratory.",
        ],
    }

    (
        out / "SCORE_PERMUTATION_SUMMARY.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    log("\n=== FINAL RESULTS ===")
    log(f"Wallets: {len(w):,}")
    log(f"Observed adjusted correlation: {observed_r:+.5f}")
    log(
        f"Permutation two-sided p: "
        f"{report['permutation_p_two_sided']:.4f}"
    )

    log(
        table.to_string(
            index=False,
            float_format=lambda z: f"{z:.4f}",
        )
    )

    log(f"\n=== COMPLETE: {out} ===")


if __name__ == "__main__":
    main()
