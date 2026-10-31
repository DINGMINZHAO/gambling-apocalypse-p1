
#!/usr/bin/env python3
"""Part I-A: Leakage-aware wallet skill persistence."""

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# Publication palette
TEAL, ORANGE, CORAL = "#8ECFC9", "#FFBE7A", "#FA7F6F"
BLUE, PURPLE, BEIGE, GRAY = (
    "#82B0D2", "#BEB8DC", "#E7DAD2", "#999999"
)
INK = "#344054"

# Existing unified 12h rule: unchanged
RULE = """
    p_market_before BETWEEN 0 AND 1
    AND price_age_seconds > 0
    AND price_age_seconds <= 43200
    AND pre_close_1h = TRUE
    AND y_yes IN (0, 1)
    AND outcome_seq IN (1, 2)
"""


def log(message):
    print(message, flush=True)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIGURE] {path.name}")


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRAY,
        "grid.color": BEIGE,
        "grid.alpha": 0.7,
        "figure.facecolor": "white",
    })


def clustered_bootstrap(contrib, n_boot=400, seed=17):
    """
    Market/event-cluster Poisson bootstrap.

    Preserves common outcome shocks within available
    market/event clusters. Wallets receive equal weight.
    """
    rows, cluster_names = pd.factorize(
        contrib["cluster_id"], sort=True
    )

    dec = contrib["decile"].to_numpy(dtype=int) - 1
    n_clusters = len(cluster_names)

    score = np.zeros((n_clusters, 10), dtype=float)
    weight = np.zeros((n_clusters, 10), dtype=float)

    np.add.at(
        score,
        (rows, dec),
        contrib["score_contrib"].to_numpy(),
    )
    np.add.at(
        weight,
        (rows, dec),
        contrib["weight_contrib"].to_numpy(),
    )

    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, 10), dtype=float)

    for start in range(0, n_boot, 50):
        end = min(start + 50, n_boot)
        mult = rng.poisson(
            1.0, size=(end - start, n_clusters)
        )

        numerator = mult @ score
        denominator = mult @ weight

        draws[start:end] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )

    lower, upper = np.nanpercentile(
        draws, [2.5, 97.5], axis=0
    )
    observed = score.sum(axis=0) / weight.sum(axis=0)
    contrast_draws = draws[:, 9] - draws[:, 0]

    return observed, lower, upper, contrast_draws, n_clusters


def make_figures(wallets, deciles, boot, figdir):
    setup_style()

    order = np.arange(1, 11)
    labels = [f"D{i}" for i in order]

    means, lower, upper, diff_draws, _ = boot

    # Figure 01: Main out-of-sample result
    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = [CORAL] + [BLUE] * 8 + [TEAL]

    ax.bar(
        order, means, color=colors,
        width=0.68, alpha=0.9,
    )
    ax.errorbar(
        order, means,
        yerr=[means - lower, upper - means],
        fmt="none",
        ecolor=INK,
        capsize=3,
        linewidth=1.2,
    )

    ax.axhline(0, color=GRAY, ls="--", lw=1)
    ax.set_xticks(order, labels)
    ax.set(
        xlabel="Historical score decile (D1 worst, D10 best)",
        ylabel="Mean future wallet score",
        title="01  Future skill by historical rank",
    )
    ax.grid(axis="y")
    save(fig, figdir / "01_oos_skill_deciles.png")

    # Figure 02: Historical vs future performance
    sample = wallets.sample(
        min(50000, len(wallets)),
        random_state=17,
    )

    fig, ax = plt.subplots(figsize=(7.7, 6))
    cmap = LinearSegmentedColormap.from_list(
        "scientific", [BEIGE, TEAL, BLUE]
    )

    hb = ax.hexbin(
        sample.train_skill,
        sample.future_skill,
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap=cmap,
        extent=(-1, 1, -1, 1),
    )

    ax.axhline(0, color=GRAY, ls="--")
    ax.axvline(0, color=GRAY, ls="--")
    ax.set(
        xlim=(-1, 1),
        ylim=(-1, 1),
        xlabel="Historical wallet score",
        ylabel="Future wallet score",
        title="02  Historical vs future wallet performance",
    )
    fig.colorbar(
        hb, ax=ax, label="Wallet count (log colors)"
    )
    save(fig, figdir / "02_historical_vs_future.png")

    # Figure 03: Fraction with positive future score
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        order,
        deciles["future_positive_pct"],
        color=TEAL,
        width=0.7,
    )
    ax.axhline(
        50, color=GRAY, ls="--",
        label="50% reference",
    )

    ax.set_xticks(order, labels)
    ax.set(
        ylim=(0, 100),
        xlabel="Historical score decile",
        ylabel="Wallets with future score > 0 (%)",
        title="03  Future positive-score frequency",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y")
    save(fig, figdir / "03_future_positive_share.png")

    # Figure 04: Future-score distributions
    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    groups = [
        (
            "D1 historical bottom",
            wallets.decile == 1,
            CORAL,
        ),
        (
            "D5-D6 historical middle",
            wallets.decile.isin([5, 6]),
            BLUE,
        ),
        (
            "D10 historical top",
            wallets.decile == 10,
            TEAL,
        ),
    ]

    for name, mask, color in groups:
        values = wallets.loc[
            mask, "future_skill"
        ].to_numpy()

        hist, edges = np.histogram(
            values, bins=70, range=(-0.75, 0.75)
        )

        if len(values):
            ax.stairs(
                hist / len(values),
                edges,
                linewidth=2,
                color=color,
                label=f"{name} (n={len(values):,})",
            )

    ax.axvline(0, color=GRAY, ls="--")
    ax.set(
        xlim=(-0.75, 0.75),
        xlabel="Future wallet score",
        ylabel="Within-group fraction per bin",
        title="04  Future-score distributions",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y")
    save(fig, figdir / "04_future_distributions.png")

    # Figure 05: Historical winners vs losers
    delta = means[9] - means[0]
    diff_ci = np.nanpercentile(
        diff_draws, [2.5, 97.5]
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.7))

    x = np.arange(3)
    values = [
        means[0],
        np.mean(means[4:6]),
        means[9],
    ]

    ax.bar(
        x, values,
        color=[CORAL, BLUE, TEAL],
        width=0.55,
    )

    ax.axhline(0, color=GRAY, ls="--")
    ax.set_xticks(
        x, ["D1 bottom", "D5-D6 middle", "D10 top"]
    )
    ax.set(
        ylabel="Mean future wallet score",
        title="05  Past losers, middle wallets and winners",
    )

    ax.text(
        0.98, 0.97,
        f"D10 - D1 = {delta:+.4f}\n"
        f"95% cluster CI [{diff_ci[0]:+.4f}, "
        f"{diff_ci[1]:+.4f}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": BEIGE,
        },
    )
    ax.grid(axis="y")
    save(fig, figdir / "05_top_bottom_contrast.png")

    # Figure 06: Historical coverage sensitivity
    fig, ax = plt.subplots(figsize=(9, 5.2))

    thresholds = [5, 10, 20, 50]
    differences = []
    counts = []

    for threshold in thresholds:
        z = wallets.loc[
            wallets.n_train >= threshold
        ].copy()

        counts.append(len(z))

        if len(z) < 100:
            differences.append(np.nan)
            continue

        z["group"] = pd.qcut(
            z.train_skill.rank(method="first"),
            10,
            labels=False,
        ) + 1

        difference = (
            z.loc[z.group == 10, "future_skill"].mean()
            - z.loc[z.group == 1, "future_skill"].mean()
        )
        differences.append(difference)

    ax.plot(
        thresholds,
        differences,
        color=PURPLE,
        marker="o",
        linewidth=2.5,
    )
    ax.axhline(0, color=GRAY, ls="--")

    ax.set_xticks(
        thresholds,
        [
            f">={k}\n(n={n:,})"
            for k, n in zip(thresholds, counts)
        ],
    )
    ax.set(
        xlabel="Minimum historical markets",
        ylabel="D10 minus D1 future score",
        title="06  Historical coverage sensitivity",
    )
    ax.grid(axis="y")
    save(fig, figdir / "06_coverage_sensitivity.png")

    # Figure 07: Market coverage of each decile
    fig, ax = plt.subplots(figsize=(10, 5.2))

    ax.plot(
        order,
        deciles["mean_n_train"],
        color=ORANGE,
        marker="o",
        linewidth=2,
        label="Mean historical markets",
    )
    ax.plot(
        order,
        deciles["mean_n_future"],
        color=BLUE,
        marker="s",
        linewidth=2,
        label="Mean future markets",
    )

    ax.set_xticks(order, labels)
    ax.set(
        xlabel="Historical score decile",
        ylabel="Distinct markets per wallet",
        title="07  Market coverage across historical deciles",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y")
    save(fig, figdir / "07_decile_market_counts.png")

    return delta, diff_ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument(
        "--split-date", default="2025-10-01"
    )
    parser.add_argument(
        "--min-train", type=int, default=5
    )
    parser.add_argument(
        "--min-future", type=int, default=5
    )
    parser.add_argument(
        "--bootstrap", type=int, default=400
    )
    parser.add_argument(
        "--memory-gb", type=int, default=6
    )
    parser.add_argument(
        "--threads", type=int, default=2
    )
    args = parser.parse_args()

    split = int(
        datetime.strptime(
            args.split_date, "%Y-%m-%d"
        ).replace(
            tzinfo=timezone.utc
        ).timestamp()
    )

    data = Path(args.drive_root) / "data"
    aligned = data / "probability_alignment"
    registry = (
        data / "research_index"
        / "market_registry.parquet"
    )

    if len(list(aligned.glob(
        "wallet_market_*.parquet"
    ))) != 15:
        raise RuntimeError(
            "Expected 15 existing alignment files"
        )

    if not registry.exists():
        raise RuntimeError(
            f"Missing market registry: {registry}"
        )

    out = (
        data / "research_results"
        / "p1a_skill_persistence"
    )
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()

    log("=== PART I-A: SKILL PERSISTENCE ===")
    log(f"Git commit: {commit}")
    log(f"Split UTC: {args.split_date}")
    log(
        f"Minimum historical/future markets: "
        f"{args.min_train}/{args.min_future}"
    )

    with tempfile.TemporaryDirectory(
        prefix="p1a_persist_"
    ) as td:

        temp = Path(td)
        (temp / "spill").mkdir()

        con = duckdb.connect(
            str(temp / "work.duckdb")
        )
        con.execute(
            f"SET memory_limit='{args.memory_gb}GB'"
        )
        con.execute(
            f"SET threads={args.threads}"
        )
        con.execute(
            f"SET temp_directory='{temp / 'spill'}'"
        )

        # Step 1: Existing resolution information
        log("[1/7] Inspecting resolution timestamps...")

        con.execute(f"""
            CREATE TABLE meta AS
            WITH parsed AS (
                SELECT
                    market_type,
                    condition_id,
                    neg_risk_market_id,
                    CASE
                        WHEN TRY_CAST(
                            resolved_at AS DOUBLE
                        ) BETWEEN 946684800 AND 4102444800
                        THEN TRY_CAST(
                            resolved_at AS DOUBLE
                        )
                        WHEN TRY_CAST(
                            resolved_at AS DOUBLE
                        ) BETWEEN 946684800000
                          AND 4102444800000
                        THEN TRY_CAST(
                            resolved_at AS DOUBLE
                        ) / 1000
                        ELSE EPOCH(
                            TRY_CAST(
                                resolved_at AS TIMESTAMPTZ
                            )
                        )
                    END AS resolved_ts
                FROM read_parquet('{registry}')
                WHERE outcome_seq = 1
            )
            SELECT
                market_type,
                condition_id,
                MIN(resolved_ts) AS resolved_ts,
                COUNT(DISTINCT resolved_ts)
                    AS resolution_versions,
                ANY_VALUE(
                    NULLIF(
                        TRIM(
                            CAST(
                                neg_risk_market_id AS VARCHAR
                            )
                        ), ''
                    )
                ) AS event_id
            FROM parsed
            GROUP BY 1, 2
        """)

        metadata = con.execute("""
            SELECT
                COUNT(*),
                COUNT(*) FILTER (
                    WHERE resolved_ts IS NOT NULL
                ),
                COUNT(*) FILTER (
                    WHERE resolution_versions > 1
                )
            FROM meta
        """).fetchone()

        log(
            f"Registry markets: {metadata[0]:,} | "
            f"Resolved timestamps: {metadata[1]:,} | "
            f"Conflicts: {metadata[2]:,}"
        )

        if metadata[1] == 0:
            raise RuntimeError(
                "No parseable resolved_at values. "
                "Inspect registry timestamps first."
            )

        # Step 2: Existing 12h sample
        log("[2/7] Scoring existing observations...")

        pattern = aligned / "wallet_market_*.parquet"

        con.execute(f"""
            CREATE TABLE obs AS
            SELECT
                LOWER(a.wallet) AS wallet,
                a.market_type,
                a.condition_id,
                CAST(a.trade_ts AS BIGINT) AS trade_ts,
                CASE
                    WHEN a.outcome_seq = 1
                    THEN a.y_yes - a.p_market_before
                    ELSE a.p_market_before - a.y_yes
                END AS score,
                m.resolved_ts,
                m.resolution_versions,
                CASE
                    WHEN m.event_id IS NOT NULL
                    THEN 'event:' || m.event_id
                    ELSE
                        a.market_type || ':' || a.condition_id
                END AS cluster_id
            FROM read_parquet('{pattern}') a
            LEFT JOIN meta m
                USING (market_type, condition_id)
            WHERE {RULE}
        """)

        n_obs = con.execute(
            "SELECT COUNT(*) FROM obs"
        ).fetchone()[0]

        log(f"Valid observations: {n_obs:,}")

        if n_obs != 7143340:
            raise RuntimeError(
                "Unified 12h sample count mismatch"
            )

        # Step 3: Time-safe market-level scores
        log("[3/7] Building time-safe wallet-market scores...")

        con.execute(f"""
            CREATE TABLE wm AS
            SELECT
                wallet,
                market_type,
                condition_id,
                ANY_VALUE(cluster_id) AS cluster_id,
                MIN(trade_ts) AS first_ts,

                AVG(score) FILTER (
                    WHERE trade_ts < {split}
                      AND resolved_ts <= {split}
                      AND resolved_ts > trade_ts
                      AND resolution_versions = 1
                ) AS train_score,

                COUNT(*) FILTER (
                    WHERE trade_ts < {split}
                      AND resolved_ts <= {split}
                      AND resolved_ts > trade_ts
                      AND resolution_versions = 1
                ) AS n_train_obs,

                AVG(score) FILTER (
                    WHERE trade_ts >= {split}
                      AND resolved_ts > trade_ts
                      AND resolution_versions = 1
                ) AS future_score,

                COUNT(*) FILTER (
                    WHERE trade_ts >= {split}
                      AND resolved_ts > trade_ts
                      AND resolution_versions = 1
                ) AS n_future_obs

            FROM obs
            GROUP BY 1, 2, 3
        """)

        con.execute("""
            CREATE TABLE train AS
            SELECT
                wallet,
                COUNT(*) AS n_train,
                AVG(train_score) AS train_skill
            FROM wm
            WHERE n_train_obs > 0
            GROUP BY wallet
        """)

        con.execute(f"""
            CREATE TABLE future AS
            SELECT
                wallet,
                COUNT(*) AS n_future,
                AVG(future_score) AS future_skill
            FROM wm
            WHERE n_future_obs > 0
              AND first_ts >= {split}
            GROUP BY wallet
        """)

        # Step 4: Rank using historical information only
        log("[4/7] Constructing historical skill deciles...")

        n_train_wallets = con.execute(
            "SELECT COUNT(*) FROM train"
        ).fetchone()[0]

        n_future_wallets = con.execute(
            "SELECT COUNT(*) FROM future"
        ).fetchone()[0]

        con.execute(f"""
            CREATE TABLE ranked_train AS
            SELECT
                wallet,
                n_train,
                train_skill,
                NTILE(10) OVER (
                    ORDER BY train_skill, wallet
                ) AS decile
            FROM train
            WHERE n_train >= {args.min_train}
        """)

        con.execute(f"""
            CREATE TABLE eligible AS
            SELECT
                t.wallet,
                t.n_train,
                t.train_skill,
                t.decile,
                f.n_future,
                f.future_skill
            FROM ranked_train t
            JOIN future f USING (wallet)
            WHERE f.n_future >= {args.min_future}
        """)

        n_wallets = con.execute(
            "SELECT COUNT(*) FROM eligible"
        ).fetchone()[0]

        n_deciles = con.execute(
            "SELECT COUNT(DISTINCT decile) FROM eligible"
        ).fetchone()[0]

        log(
            f"Historical-label wallets: "
            f"{n_train_wallets:,}"
        )
        log(
            f"Future new-market wallets: "
            f"{n_future_wallets:,}"
        )
        log(
            f"Eligible OOS wallets: {n_wallets:,}"
        )

        if n_wallets < 100 or n_deciles < 10:
            diagnostic = {
                "historical_wallets": n_train_wallets,
                "future_wallets": n_future_wallets,
                "eligible_wallets": n_wallets,
                "observed_deciles": n_deciles,
                "resolved_markets": metadata[1],
            }

            (
                out / "RESOLUTION_COVERAGE_DIAGNOSTIC.json"
            ).write_text(
                json.dumps(diagnostic, indent=2)
            )

            raise RuntimeError(
                "Insufficient OOS cohort. "
                "Inspect the resolution coverage diagnostic."
            )

        con.execute(f"""
            COPY eligible
            TO '{out / "wallet_oos_scores.parquet"}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        wallets = con.execute(
            "SELECT * FROM eligible"
        ).df()

        deciles = (
            wallets.groupby("decile")
            .agg(
                n_wallets=("wallet", "size"),
                mean_train_score=(
                    "train_skill", "mean"
                ),
                mean_future_score=(
                    "future_skill", "mean"
                ),
                future_positive_pct=(
                    "future_skill",
                    lambda s: 100 * (s > 0).mean(),
                ),
                mean_n_train=("n_train", "mean"),
                mean_n_future=("n_future", "mean"),
            )
            .reset_index()
        )

        # Step 5: Market-cluster bootstrap
        log("[5/7] Computing clustered bootstrap CIs...")

        contrib = con.execute(f"""
            SELECT
                w.cluster_id,
                e.decile,
                SUM(
                    w.future_score / e.n_future
                ) AS score_contrib,
                SUM(
                    1.0 / e.n_future
                ) AS weight_contrib
            FROM wm w
            JOIN eligible e USING (wallet)
            WHERE w.first_ts >= {split}
              AND w.n_future_obs > 0
            GROUP BY 1, 2
        """).df()

        boot = clustered_bootstrap(
            contrib,
            n_boot=args.bootstrap,
            seed=17,
        )

        log(
            f"Outcome clusters: {boot[4]:,} | "
            f"Bootstrap draws: {args.bootstrap}"
        )

        deciles["cluster_ci_low"] = boot[1]
        deciles["cluster_ci_high"] = boot[2]

        deciles.to_csv(
            out / "decile_oos_summary.csv",
            index=False,
        )

        # Step 6: Publication-quality figures
        log("[6/7] Generating seven figures...")

        contrast, contrast_ci = make_figures(
            wallets, deciles, boot, figdir
        )

        # Step 7: Final report
        log("[7/7] Saving final research summary...")

        report = {
            "git_commit": commit,
            "split_date_utc": args.split_date,
            "min_train_markets": args.min_train,
            "min_future_markets": args.min_future,
            "registry_markets": metadata[0],
            "registry_with_resolved_at": metadata[1],
            "registry_conflicting_resolved_at": metadata[2],
            "train_wallets_any": n_train_wallets,
            "future_wallets_any": n_future_wallets,
            "eligible_wallets": n_wallets,
            "shared_market_event_clusters": boot[4],
            "bootstrap_replicates": args.bootstrap,
            "train_future_wallet_score_correlation": float(
                wallets["train_skill"].corr(
                    wallets["future_skill"]
                )
            ),
            "d10_minus_d1_future_score": float(
                contrast
            ),
            "d10_minus_d1_95pct_cluster_ci": [
                float(x) for x in contrast_ci
            ],
            "caveats": [
                "Historical deciles use only pre-cutoff data.",
                "Future participation is selected ex-post.",
                "Training labels require resolved_at <= cutoff.",
                "resolved_at is not independently audited.",
                "Future scores use newly sampled markets.",
                "Clustering covers markets and known multi-events.",
                "Bootstrap does not capture every dependency.",
                "Historical 12h probability proxies may lag.",
                "No multiple-testing or profitability claim.",
            ],
        }

        report_path = (
            out / "SKILL_PERSISTENCE_SUMMARY.json"
        )
        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n=== RESEARCH RESULTS ===", flush=True)

        for key in (
            "eligible_wallets",
            "shared_market_event_clusters",
            "train_future_wallet_score_correlation",
            "d10_minus_d1_future_score",
            "d10_minus_d1_95pct_cluster_ci",
        ):
            log(f"{key}: {report[key]}")

        con.close()

    log(f"\n=== COMPLETE: {out} ===")


if __name__ == "__main__":
    main()
