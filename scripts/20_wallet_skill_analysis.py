
#!/usr/bin/env python3
"""Part I-A: exploratory wallet skill using existing alignment."""

import argparse
import json
import subprocess
import tempfile
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

# Existing unified 12h sample: unchanged
RULE = """
    p_market_before BETWEEN 0 AND 1
    AND price_age_seconds > 0
    AND price_age_seconds <= 43200
    AND pre_close_1h = TRUE
    AND y_yes IN (0, 1)
    AND outcome_seq IN (1, 2)
"""


def log(msg):
    print(msg, flush=True)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIGURE] {path.name}")


def quantiles(w, field, title, xlabel, path):
    q = w.groupby(field, observed=True)["dynamic_skill"].agg(
        n="size",
        q10=lambda s: s.quantile(.10),
        q25=lambda s: s.quantile(.25),
        med="median",
        q75=lambda s: s.quantile(.75),
        q90=lambda s: s.quantile(.90),
    )

    x = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(10, 5.3))

    ax.fill_between(
        x, q.q10, q.q90,
        color=BLUE, alpha=.22, label="P10-P90"
    )
    ax.fill_between(
        x, q.q25, q.q75,
        color=TEAL, alpha=.52, label="P25-P75"
    )
    ax.plot(
        x, q.med, color=CORAL,
        marker="o", lw=2, label="Median"
    )
    ax.axhline(0, color=GRAY, ls="--", lw=1)

    ax.set_xticks(
        x,
        [f"{k}\nn={v:,}" for k, v in zip(q.index, q.n)],
        fontsize=8,
    )
    ax.set(
        xlabel=xlabel,
        ylabel="Wallet directional score",
        ylim=(-1, 1),
        title=title,
    )
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.grid(axis="y")

    save(fig, path)
    return q


def figures(w, folder):
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
        "grid.alpha": .7,
        "figure.facecolor": "white",
    })

    labels = [
        "1", "2", "3-4", "5-9",
        "10-19", "20-49", "50-99", "100+"
    ]

    w["coverage"] = pd.cut(
        w.n_markets,
        [0, 1, 2, 4, 9, 19, 49, 99, np.inf],
        labels=labels,
    )

    w["months"] = pd.cut(
        w.active_months,
        [0, 1, 2, 3, 5, 9, 15],
        labels=["1", "2", "3", "4-5", "6-9", "10-15"],
    )

    # 01: Score distributions by market coverage
    fig, ax = plt.subplots(figsize=(10, 5.4))

    groups = [
        ("1-2 markets", w.n_markets <= 2, BLUE),
        ("3-9 markets", w.n_markets.between(3, 9), TEAL),
        ("10-49 markets", w.n_markets.between(10, 49), ORANGE),
        ("50+ markets", w.n_markets >= 50, CORAL),
    ]

    for label, mask, color in groups:
        values = w.loc[mask, "dynamic_skill"].to_numpy()
        if len(values) == 0:
            continue

        hist, bins = np.histogram(
            values, bins=80, range=(-1, 1)
        )
        ax.stairs(
            hist / len(values),
            bins,
            label=f"{label} (n={len(values):,})",
            color=color,
            lw=2,
        )

    ax.axvline(0, ls="--", lw=1, color=GRAY)
    ax.set(
        xlim=(-1, 1),
        xlabel="Dynamic market-adjusted score",
        ylabel="Wallet proportion per score bin",
        title="01  Raw skill distribution by market coverage",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y")
    save(fig, folder / "01_skill_distribution.png")

    # 02: Score dispersion vs coverage
    quantiles(
        w, "coverage",
        "02  Skill-score dispersion vs market coverage",
        "Distinct markets per wallet",
        folder / "02_score_coverage.png",
    )

    # 03: Extreme raw scores, not significance
    tail = w.groupby(
        "coverage", observed=True
    )["dynamic_skill"].agg(
        n="size",
        positive=lambda s: (s >= .25).mean() * 100,
        negative=lambda s: (s <= -.25).mean() * 100,
    )

    x = np.arange(len(tail))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(
        x - .2, tail.positive, .4,
        color=TEAL, label="Score >= +0.25"
    )
    ax.bar(
        x + .2, tail.negative, .4,
        color=CORAL, label="Score <= -0.25"
    )
    ax.set_xticks(x, tail.index)
    ax.set(
        xlabel="Distinct markets per wallet",
        ylabel="Wallet share (%)",
        title="03  Extreme raw scores (not significance)",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y")
    save(fig, folder / "03_extreme_scores.png")

    tail.to_csv(
        folder.parent / "raw_score_tails.csv"
    )

    # Wallets with repeated market observations
    repeats = w[w.repeat_markets > 0].copy()

    if len(repeats):
        # 04: Dynamic vs first-entry score
        sample = repeats.sample(
            min(60000, len(repeats)),
            random_state=17,
        )

        fig, ax = plt.subplots(figsize=(7.6, 6.1))
        cmap = LinearSegmentedColormap.from_list(
            "research", [BEIGE, TEAL, BLUE]
        )

        hb = ax.hexbin(
            sample.first_entry_skill,
            sample.dynamic_skill,
            gridsize=65,
            bins="log",
            mincnt=1,
            cmap=cmap,
            extent=(-1, 1, -1, 1),
        )

        ax.plot(
            [-1, 1], [-1, 1],
            ls="--", lw=1, color=GRAY,
        )
        ax.set(
            xlim=(-1, 1),
            ylim=(-1, 1),
            xlabel="First-entry score",
            ylabel="Dynamic score",
            title="04  First-entry vs dynamic scores",
        )
        fig.colorbar(
            hb, ax=ax,
            label="Wallet count (log color scale)"
        )
        save(fig, folder / "04_first_vs_dynamic.png")

        # 05: Effect of repeated observations
        delta = (
            repeats.dynamic_skill
            - repeats.first_entry_skill
        ).to_numpy()

        radius = min(
            1.0,
            max(
                .02,
                float(np.quantile(abs(delta), .995)) * 1.05
            ),
        )

        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.hist(
            delta,
            bins=100,
            range=(-radius, radius),
            color=PURPLE,
            density=True,
            edgecolor="white",
            lw=.2,
        )
        ax.axvline(0, ls="--", lw=1, color=GRAY)
        ax.set(
            xlim=(-radius, radius),
            xlabel="Dynamic - first-entry score",
            ylabel="Density (displayed range)",
            title="05  Repeated monthly entry effect",
        )
        ax.grid(axis="y")
        save(fig, folder / "05_repeated_entry_effect.png")

    # 06: Active months and raw scores
    quantiles(
        w, "months",
        "06  Score dispersion vs active months",
        "Active months per wallet",
        folder / "06_score_active_months.png",
    )

    # 07: Coverage of top/bottom raw-score wallets
    k = max(1, int(len(w) * .01))
    ranking_groups = [
        w,
        w.nlargest(k, "dynamic_skill"),
        w.nsmallest(k, "dynamic_skill"),
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.2))

    for j, (name, group, color) in enumerate(zip(
        [
            "All wallets",
            "Top 1% raw score",
            "Bottom 1% raw score",
        ],
        ranking_groups,
        [BLUE, TEAL, CORAL],
    )):
        p = group.coverage.value_counts(normalize=True)
        ax.bar(
            x + (j - 1) * .27,
            [100 * p.get(c, 0) for c in labels],
            .27,
            color=color,
            label=name,
        )

    ax.set_xticks(x, labels)
    ax.set(
        xlabel="Distinct markets per wallet",
        ylabel="Within-group wallet share (%)",
        title="07  Do extreme rankings favor small samples?",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y")
    save(fig, folder / "07_extreme_ranking_coverage.png")

    return {
        "wallets_with_repeated_markets": int(len(repeats)),
        "mean_abs_score_change_repeated_wallets": (
            float(abs(
                repeats.dynamic_skill
                - repeats.first_entry_skill
            ).mean()) if len(repeats) else 0
        ),
        "top_1pct_raw_score_median_markets": float(
            ranking_groups[1].n_markets.median()
        ),
        "bottom_1pct_raw_score_median_markets": float(
            ranking_groups[2].n_markets.median()
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--memory-gb", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.drive_root) / "data"
    aligned = root / "probability_alignment"

    if len(list(aligned.glob("wallet_market_*.parquet"))) != 15:
        raise RuntimeError(
            "Expected exactly 15 monthly alignment files"
        )

    profile = (
        root / "research_results"
        / "p1a_wallet_diagnostics"
        / "wallet_summary.parquet"
    )

    if not profile.exists():
        raise RuntimeError(
            f"Missing previous diagnostic file: {profile}"
        )

    out = root / "research_results" / "p1a_wallet_skill"
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parents[1]
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()

    log("=== PART I-A: WALLET SKILL ===")
    log(f"Git commit: {git_sha}")

    with tempfile.TemporaryDirectory(
        prefix="p1_skill_"
    ) as temp:

        tmp = Path(temp)
        (tmp / "spill").mkdir()

        con = duckdb.connect(str(tmp / "work.duckdb"))
        con.execute(
            f"SET memory_limit='{args.memory_gb}GB'"
        )
        con.execute(
            f"SET threads={args.threads}"
        )
        con.execute(
            f"SET temp_directory='{tmp / 'spill'}'"
        )

        # 1. Score individual observations
        log("[1/5] Scoring existing 12h observations")

        con.execute(f"""
            CREATE TABLE obs AS SELECT
                LOWER(wallet) AS wallet,
                market_type,
                condition_id,
                CAST(trade_ts AS BIGINT) AS trade_ts,
                y_yes,
                outcome_seq,
                category,
                neg_risk_market_id,
                CASE
                    WHEN outcome_seq = 1
                    THEN y_yes - p_market_before
                    ELSE p_market_before - y_yes
                END AS score,
                trade_price,
                CASE
                    WHEN outcome_seq = 1
                    THEN p_market_before
                    ELSE 1 - p_market_before
                END AS side_p
            FROM read_parquet(
                '{aligned / "wallet_market_*.parquet"}'
            )
            WHERE {RULE}
        """)

        observations = con.execute(
            "SELECT COUNT(*) FROM obs"
        ).fetchone()[0]

        if observations != 7143340:
            raise RuntimeError(
                f"Observation mismatch: {observations:,}"
            )

        log(f"[CHECK] Observations: {observations:,}")

        # 2. Wallet-market aggregation
        log("[2/5] Aggregating wallet-market scores")

        con.execute("""
            CREATE TABLE wm AS SELECT
                wallet,
                market_type,
                condition_id,
                COUNT(*) AS n_monthly_obs,
                MIN(trade_ts) AS first_trade_ts,
                AVG(score) AS dynamic_market_score,
                ARG_MIN(score, trade_ts) AS first_market_score,
                ANY_VALUE(y_yes) AS y_yes,
                ANY_VALUE(category) AS category,
                ANY_VALUE(neg_risk_market_id)
                    AS neg_risk_market_id
            FROM obs
            GROUP BY 1, 2, 3
        """)

        n_pairs = con.execute(
            "SELECT COUNT(*) FROM wm"
        ).fetchone()[0]

        if n_pairs != 6728149:
            raise RuntimeError(
                f"Wallet-market mismatch: {n_pairs:,}"
            )

        log(f"[CHECK] Wallet-market pairs: {n_pairs:,}")

        # 3. Wallet-level aggregation
        log("[3/5] Computing wallet-level skill")

        con.execute(f"""
            CREATE TABLE wallet AS
            WITH scores AS (
                SELECT
                    wallet,
                    COUNT(*) AS n_markets,
                    SUM(n_monthly_obs) AS n_observations,
                    COUNT(*) FILTER (
                        WHERE n_monthly_obs > 1
                    ) AS repeat_markets,
                    AVG(dynamic_market_score)
                        AS dynamic_skill,
                    AVG(first_market_score)
                        AS first_entry_skill
                FROM wm
                GROUP BY wallet
            )
            SELECT
                scores.*,
                p.active_months,
                p.n_pre,
                p.n_post
            FROM scores
            JOIN read_parquet('{profile}') p
                USING (wallet)
        """)

        n_wallets = con.execute(
            "SELECT COUNT(*) FROM wallet"
        ).fetchone()[0]

        if n_wallets != 1169132:
            raise RuntimeError(
                f"Wallet count mismatch: {n_wallets:,}"
            )

        log(f"[CHECK] Wallets: {n_wallets:,}")

        con.execute(f"""
            COPY wallet
            TO '{out / "wallet_skill_scores.parquet"}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        con.execute(f"""
            COPY wm
            TO '{out / "wallet_market_scores.parquet"}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # 4. Visualization
        log("[4/5] Generating research figures")

        w = con.execute("""
            SELECT
                n_markets,
                repeat_markets,
                active_months,
                dynamic_skill,
                first_entry_skill
            FROM wallet
        """).df()

        extras = figures(w, fig_dir)

        # 5. Summary
        log("[5/5] Saving research summary")

        gap, yes_n, no_n = con.execute("""
            SELECT
                AVG(ABS(trade_price - side_p)),
                COUNT(*) FILTER (
                    WHERE outcome_seq = 1
                ),
                COUNT(*) FILTER (
                    WHERE outcome_seq = 2
                )
            FROM obs
        """).fetchone()

        report = {
            "git_commit": git_sha,
            "rule": "Existing unified 12h sample",
            "score": "d * (y_yes - p_market_before)",
            "weighting": (
                "Mean monthly scores within wallet-market; "
                "equal markets within wallet"
            ),
            "observations": observations,
            "wallet_market_pairs": n_pairs,
            "wallets": n_wallets,
            "yes_buy_entries": yes_n,
            "no_buy_entries": no_n,
            "mean_abs_execution_benchmark_gap": float(gap),
            "mean_wallet_dynamic_skill": float(
                w.dynamic_skill.mean()
            ),
            "median_wallet_dynamic_skill": float(
                w.dynamic_skill.median()
            ),
            "share_wallets_positive_raw_score": float(
                (w.dynamic_skill > 0).mean()
            ),
            **extras,
            "warnings": [
                "Raw scores are not proof of skill.",
                "Markets may share correlated outcomes.",
                "Historical probabilities may lag up to 12h.",
                "First-entry means first valid sampled BUY.",
                "OOS and outcome availability checks remain.",
            ],
        }

        report_path = out / "WALLET_SKILL_SUMMARY.json"
        report_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            )
        )

        for key, value in report.items():
            if key not in ("warnings", "weighting"):
                log(f"{key}: {value}")

        con.close()

    log(f"=== COMPLETE: {out} ===")


if __name__ == "__main__":
    main()
