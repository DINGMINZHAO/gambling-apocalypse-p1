
#!/usr/bin/env python3
"""Part I-A: Wallet prediction opportunity diagnostics."""

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
import numpy as np
import pandas as pd

NAVY = "#19365A"
ORANGE = "#E68A34"
PURPLE = "#74589A"
GRAY = "#687381"

# Same unified rule as scripts/18_sample_12h.py
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


def compact(x):
    return f"{int(x):,}"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 180,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.alpha": 0.18,
    })


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"[FIGURE] {path.name}")


def heatmap(ax, matrix, xlabels, ylabels, title, xlabel, ylabel):
    ax.imshow(
        np.log10(matrix + 1),
        aspect="auto",
        cmap="Blues",
        origin="lower",
    )
    ax.set_xticks(np.arange(len(xlabels)), xlabels)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    maximum = np.log10(matrix.max() + 1) if matrix.max() else 1

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            n = matrix[r, c]
            if n:
                ax.text(
                    c, r, compact(n),
                    ha="center", va="center", fontsize=8,
                    color=(
                        "white"
                        if np.log10(n + 1) > 0.6 * maximum
                        else NAVY
                    ),
                )


def make_figures(w, rep, market, months, thresholds, figs):
    style()

    # 01 — Wallet coverage CCDF
    x, counts = np.unique(
        w["n_markets"].to_numpy(),
        return_counts=True,
    )
    survival = np.cumsum(counts[::-1])[::-1] / counts.sum()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.step(x, survival, where="post", color=NAVY, lw=2.4)
    ax.set(
        xscale="log", yscale="log",
        xlabel="Distinct markets per wallet",
        ylabel="Share of wallets with at least x markets",
        title="01  Distribution of prediction opportunities",
    )
    ax.grid(True, which="major")
    savefig(figs / "01_wallet_market_coverage_ccdf.png")

    # 02 — Repeated observations in one market
    a = rep.set_index("n_buckets")["wallet_market_pairs"]
    labels = ["1", "2", "3", "4", "5", "6+"]
    values = np.array([a.get(i, 0) for i in range(1, 7)])
    pct = 100 * values / max(values.sum(), 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        labels, pct,
        color=[NAVY] + [ORANGE] * 5,
        width=0.65,
    )
    for bar, v in zip(bars, pct):
        if v >= 0.06:
            ax.annotate(
                f"{v:.1f}%",
                (bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    ax.set(
        xlabel="Monthly observations per wallet-market pair",
        ylabel="Share of wallet-market pairs (%)",
        title="02  Repeated participation in the same market",
    )
    ax.set_ylim(0, max(pct.max() * 1.17, 1))
    ax.grid(axis="y")
    savefig(figs / "02_repeat_observations.png")

    # 03 — Market coverage vs active months
    market_bins = [0, 1, 2, 4, 9, 19, 49, 99, np.inf]
    market_labels = [
        "1", "2", "3-4", "5-9",
        "10-19", "20-49", "50-99", "100+",
    ]
    month_bins = [0, 1, 2, 3, 5, 8, 12, 15]
    month_labels = [
        "1", "2", "3", "4-5",
        "6-8", "9-12", "13-15",
    ]

    mb = pd.cut(
        w["n_markets"],
        bins=market_bins,
        labels=market_labels,
    )
    ab = pd.cut(
        w["active_months"],
        bins=month_bins,
        labels=month_labels,
    )

    matrix = pd.crosstab(ab, mb).reindex(
        index=month_labels,
        columns=market_labels,
        fill_value=0,
    ).to_numpy()

    fig, ax = plt.subplots(figsize=(11, 6.3))
    heatmap(
        ax, matrix, market_labels, month_labels,
        "03  Market coverage vs active months",
        "Distinct markets per wallet",
        "Active months per wallet",
    )
    savefig(figs / "03_coverage_by_active_months.png")

    # 04 — Temporal out-of-sample opportunity coverage
    levels = [1, 3, 5, 10, 20]
    pre = w["n_pre"].to_numpy()
    post = w["n_post"].to_numpy()

    oos_matrix = np.array([
        [
            int(((pre >= p) & (post >= q)).sum())
            for p in levels
        ]
        for q in levels
    ])

    fig, ax = plt.subplots(figsize=(8.2, 6.1))
    heatmap(
        ax,
        oos_matrix,
        list(map(str, levels)),
        list(map(str, levels)),
        "04  Wallets with opportunities in both periods",
        "Min. new markets, Jan-Sep 2025",
        "Min. new markets, Oct 2025-Mar 2026",
    )
    savefig(figs / "04_oos_coverage_heatmap.png")

    oos_df = pd.DataFrame(
        oos_matrix,
        index=levels,
        columns=levels,
    )

    # 05 — Market concentration
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for col, label, color in [
        (
            "n_observations",
            "Monthly BUY observations",
            NAVY,
        ),
        (
            "n_wallet_market_pairs",
            "Distinct wallet-market pairs",
            ORANGE,
        ),
    ]:
        ordered = np.sort(market[col].to_numpy())[::-1]
        cx = np.arange(1, len(ordered) + 1)
        cx = 100 * cx / len(ordered)
        cy = 100 * np.cumsum(ordered) / ordered.sum()
        ax.plot(cx, cy, lw=2.5, color=color, label=label)

    ax.plot(
        [0, 100], [0, 100],
        ls="--", color=GRAY, alpha=0.6,
        label="Uniform benchmark",
    )
    ax.set(
        xlabel="Top x% of markets ranked by activity",
        ylabel="Cumulative sample share (%)",
        title="05  Concentration across markets",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True)
    savefig(figs / "05_market_concentration.png")

    # 06 — Monthly wallet activity
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(months))

    ax.plot(
        x, months["active_wallets"],
        color=NAVY, marker="o", lw=2.4,
        label="Active wallets",
    )
    ax.plot(
        x, months["new_wallets"],
        color=ORANGE, marker="s", lw=2,
        label="First-seen wallets in dataset",
    )
    ax.axvline(
        8.5,
        color=PURPLE,
        ls="--",
        lw=1.4,
        label="Illustrative time split",
    )
    ax.set_xticks(
        x, months["month"],
        rotation=45, ha="right",
    )
    ax.set(
        xlabel="UTC month",
        ylabel="Wallets",
        title="06  Monthly wallet activity",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y")
    savefig(figs / "06_monthly_wallet_activity.png")

    # 07 — Candidate pool sensitivity
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(thresholds))

    ax.plot(
        x, thresholds["wallets_n_markets"],
        marker="o", lw=2.5, color=NAVY,
        label="Distinct markets >= k",
    )
    ax.plot(
        x, thresholds["wallets_plus_3months"],
        marker="s", lw=2.1, color=PURPLE,
        label="Markets >= k and active months >= 3",
    )
    ax.set_yscale("log")
    ax.set_xticks(
        x,
        thresholds["min_markets"].astype(str),
    )
    ax.set(
        xlabel="Minimum distinct markets (k)",
        ylabel="Number of wallets (log scale)",
        title="07  Candidate wallet pool sensitivity",
    )
    ax.grid(axis="y", which="major")
    ax.legend(frameon=False)
    savefig(figs / "07_wallet_threshold_sensitivity.png")

    return oos_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--split-date", default="2025-10-01")
    ap.add_argument("--memory-gb", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.drive_root).expanduser()
    source = root / "data" / "probability_alignment"
    files = sorted(source.glob("wallet_market_*.parquet"))

    if len(files) != 15:
        raise RuntimeError(
            f"Expected 15 alignment files, found {len(files)}"
        )

    split_ts = int(
        datetime.strptime(
            args.split_date, "%Y-%m-%d"
        ).replace(tzinfo=timezone.utc).timestamp()
    )

    out = (
        root / "data" / "research_results"
        / "p1a_wallet_diagnostics"
    )
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    source_glob = str(source / "wallet_market_*.parquet")
    if "'" in source_glob:
        raise ValueError("Unsupported apostrophe in source path")

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "unknown"

    log("=== PART I-A: WALLET OPPORTUNITIES ===")
    log(f"Alignment files: {len(files)}")
    log(f"Git commit: {commit}")

    with tempfile.TemporaryDirectory(
        prefix="p1a_duckdb_"
    ) as tmp:
        db = Path(tmp) / "work.duckdb"
        spill = Path(tmp) / "spill"
        spill.mkdir()

        con = duckdb.connect(str(db))
        con.execute(
            f"SET memory_limit='{args.memory_gb}GB'"
        )
        con.execute(f"SET threads={args.threads}")
        con.execute(
            f"SET temp_directory='{spill}'"
        )

        # 1. Existing unified research sample
        log("[1/7] Applying existing 12h sample rule ...")

        con.execute(f"""
            CREATE TABLE valid AS
            SELECT
                LOWER(wallet) AS wallet,
                market_type,
                condition_id,
                CAST(trade_ts AS BIGINT) AS trade_ts,
                STRFTIME(
                    TO_TIMESTAMP(trade_ts), '%Y-%m'
                ) AS month
            FROM read_parquet('{source_glob}')
            WHERE {RULE}
        """)

        n_obs = con.execute(
            "SELECT COUNT(*) FROM valid"
        ).fetchone()[0]

        log(f"Valid observations: {compact(n_obs)}")

        # 2. Wallet-market aggregation
        log("[2/7] Aggregating wallet x market ...")

        con.execute("""
            CREATE TABLE wm AS
            SELECT
                wallet,
                market_type,
                condition_id,
                COUNT(*) AS n_monthly_obs,
                MIN(trade_ts) AS first_ts,
                MIN(month) AS first_month
            FROM valid
            GROUP BY 1, 2, 3
        """)

        n_pairs = con.execute(
            "SELECT COUNT(*) FROM wm"
        ).fetchone()[0]

        log(f"Wallet-market pairs: {compact(n_pairs)}")

        # 3. Wallet and market summaries
        log("[3/7] Building wallet profiles ...")

        con.execute("""
            CREATE TABLE wallet_month AS
            SELECT
                wallet,
                month,
                COUNT(*) AS n_monthly_obs
            FROM valid
            GROUP BY 1, 2
        """)

        con.execute(f"""
            CREATE TABLE wallet AS
            SELECT t.*, a.active_months
            FROM (
                SELECT
                    wallet,
                    COUNT(*) AS n_markets,
                    SUM(n_monthly_obs) AS n_observations,
                    MIN(first_ts) AS first_ts,
                    COUNT(*) FILTER (
                        WHERE first_ts < {split_ts}
                    ) AS n_pre,
                    COUNT(*) FILTER (
                        WHERE first_ts >= {split_ts}
                    ) AS n_post
                FROM wm
                GROUP BY wallet
            ) t
            JOIN (
                SELECT
                    wallet,
                    COUNT(*) AS active_months
                FROM wallet_month
                GROUP BY wallet
            ) a USING (wallet)
        """)

        con.execute("""
            CREATE TABLE market AS
            SELECT
                market_type,
                condition_id,
                COUNT(*) AS n_wallet_market_pairs,
                SUM(n_monthly_obs) AS n_observations
            FROM wm
            GROUP BY 1, 2
        """)

        totals = con.execute("""
            SELECT
                (SELECT COUNT(*) FROM wallet),
                (SELECT COUNT(*) FROM market),
                (SELECT SUM(n_markets) FROM wallet),
                (SELECT SUM(n_observations) FROM wallet)
        """).fetchone()

        assert totals[2] == n_pairs
        assert totals[3] == n_obs

        log(f"Wallets: {compact(totals[0])}")
        log(f"Markets: {compact(totals[1])}")

        if (
            n_obs, totals[0], totals[1]
        ) != (7143340, 1169132, 14925):
            log(
                "[WARNING] Counts differ from handoff. "
                "Inspect before inference."
            )

        con.execute(f"""
            COPY wallet TO '{out / "wallet_summary.parquet"}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        con.execute(f"""
            COPY market TO '{out / "market_summary.parquet"}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # 4. Repeats and monthly statistics
        log("[4/7] Computing repeats and monthly activity ...")

        rep = con.execute("""
            SELECT
                LEAST(n_monthly_obs, 6) AS n_buckets,
                COUNT(*) AS wallet_market_pairs
            FROM wm
            GROUP BY 1 ORDER BY 1
        """).df()

        market = con.execute("""
            SELECT
                n_wallet_market_pairs,
                n_observations
            FROM market
        """).df()

        w = con.execute("""
            SELECT
                n_markets, n_observations,
                active_months, n_pre, n_post
            FROM wallet
        """).df()

        monthly_active = con.execute("""
            SELECT
                month,
                COUNT(*) AS active_wallets,
                SUM(n_monthly_obs) AS observations
            FROM wallet_month
            GROUP BY 1 ORDER BY 1
        """).df()

        new_wallets = con.execute("""
            SELECT
                STRFTIME(
                    TO_TIMESTAMP(first_ts), '%Y-%m'
                ) AS month,
                COUNT(*) AS new_wallets
            FROM wallet
            GROUP BY 1 ORDER BY 1
        """).df()

        new_markets = con.execute("""
            SELECT
                first_month AS month,
                COUNT(*) AS first_market_opportunities
            FROM wm
            GROUP BY 1 ORDER BY 1
        """).df()

        months = (
            monthly_active
            .merge(new_wallets, on="month", how="left")
            .merge(new_markets, on="month", how="left")
            .fillna(0)
        )

        rep.to_csv(
            out / "repeat_distribution.csv", index=False
        )
        months.to_csv(
            out / "monthly_activity.csv", index=False
        )

        # 5. Minimum opportunity sensitivity
        log("[5/7] Computing thresholds and time coverage ...")

        levels = [1, 2, 3, 5, 10, 20, 50, 100]

        thresholds = pd.DataFrame([
            {
                "min_markets": k,
                "wallets_n_markets": int(
                    (w.n_markets >= k).sum()
                ),
                "wallets_plus_3months": int(
                    (
                        (w.n_markets >= k)
                        & (w.active_months >= 3)
                    ).sum()
                ),
            }
            for k in levels
        ])

        thresholds.to_csv(
            out / "wallet_thresholds.csv",
            index=False,
        )

        # 6. Research figures
        log("[6/7] Rendering seven figures ...")

        oos = make_figures(
            w, rep, market, months, thresholds, figs
        )

        oos.to_csv(
            out / "oos_opportunity_matrix.csv",
            index_label="min_future_markets",
        )

        # 7. Interpretation-ready summary
        log("[7/7] Writing final diagnostics ...")

        sorted_obs = np.sort(
            market.n_observations.to_numpy()
        )[::-1]

        top1 = max(
            1, int(np.ceil(0.01 * len(market)))
        )
        top5 = max(
            1, int(np.ceil(0.05 * len(market)))
        )

        summary = {
            "scope": (
                "Unified 12h; first taker BUY "
                "per wallet-market-month"
            ),
            "git_commit": commit,
            "split_date_utc": args.split_date,
            "split_note": (
                "Descriptive new-market coverage only; "
                "not leakage-safe OOS skill inference"
            ),
            "observation_count": int(n_obs),
            "unique_wallets": int(totals[0]),
            "unique_markets": int(totals[1]),
            "wallet_market_pairs": int(n_pairs),
            "additional_repeated_market_month_observations": (
                int(n_obs - n_pairs)
            ),
            "repeated_observation_share": round(
                float((n_obs - n_pairs) / n_obs), 5
            ),
            "wallet_market_median": float(
                w.n_markets.median()
            ),
            "wallet_market_p90": float(
                w.n_markets.quantile(0.90)
            ),
            "wallet_market_p99": float(
                w.n_markets.quantile(0.99)
            ),
            "wallet_active_months_median": float(
                w.active_months.median()
            ),
            "wallets_with_both_windows_one_market": int(
                ((w.n_pre >= 1) & (w.n_post >= 1)).sum()
            ),
            "wallets_with_both_windows_five_markets": int(
                ((w.n_pre >= 5) & (w.n_post >= 5)).sum()
            ),
            "top_1pct_markets_observation_share": round(
                float(sorted_obs[:top1].sum() / n_obs), 5
            ),
            "top_5pct_markets_observation_share": round(
                float(sorted_obs[:top5].sum() / n_obs), 5
            ),
            "notes": [
                "Distinct markets may share event outcomes.",
                "Repeated monthly predictions share one outcome.",
                "Historical probabilities may lag up to 12h.",
                "Close time is a proxy, not verified disclosure.",
                "No forecasting skill has been estimated.",
            ],
        }

        report = out / "WALLET_DIAGNOSTICS_SUMMARY.json"
        report.write_text(
            json.dumps(
                summary, ensure_ascii=False, indent=2
            )
        )

        log("\n=== FINAL RESULTS ===")

        for key, value in summary.items():
            if key not in ("notes", "scope", "git_commit"):
                log(f"{key}: {value}")

        log(f"\nCOMPLETE — {out}")
        con.close()


if __name__ == "__main__":
    main()
