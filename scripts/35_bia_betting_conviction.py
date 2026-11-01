
#!/usr/bin/env python3
"""Step 35: Does betting bigger imply better judgment?

Input: Step 30 historical sequences.
No model training or raw transaction processing.
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

STEPS, DIM = 16, 5
GROUPS = ("Lower", "Similar", "Higher")
COLORS = {
    "validation": "#FFBE7A",
    "final_test": "#82B0D2",
}

COLUMNS = [
    "market_type", "condition_id", "cluster_id",
    "y_yes", "direction", "p_market_before",
    "log_trade_size", "history_count", "history",
]


def log(message):
    print(message, flush=True)


def build_rows(source, target, batch_size):
    """Keep history >=5 and compute market-adjusted outcome."""
    pf = pq.ParquetFile(source)
    writer = None
    seen = kept = 0

    try:
        for batch in pf.iter_batches(
            columns=COLUMNS, batch_size=batch_size
        ):
            n = batch.num_rows
            table = pa.Table.from_batches([batch])
            df = table.drop(["history"]).to_pandas()

            counts = df.history_count.to_numpy(np.int16)
            use = counts >= 5

            if use.any():
                seq_col = table.column(
                    "history"
                ).combine_chunks()

                if (
                    not pa.types.is_fixed_size_list(seq_col.type)
                    or seq_col.type.list_size != 80
                ):
                    raise ValueError(
                        f"Unexpected history format: {seq_col.type}"
                    )

                h = (
                    seq_col.values.to_numpy(
                        zero_copy_only=False
                    )
                    .reshape(n, STEPS, DIM)[use]
                )

                m = df.loc[use].reset_index(drop=True)
                lengths = counts[use]

                valid = (
                    np.arange(STEPS)[None, :]
                    >= STEPS - lengths[:, None]
                )

                prior_log_size = (
                    (h[:, :, 1] * valid).sum(axis=1)
                    / lengths
                )

                size_change = (
                    m.log_trade_size.to_numpy(np.float32)
                    - prior_log_size
                )

                group = np.where(
                    size_change > np.log(2),
                    "Higher",
                    np.where(
                        size_change < -np.log(2),
                        "Lower",
                        "Similar",
                    ),
                )

                side = m.direction.to_numpy()
                p_yes = m.p_market_before.to_numpy(
                    np.float64
                )

                # Probability of the outcome actually bought.
                q = np.where(side > 0, p_yes, 1 - p_yes)

                y = m.y_yes.to_numpy(np.int8)
                won = np.where(
                    side > 0, y, 1 - y
                ).astype(np.int8)

                recent = (
                    np.expm1(h[:, -1, 4]) <= 7.0
                ).astype(np.int8)

                rows = pd.DataFrame({
                    "market_type":
                        m.market_type.fillna("unknown").astype(str),
                    "condition_id":
                        m.condition_id.fillna("unknown").astype(str),
                    "cluster_id":
                        m.cluster_id.fillna("unknown").astype(str),
                    "stake_group": group,
                    "recent": recent,
                    "won": won,
                    "market_prob": q.astype(np.float32),
                    "advantage": (won - q).astype(np.float32),
                })

                piece = pa.Table.from_pandas(
                    rows, preserve_index=False
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        target, piece.schema,
                        compression="zstd",
                    )

                writer.write_table(piece)
                kept += len(rows)

            seen += n

            if seen % 500000 < n:
                log(
                    f"    Read {seen:,}/{pf.metadata.num_rows:,}"
                    f" | eligible {kept:,}"
                )

    finally:
        if writer is not None:
            writer.close()

    if seen != pf.metadata.num_rows or kept == 0:
        raise RuntimeError("Incomplete sequence scan")

    return kept


def cluster_ci(values, clusters, reps, seed):
    a = pd.DataFrame({
        "event": np.asarray(clusters),
        "value": np.asarray(values),
    })

    g = (
        a.groupby("event", sort=False)
        .value.agg(["sum", "size"])
    )

    sums = g["sum"].to_numpy()
    sizes = g["size"].to_numpy()

    if len(sums) < 2:
        return (np.nan, np.nan), len(sums)

    rng = np.random.default_rng(seed)
    draws = np.empty(reps)

    for start in range(0, reps, 40):
        k = min(40, reps - start)

        idx = rng.integers(
            0, len(sums),
            size=(k, len(sums)),
        )

        draws[start:start+k] = (
            sums[idx].sum(axis=1)
            / sizes[idx].sum(axis=1)
        )

    return np.quantile(
        draws, [0.025, 0.975]
    ), len(sums)


def score_split(con, path, split, reps):
    summaries = []
    contrasts = []

    for recent_only in (False, True):
        segment = (
            "recent_7d" if recent_only else "all_h5"
        )

        condition = (
            "WHERE recent = 1" if recent_only else ""
        )

        market = con.execute(f"""
            SELECT
                market_type,
                condition_id,
                MIN(cluster_id) AS cluster_id,
                stake_group,
                COUNT(*) AS n,
                AVG(won) AS win_rate,
                AVG(market_prob) AS implied_rate,
                AVG(advantage) AS advantage
            FROM read_parquet(?)
            {condition}
            GROUP BY market_type, condition_id, stake_group
        """, [str(path)]).df()

        keys = ["market_type", "condition_id"]

        for i, name in enumerate(GROUPS):
            m = market.loc[
                market.stake_group == name
            ]

            ci, events = cluster_ci(
                m.advantage,
                m.cluster_id,
                reps,
                35 + i + int(recent_only) * 100,
            )

            summaries.append({
                "split": split,
                "segment": segment,
                "stake_group": name,
                "rows": int(m.n.sum()),
                "markets": len(m),
                "events": events,
                "win_rate": float(m.win_rate.mean()),
                "implied_rate": float(m.implied_rate.mean()),
                "advantage": float(m.advantage.mean()),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
            })

        # Fairer comparison: same markets, equal market weight.
        wide = market.pivot(
            index=keys,
            columns="stake_group",
            values="advantage",
        )

        market_cluster = (
            market.drop_duplicates(keys)
            .set_index(keys)["cluster_id"]
        )

        for j, baseline in enumerate(
            ("Similar", "Lower")
        ):
            matched = wide[
                ["Higher", baseline]
            ].dropna()

            difference = (
                matched["Higher"] - matched[baseline]
            )

            events = market_cluster.loc[
                matched.index
            ]

            ci, n_events = cluster_ci(
                difference,
                events,
                reps,
                350 + j + int(recent_only) * 100,
            )

            contrasts.append({
                "split": split,
                "segment": segment,
                "contrast": f"Higher vs {baseline}",
                "difference": float(difference.mean()),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "common_markets": len(matched),
                "events": n_events,
            })

    return summaries, contrasts


def make_figures(groups, contrasts, out):
    plt.rcParams.update({
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    specs = [
        (
            groups,
            "01_group_advantage.png",
            "Market-adjusted outcome by stake change",
            list(GROUPS),
            "stake_group",
            "advantage",
            "Realized minus implied probability (pp)",
        ),
        (
            contrasts,
            "02_common_market_contrasts.png",
            "Higher stake versus other groups",
            ["Higher vs Similar", "Higher vs Lower"],
            "contrast",
            "difference",
            "Within-market advantage gap (pp)",
        ),
    ]

    for (
        data, filename, title, categories,
        key, value_key, ylabel
    ) in specs:

        fig, ax = plt.subplots(
            figsize=(8.8, 4.7)
        )

        for offset, split in (
            (-0.13, "validation"),
            (0.13, "final_test"),
        ):
            sub = data[
                (data.split == split)
                & (data.segment == "all_h5")
            ].set_index(key).reindex(categories)

            values = sub[value_key].to_numpy() * 100

            low = np.maximum(
                0,
                values - sub.ci_low.to_numpy() * 100,
            )

            high = np.maximum(
                0,
                sub.ci_high.to_numpy() * 100 - values,
            )

            ax.errorbar(
                np.arange(len(categories)) + offset,
                values,
                yerr=[low, high],
                fmt="o",
                capsize=4,
                markersize=7,
                color=COLORS[split],
                label=split,
            )

        ax.axhline(
            0, ls="--", lw=1, color="#999999"
        )

        ax.set_xticks(
            np.arange(len(categories)),
            categories,
        )

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.12)
        ax.legend(frameon=False)

        fig.tight_layout()
        fig.savefig(
            out / filename, dpi=200
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--drive-root", required=True
    )
    parser.add_argument(
        "--batch-size", type=int, default=32000
    )
    parser.add_argument(
        "--bootstrap", type=int, default=400
    )
    parser.add_argument(
        "--grace-days", type=int, default=7
    )

    args = parser.parse_args()

    base = (
        Path(args.drive_root)
        / "data" / "research_results"
    )

    seq_dir = (
        base
        / f"p1a_bia_sequences_{args.grace_days}d"
    )

    out = (
        base
        / f"p1a_bia_betting_conviction_{args.grace_days}d"
    )
    out.mkdir(parents=True, exist_ok=True)

    cache = Path("/content/bia35_cache")
    cache.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (seq_dir / "SEQUENCE_MANIFEST.json").read_text()
    )

    con = duckdb.connect()

    summaries = []
    contrasts = []

    log("[START] Step 35: Betting Conviction vs Market")

    for split in ("validation", "final_test"):
        src = seq_dir / f"{split}.parquet"
        dst = cache / f"{split}.parquet"

        if (
            not dst.exists()
            or dst.stat().st_size != src.stat().st_size
        ):
            log(f"[CACHE] {split} -> local runtime")
            shutil.copy2(src, dst)

        actual = pq.ParquetFile(
            dst
        ).metadata.num_rows

        if actual != manifest["split_rows"][split]:
            raise RuntimeError(
                "Unexpected sequence row count"
            )

        tmp = cache / f"conviction_{split}.parquet"

        eligible = build_rows(
            dst, tmp, args.batch_size
        )

        if eligible != manifest["history_ge_5"][split]:
            raise RuntimeError(
                f"History>=5 coverage mismatch: {eligible}"
            )

        log(
            f"[EVAL] {split}: {eligible:,} observations"
        )

        a, b = score_split(
            con, tmp, split, args.bootstrap
        )

        summaries.extend(a)
        contrasts.extend(b)

    gs = pd.DataFrame(summaries)
    cs = pd.DataFrame(contrasts)

    gs.to_csv(
        out / "group_market_adjusted_scores.csv",
        index=False,
    )

    cs.to_csv(
        out / "common_market_contrasts.csv",
        index=False,
    )

    make_figures(gs, cs, out)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()

    report = {
        "git_commit": commit,
        "sequence_git_commit": manifest["git_commit"],
        "definition": (
            "Realized chosen outcome minus "
            "pretrade market-implied probability"
        ),
        "stake_groups": (
            "Current log1p(stake) minus mean historical "
            "log1p(stake), thresholds +/-ln(2)"
        ),
        "sample": (
            "First qualifying wallet-market BUY, "
            "history_count >= 5"
        ),
        "bootstrap": args.bootstrap,
        "exploratory": True,
        "limitations": [
            "Previously inspected final test",
            "Market probability is a lagged proxy",
            "Not trading PnL",
            "Stake size is not direct evidence of confidence",
        ],
    }

    (
        out / "CONVICTION_MANIFEST.json"
    ).write_text(
        json.dumps(report, indent=2)
    )

    print(
        "\n=== MARKET-ADJUSTED OUTCOME "
        "(percentage points) ==="
    )

    view = gs.copy()
    for c in (
        "win_rate", "implied_rate",
        "advantage", "ci_low", "ci_high"
    ):
        view[c] = (
            view[c] * 100
        ).round(3)

    print(
        view[[
            "split", "segment", "stake_group",
            "rows", "markets", "win_rate",
            "implied_rate", "advantage",
            "ci_low", "ci_high",
        ]].to_string(index=False),
        flush=True,
    )

    print(
        "\n=== COMMON-MARKET HIGHER-STAKE "
        "CONTRAST (percentage points) ==="
    )

    view = cs.copy()
    for c in ("difference", "ci_low", "ci_high"):
        view[c] = (
            view[c] * 100
        ).round(3)

    print(
        view[[
            "split", "segment", "contrast",
            "difference", "ci_low", "ci_high",
            "common_markets",
        ]].to_string(index=False),
        flush=True,
    )

    log(f"[DONE] {out}")
    con.close()


if __name__ == "__main__":
    main()
