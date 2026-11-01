
#!/usr/bin/env python3
"""Step 34: Does a sudden stake increase make history informative?

Sources:
- Step 30: point-in-time historical sequences
- Step 32: frozen C0/C1 ensemble predictions

No new training or raw transaction reconstruction.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS, DIM = 16, 5
EPS = 1e-5

ORANGE = "#FFBE7A"
BLUE = "#82B0D2"
GRAY = "#999999"

COLS = [
    "obs_id", "market_type", "condition_id",
    "cluster_id", "y_yes", "history_count",
    "history", "direction", "log_trade_size",
    "p_market_before", "price_age_hours",
    "n_past_resolved_markets",
    "historical_shrunk_skill",
]


def log(message):
    print(message, flush=True)


def local_copy(source, target, expected=None):
    if not source.exists():
        raise FileNotFoundError(source)

    if (
        not target.exists()
        or target.stat().st_size != source.stat().st_size
    ):
        log(f"[CACHE] Copy {source.name} to runtime disk")
        shutil.copy2(source, target)

    if expected is not None:
        actual = pq.ParquetFile(
            target
        ).metadata.num_rows

        if actual != expected:
            raise ValueError(
                f"{target}: {actual} != {expected}"
            )

    return target


def build_rows(
    sequence_file, prediction_file,
    target, expected, batch_size
):
    """Create lightweight rows for history >= 5."""

    pred = pd.read_parquet(
        prediction_file,
        columns=["obs_id", "p_C0", "p_C1"],
    )

    seq = pq.ParquetFile(sequence_file)

    if (
        len(pred) != expected
        or seq.metadata.num_rows != expected
    ):
        raise ValueError(
            "Sequence/prediction row-count mismatch"
        )

    pred_ids = pred.obs_id.to_numpy(np.int64)
    p0 = pred.p_C0.to_numpy(np.float64)
    p1 = pred.p_C1.to_numpy(np.float64)

    processed = 0
    retained = 0
    writer = None

    try:
        for batch in seq.iter_batches(
            columns=COLS,
            batch_size=batch_size,
        ):
            n = batch.num_rows

            meta = (
                pa.Table.from_batches([batch])
                .drop(["history"])
                .to_pandas()
            )

            if not np.array_equal(
                meta.obs_id.to_numpy(np.int64),
                pred_ids[processed:processed+n],
            ):
                raise RuntimeError(
                    f"obs_id mismatch at {processed}"
                )

            keep = (
                meta.history_count.to_numpy(np.int16)
                >= 5
            )

            if keep.any():
                ht = batch.column(
                    batch.schema.get_field_index("history")
                )

                if (
                    not pa.types.is_fixed_size_list(ht.type)
                    or ht.type.list_size != STEPS * DIM
                ):
                    raise ValueError(
                        f"Unexpected history type: {ht.type}"
                    )

                hist = (
                    ht.values.to_numpy(
                        zero_copy_only=False
                    )
                    .reshape(n, STEPS, DIM)[keep]
                )

                m = meta.loc[keep].reset_index(
                    drop=True
                )

                hn = m.history_count.to_numpy(
                    np.int16
                )

                # Step 30 stores valid events right-aligned.
                valid = (
                    np.arange(STEPS)[None, :]
                    >= STEPS - hn[:, None]
                )

                previous_size = (
                    (hist[:, :, 1] * valid).sum(axis=1)
                    / hn
                )

                size_change = (
                    m.log_trade_size.to_numpy(np.float32)
                    - previous_size
                )

                stake = np.where(
                    size_change > np.log(2),
                    "Higher",
                    np.where(
                        size_change < -np.log(2),
                        "Lower",
                        "Similar",
                    ),
                )

                days = np.expm1(hist[:, -1, 4])
                recent = days <= 7.0

                y = m.y_yes.to_numpy(np.float64)

                a = np.clip(
                    p0[processed:processed+n][keep],
                    EPS, 1-EPS,
                )
                b = np.clip(
                    p1[processed:processed+n][keep],
                    EPS, 1-EPS,
                )

                chosen_p = np.where(
                    m.direction.to_numpy() > 0,
                    m.p_market_before.to_numpy(),
                    1 - m.p_market_before.to_numpy(),
                )

                output = pd.DataFrame({
                    "market_type":
                        m.market_type.fillna("unknown")
                        .astype(str).to_numpy(),
                    "condition_id":
                        m.condition_id.fillna("unknown")
                        .astype(str).to_numpy(),
                    "cluster_id":
                        m.cluster_id.fillna("unknown")
                        .astype(str).to_numpy(),
                    "stake_group": stake,
                    "recent": recent.astype(np.int8),
                    "history_count": hn.astype(np.int8),
                    "current_log_size":
                        m.log_trade_size.to_numpy(
                            np.float32
                        ),
                    "chosen_probability":
                        chosen_p.astype(np.float32),
                    "price_age_hours":
                        m.price_age_hours.to_numpy(
                            np.float32
                        ),
                    "resolved_history":
                        m.n_past_resolved_markets.to_numpy(
                            np.int32
                        ),
                    "historical_shrunk_skill":
                        m.historical_shrunk_skill.to_numpy(
                            np.float32
                        ),
                    "direction":
                        m.direction.to_numpy(np.int8),
                    "gain_brier":
                        (
                            (y-a)**2 - (y-b)**2
                        ).astype(np.float32),
                    "gain_logloss":
                        (
                            -y*np.log(a)
                            -(1-y)*np.log1p(-a)
                            +y*np.log(b)
                            +(1-y)*np.log1p(-b)
                        ).astype(np.float32),
                })

                tab = pa.Table.from_pandas(
                    output,
                    preserve_index=False,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        target,
                        tab.schema,
                        compression="zstd",
                    )

                writer.write_table(tab)
                retained += len(output)

            processed += n

            if processed % 500000 < n:
                log(
                    f"    Scanned {processed:,} / "
                    f"{expected:,} | "
                    f"History >=5: {retained:,}"
                )

    finally:
        if writer is not None:
            writer.close()

    if processed != expected or retained == 0:
        raise RuntimeError(
            "Incomplete analysis-row generation"
        )

    return retained


def market_group_table(con, path, feature):
    expressions = {
        "stake_group": "stake_group",
        "recency": """
            CASE WHEN recent = 1
            THEN 'Recent' ELSE 'Older' END
        """,
        "stake_x_recency": """
            stake_group || '_' ||
            CASE WHEN recent = 1
            THEN 'Recent' ELSE 'Older' END
        """,
    }

    expression = expressions[feature]

    return con.execute(f"""
        SELECT
            market_type,
            condition_id,
            MIN(cluster_id) AS cluster_id,
            {expression} AS grp,
            COUNT(*) AS rows,
            AVG(gain_brier) AS brier,
            AVG(gain_logloss) AS logloss
        FROM read_parquet(?)
        GROUP BY 1, 2, 4
    """, [str(path)]).df()


def cluster_bootstrap(values, clusters, times, seed):
    t = pd.DataFrame({
        "cluster": clusters,
        "diff": np.asarray(values),
    })

    g = (
        t.groupby("cluster", sort=False)["diff"]
        .agg(["sum", "size"])
    )

    sums = g["sum"].to_numpy()
    sizes = g["size"].to_numpy()

    if len(sums) < 2:
        raise RuntimeError(
            "Too few event clusters"
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(times)

    for start in range(0, times, 40):
        n = min(40, times-start)

        sample = rng.integers(
            0,
            len(sums),
            size=(n, len(sums)),
        )

        draws[start:start+n] = (
            sums[sample].sum(axis=1)
            / sizes[sample].sum(axis=1)
        )

    return np.quantile(
        draws, [0.025, 0.975]
    ), len(sums)


def paired_contrast(
    grouped, old, new, feature,
    metric, split, times, seed
):
    # Markets must contain BOTH comparison groups.
    pivot = grouped.pivot(
        index=["market_type", "condition_id"],
        columns="grp",
        values=metric,
    )

    if old not in pivot or new not in pivot:
        raise ValueError(
            f"Missing comparison groups: {new}/{old}"
        )

    pair = pivot[[old, new]].dropna()

    if pair.empty:
        raise RuntimeError(
            f"No shared markets for {new}/{old}"
        )

    clusters = (
        grouped.drop_duplicates([
            "market_type", "condition_id"
        ])
        .set_index([
            "market_type", "condition_id"
        ])
        .loc[pair.index, "cluster_id"]
    )

    delta = (
        pair[new].to_numpy()
        - pair[old].to_numpy()
    )

    ci, nclusters = cluster_bootstrap(
        delta,
        clusters.to_numpy(),
        times,
        seed,
    )

    return {
        "split": split,
        "feature": feature,
        "contrast": f"{new} vs {old}",
        "metric": metric,
        "delta": float(delta.mean()),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "common_markets": len(pair),
        "clusters": nclusters,
    }


REGRESSORS = [
    "higher",
    "lower",
    "recent",
    "higher_x_recent",
    "lower_x_recent",
    "current_log_size",
    "chosen_probability",
    "price_age_hours",
    "log_history_count",
    "log_resolved_history",
    "historical_shrunk_skill",
    "direction",
]


def fixed_effects_regression(d):
    """Market FE + equal-market WLS + event-cluster SE.

    The dependent variable is the C1-vs-C0
    Brier gain multiplied by 10,000.
    """

    market_key = pd.MultiIndex.from_arrays([
        d.market_type,
        d.condition_id,
    ])

    market, unique = pd.factorize(
        market_key, sort=False
    )

    nmarkets = len(unique)

    counts = np.bincount(
        market, minlength=nmarkets
    ).astype(np.float64)

    # Every market has total weight 1.
    weight = 1.0 / counts[market]

    higher = (
        d.stake_group.to_numpy() == "Higher"
    ).astype(np.float64)

    lower = (
        d.stake_group.to_numpy() == "Lower"
    ).astype(np.float64)

    recent = d.recent.to_numpy(np.float64)

    X = np.column_stack([
        higher,
        lower,
        recent,
        higher * recent,
        lower * recent,
        d.current_log_size.to_numpy(np.float64),
        d.chosen_probability.to_numpy(np.float64),
        d.price_age_hours.to_numpy(np.float64),
        np.log1p(
            d.history_count.to_numpy(np.float64)
        ),
        np.log1p(
            d.resolved_history.to_numpy(np.float64)
        ),
        d.historical_shrunk_skill.to_numpy(
            np.float64
        ),
        d.direction.to_numpy(np.float64),
    ])

    y = (
        d.gain_brier.to_numpy(np.float64)
        * 10000
    )

    if (
        not np.isfinite(X).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError(
            "Non-finite regression inputs"
        )

    # Within-market transformation:
    # absorb one fixed effect per market without
    # constructing thousands of dummy columns.
    for j in range(X.shape[1]):
        avg = (
            np.bincount(
                market,
                weights=X[:, j],
                minlength=nmarkets,
            ) / counts
        )
        X[:, j] -= avg[market]

    y -= (
        np.bincount(
            market,
            weights=y,
            minlength=nmarkets,
        ) / counts
    )[market]

    # Scale continuous controls to stabilize inversion.
    scales = np.ones(X.shape[1])

    for j in range(5, X.shape[1]):
        scales[j] = max(
            np.sqrt(
                (
                    weight * X[:, j]**2
                ).sum() / nmarkets
            ),
            1e-8,
        )
        X[:, j] /= scales[j]

    bread = X.T @ (weight[:, None] * X)

    rank = np.linalg.matrix_rank(bread)

    if rank != X.shape[1]:
        raise RuntimeError(
            f"Rank-deficient FE design: "
            f"{rank}/{X.shape[1]}"
        )

    inv_bread = np.linalg.inv(bread)

    beta_scaled = (
        inv_bread @ (X.T @ (weight * y))
    )

    residual = y - X @ beta_scaled

    events, names = pd.factorize(
        d.cluster_id,
        sort=False,
    )

    nclusters = len(names)

    scores = np.stack([
        np.bincount(
            events,
            weights=(
                weight * X[:, j] * residual
            ),
            minlength=nclusters,
        )
        for j in range(X.shape[1])
    ], axis=1)

    meat = scores.T @ scores

    covariance_scaled = (
        inv_bread @ meat @ inv_bread
    )

    if nclusters > 1:
        covariance_scaled *= (
            nclusters / (nclusters - 1)
        )

    factor = 1 / scales

    beta = beta_scaled * factor

    covariance = (
        factor[:, None]
        * covariance_scaled
        * factor[None, :]
    )

    se = np.sqrt(
        np.maximum(
            np.diag(covariance), 0
        )
    )

    coefficients = pd.DataFrame({
        "term": REGRESSORS,
        "coefficient_x10000": beta,
        "std_error_x10000": se,
        "ci_low_x10000": beta - 1.96*se,
        "ci_high_x10000": beta + 1.96*se,
    })

    contrasts = {
        "Higher vs Similar | older": {
            "higher": 1
        },
        "Higher vs Similar | recent": {
            "higher": 1,
            "higher_x_recent": 1,
        },
        "Higher effect: recent minus older": {
            "higher_x_recent": 1
        },
        "Recent vs Older | similar stake": {
            "recent": 1
        },
        "Recent effect | higher stake": {
            "recent": 1,
            "higher_x_recent": 1,
        },
    }

    rows = []

    for name, terms in contrasts.items():
        c = np.zeros(len(REGRESSORS))

        for term, coef in terms.items():
            c[REGRESSORS.index(term)] = coef

        estimate = float(c @ beta)

        stderr = float(
            np.sqrt(
                max(c @ covariance @ c, 0)
            )
        )

        rows.append({
            "contrast": name,
            "estimate_x10000": estimate,
            "std_error_x10000": stderr,
            "ci_low_x10000":
                estimate - 1.96*stderr,
            "ci_high_x10000":
                estimate + 1.96*stderr,
        })

    diagnostics = {
        "rows": len(y),
        "markets": nmarkets,
        "event_clusters": nclusters,
        "rank": int(rank),
        "within_market_equal_weight": True,
    }

    return (
        coefficients,
        pd.DataFrame(rows),
        diagnostics,
    )


def make_figure(direct, effects, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.1)
    )

    specifications = [
        (
            axes[0],
            direct[direct.metric == "brier"],
            "delta", "ci_low", "ci_high",
            "Unadjusted within-market pairs",
            [
                "Higher vs Similar",
                "Recent vs Older",
                "Higher_Recent vs Similar_Recent",
            ],
            10000,
        ),
        (
            axes[1],
            effects,
            "estimate_x10000",
            "ci_low_x10000",
            "ci_high_x10000",
            "Market FE + context adjustment",
            [
                "Higher vs Similar | older",
                "Higher vs Similar | recent",
                "Higher effect: recent minus older",
            ],
            1,
        ),
    ]

    for (
        ax, df, point, low, high,
        title, groups, scale
    ) in specifications:

        for i, (split, color) in enumerate([
            ("validation", ORANGE),
            ("final_test", BLUE),
        ]):
            t = (
                df[df.split == split]
                .set_index("contrast")
                .reindex(groups)
            )

            values = t[point].to_numpy() * scale
            lows = t[low].to_numpy() * scale
            highs = t[high].to_numpy() * scale

            x = (
                np.arange(len(groups))
                + (i - 0.5) * 0.17
            )

            ax.errorbar(
                x,
                values,
                yerr=[
                    np.maximum(values-lows, 0),
                    np.maximum(highs-values, 0),
                ],
                fmt="o",
                capsize=4,
                lw=1.8,
                color=color,
                label=split,
            )

        ax.axhline(
            0,
            ls="--",
            lw=1,
            color=GRAY,
        )

        labels = [
            s.replace(" | ", "\n")
             .replace(" vs ", " / ")
             .replace("_", " ")
            for s in groups
        ]

        ax.set_xticks(
            range(len(groups)),
            labels,
            rotation=8,
            ha="right",
        )

        ax.set_title(title)
        ax.set_ylabel(
            "Difference in Brier gain × 10,000"
        )
        ax.grid(axis="y", alpha=0.12)

    axes[0].legend(frameon=False)

    fig.suptitle(
        "Does Increasing Stake Make History "
        "More Informative?"
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--grace-days", type=int, default=7)
    ap.add_argument(
        "--cache-root",
        default="/content/bia34_cache",
    )
    ap.add_argument("--batch-size", type=int, default=32000)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--memory-gb", type=int, default=6)

    args = ap.parse_args()

    base = (
        Path(args.drive_root)
        / "data" / "research_results"
    )

    seq_folder = (
        base
        / f"p1a_bia_sequences_{args.grace_days}d"
    )

    model_folder = (
        base
        / f"p1a_bia_history_control_{args.grace_days}d"
    )

    out = (
        base
        / f"p1a_bia_stake_heterogeneity_{args.grace_days}d"
    )

    out.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache_root)
    cache.mkdir(parents=True, exist_ok=True)

    sequence_manifest = json.loads(
        (
            seq_folder / "SEQUENCE_MANIFEST.json"
        ).read_text()
    )

    control_manifest = json.loads(
        (
            model_folder / "CONTROL_MANIFEST.json"
        ).read_text()
    )

    if (
        sequence_manifest["split_rows"]
        != control_manifest["rows"]
    ):
        raise ValueError(
            "Step 30/32 row manifests disagree"
        )

    con = duckdb.connect()
    con.execute(
        f"SET memory_limit='{args.memory_gb}GB'"
    )
    con.execute("SET threads=4")

    rows = []
    coef_tables = []
    effect_tables = []
    diagnostics = {}

    log("[START] Step 34 | Stake heterogeneity")

    comparison_specs = [
        ("stake_group", "Similar", "Higher"),
        ("stake_group", "Lower", "Higher"),
        ("recency", "Older", "Recent"),
        (
            "stake_x_recency",
            "Similar_Recent",
            "Higher_Recent",
        ),
        (
            "stake_x_recency",
            "Similar_Older",
            "Higher_Older",
        ),
    ]

    for si, split in enumerate(
        ("validation", "final_test")
    ):
        log(f"[{split}] Loading existing data")

        expected = int(
            sequence_manifest["split_rows"][split]
        )

        seq = local_copy(
            seq_folder / f"{split}.parquet",
            cache / f"{split}.parquet",
            expected,
        )

        prediction = (
            model_folder
            / f"{split}_ensemble_predictions.parquet"
        )

        if not prediction.exists():
            raise FileNotFoundError(prediction)

        data_path = (
            cache / f"analysis34_{split}.parquet"
        )

        kept = build_rows(
            seq,
            prediction,
            data_path,
            expected,
            args.batch_size,
        )

        expected_h5 = int(
            sequence_manifest["history_ge_5"][split]
        )

        if kept != expected_h5:
            raise ValueError(
                f"History coverage: "
                f"{kept} vs {expected_h5}"
            )

        log(
            f"[{split}] Eligible: {kept:,}"
        )

        tables = {
            feature: market_group_table(
                con, data_path, feature
            )
            for feature in (
                "stake_group",
                "recency",
                "stake_x_recency",
            )
        }

        log(
            f"[{split}] Paired market comparisons"
        )

        for j, (
            feature, old, new
        ) in enumerate(comparison_specs):

            for metric in ("brier", "logloss"):
                r = paired_contrast(
                    tables[feature],
                    old,
                    new,
                    feature,
                    metric,
                    split,
                    args.bootstrap,
                    3400 + si*100 + j*2
                    + (metric == "logloss"),
                )

                rows.append(r)

                if metric == "brier":
                    log(
                        f"    {r['contrast']:<34} "
                        f"Markets={r['common_markets']:,} | "
                        f"Gain gap ×1e4="
                        f"{r['delta']*1e4:+.3f}"
                    )

        log(
            f"[{split}] Market Fixed Effects"
        )

        columns = [
            "market_type",
            "condition_id",
            "cluster_id",
            "stake_group",
            "recent",
            "history_count",
            "current_log_size",
            "chosen_probability",
            "price_age_hours",
            "resolved_history",
            "historical_shrunk_skill",
            "direction",
            "gain_brier",
        ]

        d = pd.read_parquet(
            data_path, columns=columns
        )

        coef, effects, info = (
            fixed_effects_regression(d)
        )

        coef.insert(0, "split", split)
        effects.insert(0, "split", split)

        coef_tables.append(coef)
        effect_tables.append(effects)
        diagnostics[split] = info

        del d, tables

    direct = pd.DataFrame(rows)

    adjusted = pd.concat(
        coef_tables, ignore_index=True
    )

    effects = pd.concat(
        effect_tables, ignore_index=True
    )

    direct.to_csv(
        out / "paired_market_contrasts.csv",
        index=False,
    )

    adjusted.to_csv(
        out / "fixed_effects_coefficients.csv",
        index=False,
    )

    effects.to_csv(
        out / "adjusted_contrasts.csv",
        index=False,
    )

    make_figure(
        direct,
        effects,
        out / "01_stake_context_effects.png",
    )

    repo = Path(__file__).resolve().parents[1]

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()

    report = {
        "git_commit": commit,
        "sequence_commit":
            sequence_manifest.get("git_commit"),
        "model_commit":
            control_manifest.get("git_commit"),
        "source": (
            "Step 30 sequences + "
            "Step 32 frozen ensemble predictions"
        ),
        "sample": "history_count >= 5",
        "stake_definition": (
            "Current log1p(USDC) minus mean "
            "historical log1p(USDC). "
            "Higher: >ln(2), Lower: <-ln(2)."
        ),
        "recency_definition": (
            "Most recent qualifying BUY <=7 days"
        ),
        "direct_contrast": (
            "Equal market weight among markets "
            "containing both groups; "
            "Event-cluster bootstrap"
        ),
        "adjusted": (
            "Within-market fixed effects, "
            "equal-market WLS, "
            "Event-cluster robust covariance"
        ),
        "regressors": REGRESSORS,
        "bootstrap": args.bootstrap,
        "diagnostics": diagnostics,
        "exploratory": True,
        "limitations": [
            "Final test previously inspected",
            "Multiple exploratory comparisons",
            "Stake size does not directly measure confidence",
            "Qualifying BUY history only, up to 16",
            "Lagged historical market probabilities",
            "Conditional association, not causality",
        ],
    }

    (
        out / "STAKE_MANIFEST.json"
    ).write_text(
        json.dumps(report, indent=2)
    )

    log(
        "\n=== DIRECT MARKET-PAIRED "
        "BRIER CONTRASTS × 10,000 ==="
    )

    show = direct[
        direct.metric == "brier"
    ].copy()

    for c in ("delta", "ci_low", "ci_high"):
        show[c] = (
            show[c] * 10000
        ).round(3)

    print(
        show[[
            "split", "contrast",
            "delta", "ci_low", "ci_high",
            "common_markets",
        ]].to_string(index=False),
        flush=True,
    )

    log(
        "\n=== ADJUSTED MARKET-FE CONTRASTS "
        "× 10,000 ==="
    )

    print(
        effects[[
            "split",
            "contrast",
            "estimate_x10000",
            "ci_low_x10000",
            "ci_high_x10000",
        ]].round(3).to_string(index=False),
        flush=True,
    )

    con.close()

    log(f"[DONE] {out}")


if __name__ == "__main__":
    main()
