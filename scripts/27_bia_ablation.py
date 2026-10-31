
#!/usr/bin/env python3
"""Q2: Experience vs Skill ablation and directional analysis."""

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit, logit

TEAL, ORANGE, CORAL = "#8ECFC9", "#FFBE7A", "#FA7F6F"
BLUE, PURPLE, GRAY = "#82B0D2", "#BEB8DC", "#999999"
EPS = 1e-5


def log(message):
    print(message, flush=True)


def load_previous_script():
    path = Path(__file__).with_name("25_bia_models.py")
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("bia25", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def market_means(frame, predictions, market_ids, counts):
    y = frame.y_yes.to_numpy(dtype=float)
    results = {}

    for name, p in predictions.items():
        p = np.clip(p, EPS, 1 - EPS)
        brier = (p - y) ** 2
        ll = -y * np.log(p) - (1 - y) * np.log1p(-p)

        results[name] = {
            "brier": np.bincount(
                market_ids, weights=brier
            ) / counts,
            "logloss": np.bincount(
                market_ids, weights=ll
            ) / counts,
        }

    return results


def cluster_interval(deltas, clusters, n_boot, seed):
    """Resample event clusters with equal market weighting."""
    event_ids, unique = pd.factorize(
        clusters, sort=False
    )
    totals = np.bincount(event_ids, weights=deltas)
    sizes = np.bincount(event_ids)

    rng = np.random.default_rng(seed)
    draws = []
    k = len(unique)

    for start in range(0, n_boot, 50):
        batch = min(50, n_boot - start)
        ids = rng.integers(0, k, (batch, k))

        values = (
            totals[ids].sum(axis=1)
            / sizes[ids].sum(axis=1)
        )
        draws.extend(values.tolist())

    low, high = np.quantile(draws, [.025, .975])
    return float(low), float(high), k


def get_skill_coefficients(path):
    frame = pd.read_csv(path)
    frame = frame[
        frame.model == "M3_wallet_intelligence"
    ]

    result = {}
    for feature in [
        "historical_shrunk_skill",
        "direction_x_skill",
    ]:
        row = frame[
            frame.feature == "numeric__" + feature
        ]
        if len(row) != 1:
            raise RuntimeError(
                f"Missing M3 coefficient: {feature}"
            )
        result[feature] = float(
            row.coefficient.iloc[0]
        )

    return result


def make_figures(comparisons, groups, figdir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Figure 1: Experience and skill contributions.
    fig, ax = plt.subplots(figsize=(8.3, 4.9))

    order = [
        "Experience vs M2",
        "Skill vs Experience",
    ]

    for j, (split, color) in enumerate([
        ("validation", ORANGE),
        ("final_test", BLUE),
    ]):
        sub = comparisons[
            (comparisons.split == split)
            & (comparisons.metric == "Brier")
        ].set_index("comparison").loc[order]

        x = np.arange(2) + (j - .5) * .25
        y = sub.delta.to_numpy() * 10000

        low = (
            sub.delta - sub.ci_low
        ).to_numpy() * 10000

        high = (
            sub.ci_high - sub.delta
        ).to_numpy() * 10000

        ax.errorbar(
            x, y,
            yerr=[
                np.maximum(low, 0),
                np.maximum(high, 0),
            ],
            fmt="o",
            capsize=4,
            lw=1.9,
            markersize=7,
            color=color,
            label=split,
        )

    ax.axhline(0, color=GRAY, ls="--")
    ax.set_xticks(
        range(2),
        ["+ Wallet experience", "+ Historical skill"],
    )
    ax.set_ylabel(
        "Brier improvement × 10,000"
    )
    ax.set_title(
        "Experience vs Skill | Event-cluster 95% CI"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.13)

    fig.savefig(
        figdir / "01_experience_vs_skill.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2: Signed skill contribution.
    order = [
        "Negative / BUY YES",
        "Negative / BUY NO",
        "Positive / BUY YES",
        "Positive / BUY NO",
    ]

    test = groups[
        groups.split == "final_test"
    ].set_index("group").reindex(order)

    colors = [CORAL, CORAL, TEAL, TEAL]

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.barh(
        order,
        test.mean_skill_delta_p.to_numpy() * 10000,
        color=colors,
    )

    ax.axvline(0, color=GRAY, ls="--")
    ax.set_xlabel(
        "YES probability shift from skill features (bps)"
    )
    ax.set_title(
        "How Does Historical Skill Adjust Predictions?"
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.12)

    fig.savefig(
        figdir / "02_signed_skill_adjustments.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 3: Skill-specific predictive improvement.
    fig, ax = plt.subplots(figsize=(8.6, 4.9))

    ax.barh(
        order,
        test.delta_brier.to_numpy() * 10000,
        color=colors,
    )

    ax.axvline(0, color=GRAY, ls="--")
    ax.set_xlabel(
        "M3 vs Experience-only Brier improvement × 10,000"
    )
    ax.set_title(
        "Does Historical Skill Improve Prediction Accuracy?"
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.12)

    fig.savefig(
        figdir / "03_skill_gain_by_side.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--grace-days", type=int, default=7)
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()

    if args.bootstrap < 100:
        raise ValueError("Bootstrap must be >= 100")

    bia = load_previous_script()

    base = (
        Path(args.drive_root).expanduser()
        / "data" / "research_results"
    )
    features = (
        base / f"p1a_bia_features_{args.grace_days}d"
    )
    original = (
        base / f"p1a_bia_models_{args.grace_days}d"
    )
    out = (
        base / f"p1a_bia_ablation_{args.grace_days}d"
    )
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------
    # 1. Fit experience-only model
    # ---------------------------------------

    log("[1/5] Fitting Experience-only model...")

    train = bia.load_frame(
        features / "train.parquet"
    )

    exp_columns = (
        bia.MODEL_SPECS["M2_anonymous_behavior"]
        + ["log_past_markets", "log_past_resolved"]
    )

    ids, counts = bia.market_codes(train)
    weights = bia.balanced_weights(ids, counts)

    model = bia.make_model(exp_columns)
    model.fit(
        train[exp_columns + bia.M1_CAT],
        train.y_yes.to_numpy(dtype=int),
        classifier__sample_weight=weights,
    )

    coefs = get_skill_coefficients(
        original / "model_coefficients.csv"
    )

    skill = train.historical_shrunk_skill.to_numpy(
        dtype=float
    )
    direction_skill = (
        skill * train.direction.to_numpy(dtype=float)
    )

    sd_skill = float(skill.std())
    sd_interaction = float(direction_skill.std())

    if min(sd_skill, sd_interaction) < 1e-12:
        raise RuntimeError("Skill variance too small")

    log(f"      Training rows: {len(train):,}")
    log(f"      Markets: {len(counts):,}")

    del train, ids, counts, weights
    del skill, direction_skill

    comparison_rows = []
    score_rows = []
    group_rows = []

    # ---------------------------------------
    # 2. Evaluation
    # ---------------------------------------

    for j, split in enumerate([
        "validation", "final_test"
    ]):
        log(f"[2/5] Evaluating {split}...")

        df = bia.load_frame(
            features / f"{split}.parquet"
        )

        saved = pd.read_parquet(
            original / f"{split}_predictions.parquet",
            columns=[
                "wallet", "market_type",
                "condition_id", "trade_ts",
                "p_M2_anonymous_behavior",
                "p_M3_wallet_intelligence",
            ],
        )

        keys = [
            "wallet", "market_type",
            "condition_id", "trade_ts",
        ]

        if not df[keys].reset_index(
            drop=True
        ).equals(saved[keys].reset_index(drop=True)):
            raise RuntimeError(
                f"{split}: prediction row alignment mismatch"
            )

        p2 = saved.p_M2_anonymous_behavior.to_numpy(
            dtype=float
        )
        p3 = saved.p_M3_wallet_intelligence.to_numpy(
            dtype=float
        )

        pe = model.predict_proba(
            df[exp_columns + bia.M1_CAT]
        )[:, 1]

        market_ids, market_counts = (
            bia.market_codes(df)
        )

        losses = market_means(
            df,
            {
                "M2": p2,
                "Experience": pe,
                "M3": p3,
            },
            market_ids,
            market_counts,
        )

        clusters = df.drop_duplicates(
            "market_key"
        ).cluster_id.to_numpy()

        if len(clusters) != len(market_counts):
            raise RuntimeError(
                "Market-cluster alignment mismatch"
            )

        for name in ["M2", "Experience", "M3"]:
            score_rows.append({
                "split": split,
                "model": name,
                "brier": float(
                    losses[name]["brier"].mean()
                ),
                "logloss": float(
                    losses[name]["logloss"].mean()
                ),
                "markets": len(market_counts),
            })

        comparisons = [
            ("Experience vs M2", "Experience", "M2"),
            ("Skill vs Experience", "M3", "Experience"),
            ("M3 vs M2", "M3", "M2"),
        ]

        for k, (label, new, old) in enumerate(comparisons):
            for metric in ["brier", "logloss"]:
                delta = (
                    losses[old][metric]
                    - losses[new][metric]
                )

                low, high, n_clusters = cluster_interval(
                    delta, clusters,
                    args.bootstrap,
                    40 + j * 41 + k * 11
                    + int(metric == "logloss"),
                )

                row = {
                    "split": split,
                    "comparison": label,
                    "metric": (
                        "Brier" if metric == "brier"
                        else "LogLoss"
                    ),
                    "delta": float(delta.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "markets": len(delta),
                    "event_clusters": n_clusters,
                }

                comparison_rows.append(row)

                log(
                    f"      {row['metric']:<7} "
                    f"{label:<21} "
                    f"{row['delta']:+.7f} "
                    f"CI [{low:+.7f}, {high:+.7f}]"
                )

        # ---------------------------------------
        # 3. Signed skill contribution
        # ---------------------------------------

        log(f"[3/5] Skill direction: {split}...")

        h = df.historical_shrunk_skill.to_numpy(
            dtype=float
        )
        d = df.direction.to_numpy(dtype=float)

        # Contribution of the two skill terms
        # to fitted M3 log-odds.
        log_odds_effect = (
            coefs["historical_shrunk_skill"]
            * h / sd_skill
            + coefs["direction_x_skill"]
            * h * d / sd_interaction
        )

        # Counterfactual within the SAME M3:
        # set the two skill features to zero,
        # holding all other features fixed.
        p_zero = expit(
            logit(np.clip(p3, EPS, 1 - EPS))
            - log_odds_effect
        )

        signed_effect = p3 - p_zero

        eligible = (
            df.n_past_resolved_markets.to_numpy()
            >= 5
        )

        past_sign = np.where(
            h < 0, "Negative", "Positive"
        )
        side = np.where(
            d > 0, "BUY YES", "BUY NO"
        )

        labels = np.char.add(
            np.char.add(
                past_sign.astype(str), " / "
            ),
            side.astype(str),
        )

        y = df.y_yes.to_numpy(dtype=float)

        # M3 vs experience-only, same observations
        delta_brier = (
            (pe - y) ** 2
            - (p3 - y) ** 2
        )

        groups = [
            "Negative / BUY YES",
            "Negative / BUY NO",
            "Positive / BUY YES",
            "Positive / BUY NO",
        ]

        for group in groups:
            mask = (
                eligible
                & (labels == group)
                & (h != 0)
            )

            if not mask.any():
                continue

            sub = pd.DataFrame({
                "market_key": df.loc[
                    mask, "market_key"
                ].to_numpy(),
                "signed": signed_effect[mask],
                "delta_brier": delta_brier[mask],
            })

            per_market = sub.groupby(
                "market_key", sort=False
            ).agg(
                shift=("signed", "mean"),
                gain=("delta_brier", "mean"),
            )

            group_rows.append({
                "split": split,
                "group": group,
                "rows": int(mask.sum()),
                "markets": len(per_market),
                "mean_skill_delta_p": float(
                    per_market["shift"].mean()
                ),
                "share_upward_row": float(
                    np.mean(signed_effect[mask] > 0)
                ),
                "delta_brier": float(
                    per_market["gain"].mean()
                ),
            })

        del df, saved

    # ---------------------------------------
    # 4. Save results
    # ---------------------------------------

    log("[4/5] Saving tables and figures...")

    comparisons_df = pd.DataFrame(comparison_rows)
    groups_df = pd.DataFrame(group_rows)

    pd.DataFrame(score_rows).to_csv(
        out / "ablation_scores.csv",
        index=False,
    )

    comparisons_df.to_csv(
        out / "ablation_improvements.csv",
        index=False,
    )

    groups_df.to_csv(
        out / "skill_direction_diagnostics.csv",
        index=False,
    )

    make_figures(
        comparisons_df, groups_df, figures
    )

    script_folder = Path(__file__).resolve().parent
    repo = (
        script_folder.parent
        if (script_folder.parent / ".git").exists()
        else script_folder
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo, text=True,
    ).strip()

    manifest = {
        "git_commit": commit,
        "additional_model": "M2 + Experience",
        "baseline_models": "Existing M2 and M3",
        "weighting": "Equal market weight",
        "bootstrap": "Event-cluster",
        "bootstrap_resamples": args.bootstrap,
        "skill_counterfactual": (
            "Within fitted M3, set skill features "
            "to zero while keeping other features fixed"
        ),
        "status": (
            "Exploratory post-hoc analysis; "
            "final test already inspected"
        ),
        "skill_coefficients": coefs,
        "training_sd": {
            "skill": sd_skill,
            "direction_x_skill": sd_interaction,
        },
    }

    (out / "ABLATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2)
    )

    log(f"[5/5] DONE: {out}")


if __name__ == "__main__":
    main()
