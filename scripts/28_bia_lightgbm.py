
#!/usr/bin/env python3
"""Part I-A Q2: LightGBM behavioral information models."""

import argparse
import json
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAT = ["category", "market_type"]
BASE = ["p_market_before", "price_age_hours"]
ACTION = [
    "direction", "log_trade_size", "direction_x_size"
]
EXPERIENCE = ["log_past_markets", "log_past_resolved"]
SKILL = ["historical_shrunk_skill", "direction_x_skill"]

SPECS = {
    "L1_market": BASE + CAT,
    "L2_action": BASE + ACTION + CAT,
    "L3_experience": BASE + ACTION + EXPERIENCE + CAT,
    "L4_skill": BASE + ACTION + EXPERIENCE + SKILL + CAT,
}

COMPARISONS = [
    ("L2_action", "L1_market"),
    ("L3_experience", "L2_action"),
    ("L4_skill", "L3_experience"),
    ("L4_skill", "L2_action"),
]

READ = sorted(set(
    BASE + ACTION + EXPERIENCE + SKILL + CAT + [
        "wallet", "condition_id", "cluster_id",
        "trade_ts", "y_yes",
        "n_past_sampled_markets",
        "n_past_resolved_markets",
    ]
))

EPS = 1e-5
TEAL = "#8ECFC9"
ORANGE = "#FFBE7A"
BLUE = "#82B0D2"
GRAY = "#999999"


def log(msg):
    print(msg, flush=True)


def read_data(path, categories=None):
    derived = {
        "direction_x_size", "direction_x_skill",
        "log_past_markets", "log_past_resolved"
    }

    df = pd.read_parquet(
        path,
        columns=[c for c in READ if c not in derived]
    )

    if df.empty:
        raise RuntimeError(f"Empty dataset: {path}")

    df["market_key"] = (
        df.market_type.fillna("unknown")
        .astype(str)
        .str.cat(df.condition_id.astype(str), sep=":")
    )

    df["direction_x_size"] = (
        df.direction * df.log_trade_size
    )
    df["direction_x_skill"] = (
        df.direction * df.historical_shrunk_skill
    )
    df["log_past_markets"] = np.log1p(
        df.n_past_sampled_markets
    )
    df["log_past_resolved"] = np.log1p(
        df.n_past_resolved_markets
    )

    for col in CAT:
        s = df[col].fillna("unknown").astype(str)

        if categories is None:
            df[col] = s.astype("category")
        else:
            df[col] = pd.Categorical(
                s, categories=categories[col]
            )

    numeric = list(set(BASE + ACTION + EXPERIENCE + SKILL))

    if not np.isfinite(
        df[numeric].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Non-finite features detected")

    if not df.y_yes.isin([0, 1]).all():
        raise RuntimeError("Invalid binary labels")

    return df


def market_weights(df):
    ids, unique = pd.factorize(
        df.market_key, sort=False
    )
    counts = np.bincount(ids)

    weights = (
        len(df) / (len(counts) * counts[ids])
    )

    return weights, len(unique)


def evaluate(df, prediction):
    y = df.y_yes.to_numpy(dtype=float)
    p = np.clip(prediction, EPS, 1 - EPS)

    brier = (p - y) ** 2
    logloss = (
        -y * np.log(p)
        - (1 - y) * np.log1p(-p)
    )

    ids, markets = pd.factorize(
        df.market_key, sort=False
    )
    counts = np.bincount(ids)

    result = pd.DataFrame({
        "market_key": markets,
        "brier": (
            np.bincount(ids, weights=brier) / counts
        ),
        "logloss": (
            np.bincount(ids, weights=logloss) / counts
        ),
    })

    clusters = df.groupby(
        "market_key", sort=False
    ).cluster_id.first()

    result["cluster_id"] = (
        clusters.reindex(markets).to_numpy()
    )

    scores = {
        "brier": float(result.brier.mean()),
        "logloss": float(result.logloss.mean()),
        "rows": int(len(df)),
        "markets": int(len(markets)),
    }

    return result.set_index("market_key"), scores


def cluster_bootstrap(frame, n_boot=400, seed=42):
    group = (
        frame.groupby("cluster_id")
        .delta.agg(["sum", "size"])
    )

    sums = group["sum"].to_numpy()
    sizes = group["size"].to_numpy()

    k = len(group)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)

    for start in range(0, n_boot, 50):
        end = min(start + 50, n_boot)

        ids = rng.integers(
            0, k, size=(end - start, k)
        )

        draws[start:end] = (
            sums[ids].sum(axis=1)
            / sizes[ids].sum(axis=1)
        )

    return np.quantile(draws, [.025, .975]), k


def make_figures(gains, importance, out):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    order = [
        f"{new} vs {old}"
        for new, old in COMPARISONS
    ]

    labels = [
        "Action\nvs Market",
        "Experience\nvs Action",
        "Skill\nvs Experience",
        "Full Wallet\nvs Action",
    ]

    for metric in ("brier", "logloss"):
        fig, ax = plt.subplots(figsize=(9, 5.2))

        for offset, split, color in [
            (-0.13, "validation", ORANGE),
            (0.13, "final_test", BLUE),
        ]:
            t = (
                gains[
                    (gains.split == split)
                    & (gains.metric == metric)
                ]
                .set_index("comparison")
                .reindex(order)
            )

            x = np.arange(len(order)) + offset
            y = t.delta.to_numpy() * 10000

            low = (
                (t.delta - t.ci_low).to_numpy()
                * 10000
            )
            high = (
                (t.ci_high - t.delta).to_numpy()
                * 10000
            )

            ax.errorbar(
                x, y,
                yerr=[
                    np.maximum(low, 0),
                    np.maximum(high, 0),
                ],
                fmt="o",
                color=color,
                markersize=7,
                capsize=4,
                lw=1.9,
                label=split,
            )

        ax.axhline(0, color=GRAY, ls="--")
        ax.set_xticks(range(len(order)), labels)
        ax.set_ylabel(
            f"{metric.title()} Improvement × 10,000"
        )
        ax.set_title(
            "LightGBM | Incremental Behavioral Information"
        )
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.12)

        fig.tight_layout()
        fig.savefig(
            out / f"01_{metric}_incremental.png",
            dpi=200,
        )
        plt.close(fig)

    top = (
        importance[
            importance.model == "L4_skill"
        ]
        .sort_values("gain", ascending=False)
        .head(12)
        .iloc[::-1]
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.barh(top.feature, top.gain, color=TEAL)
    ax.set_xlabel("Total Split Gain")
    ax.set_title(
        "Wallet Intelligence | Feature Importance"
    )

    fig.tight_layout()
    fig.savefig(
        out / "02_feature_importance.png",
        dpi=200,
    )
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--grace-days", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=400)
    args = ap.parse_args()

    base = (
        Path(args.drive_root).expanduser()
        / "data" / "research_results"
    )

    source = (
        base / f"p1a_bia_features_{args.grace_days}d"
    )
    out = (
        base / f"p1a_bia_lightgbm_{args.grace_days}d"
    )
    out.mkdir(parents=True, exist_ok=True)

    for name in ("train", "validation", "final_test"):
        path = source / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)

    # -------------------------------------
    # 1. Read existing feature datasets
    # -------------------------------------

    log("[1/5] Loading feature datasets...")

    train = read_data(source / "train.parquet")
    categories = {
        c: train[c].cat.categories for c in CAT
    }
    val = read_data(
        source / "validation.parquet", categories
    )

    w_train, n_train_markets = market_weights(train)
    w_val, n_val_markets = market_weights(val)

    log(
        f"Training: {len(train):,} observations, "
        f"{n_train_markets:,} markets"
    )
    log(
        f"Validation: {len(val):,} observations, "
        f"{n_val_markets:,} markets"
    )

    # -------------------------------------
    # 2. Train nested LightGBM models
    # -------------------------------------

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.045,
        "num_leaves": 15,
        "min_data_in_leaf": 400,
        "lambda_l2": 20.0,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "max_bin": 127,
        "num_threads": args.threads,
        "seed": 42,
        "verbosity": -1,
    }

    models = {}
    best_rounds = {}
    importance_rows = []

    log("[2/5] Training LightGBM models...")

    for name, columns in SPECS.items():
        log(f"      Training {name}...")

        training_set = lgb.Dataset(
            train[columns],
            label=train.y_yes,
            weight=w_train,
            categorical_feature=CAT,
        )

        validation_set = lgb.Dataset(
            val[columns],
            label=val.y_yes,
            weight=w_val,
            categorical_feature=CAT,
            reference=training_set,
        )

        model = lgb.train(
            params,
            training_set,
            num_boost_round=650,
            valid_sets=[validation_set],
            valid_names=["validation"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=50,
                    verbose=False,
                ),
                lgb.log_evaluation(period=100),
            ],
        )

        models[name] = model
        best_rounds[name] = model.best_iteration

        model.save_model(str(out / f"{name}.txt"))

        importance_rows.extend([
            {
                "model": name,
                "feature": feature,
                "gain": float(gain),
            }
            for feature, gain in zip(
                model.feature_name(),
                model.feature_importance(
                    importance_type="gain"
                ),
            )
        ])

        log(
            f"      Done: {name} | "
            f"Best iteration={model.best_iteration}"
        )

        del training_set, validation_set

    importance = pd.DataFrame(importance_rows)
    importance.to_csv(
        out / "feature_importance.csv", index=False
    )

    del train, w_train, w_val

    # -------------------------------------
    # 3. Evaluate models
    # -------------------------------------

    log("[3/5] Evaluating out-of-sample results...")

    scores = []
    comparisons = []

    for split in ("validation", "final_test"):
        if split == "validation":
            df = val
        else:
            df = read_data(
                source / "final_test.parquet",
                categories,
            )

        predictions = {
            name: model.predict(
                df[SPECS[name]],
                num_iteration=best_rounds[name],
                num_threads=args.threads,
            )
            for name, model in models.items()
        }

        market_tables = {}

        for name, p in predictions.items():
            table, result = evaluate(df, p)

            market_tables[name] = table

            scores.append({
                "split": split,
                "model": name,
                **result,
            })

            log(
                f"      {split} | {name} | "
                f"Brier={result['brier']:.6f} | "
                f"LogLoss={result['logloss']:.6f}"
            )

        for idx, (new, old) in enumerate(COMPARISONS):
            a = market_tables[new]
            b = market_tables[old]

            for metric in ("brier", "logloss"):
                delta = b[metric] - a[metric]

                frame = pd.DataFrame({
                    "cluster_id": a.cluster_id,
                    "delta": delta,
                })

                ci, n_clusters = cluster_bootstrap(
                    frame,
                    n_boot=args.bootstrap,
                    seed=(
                        101 + idx * 13
                        + (split == "final_test") * 100
                        + (metric == "logloss")
                    ),
                )

                comparisons.append({
                    "split": split,
                    "comparison": f"{new} vs {old}",
                    "metric": metric,
                    "delta": float(delta.mean()),
                    "ci_low": float(ci[0]),
                    "ci_high": float(ci[1]),
                    "markets": len(frame),
                    "clusters": n_clusters,
                })

        saved = df[[
            "wallet", "market_type", "condition_id",
            "cluster_id", "trade_ts", "y_yes",
            "p_market_before", "direction",
            "historical_shrunk_skill",
            "n_past_resolved_markets",
            "price_age_hours", "category",
        ]].copy()

        for name, prediction in predictions.items():
            saved[f"p_{name}"] = prediction.astype(
                np.float32
            )

        saved.to_parquet(
            out / f"{split}_predictions.parquet",
            compression="zstd",
            index=False,
        )

        del df, saved, predictions, market_tables

    # -------------------------------------
    # 4. Save statistical results
    # -------------------------------------

    log("[4/5] Saving model comparison...")

    scores_df = pd.DataFrame(scores)
    gains_df = pd.DataFrame(comparisons)

    scores_df.to_csv(
        out / "lightgbm_scores.csv", index=False
    )
    gains_df.to_csv(
        out / "lightgbm_incremental.csv", index=False
    )

    make_figures(gains_df, importance, out)

    # -------------------------------------
    # 5. Manifest
    # -------------------------------------

    repo = Path(__file__).resolve().parents[1]

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo, text=True,
    ).strip()

    manifest = {
        "git_commit": commit,
        "models": SPECS,
        "parameters": params,
        "best_rounds": best_rounds,
        "bootstrap": args.bootstrap,
        "evaluation": "Market-equal Brier and Log Loss",
        "inference": "Event-cluster bootstrap",
        "status": (
            "Exploratory: final holdout "
            "previously inspected"
        ),
    }

    (out / "LIGHTGBM_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2)
    )

    log("\n=== INCREMENTAL RESULTS ===")

    for r in comparisons:
        log(
            f"{r['split']:<11} "
            f"{r['metric']:<8} "
            f"{r['comparison']:<34} "
            f"Delta={r['delta']:+.7f} "
            f"CI=[{r['ci_low']:+.7f}, "
            f"{r['ci_high']:+.7f}]"
        )

    log(f"\n[5/5] DONE: {out}")


if __name__ == "__main__":
    main()
