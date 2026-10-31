
#!/usr/bin/env python3
"""Part I-A Q2: compare market-only vs wallet-behavior models."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPS = 1e-5

M1_NUM = ["market_logit", "price_age_hours"]
M1_CAT = ["category", "market_type"]

M2_EXTRA = [
    "direction",
    "log_trade_size",
    "direction_x_size",
]

M3_EXTRA = [
    "log_past_markets",
    "log_past_resolved",
    "historical_shrunk_skill",
    "direction_x_skill",
]

MODEL_SPECS = {
    "M1_calibrated_market": M1_NUM,
    "M2_anonymous_behavior": M1_NUM + M2_EXTRA,
    "M3_wallet_intelligence": M1_NUM + M2_EXTRA + M3_EXTRA,
}

READ_COLS = [
    "wallet", "market_type", "condition_id",
    "cluster_id", "trade_ts", "label_available_ts",
    "y_yes", "p_market_before", "category",
    "price_age_hours", "direction", "log_trade_size",
    "n_past_sampled_markets", "n_past_resolved_markets",
    "historical_shrunk_skill",
]


def log(message):
    print(message, flush=True)


def load_frame(path):
    df = pd.read_parquet(path, columns=READ_COLS)

    if df.empty:
        raise RuntimeError(f"No rows in {path}")

    df["category"] = (
        df["category"].fillna("unknown").astype(str)
    )
    df["market_type"] = (
        df["market_type"].fillna("unknown").astype(str)
    )

    df["market_key"] = df["market_type"].str.cat(
        df["condition_id"].astype(str),
        sep=":",
    )

    p = np.clip(
        df["p_market_before"].to_numpy(dtype=float),
        EPS, 1 - EPS,
    )
    df["market_logit"] = np.log(p / (1 - p))

    df["direction_x_size"] = (
        df["direction"] * df["log_trade_size"]
    )
    df["log_past_markets"] = np.log1p(
        df["n_past_sampled_markets"]
    )
    df["log_past_resolved"] = np.log1p(
        df["n_past_resolved_markets"]
    )
    df["direction_x_skill"] = (
        df["direction"] * df["historical_shrunk_skill"]
    )

    numeric = M1_NUM + M2_EXTRA + M3_EXTRA

    if not np.isfinite(
        df[numeric].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(f"Non-finite features in {path}")

    if not df["y_yes"].isin([0, 1]).all():
        raise RuntimeError(f"Invalid labels in {path}")

    return df


def market_codes(df):
    ids, uniques = pd.factorize(
        df["market_key"], sort=False
    )
    counts = np.bincount(
        ids, minlength=len(uniques)
    )
    return ids, counts


def balanced_weights(ids, counts):
    # Every market receives the same total weight.
    # Normalize weights to mean 1.
    n = len(ids)

    return (
        n / (len(counts) * counts[ids])
    ).astype(np.float64)


def make_model(num_columns):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                num_columns,
            ),
            (
                "category",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=300,
                    sparse_output=True,
                ),
                M1_CAT,
            ),
        ],
        sparse_threshold=1.0,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                solver="lbfgs",
                max_iter=200,
                C=1.0,
                tol=1e-4,
            ),
        ),
    ])


def evaluate(y, predictions, ids, counts):
    p = np.clip(predictions, EPS, 1 - EPS)

    brier = (p - y) ** 2

    logloss = -(
        y * np.log(p)
        + (1 - y) * np.log1p(-p)
    )

    n_markets = len(counts)

    market_brier = np.bincount(
        ids, weights=brier, minlength=n_markets
    ) / counts

    market_logloss = np.bincount(
        ids, weights=logloss, minlength=n_markets
    ) / counts

    return {
        "rows": int(len(y)),
        "markets": int(n_markets),
        "brier_market": float(
            np.mean(market_brier)
        ),
        "logloss_market": float(
            np.mean(market_logloss)
        ),
        "brier_row": float(np.mean(brier)),
        "logloss_row": float(np.mean(logloss)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument(
        "--grace-days", type=int, default=7
    )
    args = ap.parse_args()

    root = Path(args.drive_root).expanduser()

    source = (
        root / "data" / "research_results"
        / f"p1a_bia_features_{args.grace_days}d"
    )

    out = (
        root / "data" / "research_results"
        / f"p1a_bia_models_{args.grace_days}d"
    )
    out.mkdir(parents=True, exist_ok=True)

    for name in ("train", "validation", "final_test"):
        path = source / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)

    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo, text=True,
    ).strip()

    log(
        f"[START] BIA model comparison | "
        f"Git {commit[:12]}"
    )

    # -------------------------------------
    # 1. Training data
    # -------------------------------------

    log("[1/5] Loading training observations...")

    train = load_frame(
        source / "train.parquet"
    )

    train_ids, train_counts = market_codes(train)

    weights = balanced_weights(
        train_ids, train_counts
    )

    log(
        f"      {len(train):,} rows | "
        f"{len(train_counts):,} markets"
    )

    y_train = train["y_yes"].to_numpy(dtype=int)

    fitted = {}
    coefficient_rows = []

    # -------------------------------------
    # 2. Fit statistical models
    # -------------------------------------

    log("[2/5] Training logistic models...")

    for name, columns in MODEL_SPECS.items():
        log(f"      Fitting {name} ...")

        model = make_model(columns)

        model.fit(
            train[columns + M1_CAT],
            y_train,
            classifier__sample_weight=weights,
        )

        fitted[name] = model

        feature_names = (
            model.named_steps["preprocessor"]
            .get_feature_names_out()
        )
        coefficients = (
            model.named_steps["classifier"].coef_[0]
        )

        coefficient_rows.extend({
            "model": name,
            "feature": str(feature),
            "coefficient": float(value),
        } for feature, value in zip(
            feature_names, coefficients
        ))

        log(f"      Done: {name}")

    pd.DataFrame(coefficient_rows).to_csv(
        out / "model_coefficients.csv",
        index=False,
    )

    train_market_keys = set(
        train["market_key"].unique()
    )
    train_event_keys = set(
        train["cluster_id"].unique()
    )

    del train, train_ids, train_counts
    del weights, y_train

    # -------------------------------------
    # 3. Validation and final test
    # -------------------------------------

    log(
        "[3/5] Evaluating validation "
        "and final holdout..."
    )

    scores = []
    overlaps = {}

    for split in ("validation", "final_test"):
        df = load_frame(
            source / f"{split}.parquet"
        )

        ids, counts = market_codes(df)
        y = df["y_yes"].to_numpy(dtype=int)

        log(
            f"      {split}: {len(df):,} rows | "
            f"{len(counts):,} markets"
        )

        overlaps[split] = {
            "markets_shared_with_training": int(
                len(train_market_keys.intersection(
                    df["market_key"].unique()
                ))
            ),
            "event_clusters_shared_with_training": int(
                len(train_event_keys.intersection(
                    df["cluster_id"].unique()
                ))
            ),
        }

        if overlaps[split]["markets_shared_with_training"]:
            log(
                "      [WARNING] Shared markets: "
                f"{overlaps[split]['markets_shared_with_training']:,}"
            )

        # M0: unchanged historical market probability
        predictions = {
            "M0_raw_market":
                df["p_market_before"].to_numpy(dtype=float)
        }

        # M1, M2, M3: same observations, extra features
        for name, columns in MODEL_SPECS.items():
            predictions[name] = (
                fitted[name].predict_proba(
                    df[columns + M1_CAT]
                )[:, 1]
            )

        for name, p in predictions.items():
            scores.append({
                "split": split,
                "model": name,
                **evaluate(y, p, ids, counts),
            })

        # Store predictions for subsequent analysis
        saved = df[[
            "wallet",
            "market_type",
            "condition_id",
            "cluster_id",
            "trade_ts",
            "y_yes",
            "p_market_before",
            "direction",
            "log_trade_size",
            "n_past_resolved_markets",
            "historical_shrunk_skill",
            "category",
            "price_age_hours",
        ]].copy()

        for name, p in predictions.items():
            saved["p_" + name] = p.astype(
                np.float32
            )

        saved.to_parquet(
            out / f"{split}_predictions.parquet",
            compression="zstd",
            index=False,
        )

        del df, saved, ids, counts, y, predictions

    # -------------------------------------
    # 4. Model comparison
    # -------------------------------------

    log("[4/5] Comparing incremental improvements...")

    table = pd.DataFrame(scores)
    table.to_csv(
        out / "model_scores.csv", index=False
    )

    comparisons = [
        ("M1_calibrated_market", "M0_raw_market"),
        ("M2_anonymous_behavior", "M1_calibrated_market"),
        ("M3_wallet_intelligence", "M2_anonymous_behavior"),
    ]

    improvements = []
    indexed = table.set_index(["split", "model"])

    for split in ("validation", "final_test"):
        log(
            f"\n=== {split.upper()} "
            "(market-equal weighting) ==="
        )

        for row in scores:
            if row["split"] == split:
                log(
                    f"{row['model']:<25} "
                    f"Brier={row['brier_market']:.6f} "
                    f"LogLoss={row['logloss_market']:.6f}"
                )

        for candidate, baseline in comparisons:
            a = indexed.loc[(split, candidate)]
            b = indexed.loc[(split, baseline)]

            db = float(
                b["brier_market"] - a["brier_market"]
            )
            dl = float(
                b["logloss_market"] - a["logloss_market"]
            )

            improvements.append({
                "split": split,
                "candidate": candidate,
                "baseline": baseline,
                "delta_brier": db,
                "delta_logloss": dl,
            })

            log(
                f"{candidate} vs {baseline}: "
                f"Delta Brier={db:+.6f} | "
                f"Delta LogLoss={dl:+.6f}"
            )

    pd.DataFrame(improvements).to_csv(
        out / "incremental_improvements.csv",
        index=False,
    )

    # -------------------------------------
    # 5. Save manifest
    # -------------------------------------

    log("[5/5] Saving manifest...")

    manifest = {
        "git_commit": commit,
        "source": str(source),
        "models": {
            name: columns + M1_CAT
            for name, columns in MODEL_SPECS.items()
        },
        "training": (
            "Market-equal weighting; "
            "LogisticRegression C=1"
        ),
        "evaluation": (
            "Equal-weighted market-level "
            "Brier and Log Loss"
        ),
        "overlap_diagnostics": overlaps,
        "scores": scores,
        "improvements": improvements,
        "limitations": [
            "Historical probability can lag by up to 12h",
            "Cluster bootstrap not yet performed",
            "Event overlap is diagnostic only",
        ],
    }

    (out / "BIA_MODELS_MANIFEST.json").write_text(
        json.dumps(
            manifest, indent=2, ensure_ascii=False
        )
    )

    log(f"\n[DONE] {out}")
    log("[NEXT] Behavioral interpretation and visualization")


if __name__ == "__main__":
    main()
