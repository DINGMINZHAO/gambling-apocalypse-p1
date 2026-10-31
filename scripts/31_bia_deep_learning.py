
#!/usr/bin/env python3
"""Part I-A Q2: Deep Behavioral Intelligence."""

import argparse
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

EPS = 1e-5
STEPS, DIM = 16, 5

COLORS = {
    "Market": "#82B0D2",
    "S0": "#999999",
    "S1": "#8ECFC9",
    "S2": "#FA7F6F",
}

STATIC = [
    "p_market_before", "price_age_hours",
    "direction", "log_trade_size",
    "n_past_sampled_markets",
    "n_past_resolved_markets",
    "historical_shrunk_skill",
]

META = [
    "obs_id", "market_type", "condition_id",
    "cluster_id", "category", "y_yes",
    "history_count",
] + STATIC


def log(msg):
    print(msg, flush=True)


def categorical_map(values, maximum=80, minimum=100):
    counts = (
        values.fillna("unknown")
        .astype(str).value_counts()
    )
    selected = (
        counts[counts >= minimum]
        .head(maximum).index.tolist()
    )
    return {
        name: i + 1
        for i, name in enumerate(selected)
    }


def build_numeric(df):
    p = np.clip(
        df["p_market_before"].to_numpy(np.float32),
        EPS, 1 - EPS,
    )
    direction = df["direction"].to_numpy(np.float32)
    size = df["log_trade_size"].to_numpy(np.float32)
    skill = df[
        "historical_shrunk_skill"
    ].to_numpy(np.float32)

    return np.column_stack([
        np.log(p / (1 - p)),
        df["price_age_hours"].to_numpy(np.float32),
        direction,
        size,
        direction * size,
        np.log1p(
            df["n_past_sampled_markets"].to_numpy(
                np.float32
            )
        ),
        np.log1p(
            df["n_past_resolved_markets"].to_numpy(
                np.float32
            )
        ),
        skill,
        direction * skill,
    ]).astype(np.float32)


def load_split(path, maps=None):
    log(f"    Reading {path.name} ...")

    table = pq.read_table(
        path, columns=META + ["history"]
    )
    n = table.num_rows

    h = table.column("history").combine_chunks()

    if (
        not hasattr(h.type, "list_size")
        or h.type.list_size != STEPS * DIM
    ):
        raise ValueError(
            f"Expected history[80], got {h.type}"
        )

    history = (
        h.values.to_numpy(zero_copy_only=False)
        .reshape(n, STEPS, DIM).copy()
    )

    df = table.drop(["history"]).to_pandas()
    del table, h

    if maps is None:
        maps = {
            "category": categorical_map(
                df["category"], 80, 100
            ),
            "market_type": categorical_map(
                df["market_type"], 12, 1
            ),
        }

    cat_ids = np.column_stack([
        df[c].fillna("unknown").astype(str)
        .map(maps[c]).fillna(0)
        .to_numpy(np.int64)
        for c in ("category", "market_type")
    ])

    nums = build_numeric(df)
    y = df["y_yes"].to_numpy(np.float32)
    lengths = df[
        "history_count"
    ].to_numpy(np.int64)

    if (
        not np.isfinite(nums).all()
        or not np.isfinite(history).all()
    ):
        raise ValueError("Nonfinite model inputs")

    if (
        not np.isin(y, [0, 1]).all()
        or (lengths < 0).any()
        or (lengths > STEPS).any()
    ):
        raise ValueError("Invalid labels/history lengths")

    market_key = pd.MultiIndex.from_arrays([
        df["market_type"].fillna("unknown").astype(str),
        df["condition_id"].fillna("unknown").astype(str),
    ])

    ids, _ = pd.factorize(market_key, sort=False)
    ids = ids.astype(np.int32)
    counts = np.bincount(ids)

    first_idx = np.unique(
        ids, return_index=True
    )[1]

    clusters = (
        df["cluster_id"].fillna("unknown")
        .astype(str).to_numpy()[first_idx]
    )

    # Every market receives equal total training weight.
    weights = (
        n / (len(counts) * counts[ids])
    ).astype(np.float32)

    result = {
        "nums": nums,
        "cats": cat_ids,
        "history": history,
        "lengths": lengths,
        "y": y,
        "weights": weights,
        "ids": ids,
        "counts": counts,
        "clusters": clusters,
        "obs_id": df["obs_id"].to_numpy(np.int64),
        "market_type": (
            df["market_type"].fillna("unknown")
            .astype(str).to_numpy()
        ),
        "condition_id": (
            df["condition_id"].fillna("unknown")
            .astype(str).to_numpy()
        ),
        "cluster_id": (
            df["cluster_id"].fillna("unknown")
            .astype(str).to_numpy()
        ),
        "p_market": (
            df["p_market_before"]
            .to_numpy(np.float32)
        ),
    }

    del df

    log(
        f"    Loaded {n:,} rows | "
        f"{len(counts):,} markets | "
        f"history >=1: {(lengths >= 1).mean():.1%} | "
        f">=5: {(lengths >= 5).mean():.1%}"
    )

    return result, maps


def history_stats(hist, lengths, chunk=50000):
    sums = np.zeros(DIM, np.float64)
    squares = np.zeros(DIM, np.float64)
    n_events = 0

    for start in range(0, len(lengths), chunk):
        end = min(start + chunk, len(lengths))

        mask = (
            np.arange(STEPS)[None, :]
            >= STEPS - lengths[start:end, None]
        )

        x = hist[start:end][mask].astype(np.float64)

        sums += x.sum(axis=0)
        squares += np.square(x).sum(axis=0)
        n_events += len(x)

    if n_events == 0:
        raise ValueError("No training history available")

    mean = sums / n_events
    std = np.sqrt(
        np.maximum(
            squares / n_events - mean ** 2,
            1e-6,
        )
    )

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
        int(n_events),
    )


def tensor_loader(d, batch, shuffle, pin):
    ds = TensorDataset(
        torch.from_numpy(d["nums"]),
        torch.from_numpy(d["cats"]),
        torch.from_numpy(d["history"]),
        torch.from_numpy(d["lengths"]),
        torch.from_numpy(d["y"]),
        torch.from_numpy(d["weights"]),
    )

    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin,
    )


class BehavioralNet(nn.Module):
    def __init__(
        self, mode, n_cat, n_type,
        num_mean, num_std, hist_mean, hist_std,
    ):
        super().__init__()
        self.mode = mode

        self.register_buffer(
            "num_mean", torch.tensor(num_mean)
        )
        self.register_buffer(
            "num_std", torch.tensor(num_std)
        )
        self.register_buffer(
            "hist_mean", torch.tensor(hist_mean)
        )
        self.register_buffer(
            "hist_std", torch.tensor(hist_std)
        )

        self.emb_cat = nn.Embedding(n_cat + 1, 8)
        self.emb_type = nn.Embedding(n_type + 1, 4)

        self.static_encoder = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        if mode != "S0":
            self.event_encoder = nn.Sequential(
                nn.Linear(DIM, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )

            if mode == "S2":
                self.gru = nn.GRU(
                    32, 32, batch_first=True
                )

            self.hist_out = nn.Sequential(
                nn.Linear(33, 32),
                nn.ReLU(),
            )

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x, cats, history, lengths):
        x = (x - self.num_mean) / self.num_std

        static = torch.cat([
            x,
            self.emb_cat(cats[:, 0]),
            self.emb_type(cats[:, 1]),
        ], dim=-1)

        static = self.static_encoder(static)

        if self.mode == "S0":
            rep = torch.zeros_like(static)

        else:
            batch, steps, dim = history.shape

            positions = torch.arange(
                steps, device=history.device
            )[None, :]

            if self.mode == "S1":
                mask = (
                    positions >= steps - lengths[:, None]
                )
                ordered = history

            else:
                # Step 30 stores histories right-aligned.
                # Shift real events left before GRU packing.
                indices = (
                    positions + steps - lengths[:, None]
                ).clamp(max=steps - 1)

                ordered = history.gather(
                    1,
                    indices[..., None].expand(
                        -1, -1, dim
                    ),
                )

                mask = positions < lengths[:, None]

            z = (
                ordered - self.hist_mean
            ) / self.hist_std

            z = torch.where(
                mask[..., None],
                z,
                torch.zeros_like(z),
            )

            z = self.event_encoder(z)

            if self.mode == "S1":
                rep = (
                    z * mask[..., None]
                ).sum(dim=1) / lengths.clamp(min=1)[:, None]

            else:
                packed = pack_padded_sequence(
                    z,
                    lengths.clamp(min=1).cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )

                _, hidden = self.gru(packed)
                rep = hidden[-1]

            rep = self.hist_out(torch.cat([
                rep,
                torch.log1p(
                    lengths.float()
                )[:, None],
            ], dim=-1))

            # Cold-start wallets: no historical signal.
            rep = (
                rep * (lengths > 0).float()[:, None]
            )

        return self.head(
            torch.cat([static, rep], dim=-1)
        ).squeeze(-1)


def predict(model, loader, device):
    model.eval()
    parts = []

    with torch.inference_mode():
        for batch in loader:
            x, cats, hist, length = (
                z.to(device, non_blocking=True)
                for z in batch[:4]
            )

            p = torch.sigmoid(
                model(x, cats, hist, length)
            )

            parts.append(p.cpu().numpy())

    return np.concatenate(parts).astype(np.float32)


def market_metrics(d, p, subset=None):
    if subset is None:
        subset = np.ones(
            len(d["y"]), dtype=bool
        )

    ids = d["ids"][subset]
    y = d["y"][subset]

    p = np.clip(
        np.asarray(p)[subset].astype(np.float64),
        EPS, 1 - EPS,
    )

    counts = np.bincount(
        ids, minlength=len(d["counts"])
    )
    included = counts > 0

    b = (p - y) ** 2
    l = (
        -y * np.log(p)
        - (1 - y) * np.log1p(-p)
    )

    mb = (
        np.bincount(
            ids, weights=b, minlength=len(counts)
        )[included] / counts[included]
    )

    ml = (
        np.bincount(
            ids, weights=l, minlength=len(counts)
        )[included] / counts[included]
    )

    return {
        "brier": float(mb.mean()),
        "logloss": float(ml.mean()),
        "rows": int(len(y)),
        "markets": int(included.sum()),
    }, (mb, ml, included)


def bootstrap_delta(old, new, clusters, times, seed):
    g = pd.DataFrame({
        "cluster": clusters,
        "delta": np.asarray(old) - np.asarray(new),
    })

    agg = (
        g.groupby("cluster", sort=False)["delta"]
        .agg(["sum", "size"])
    )

    sums = agg["sum"].to_numpy()
    sizes = agg["size"].to_numpy()

    rng = np.random.default_rng(seed)
    draws = np.empty(times, np.float64)

    for start in range(0, times, 40):
        end = min(start + 40, times)

        sampled = rng.integers(
            0, len(sums),
            size=(end - start, len(sums)),
        )

        draws[start:end] = (
            sums[sampled].sum(axis=1)
            / sizes[sampled].sum(axis=1)
        )

    return (
        np.quantile(draws, [0.025, 0.975]),
        len(sums),
    )


def train_one(
    mode, train, val, args, stats, vocab, out, device
):
    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = BehavioralNet(
        mode, vocab[0], vocab[1], *stats
    ).to(device)

    train_dl = tensor_loader(
        train, args.batch_size, True,
        device.type == "cuda",
    )
    val_dl = tensor_loader(
        val, args.eval_batch_size, False,
        device.type == "cuda",
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,
    )

    best = float("inf")
    bad = 0
    best_state = None
    best_epoch = 0
    trace = []

    log(
        f"[TRAIN] {mode} | "
        f"{sum(p.numel() for p in model.parameters()):,} "
        "parameters"
    )

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running = 0.0
        running_w = 0.0

        for i, batch in enumerate(train_dl, start=1):
            x, cats, hist, length, y, weight = (
                z.to(device, non_blocking=True)
                for z in batch
            )

            optim.zero_grad(set_to_none=True)

            logits = model(
                x, cats, hist, length
            )

            losses = (
                nn.functional
                .binary_cross_entropy_with_logits(
                    logits, y, reduction="none"
                )
            )

            loss = (
                (losses * weight).sum()
                / weight.sum().clamp(min=1e-8)
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optim.step()

            running += float(
                (losses.detach() * weight).sum().item()
            )
            running_w += float(weight.sum().item())

            if i % args.print_every == 0:
                log(
                    f"    {mode} epoch {epoch}: "
                    f"batch {i:,}/{len(train_dl):,}"
                )

        val_p = predict(model, val_dl, device)

        scores, _ = market_metrics(val, val_p)

        row = {
            "model": mode,
            "epoch": epoch,
            "train_weighted_logloss": (
                running / running_w
            ),
            "val_market_brier": scores["brier"],
            "val_market_logloss": scores["logloss"],
        }

        trace.append(row)

        log(
            f"    {mode} epoch {epoch:02d} | "
            f"train CE={row['train_weighted_logloss']:.6f} | "
            f"val Brier={scores['brier']:.6f} | "
            f"val LogLoss={scores['logloss']:.6f} | "
            f"{time.time()-start:.0f}s"
        )

        if scores["brier"] < best - args.min_delta:
            best = scores["brier"]
            best_epoch = epoch
            bad = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= args.patience:
                log(
                    f"    {mode} early stopping "
                    f"at epoch {epoch}"
                )
                break

    if best_state is None:
        raise RuntimeError(
            "Training failed to select checkpoint"
        )

    model.load_state_dict(best_state)

    torch.save({
        "mode": mode,
        "epoch": best_epoch,
        "state_dict": best_state,
        "vocab": vocab,
    }, out / f"{mode}_best.pt")

    pd.DataFrame(trace).to_csv(
        out / f"{mode}_training.csv",
        index=False,
    )

    log(
        f"    BEST {mode}: epoch={best_epoch} | "
        f"val Market-Brier={best:.6f}"
    )

    return model, best_epoch


def old_baselines(base, split, expected_rows, expected_markets):
    specs = [
        (
            "Logistic",
            base / "p1a_bia_models_7d/model_scores.csv",
            "brier_market",
            "logloss_market",
        ),
        (
            "LightGBM",
            base / "p1a_bia_lightgbm_7d/lightgbm_scores.csv",
            "brier",
            "logloss",
        ),
    ]

    rows = []

    for family, path, b_col, l_col in specs:
        if not path.exists():
            log(
                f"[WARNING] Optional comparison "
                f"unavailable: {path}"
            )
            continue

        frame = pd.read_csv(path)

        for _, row in frame[
            frame["split"] == split
        ].iterrows():

            if (
                int(row["rows"]) != expected_rows
                or int(row["markets"]) != expected_markets
            ):
                log(
                    f"[WARNING] Skip unmatched "
                    f"{family}/{row['model']}"
                )
                continue

            rows.append({
                "split": split,
                "subgroup": "all",
                "family": family,
                "model": row["model"],
                "brier": float(row[b_col]),
                "logloss": float(row[l_col]),
                "rows": expected_rows,
                "markets": expected_markets,
            })

    return rows


def evaluate_split(
    name, d, models, device, args, base, out
):
    log(f"[EVAL] {name}")

    loader = tensor_loader(
        d, args.eval_batch_size, False,
        device.type == "cuda",
    )

    predictions = {
        "Market": d["p_market"]
    }

    for mode in ("S0", "S1", "S2"):
        predictions[mode] = predict(
            models[mode], loader, device
        )

    pd.DataFrame({
        "obs_id": d["obs_id"],
        "market_type": d["market_type"],
        "condition_id": d["condition_id"],
        "cluster_id": d["cluster_id"],
        "y_yes": d["y"].astype(np.int8),
        "history_count": (
            d["lengths"].astype(np.int8)
        ),
        "p_market_before": predictions["Market"],
        **{
            f"p_{mode}": predictions[mode]
            for mode in ("S0", "S1", "S2")
        },
    }).to_parquet(
        out / f"{name}_predictions.parquet",
        index=False,
        compression="zstd",
    )

    scores, gains = [], []

    subgroups = [
        (
            "all",
            np.ones(len(d["y"]), dtype=bool),
        ),
        ("history_0", d["lengths"] == 0),
        ("history_ge_1", d["lengths"] >= 1),
        ("history_ge_5", d["lengths"] >= 5),
    ]

    for subgroup, subset in subgroups:
        if subset.sum() == 0:
            continue

        loss = {}
        included = None

        for mode, p in predictions.items():
            result, (b, l, included) = (
                market_metrics(d, p, subset)
            )

            scores.append({
                "split": name,
                "subgroup": subgroup,
                "family": (
                    "RawMarket"
                    if mode == "Market"
                    else "DeepLearning"
                ),
                "model": mode,
                **result,
            })

            loss[mode] = {
                "brier": b,
                "logloss": l,
            }

            log(
                f"    {subgroup:<13} {mode:<7} "
                f"Market Brier={result['brier']:.6f}"
            )

        cluster_subset = d["clusters"][included]

        comparisons = [
            ("S1", "S0"),
            ("S2", "S1"),
            ("S2", "S0"),
        ]

        for new, old in comparisons:
            for metric in ("brier", "logloss"):
                ci, n_clusters = bootstrap_delta(
                    loss[old][metric],
                    loss[new][metric],
                    cluster_subset,
                    args.bootstrap,
                    args.seed + len(gains) * 17,
                )

                delta = float(
                    np.mean(
                        loss[old][metric]
                        - loss[new][metric]
                    )
                )

                gains.append({
                    "split": name,
                    "subgroup": subgroup,
                    "comparison": f"{new} vs {old}",
                    "metric": metric,
                    "delta": delta,
                    "ci_low": float(ci[0]),
                    "ci_high": float(ci[1]),
                    "clusters": n_clusters,
                    "markets": len(cluster_subset),
                    "rows": int(subset.sum()),
                })

                if subgroup == "all":
                    log(
                        f"    {new} vs {old} ({metric}) | "
                        f"gain={delta:+.7f} "
                        f"95% CI=[{ci[0]:+.7f}, "
                        f"{ci[1]:+.7f}]"
                    )

    scores.extend(
        old_baselines(
            base, name,
            len(d["y"]),
            len(d["counts"]),
        )
    )

    return scores, gains


def make_figures(scores, gains, out):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    comparisons = [
        "S1 vs S0",
        "S2 vs S1",
        "S2 vs S0",
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for i, (split, color) in enumerate([
        ("validation", "#FFBE7A"),
        ("final_test", "#82B0D2"),
    ]):
        sub = gains[
            (gains.split == split)
            & (gains.subgroup == "all")
            & (gains.metric == "brier")
        ].set_index(
            "comparison"
        ).reindex(comparisons)

        if sub.delta.isna().all():
            continue

        x = (
            np.arange(len(comparisons))
            + (i - 0.5) * 0.18
        )

        means = sub.delta.to_numpy() * 10000

        lower = np.maximum(
            0,
            (sub.delta - sub.ci_low).to_numpy()
            * 10000,
        )

        upper = np.maximum(
            0,
            (sub.ci_high - sub.delta).to_numpy()
            * 10000,
        )

        ax.errorbar(
            x, means,
            yerr=[lower, upper],
            fmt="o",
            color=color,
            markersize=7,
            capsize=4,
            label=split,
        )

    ax.axhline(
        0, color="#999999", linestyle="--"
    )

    ax.set_xticks(
        np.arange(3),
        [
            "Historical information\nS1 vs S0",
            "Sequential encoding\nS2 vs S1",
            "Full sequence\nS2 vs S0",
        ],
    )

    ax.set_ylabel(
        "Market-equal Brier improvement × 10,000"
    )
    ax.set_title(
        "Does Wallet History Improve Prediction?"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.12)

    fig.tight_layout()
    fig.savefig(
        out / "01_incremental_brier.png", dpi=200
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.9))

    subgroups = [
        "all", "history_ge_1", "history_ge_5"
    ]

    for mode in ("Market", "S0", "S1", "S2"):
        sub = scores[
            (scores.split == "final_test")
            & (scores.model == mode)
        ]

        sub = (
            sub.set_index("subgroup")
            .reindex(subgroups)
        )

        ax.plot(
            range(len(subgroups)),
            sub.brier.to_numpy(),
            "o-",
            color=COLORS[mode],
            label=mode,
            linewidth=2,
        )

    ax.set_xticks(
        range(3),
        [
            "All wallets",
            "At least 1 past action",
            "At least 5 past actions",
        ],
    )

    ax.set_ylabel(
        "Market-equal Brier score (lower is better)"
    )
    ax.set_title(
        "Predictive Accuracy by Historical Coverage"
    )
    ax.grid(axis="y", alpha=0.12)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        out / "02_history_coverage_brier.png",
        dpi=200,
    )
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--grace-days", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--eval-batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--print-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cache-root",
        default="/content/bia31_cache",
    )

    args = ap.parse_args()

    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "GPU runtime required for this experiment"
        )

    log(
        f"[START] Step 31 | "
        f"GPU={torch.cuda.get_device_name(0)}"
    )

    base = (
        Path(args.drive_root)
        / "data" / "research_results"
    )

    source = (
        base
        / f"p1a_bia_sequences_{args.grace_days}d"
    )

    out = (
        base
        / f"p1a_bia_deep_learning_{args.grace_days}d"
    )
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (source / "SEQUENCE_MANIFEST.json").read_text()
    )

    if (
        manifest["sequence_length"] != STEPS
        or len(manifest["sequence_features"]) != DIM
    ):
        raise ValueError("Sequence schema mismatch")

    log(
        "[1/5] History coverage: "
        f"{manifest['history_ge_1']} | "
        f">=5: {manifest['history_ge_5']}"
    )

    # Copy only existing sequence files, not raw data.
    cache = Path(args.cache_root)
    cache.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "final_test"):
        original = source / f"{split}.parquet"
        target = cache / original.name

        if not original.exists():
            raise FileNotFoundError(original)

        if (
            not target.exists()
            or target.stat().st_size
            != original.stat().st_size
        ):
            log(
                f"[CACHE] Copy {split} "
                "from Drive to Colab disk ..."
            )
            shutil.copy2(original, target)

        rows = pq.ParquetFile(
            target
        ).metadata.num_rows

        if rows != manifest["split_rows"][split]:
            raise ValueError(
                f"{split}: unexpected row count {rows}"
            )

    log("[2/5] Loading training and validation ...")

    train, maps = load_split(
        cache / "train.parquet"
    )

    val, _ = load_split(
        cache / "validation.parquet", maps
    )

    # All normalization statistics fit on train only.
    mean = train["nums"].mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)

    std = train["nums"].std(
        axis=0, dtype=np.float64
    ).astype(np.float32).clip(min=1e-3)

    hmean, hstd, n_hist = history_stats(
        train["history"], train["lengths"]
    )

    stats = (mean, std, hmean, hstd)

    vocab = (
        len(maps["category"]),
        len(maps["market_type"]),
    )

    log(
        f"    Training historical events: {n_hist:,}"
    )

    log("[3/5] Training S0 / S1 / S2 ...")

    models = {}
    epochs = {}

    for mode in ("S0", "S1", "S2"):
        models[mode], epochs[mode] = train_one(
            mode, train, val, args,
            stats, vocab, out, device,
        )

    log("[4/5] Evaluating validation and final test ...")

    scores, gains = evaluate_split(
        "validation", val, models,
        device, args, base, out,
    )

    del train, val
    gc.collect()

    test, _ = load_split(
        cache / "final_test.parquet", maps
    )

    s, g = evaluate_split(
        "final_test", test, models,
        device, args, base, out,
    )

    scores.extend(s)
    gains.extend(g)

    del test
    gc.collect()

    scores = pd.DataFrame(scores)
    gains = pd.DataFrame(gains)

    scores.to_csv(
        out / "model_scores.csv", index=False
    )
    gains.to_csv(
        out / "incremental_bootstrap.csv",
        index=False,
    )

    log("[5/5] Saving figures and manifest ...")

    make_figures(scores, gains, out)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()

    result = {
        "git_commit": commit,
        "source_sequence_commit": (
            manifest.get("git_commit")
        ),
        "source_rows": manifest["split_rows"],
        "history_ge_1": manifest["history_ge_1"],
        "history_ge_5": manifest["history_ge_5"],
        "selected_epochs": epochs,
        "model_architecture": (
            "Static MLP / Masked Mean Pooling / GRU(32)"
        ),
        "training_objective": (
            "Market-equal weighted BCE"
        ),
        "early_stopping": (
            "Validation market-equal Brier"
        ),
        "evaluation": (
            "Market-equal Brier and Log Loss; "
            "event-cluster bootstrap"
        ),
        "normalization": "Training data only",
        "numeric_mean": mean.tolist(),
        "numeric_std": std.tolist(),
        "history_mean": hmean.tolist(),
        "history_std": hstd.tolist(),
        "categories": maps,
        "hyperparameters": vars(args),
        "exploratory_oos": True,
        "limitations": [
            "Lagged market probabilities, not executable quotes",
            "Qualified buys only, not complete wallet histories",
            "GRU versus pooling also changes model capacity",
            "Previously inspected final test",
        ],
    }

    (
        out / "DEEP_LEARNING_MANIFEST.json"
    ).write_text(
        json.dumps(result, indent=2)
    )

    log(f"[DONE] {out}")

    log("\nFINAL TEST — ALL OBSERVATIONS")

    print(
        scores[
            (scores.split == "final_test")
            & (scores.subgroup == "all")
        ][[
            "family", "model", "brier",
            "logloss", "rows", "markets",
        ]].to_string(index=False)
    )

    print(
        gains[
            (gains.split == "final_test")
            & (gains.subgroup == "all")
            & (gains.metric == "brier")
        ][[
            "comparison", "delta",
            "ci_low", "ci_high",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
