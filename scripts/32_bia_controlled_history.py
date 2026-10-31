
#!/usr/bin/env python3
"""Part I-A Q2: matched-capacity history-content experiment.

C0: S1 architecture, actual history lengths, constant zero event content.
C1: identical S1 architecture with real point-in-time history content.
"""

import argparse
import gc
import importlib.util
import json
import random
import shutil
import subprocess
import sys
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
from torch.utils.data import DataLoader, TensorDataset

MODES = ("C0", "C1")
SPLITS = ("validation", "final_test")
COLORS = {
    "validation": "#FFBE7A",
    "final_test": "#82B0D2",
}


def log(s):
    print(s, flush=True)


def import_step31():
    path = Path(__file__).resolve().with_name(
        "31_bia_deep_learning.py"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Step 31 must be alongside Step 32: {path}"
        )
    spec = importlib.util.spec_from_file_location(
        "bia31", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(d, batch_size, shuffle, seed, pin):
    ds = TensorDataset(*[
        torch.from_numpy(d[k])
        for k in (
            "nums", "cats", "history",
            "lengths", "y", "weights"
        )
    ])
    generator = (
        torch.Generator().manual_seed(seed)
        if shuffle else None
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=pin,
    )


def logits_for(model, batch, mode, device):
    x, cats, hist, length = (
        z.to(device, non_blocking=True)
        for z in batch[:4]
    )

    if mode == "C0":
        # Keep actual history_count.
        # Replace ONLY event content.
        hist = torch.zeros_like(hist)

    return model(x, cats, hist, length)


@torch.inference_mode()
def predict(model, loader, mode, device):
    model.eval()
    predictions = []

    for batch in loader:
        p = torch.sigmoid(
            logits_for(model, batch, mode, device)
        )
        predictions.append(p.cpu().numpy())

    return np.concatenate(
        predictions
    ).astype(np.float32)


def build_model(bia, vocab, stats, device):
    return bia.BehavioralNet(
        "S1", vocab[0], vocab[1], *stats
    ).to(device)


def checkpoint_path(out, seed, mode):
    return out / f"{mode}_seed{seed}_best.pt"


def fit_model(
    bia, train, val, mode, seed, args,
    stats, vocab, out, signature, device
):
    path = checkpoint_path(out, seed, mode)

    seed_all(seed)
    model = build_model(
        bia, vocab, stats, device
    )

    if path.exists():
        cp = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        if cp.get("signature") != signature:
            raise RuntimeError(
                f"Checkpoint configuration mismatch: {path}"
            )
        model.load_state_dict(cp["state_dict"])
        log(
            f"[RESUME] {mode} seed={seed} | "
            f"best epoch={cp['epoch']}"
        )
        return model, int(cp["epoch"])

    train_dl = make_loader(
        train,
        args.batch_size,
        True,
        seed,
        device.type == "cuda",
    )

    val_dl = make_loader(
        val,
        args.eval_batch_size,
        False,
        seed,
        device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,
    )

    best = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    curve = []

    log(
        f"[TRAIN] {mode} | seed={seed} | parameters="
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0

        for batch_no, batch in enumerate(
            train_dl, start=1
        ):
            y = batch[4].to(
                device, non_blocking=True
            )
            weight = batch[5].to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            logits = logits_for(
                model, batch, mode, device
            )

            individual_losses = (
                nn.functional
                .binary_cross_entropy_with_logits(
                    logits, y, reduction="none"
                )
            )

            total = (
                individual_losses * weight
            ).sum()

            loss = total / weight.sum().clamp(
                min=1e-8
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            weighted_loss_sum += (
                total.detach().item()
            )
            weight_sum += weight.sum().item()

            if batch_no % args.print_every == 0:
                log(
                    f"    {mode} seed={seed} "
                    f"epoch={epoch} "
                    f"batch={batch_no}/{len(train_dl)}"
                )

        val_p = predict(
            model, val_dl, mode, device
        )

        val_scores, _ = bia.market_metrics(
            val, val_p
        )

        curve.append({
            "mode": mode,
            "seed": seed,
            "epoch": epoch,
            "train_weighted_logloss": (
                weighted_loss_sum / weight_sum
            ),
            "validation_brier": (
                val_scores["brier"]
            ),
            "validation_logloss": (
                val_scores["logloss"]
            ),
        })

        log(
            f"    {mode} seed={seed} "
            f"epoch={epoch:02d} | "
            f"Train CE={weighted_loss_sum / weight_sum:.6f} | "
            f"Val Brier={val_scores['brier']:.6f} | "
            f"Val LogLoss={val_scores['logloss']:.6f} | "
            f"{time.time()-start:.0f}s"
        )

        if (
            val_scores["brier"]
            < best - args.min_delta
        ):
            best = val_scores["brier"]
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1

            if patience_counter >= args.patience:
                log(
                    f"    [STOP] {mode} "
                    f"seed={seed} at epoch {epoch}"
                )
                break

    if best_state is None:
        raise RuntimeError(
            "No checkpoint selected"
        )

    model.load_state_dict(best_state)

    torch.save({
        "state_dict": best_state,
        "epoch": best_epoch,
        "seed": seed,
        "mode": mode,
        "signature": signature,
    }, path)

    pd.DataFrame(curve).to_csv(
        out / f"{mode}_seed{seed}_training.csv",
        index=False,
    )

    log(
        f"[BEST] {mode} seed={seed} | "
        f"epoch={best_epoch} | "
        f"Validation Brier={best:.6f}"
    )

    return model, best_epoch


def evaluate_pair(
    bia, split, d, c0, c1,
    seed_label, bootstrap, seed
):
    scores = []
    deltas = []

    masks = {
        "all": np.ones(
            len(d["y"]), dtype=bool
        ),
        "cold_start": d["lengths"] == 0,
        "history_ge_1": d["lengths"] >= 1,
        "history_ge_5": d["lengths"] >= 5,
    }

    for subgroup, mask in masks.items():
        if not mask.any():
            continue

        m0, (b0, l0, included) = (
            bia.market_metrics(d, c0, mask)
        )
        m1, (b1, l1, _) = (
            bia.market_metrics(d, c1, mask)
        )

        for mode, values in (
            ("C0", m0),
            ("C1", m1),
        ):
            scores.append({
                "split": split,
                "seed": str(seed_label),
                "subgroup": subgroup,
                "mode": mode,
                **values,
            })

        for j, (metric, old, new) in enumerate((
            ("brier", b0, b1),
            ("logloss", l0, l1),
        )):
            ci, clusters = bia.bootstrap_delta(
                old,
                new,
                d["clusters"][included],
                bootstrap,
                seed + j * 1000 + len(deltas) * 37,
            )

            delta = float(
                np.mean(old - new)
            )

            deltas.append({
                "split": split,
                "seed": str(seed_label),
                "subgroup": subgroup,
                "metric": metric,
                "delta": delta,
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "clusters": clusters,
                "rows": int(mask.sum()),
                "markets": int(included.sum()),
            })

            if metric == "brier":
                log(
                    f"[SCORE] {split:10s} "
                    f"seed={str(seed_label):<8} "
                    f"{subgroup:13s} | "
                    f"C1-C0 gain x1e4="
                    f"{delta*1e4:+.3f} | "
                    f"CI=[{ci[0]*1e4:+.3f}, "
                    f"{ci[1]*1e4:+.3f}]"
                )

    return scores, deltas


def plot_results(gains, out):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(
        figsize=(9.5, 4.9)
    )

    groups = (
        "all",
        "history_ge_1",
        "history_ge_5",
    )

    labels = (
        "All wallets",
        "History >= 1",
        "History >= 5",
    )

    for offset, split in (
        (-0.14, "validation"),
        (0.14, "final_test"),
    ):
        df = gains[
            (gains.split == split)
            & (gains.metric == "brier")
            & (gains.seed == "ensemble")
        ].set_index(
            "subgroup"
        ).reindex(groups)

        means = (
            df.delta.to_numpy() * 1e4
        )

        lo = np.maximum(
            (
                df.delta - df.ci_low
            ).to_numpy() * 1e4,
            0,
        )

        hi = np.maximum(
            (
                df.ci_high - df.delta
            ).to_numpy() * 1e4,
            0,
        )

        ax.errorbar(
            np.arange(3) + offset,
            means,
            yerr=[lo, hi],
            fmt="o",
            capsize=4,
            lw=2,
            markersize=7,
            color=COLORS[split],
            label=split,
        )

    ax.axhline(
        0, lw=1,
        ls="--",
        color="#999999",
    )

    ax.set_xticks(
        range(3), labels
    )

    ax.set_ylabel(
        "Market-equal Brier improvement x 10,000"
    )

    ax.set_title(
        "Does Historical Behavioral Content "
        "Add Predictive Value?"
    )

    ax.grid(axis="y", alpha=0.12)
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        out / "01_history_content_incremental.png",
        dpi=200,
    )
    plt.close(fig)


def copy_parquet(
    source, cache, split, expected_rows
):
    src = source / f"{split}.parquet"
    target = cache / f"{split}.parquet"

    if not src.exists():
        raise FileNotFoundError(src)

    if (
        not target.exists()
        or target.stat().st_size
        != src.stat().st_size
    ):
        log(
            f"[CACHE] Copy existing {split} "
            "sequence Parquet to Colab disk"
        )
        shutil.copy2(src, target)

    rows = pq.ParquetFile(
        target
    ).metadata.num_rows

    if rows != expected_rows:
        raise ValueError(
            f"{split} rows mismatch: "
            f"{rows} vs {expected_rows}"
        )

    return target


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--drive-root", required=True
    )
    ap.add_argument(
        "--grace-days", type=int, default=7
    )
    ap.add_argument(
        "--seeds", default="13,42,2026"
    )
    ap.add_argument(
        "--batch-size", type=int, default=4096
    )
    ap.add_argument(
        "--eval-batch-size",
        type=int, default=8192
    )
    ap.add_argument(
        "--epochs", type=int, default=12
    )
    ap.add_argument(
        "--patience", type=int, default=3
    )
    ap.add_argument(
        "--min-delta", type=float, default=1e-5
    )
    ap.add_argument(
        "--lr", type=float, default=0.001
    )
    ap.add_argument(
        "--bootstrap", type=int, default=400
    )
    ap.add_argument(
        "--print-every", type=int, default=200
    )
    ap.add_argument(
        "--cache-root",
        default="/content/bia31_cache",
    )

    args = ap.parse_args()

    seeds = tuple(
        int(s.strip())
        for s in args.seeds.split(",")
    )

    if (
        len(seeds) != 3
        or len(set(seeds)) != 3
    ):
        raise ValueError(
            "Expected three distinct seeds"
        )

    if args.bootstrap < 100:
        raise ValueError(
            "Use at least 100 bootstrap replicates"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "Please select a Colab GPU runtime"
        )

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False

    bia = import_step31()

    repo = Path(__file__).resolve().parents[1]

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()

    base = (
        Path(args.drive_root).expanduser()
        / "data" / "research_results"
    )

    source = (
        base
        / f"p1a_bia_sequences_{args.grace_days}d"
    )

    out = (
        base
        / f"p1a_bia_history_control_{args.grace_days}d"
    )
    out.mkdir(
        parents=True, exist_ok=True
    )

    cache = Path(args.cache_root)
    cache.mkdir(
        parents=True, exist_ok=True
    )

    source_manifest = json.loads(
        (
            source / "SEQUENCE_MANIFEST.json"
        ).read_text()
    )

    if (
        source_manifest["sequence_length"] != bia.STEPS
        or len(
            source_manifest["sequence_features"]
        ) != bia.DIM
    ):
        raise RuntimeError(
            "Step 30 schema incompatible with Step 31"
        )

    cfg = {
        "git_commit": commit,
        "source_commit": (
            source_manifest["git_commit"]
        ),
        "rows": source_manifest["split_rows"],
        "seeds": seeds,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "lr": args.lr,
        "bootstrap": args.bootstrap,
        "design": (
            "S1/C0 same architecture; "
            "C0 raw event vectors zeroed, "
            "lengths kept; C1 real event vectors"
        ),
    }

    cfg_path = out / "RUN_CONFIG.json"
    signature = json.dumps(
        cfg, sort_keys=True
    )

    if cfg_path.exists():
        previous = json.dumps(
            json.loads(cfg_path.read_text()),
            sort_keys=True,
        )
        if previous != signature:
            raise RuntimeError(
                "Existing output has different "
                f"configuration: {cfg_path}"
            )
    else:
        cfg_path.write_text(
            json.dumps(cfg, indent=2)
        )

    log(
        f"[START] Step 32 | git={commit[:12]} | "
        f"GPU={torch.cuda.get_device_name(0)}"
    )
    log(
        f"        seeds={seeds} | "
        "C0(count-only) vs C1(full history)"
    )

    train_path = copy_parquet(
        source, cache, "train",
        source_manifest["split_rows"]["train"],
    )

    val_path = copy_parquet(
        source, cache, "validation",
        source_manifest["split_rows"]["validation"],
    )

    train, maps = bia.load_split(
        train_path
    )

    val, _ = bia.load_split(
        val_path, maps
    )

    # Fit normalization on training data only.
    mean = train["nums"].mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)

    std = train["nums"].std(
        axis=0, dtype=np.float64
    ).astype(np.float32).clip(min=1e-3)

    hmean, hstd, n_events = (
        bia.history_stats(
            train["history"],
            train["lengths"],
        )
    )

    stats = (
        mean, std, hmean, hstd
    )

    vocab = (
        len(maps["category"]),
        len(maps["market_type"]),
    )

    log(
        f"[DATA] {len(train):,} train | "
        f"{len(val):,} validation | "
        f"{n_events:,} historical training events"
    )

    best_epochs = {}
    val_predictions = {}

    val_dl = make_loader(
        val,
        args.eval_batch_size,
        False,
        0,
        device.type == "cuda",
    )

    # First fit all six models using training/validation.
    for seed in seeds:
        pair = {}

        for mode in MODES:
            model, best_epoch = fit_model(
                bia, train, val, mode, seed,
                args, stats, vocab, out,
                signature, device,
            )

            best_epochs[
                f"{mode}_{seed}"
            ] = best_epoch

            pair[mode] = predict(
                model, val_dl, mode, device
            )

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

        val_predictions[seed] = pair

        pd.DataFrame({
            "obs_id": val["obs_id"],
            "p_C0": pair["C0"],
            "p_C1": pair["C1"],
        }).to_parquet(
            out / (
                f"validation_seed{seed}"
                "_predictions.parquet"
            ),
            index=False,
            compression="zstd",
        )

    all_scores = []
    all_gains = []

    for seed in seeds:
        a, b = evaluate_pair(
            bia,
            "validation",
            val,
            val_predictions[seed]["C0"],
            val_predictions[seed]["C1"],
            seed,
            args.bootstrap,
            seed,
        )
        all_scores.extend(a)
        all_gains.extend(b)

    val_ens = {
        mode: np.mean(
            np.stack([
                val_predictions[seed][mode]
                for seed in seeds
            ], axis=0),
            axis=0,
        ).astype(np.float32)
        for mode in MODES
    }

    a, b = evaluate_pair(
        bia,
        "validation",
        val,
        val_ens["C0"],
        val_ens["C1"],
        "ensemble",
        args.bootstrap,
        1601,
    )

    all_scores.extend(a)
    all_gains.extend(b)

    pd.DataFrame({
        "obs_id": val["obs_id"],
        "p_C0": val_ens["C0"],
        "p_C1": val_ens["C1"],
    }).to_parquet(
        out / "validation_ensemble_predictions.parquet",
        index=False,
        compression="zstd",
    )

    del train, val, val_dl
    del val_predictions, val_ens
    gc.collect()

    log(
        "[TEST] All checkpoints fixed by validation; "
        "final-test inference begins"
    )

    test_path = copy_parquet(
        source, cache, "final_test",
        source_manifest["split_rows"]["final_test"],
    )

    test, _ = bia.load_split(
        test_path, maps
    )

    test_dl = make_loader(
        test,
        args.eval_batch_size,
        False,
        0,
        device.type == "cuda",
    )

    test_predictions = {}

    # No new model tuning after reading final test.
    for seed in seeds:
        pair = {}

        for mode in MODES:
            cp = torch.load(
                checkpoint_path(
                    out, seed, mode
                ),
                map_location="cpu",
                weights_only=True,
            )

            if cp.get("signature") != signature:
                raise RuntimeError(
                    "Checkpoint signature mismatch"
                )

            model = build_model(
                bia, vocab, stats, device
            )

            model.load_state_dict(
                cp["state_dict"]
            )

            pair[mode] = predict(
                model, test_dl, mode, device
            )

            del model, cp

        test_predictions[seed] = pair

        pd.DataFrame({
            "obs_id": test["obs_id"],
            "p_C0": pair["C0"],
            "p_C1": pair["C1"],
        }).to_parquet(
            out / (
                f"final_test_seed{seed}"
                "_predictions.parquet"
            ),
            index=False,
            compression="zstd",
        )

        a, b = evaluate_pair(
            bia,
            "final_test",
            test,
            pair["C0"],
            pair["C1"],
            seed,
            args.bootstrap,
            seed + 1700,
        )

        all_scores.extend(a)
        all_gains.extend(b)

    test_ens = {
        mode: np.mean(
            np.stack([
                test_predictions[seed][mode]
                for seed in seeds
            ], axis=0),
            axis=0,
        ).astype(np.float32)
        for mode in MODES
    }

    a, b = evaluate_pair(
        bia,
        "final_test",
        test,
        test_ens["C0"],
        test_ens["C1"],
        "ensemble",
        args.bootstrap,
        1707,
    )

    all_scores.extend(a)
    all_gains.extend(b)

    pd.DataFrame({
        "obs_id": test["obs_id"],
        "p_C0": test_ens["C0"],
        "p_C1": test_ens["C1"],
    }).to_parquet(
        out / "final_test_ensemble_predictions.parquet",
        index=False,
        compression="zstd",
    )

    scores = pd.DataFrame(all_scores)
    gains = pd.DataFrame(all_gains)

    scores.to_csv(
        out / "controlled_scores.csv",
        index=False,
    )

    gains.to_csv(
        out / "controlled_incremental.csv",
        index=False,
    )

    plot_results(
        gains, out
    )

    (
        out / "CONTROL_MANIFEST.json"
    ).write_text(
        json.dumps({
            **cfg,
            "best_epochs": best_epochs,
            "evaluation": (
                "Market-equal Brier and LogLoss "
                "with event-cluster bootstrap"
            ),
            "seed_variation": (
                "Three paired training seeds; "
                "ensemble formed by probability average"
            ),
            "interpretation": (
                "Exploratory OOS; final test previously "
                "inspected; validation reused "
                "for early stopping"
            ),
            "caveats": [
                "C0 has history length and static "
                "summaries, no event content",
                "C1 vs C0 isolates incremental history "
                "content in matched architecture",
                "This does not prove tradeable alpha",
                "Market proxy can lag 12 hours",
                "Sequence includes qualifying buys only",
            ],
        }, indent=2)
    )

    log(f"[DONE] {out}")

    sub = gains[
        (gains.metric == "brier")
        & (gains.subgroup == "all")
    ]

    print(
        "\n=== CORE BRIER COMPARISON "
        "(positive = C1 better) ===",
        flush=True,
    )

    print(
        sub[[
            "split", "seed", "delta",
            "ci_low", "ci_high",
        ]].to_string(index=False),
        flush=True,
    )

    print(
        "\n=== THREE-SEED STABILITY ===",
        flush=True,
    )

    sub3 = gains[
        (gains.metric == "brier")
        & (gains.subgroup == "all")
        & (gains.seed != "ensemble")
    ].copy()

    sub3["gain_x10000"] = (
        sub3["delta"] * 1e4
    )

    stable = (
        sub3.groupby("split")["gain_x10000"]
        .agg(["mean", "std", "min", "max"])
    )

    stable["positive_seeds"] = (
        sub3.groupby("split")["gain_x10000"]
        .apply(
            lambda x: int((x > 0).sum())
        )
    )

    print(
        stable.round(3).to_string(),
        flush=True,
    )

    print(
        "\n=== FINAL-TEST ENSEMBLE "
        "ABSOLUTE SCORES ===",
        flush=True,
    )

    s = scores[
        (scores.split == "final_test")
        & (scores.seed == "ensemble")
        & (scores.subgroup == "all")
    ]

    print(
        s[[
            "mode", "brier", "logloss",
            "rows", "markets",
        ]].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
