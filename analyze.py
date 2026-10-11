#!/usr/bin/env python3
"""Plots, paired McNemar tests, paired bootstrap CIs for Stanford Cars PEFT runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def paired_bootstrap_difference(y, lora_pred, dora_pred, n_resamples=2000, seed=2026):
    """95% percentile CI on per-image (DoRA correct - LoRA correct) paired differences."""
    diff = (dora_pred == y).astype(float) - (lora_pred == y).astype(float)
    rng = np.random.default_rng(seed)
    result = np.empty(n_resamples, dtype=float)
    n = len(diff)
    for i in range(n_resamples):
        result[i] = diff[rng.integers(n, size=n)].mean()
    return float(np.quantile(result, .025)), float(np.quantile(result, .975))


def holm_adjust(p_values):
    """Holm family-wise-error-rate adjustment, across the shown per-run comparisons."""
    n = len(p_values)
    adjusted = np.empty(n, dtype=float)
    order = np.argsort(p_values)
    last = 0.0
    for i, idx in enumerate(order):
        last = max(last, (n-i)*p_values[idx])
        adjusted[idx] = min(1.0, last)
    return adjusted.tolist()


def compare_paired(root, summary, split="test", n_resamples=2000):
    pairs = []
    for (rank, seed), group in summary.groupby(["rank", "seed"]):
        methods = set(group["method"])
        if not {"lora", "dora"}.issubset(methods):
            continue
        lrow = group.loc[group["method"] == "lora"].iloc[0]
        drow = group.loc[group["method"] == "dora"].iloc[0]
        la = root / lrow["run_dir"] / f"predictions_{split}.csv"
        da = root / drow["run_dir"] / f"predictions_{split}.csv"
        if not (la.exists() and da.exists()):
            continue
        a, b = pd.read_csv(la), pd.read_csv(da)
        if len(a) != len(b) or not np.array_equal(a["row_idx"], b["row_idx"]):
            raise RuntimeError(f"Mismatched row indices for rank {rank} seed {seed}")
        if not np.array_equal(a["label"], b["label"]):
            raise RuntimeError(f"Mismatched true labels for rank {rank} seed {seed}")
        if "source_id" in a and "source_id" in b and not np.array_equal(a["source_id"], b["source_id"]):
            raise RuntimeError("Mismatched dataset image IDs")
        truth = a["label"].to_numpy()
        x, y = a["prediction"].to_numpy(), b["prediction"].to_numpy()
        l_correct, d_correct = x == truth, y == truth
        l_only = int(np.sum(l_correct & ~d_correct))
        d_only = int(np.sum(~l_correct & d_correct))
        n_discordant = l_only + d_only
        # H0: conditional on disagreement, each method wins equally often
        p = float(binomtest(min(l_only, d_only), n_discordant, .5).pvalue) if n_discordant else 1.0
        lo, hi = paired_bootstrap_difference(truth, x, y, n_resamples,
                                             seed=2026+int(rank)*31+int(seed))
        pairs.append({
            "split": split, "rank": int(rank), "seed": int(seed), "n": len(truth),
            "lora_accuracy": float(l_correct.mean()),
            "dora_accuracy": float(d_correct.mean()),
            "dora_minus_lora_pp": 100*float(d_correct.mean()-l_correct.mean()),
            "bootstrap_95_low_pp": 100*lo, "bootstrap_95_high_pp": 100*hi,
            "lora_only_correct": l_only, "dora_only_correct": d_only,
            "discordant": n_discordant, "mcnemar_exact_p": p,
        })
    if pairs:
        adjusted = holm_adjust([r["mcnemar_exact_p"] for r in pairs])
        for r, p in zip(pairs, adjusted):
            r["holm_adjusted_p"] = p
    return pd.DataFrame(pairs)


def plot_history(summary, root, plots):
    # Each method/rank/seed is displayed as a separate line; no hidden averaging.
    history_frames = []
    for _, row in summary.iterrows():
        path = root / row["history_csv"]
        if path.exists():
            history_frames.append(pd.read_csv(path))
    if not history_frames:
        return
    df = pd.concat(history_frames, ignore_index=True)
    for col, ylabel, filename in [
        ("val_accuracy", "Validation accuracy (%)", "val_accuracy_by_epoch.png"),
        ("val_loss", "Validation cross-entropy", "val_loss_by_epoch.png"),
        ("train_loss", "Training cross-entropy", "train_loss_by_epoch.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for (method, rank, seed), group in df.groupby(["method", "rank", "seed"], sort=True):
            y = group[col]*100 if col == "val_accuracy" else group[col]
            ax.plot(group["epoch"], y, label=f"{method} r={rank} seed={seed}")
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(True, alpha=.25)
        ax.legend(loc="best", fontsize="x-small", ncol=2)
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=160)
        plt.close(fig)
    df.to_csv(root / "all_epochs.csv", index=False)


def plot_summary(summary, plots):
    # Accuracy (%) versus trainable parameters, independent of rank.
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, grp in summary.groupby("method", sort=True):
        grp = grp.sort_values(["rank", "seed"])
        ax.scatter(grp["trainable_params"] / 1e6, grp["best_val_accuracy"] * 100,
                   label=method, alpha=.85)
    ax.set(xlabel="Trainable parameters (millions)", ylabel="Best validation accuracy (%)")
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "accuracy_vs_params.png", dpi=160)
    plt.close(fig)
    for x, xlab, outfile in [
        ("train_time_sec", "Training time (seconds)", "accuracy_vs_time.png"),
        ("peak_allocated_gib", "Peak PyTorch allocated VRAM (GiB)", "accuracy_vs_vram.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, grp in summary.groupby("method", sort=True):
            ax.scatter(grp[x], grp["best_val_accuracy"] * 100, label=method, alpha=.85)
        ax.set(xlabel=xlab, ylabel="Best validation accuracy (%)")
        ax.grid(True, alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / outfile, dpi=160)
        plt.close(fig)
    if summary["rank"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, grp in summary.groupby("method", sort=True):
            gr = grp.groupby("rank")["best_val_accuracy"].agg(["mean","std"])
            err = gr["std"].fillna(0) * 100
            ax.errorbar(gr.index.to_numpy(), gr["mean"] * 100, yerr=err, marker="o", capsize=3, label=method)
        ax.set(xlabel="Adapter rank", ylabel="Validation accuracy (%), mean ± SD across seeds")
        ax.grid(True, alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "rank_sweep_val.png", dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="runs/stanford_cars_vit_b16")
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    root = Path(args.input).resolve()
    summary_path = root / "summary.csv"
    if not summary_path.exists():
        p.error(f"No summary found at {summary_path}")
    summary = pd.read_csv(summary_path)
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    if summary.empty:
        raise RuntimeError("summary.csv empty")
    plot_history(summary, root, plots)
    plot_summary(summary, plots)
    summary_stats = summary.groupby(["method", "rank"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        val_mean=("best_val_accuracy", "mean"),
        val_sd=("best_val_accuracy", "std"),
        test_mean=("test_accuracy", "mean"),
        test_sd=("test_accuracy", "std"),
        params=("trainable_params", "first"),
        mean_train_seconds=("train_time_sec", "mean"),
        mean_peak_vram_gib=("peak_allocated_gib", "mean"),
    )
    summary_stats.to_csv(root / "aggregated.csv", index=False)
    all_pairs = []
    for split in ("val", "test"):
        paired = compare_paired(root, summary, split=split, n_resamples=args.bootstrap)
        if not paired.empty:
            paired.to_csv(root / f"paired_{split}_stats.csv", index=False)
            all_pairs.extend(paired.to_dict("records"))
    (root / "statistics.json").write_text(json.dumps(all_pairs, ensure_ascii=False, indent=2))
    lines = [
        "# LoRA vs DoRA — Stanford Cars",
        "",
        "All scores were produced by the local experiment; they are not published benchmark scores.",
        "Train/test protocol and versions are in experiment_config.json, split_manifest.json and system.json.",
        "",
        "## Summary by method and rank",
        "",
        "```",
        summary_stats.to_string(index=False, float_format=lambda x: f"{x:.5f}"),
        "```",
    ]
    for split in ("val", "test"):
        file = root / f"paired_{split}_stats.csv"
        if file.exists():
            lines.extend(["", f"## Paired {split} comparisons", "", "```",
                          pd.read_csv(file).to_string(index=False, float_format=lambda x: f"{x:.5f}"), "```"])
    lines += [
        "", "## Methodological notes", "",
        "- DoRA has extra magnitude parameters, so same rank does NOT mean exact parameter matching.",
        "- Exact McNemar test examines discordant correctness on the SAME images for one seed/rank; two-sided p-value.",
        "- Bootstrap 95% CI resamples images in paired fashion; it does NOT capture uncertainty from rerunning training.",
        "- Holm adjustment within the reported val or test comparisons controls multiple tests across ranks/seeds.",
        "- Repeated seed scores should be interpreted together. Three seeds do not establish a powerful significance test across training runs.",
        "- Do not choose rank, epoch, or training hyperparameters based on test metrics. Choose on validation; reserve test for the final protocol.",
        "- Peak GPU memory is PyTorch's allocated CUDA memory, not the full nvidia-smi process usage.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Created:", root / "REPORT.md")
    print("Charts:", plots)
    print("Paired comparisons:", root / "paired_val_stats.csv", root / "paired_test_stats.csv")
    print(summary_stats.to_string(index=False))


if __name__ == "__main__":
    main()
