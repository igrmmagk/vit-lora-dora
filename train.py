#!/usr/bin/env python3
"""Stanford Cars / ViT-B/16: paired LoRA vs DoRA experiment.

Based on user's sample(1).ipynb. Run `python train.py --help`.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    get_cosine_schedule_with_warmup,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/vit-base-patch16-224-in21k")
    p.add_argument("--dataset", default="mteb/StanfordCars")
    p.add_argument("--output", default="runs/stanford_cars_vit_b16")
    p.add_argument("--methods", nargs="+", choices=["lora", "dora"], default=["lora", "dora"])
    p.add_argument("--ranks", nargs="+", type=int, default=[8])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--adapter-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--resize-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--test-mode", choices=["on", "off"], default="on",
                   help="Evaluate best checkpoint on held-out test (do not use test to select a rank).")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="Smoke test only; limiting batches makes results non-comparable.")
    args = p.parse_args()
    if min(args.ranks) < 1 or min(args.seeds) < 0 or not (0 < args.val_fraction < 0.5):
        p.error("Ranks must be positive, seeds nonnegative, validation fraction in (0, 0.5)")
    if args.epochs < 1 or args.grad_accum < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        p.error("Epochs, accumulation and batch sizes must be positive")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        p.error("--max-train-batches must be positive")
    if not (0 <= args.warmup_ratio < 1):
        p.error("--warmup-ratio must be in [0, 1)")
    args.ranks = sorted(set(args.ranks))
    args.seeds = sorted(set(args.seeds))
    args.methods = list(dict.fromkeys(args.methods))
    return args


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility depends on CUDA kernels, GPU, and library versions.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def command_output(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=8).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def system_info(device):
    import datasets, peft, sklearn, torchvision, transformers
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "bf16_available": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    }


def choose_precision(requested, device):
    if device.type != "cuda":
        if requested in ("bf16", "fp16"):
            raise ValueError("Mixed precision here requires CUDA")
        return "fp32"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unavailable; use --precision fp16 or fp32")
    return requested


def autocast_context(device, precision):
    if device.type == "cuda" and precision in ("bf16", "fp16"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
    return nullcontext()


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class CarsDataset(Dataset):
    def __init__(self, hf_data, transform):
        self.rows = hf_data
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[int(i)]
        x = self.transform(row["image"].convert("RGB"))
        return {"pixel_values": x, "labels": int(row["label"])}


def get_data(args, out):
    print(f"Loading dataset: {args.dataset}", flush=True)
    raw = load_dataset(args.dataset)
    if not {"train", "test"}.issubset(set(raw)):
        raise ValueError(f"Dataset must provide train/test; found {list(raw)}")
    for split in ("train", "test"):
        if not {"image", "label"}.issubset(set(raw[split].column_names)):
            raise ValueError(f"Missing image/label columns on {split}")
    label_feature = raw["train"].features["label"]
    label_names = list(label_feature.names) if hasattr(label_feature, "names") else None
    classes = sorted(set(int(z) for z in raw["train"]["label"]))
    test_classes = sorted(set(int(z) for z in raw["test"]["label"]))
    if classes != list(range(len(classes))) or test_classes != classes:
        raise ValueError("Class labels must cover 0..num_labels-1 in train and test")
    if len(classes) != 196:
        raise ValueError(f"Expected Stanford Cars 196 classes, got {len(classes)}")
    indices = np.arange(len(raw["train"]))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_fraction, random_state=args.split_seed,
        stratify=np.asarray(raw["train"]["label"], dtype=np.int64),
    )
    train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
    split_manifest = {
        "dataset": args.dataset,
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "raw_train_fingerprint": raw["train"]._fingerprint,
        "raw_test_fingerprint": raw["test"]._fingerprint,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": list(range(len(raw["test"]))),
        "class_names": label_names,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(raw["test"]),
    }
    manifest_file = out / "split_manifest.json"
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text())
        if old != split_manifest:
            raise RuntimeError("Existing split_manifest.json differs; choose another --output")
    else:
        save_json(manifest_file, split_manifest)
    processor = AutoImageProcessor.from_pretrained(args.model)
    mean, std = processor.image_mean, processor.image_std
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    data = {
        "train": CarsDataset(raw["train"].select(train_idx), train_transform),
        "val": CarsDataset(raw["train"].select(val_idx), eval_transform),
        "test": CarsDataset(raw["test"], eval_transform),
    }
    print(f"Classes={len(classes)} train={len(train_idx)} val={len(val_idx)} test={len(data['test'])}", flush=True)
    return data


def get_loaders(data, args, seed, device):
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        data["train"], batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"), generator=train_generator,
    )
    eval_kw = dict(batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers,
                   pin_memory=(device.type == "cuda"))
    return train_loader, DataLoader(data["val"], **eval_kw), DataLoader(data["test"], **eval_kw)


def infer_target_modules(model):
    names = [n for n, _ in model.named_modules()]
    for targets in (["q_proj", "v_proj"], ["query", "value"]):
        match = [n for n in names if any(n.endswith("." + t) for t in targets)]
        # ViT-B/16 with 12 attention blocks: Q and V in each block
        if len(match) == 24:
            return targets
    possible = [n for n in names if any(x in n for x in ("attention", "query", "proj", "value"))]
    raise RuntimeError(f"Expected 24 ViT attention Q/V modules; candidates:\n" + "\n".join(possible[:60]))


def classifier_hash(model):
    digest = hashlib.sha256()
    for n, p in model.classifier.named_parameters():
        digest.update(n.encode("utf-8"))
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(args, method, rank, device):
    base = AutoModelForImageClassification.from_pretrained(
        args.model, num_labels=196, ignore_mismatched_sizes=True,
    )
    initial_head_hash = classifier_hash(base)
    targets = infer_target_modules(base)
    config = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
        target_modules=targets, modules_to_save=["classifier"],
        bias="none", use_dora=(method == "dora"),
    )
    model = get_peft_model(base, config).to(device)
    adapted = [name for name, mod in model.named_modules() if hasattr(mod, "lora_A")]
    if len(adapted) != 24:
        raise RuntimeError(f"Got {len(adapted)} LoRA-adapted layers, expected 24")
    counts = {"head": 0, "adapter": 0, "magnitude": 0, "total_trainable": 0, "total_all": 0}
    for name, param in model.named_parameters():
        counts["total_all"] += param.numel()
        if param.requires_grad:
            key = "head" if "classifier" in name else "adapter"
            counts[key] += param.numel()
            counts["total_trainable"] += param.numel()
            if "magnitude" in name.lower():
                counts["magnitude"] += param.numel()
    if counts["head"] != 196 * (768 + 1):
        raise RuntimeError(f"Unexpected head count {counts['head']}")
    expected_adapter = 12 * 2 * rank * (768 + 768) + (12 * 2 * 768 if method == "dora" else 0)
    if counts["adapter"] != expected_adapter:
        raise RuntimeError(f"Unexpected {method} adapter params {counts['adapter']} (expected {expected_adapter})")
    return model, counts, initial_head_hash, targets, adapted


def make_optimizer(model, args):
    head = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" in n]
    adapters = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n]
    if not head or not adapters:
        raise RuntimeError("Missing trainable head or adapters")
    optimizer = AdamW([
        {"params": adapters, "lr": args.adapter_lr, "initial_lr": args.adapter_lr},
        {"params": head, "lr": args.head_lr, "initial_lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    return optimizer, adapters + head


def train_one_epoch(model, loader, optimizer, scheduler, scaler, trainable, args, device, precision, max_batches=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, n_seen, total_correct, updates = 0.0, 0, 0, 0
    max_batches = min(max_batches or len(loader), len(loader))
    batch_iter = iter(loader)
    progress = tqdm(range(max_batches), desc="train", leave=False)
    for step in progress:
        batch = next(batch_iter)
        labels = batch["labels"].to(device, non_blocking=True)
        x = batch["pixel_values"].to(device, non_blocking=True)
        micro_n = labels.shape[0]
        # For a last partial accumulation group, weight each microbatch by its
        # actual number of samples, reproducing a sample-mean loss.
        group_start = (step // args.grad_accum) * args.grad_accum
        group_end = min(group_start + args.grad_accum, max_batches)
        # All microbatches except the last DataLoader batch have args.batch_size samples.
        group_count = (group_end - group_start) * args.batch_size
        if group_end == len(loader) and len(loader.dataset) % args.batch_size:
            group_count -= args.batch_size - len(loader.dataset) % args.batch_size
        with autocast_context(device, precision):
            output = model(pixel_values=x, labels=labels)
            loss = output.loss
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite training loss at step {step}: {loss.item()}")
        (scaler.scale(loss * (micro_n / group_count))).backward()
        total_loss += loss.detach().item() * micro_n
        total_correct += (output.logits.detach().argmax(-1) == labels).sum().item()
        n_seen += micro_n
        if step + 1 == group_end:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        if step % 20 == 0:
            progress.set_postfix(loss=f"{total_loss/n_seen:.3f}")
    return {"train_loss": total_loss/n_seen, "train_accuracy": total_correct/n_seen,
            "train_seen": n_seen, "optimizer_steps": updates}


@torch.inference_mode()
def evaluate(model, loader, device, precision, collect_predictions=False):
    model.eval()
    total_loss, total_correct, seen = 0.0, 0, 0
    logits_parts, label_parts = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            result = model(pixel_values=x, labels=labels)
        if not torch.isfinite(result.loss).item():
            raise FloatingPointError("Nonfinite eval loss")
        n = len(labels)
        total_loss += result.loss.detach().item() * n
        total_correct += (result.logits.argmax(-1) == labels).sum().item()
        seen += n
        if collect_predictions:
            logits_parts.append(result.logits.float().cpu().numpy())
            label_parts.append(labels.cpu().numpy())
    report = {"loss": total_loss/seen, "accuracy": total_correct/seen, "n": seen}
    if collect_predictions:
        logits = np.concatenate(logits_parts).astype(np.float32)
        labels_np = np.concatenate(label_parts).astype(np.int64)
        return report, logits, labels_np
    return report


def save_predictions(directory, split_name, logits, labels, dataset):
    preds = logits.argmax(axis=1).astype(np.int64)
    # Avoid softmax overflow; stable transform for confidence
    logits64 = logits.astype(np.float64)
    shifted = logits64 - logits64.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    rows = []
    for i, (p, y) in enumerate(zip(preds, labels)):
        row = {"row_idx": i, "label": int(y), "prediction": int(p),
               "correct": int(p == y), "confidence": float(probs[i, p]),
               "prob_true_class": float(probs[i, y])}
        # mteb/StanfordCars exposes an image id, when available.
        if "id" in dataset.rows.column_names:
            row["source_id"] = int(dataset.rows[i]["id"])
        rows.append(row)
    save_csv(directory / f"predictions_{split_name}.csv", rows)
    np.savez_compressed(directory / f"logits_{split_name}.npz",
                        logits=logits.astype(np.float16), labels=labels,
                        predictions=preds, indices=np.arange(len(labels)))


def reload_best(args, path, device):
    base = AutoModelForImageClassification.from_pretrained(
        args.model, num_labels=196, ignore_mismatched_sizes=True,
    )
    return PeftModel.from_pretrained(base, str(path), is_trainable=False).to(device)


def run_one(args, data, method, rank, seed, device, precision, root):
    tag = f"{method}_r{rank}_seed{seed}"
    out = root / tag
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    config = {**vars(args), "method": method, "rank": rank, "seed": seed, "precision_effective": precision}
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError(f"Experiment config differs for {tag}; use another --output directory")
    else:
        save_json(config_path, config)
    if result_path.exists():
        print(f"[SKIP] Already completed: {tag} (remove run dir to rerun)", flush=True)
        return json.loads(result_path.read_text())
    if (out / "history.csv").exists():
        raise RuntimeError(f"Found incomplete run {out}. Do not silently overwrite; move or remove its directory.")
    set_seed(seed)
    train_loader, val_loader, test_loader = get_loaders(data, args, seed, device)
    model, counts, head_hash, targets, adapted = build_model(args, method, rank, device)
    print(f"[{tag}] trainable={counts['total_trainable']:,} (adapter={counts['adapter']:,}, head={counts['head']:,})", flush=True)
    opt, trainable = make_optimizer(model, args)
    batches_per_epoch = min(len(train_loader), args.max_train_batches or len(train_loader))
    updates_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    steps_total = updates_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(1, round(steps_total * args.warmup_ratio)),
        num_training_steps=steps_total,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    (out / "best_adapter").mkdir(parents=True, exist_ok=True)
    init_info = {"method":method,"rank":rank,"seed":seed,"targets":targets,
                 "adapted_module_names":adapted,"init_classifier_sha256":head_hash,
                 "counts":counts,"first_epoch_utc":utc_now(), "steps_per_epoch":updates_epoch,
                 "total_optimizer_steps":steps_total}
    save_json(out / "architecture.json", init_info)
    history = []
    best_acc, best_loss, best_epoch = -1.0, float("inf"), 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cuda_sync(device)
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, opt, scheduler, scaler, trainable,
                             args, device, precision, args.max_train_batches)
        val = evaluate(model, val_loader, device, precision)
        cuda_sync(device)
        is_best = val["accuracy"] > best_acc or (
            val["accuracy"] == best_acc and val["loss"] < best_loss)
        if is_best:
            best_acc, best_loss, best_epoch = val["accuracy"], val["loss"], epoch
            model.save_pretrained(str(out / "best_adapter"), safe_serialization=True)
        epoch_time = time.perf_counter() - epoch_t0
        entry = {
            "method":method,"rank":rank,"seed":seed,"epoch":epoch,
            **tr, "val_loss":val["loss"], "val_accuracy":val["accuracy"],
            "lr_adapter_after_epoch":opt.param_groups[0]["lr"],
            "lr_head_after_epoch":opt.param_groups[1]["lr"],
            "epoch_seconds":epoch_time, "best_so_far":best_acc,
        }
        history.append(entry)
        save_csv(out / "history.csv", history)
        print(f"[{tag}] {epoch:02d}/{args.epochs} train_loss={tr['train_loss']:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4%} "
              f"best={best_acc:.4%} epoch_s={epoch_time:.1f}", flush=True)
    cuda_sync(device)
    train_seconds = time.perf_counter() - t0
    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else None
    # Rebuild from the saved adapter to ensure the evaluation uses best epoch,
    # not the last weights. The classifier is part of modules_to_save.
    del model, opt, scheduler, scaler, trainable, train_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = reload_best(args, out / "best_adapter", device)
    val_best, val_logits, val_labels = evaluate(model, val_loader, device, precision, collect_predictions=True)
    if abs(val_best["accuracy"] - best_acc) > 1e-8:
        raise RuntimeError("Best adapter reload changed validation accuracy; checkpoint is inconsistent")
    save_predictions(out, "val", val_logits, val_labels, data["val"])
    test = None
    if args.test_mode == "on":
        test, test_logits, test_labels = evaluate(model, test_loader, device, precision, collect_predictions=True)
        save_predictions(out, "test", test_logits, test_labels, data["test"])
    result = {
        "method":method,"rank":rank,"seed":seed,"status":"complete",
        "model":args.model,"dataset":args.dataset,"precision":precision,
        "trainable_params":counts["total_trainable"],"adapter_params":counts["adapter"],
        "magnitude_params":counts["magnitude"],"head_params":counts["head"],
        "total_model_params":counts["total_all"],
        "best_epoch":best_epoch,"best_val_accuracy":val_best["accuracy"],
        "best_val_loss":val_best["loss"],
        "test_accuracy":test["accuracy"] if test else None,
        "test_loss":test["loss"] if test else None,
        "train_time_sec":train_seconds,"peak_allocated_gib":peak_gib,
        "peak_reserved_gib":peak_reserved_gib,
        "init_classifier_sha256":head_hash,
        "checkpoint_dir":str(Path(tag) / "best_adapter"),
        "history_csv":str(Path(tag) / "history.csv"),"run_dir":tag,
        "finished_utc":utc_now(),
        "smoke_test":args.max_train_batches is not None,
    }
    save_json(result_path, result)
    print(f"[DONE {tag}] val={result['best_val_accuracy']:.4%} "
          f"test={result['test_accuracy']} time={train_seconds:.1f}s "
          f"peak_allocated={peak_gib}", flush=True)
    del model, val_loader, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This server experiment expects an NVIDIA CUDA GPU. Check the PyTorch CUDA installation.")
    precision = choose_precision(args.precision, device)
    existing = root / "experiment_config.json"
    conf = vars(args).copy()
    conf.pop("output")  # output directory name is not part of experiment methodology
    conf["precision_effective"] = precision
    if existing.exists():
        if json.loads(existing.read_text()) != conf:
            raise RuntimeError("Experiment configuration changed; use a new --output directory")
    else:
        save_json(existing, conf)
    info = system_info(device)
    if not (root / "system.json").exists():
        save_json(root / "system.json", info)
    if not (root / "pip_freeze.txt").exists():
        pip_freeze = command_output([sys.executable, "-m", "pip", "freeze"])
        (root / "pip_freeze.txt").write_text((pip_freeze or "") + "\n")
    print("Device:", info["gpu_name"], "| precision:", precision, "| output:", root, flush=True)
    if args.max_train_batches:
        print("WARNING: smoke test mode, metrics are NOT a benchmark", flush=True)
    if args.test_mode == "on" and (len(args.ranks)>1 or len(args.seeds)>1):
        print("CAUTION: test predictions for all runs are for fixed, preregistered configurations; do not choose ranks/seeds by test accuracy", flush=True)
    data = get_data(args, root)
    results = []
    # Pair methods within each (rank,seed). This makes corresponding conditions easy to compare.
    for rank in args.ranks:
        for seed in args.seeds:
            for method in args.methods:
                print("\n" + "=" * 70 + f"\n{method.upper()} | r={rank} seed={seed}\n" + "="*70, flush=True)
                results.append(run_one(args, data, method, rank, seed, device, precision, root))
                save_csv(root / "summary.csv", results)
                if len(args.methods) == 2:
                    same = [z for z in results if z["rank"]==rank and z["seed"]==seed]
                    if len(same) == 2 and len({z["init_classifier_sha256"] for z in same}) != 1:
                        raise RuntimeError("LoRA/DoRA initial classifier weights do not match")
    print("\n=== Completed ===", flush=True)
    for r in results:
        print(f"{r['method']:4} rank={r['rank']:2} seed={r['seed']:4} "
              f"val={r['best_val_accuracy']:.4%} test={r['test_accuracy']} "
              f"params={r['trainable_params']:,} sec={r['train_time_sec']:.1f}", flush=True)
    print(f"Saved results: {root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
