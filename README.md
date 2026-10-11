# ViT-B/16: LoRA vs DoRA on Stanford Cars (RTX A4000)

Based on the provided `sample(1).ipynb`; here the notebook-only tests have been removed, and training runs once per `(method, rank, seed)`.

## Design

- Dataset: [`mteb/StanfordCars`](https://huggingface.co/datasets/mteb/StanfordCars), **196 classes**, 8,144 original train images and 8,041 untouched test images. Validation is stratified **15% of original train**, with fixed split seed 42. Images 224×224.
- Backbone: `google/vit-base-patch16-224-in21k`; transformer backbone frozen. New classifier 768→196 is trainable for **both** methods.
- Adapted modules: all 12 blocks' attention Q/V projections; fallback supports both `q_proj/v_proj` and `query/value` naming with exact 24-layer assertion.
- Compared settings (initial): LoRA and DoRA, rank=8, alpha=16, dropout=0, batch=16, grad accumulation=2, 20 epochs, seed=42. Both use identical train split, augmentations, optimizer, LR schedule, initial head, training epoch budget and checkpoint-selection rule.
- AdamW, adapter LR 1e-4, head LR 5e-4, weight decay .05, cosine LR with 5% warmup, best checkpoint chosen **only** on validation accuracy (lower loss breaks a tie).
- Precision: BF16 where supported; FP16 + GradScaler otherwise. **No quantization or torchao required**.
- Default `--test-mode on` evaluates each best-validation checkpoint once on the held-out test. Do not use these test accuracies to select rank or hyperparameters. For sweeps use `--test-mode off` and reserve test until the finalized configuration.

The parameter budget for rank=8 should be: **LoRA 294,912 adapter + 150,724 head = 445,636 trainable**; **DoRA 313,344 adapter (18,432 magnitude) + 150,724 head = 464,068 trainable**. The script checks these numbers, and fails rather than silently training the wrong layers.

## Step A — add the files to your team repository (local computer)

You already use: https://github.com/STAN147/ProjectSchool_team17

Create the folder `experiments/cars_peft` in your local repo, and copy into it these files: `train.py`, `analyze.py`, `requirements.txt`, `launch.sh`, `.gitignore`, `README.md`. Do **not** commit `runs/` with model checkpoints / large predictions.

```bash
git add experiments/cars_peft
git commit -m "Add ViT Stanford Cars LoRA DoRA experiment"
git push origin main
```

**Fast alternative: no git push required.** Download `vit_lora_dora_a4000.zip` from this chat to your local machine. From Windows PowerShell in the download folder:

```powershell
scp .\vit_lora_dora_a4000.zip LOGIN@SERVER:~/
```

Then on the server:

```bash
ssh LOGIN@SERVER
git clone https://github.com/STAN147/ProjectSchool_team17.git
mkdir -p ProjectSchool_team17/experiments
python3 -m zipfile -e ~/vit_lora_dora_a4000.zip ProjectSchool_team17/experiments
cd ProjectSchool_team17/experiments/cars_peft
```

The archive contains `cars_peft/` at its root. Thus you can clone your actual team repository and run the experiment without pushing changes beforehand. To version your experiment properly, add/commit/push these files later.

## Step B — clone / update on the server

```bash
ssh LOGIN@SERVER
nvidia-smi
git clone https://github.com/STAN147/ProjectSchool_team17.git
cd ProjectSchool_team17/experiments/cars_peft
```

If already cloned: `cd ProjectSchool_team17 && git pull --ff-only && cd experiments/cars_peft`.

Create a **fresh** Python venv; Python 3.11 or 3.12 recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Example PyTorch build, compatible with CUDA 12.6 capable NVIDIA drivers:
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
# torchao is not needed; remove if accidentally installed:
python -m pip uninstall -y torchao
python -m pip check
python -c 'import torch, peft, transformers; print(torch.__version__, peft.__version__, transformers.__version__); print(torch.cuda.is_available(),torch.cuda.get_device_name(0)); print("bf16:",torch.cuda.is_bf16_supported())'
```

The CUDA **wheel** must be compatible with the installed NVIDIA driver (`nvidia-smi`). If the `cu126` example fails or the driver is older, choose the CUDA build supported by the server from the [official PyTorch installation page](https://pytorch.org/get-started/locally/); do **not** randomly change the system CUDA/driver. Avoid installing dependencies into an existing venv containing incompatible `torchao==0.10.0`.

## Step C — run LoRA, then DoRA on one A4000

Recommended: use `tmux` so disconnecting SSH does not interrupt the job:

```bash
tmux new -s cars
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
bash launch.sh 2>&1 | tee training.log
# When complete: Ctrl+B, then D to detach. Reattach with `tmux attach -t cars`.
```

`bash launch.sh` sequentially trains **LoRA r=8/seed=42 then DoRA r=8/seed=42**, then generates charts, paired tests, and `REPORT.md`.

Check progress with `tail -f training.log` in another SSH session; GPU usage `nvidia-smi`.

### Optional smoke test BEFORE the full run

A quick integration test, not valid benchmark metrics:

```bash
python -u train.py --output runs/smoke --epochs 1 --max-train-batches 2 --test-mode off
```

**Use a different output directory** for smoke vs full experiments.

### More ranks and seeds (after fixed first comparison)

For meaningful between-seed stability use at least 3 seed values; to select rank use validation only:

```bash
python -u train.py --output runs/rank_sweep --methods lora dora \
  --ranks 2 4 8 16 --seeds 42 123 3407 --epochs 20 \
  --batch-size 16 --grad-accum 2 --precision auto --test-mode off \
  2>&1 | tee rank_sweep.log
python analyze.py --input runs/rank_sweep
```

Set `--batch-size 8 --grad-accum 4` if memory is insufficient. That keeps nominal effective batch=32; training time changes. For strict comparison of methods, use the **same settings** for LoRA and DoRA in one experiment. Do not combine results from two different output roots/protocols without documenting their differences.

**On OOM:** choose a NEW `--output` directory with a reduced batch size, or remove the incomplete per-run folder and `experiment_config.json`/`split_manifest.json` if you intend to restart with changed settings. Completed per-run files are not overwritten automatically. There is no mid-epoch resume; best adapter and epoch logs are persisted as the run progresses.

## What is saved

```
runs/stanford_cars_vit_b16/
  experiment_config.json          # CLI/settings, including model, schedule, all hyperparams
  system.json                     # GPU, software versions, CUDA, git revision/status
  pip_freeze.txt                  # environment lock snapshot
  split_manifest.json             # exact train/val row indices, fingerprints, class names
  summary.csv                     # one row per completed method/rank/seed, incremental
  lora_r8_seed42/
    config.json                   # exact method+rank+seed config
    architecture.json             # 24 targeted layers, head hash, trainable params
    best_adapter/                 # PEFT adapter + trained classification head
    history.csv                   # per-epoch loss, val accuracy/loss, LR, time, optimizer steps
    result.json                   # best epoch, test/val accuracy, time, peak PyTorch VRAM
    predictions_val.csv           # per-image truth, pred, confidence, correctness, image id
    predictions_test.csv          # same on test if --test-mode on
    logits_val.npz                # all 196 logits for every validation example
    logits_test.npz               # test logits if --test-mode on
  dora_r8_seed42/                 # same structure
  all_epochs.csv                  # concatenated epoch histories, produced by analyze.py
  aggregated.csv                  # mean/SD by method & rank if multiple seeds
  paired_val_stats.csv            # exact McNemar + paired-image bootstrap 95% CI
  paired_test_stats.csv           # if test predictions available
  statistics.json
  REPORT.md
  plots/
    val_accuracy_by_epoch.png
    val_loss_by_epoch.png
    train_loss_by_epoch.png
    accuracy_vs_params.png
    accuracy_vs_time.png
    accuracy_vs_vram.png
    rank_sweep_val.png            # if >=2 ranks
```

Each `best_adapter` is a **small PEFT checkpoint**; the ImageNet-pretrained base model is referenced by its Hugging Face name, not copied. `adapter_model.safetensors` includes both LoRA/DoRA and the classifier because `modules_to_save=["classifier"]`.

After training, `analyze.py` constructs McNemar's exact paired test for **identical images** plus paired-image bootstrap 95% CIs for DoRA−LoRA accuracy. For multiple comparisons it reports Holm-adjusted p-values. These CIs describe **test-image sampling uncertainty conditional on fixed trained models**, not variation across training seeds. Report per-seed results and across-seed variance separately. There is no guarantee DoRA beats LoRA.

## Get results back to your computer

On the **local** machine:

```bash
scp -r LOGIN@SERVER:~/ProjectSchool_team17/experiments/cars_peft/runs/stanford_cars_vit_b16 ./cars_results
```

For a smaller download, exclude checkpoints and logits by making a reporting archive on the server:

```bash
cd ~/ProjectSchool_team17/experiments/cars_peft
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root=Path('runs/stanford_cars_vit_b16')
with ZipFile('cars_report_only.zip','w',compression=ZIP_DEFLATED) as z:
    for f in root.rglob('*'):
        if f.is_file() and not any(p in f.parts for p in ('best_adapter',)) and f.suffix not in ('.npz',):
            z.write(f, f.relative_to(root.parent))
print('cars_report_only.zip')
PY
```

Download `cars_report_only.zip` with `scp`. Keep the **full** `runs/` on the server for future plots, audit and independent verification.

## Research interpretation

**H1:** At equal rank and training budget, DoRA achieves higher held-out accuracy than LoRA. Test this *empirically*, with paired data and ideally multiple seeds.

**H2:** Differences are stronger at low rank. Test with r=2/4/8/16 (rank sweep); same rank is **not** the same number of trainable parameters because DoRA also learns magnitude.

**H3:** Any improvement justifies additional computational cost. Compare time, peak GPU memory and adapter/head trainable parameter counts. PyTorch allocated memory is not the full driver-reported consumption.

Do not tune models after inspecting final test metrics. If tuning is needed, run `--test-mode off`, select the hyperparameters from validation, then perform one held-out test under a fixed finalized protocol.
