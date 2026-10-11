#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
# Full fixed-condition LoRA + DoRA pair. Both methods train sequentially on one GPU.
python -u train.py --output runs/stanford_cars_vit_b16 --methods lora dora \
  --ranks 8 --seeds 42 --epochs 20 --batch-size 16 --grad-accum 2 \
  --adapter-lr 1e-4 --head-lr 5e-4 --precision auto --test-mode on
python analyze.py --input runs/stanford_cars_vit_b16
