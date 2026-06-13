#!/usr/bin/env bash
# Queue + launch Qwen3-4B-Base SFT v7 on 5090.
#
# Prereqs (verify before launch):
#   - GPU has ≥28 GB free (kill CTD / other training)
#   - backend/training/runs/manga-bubbles/data_v7.parquet exists
#   - backend/training/runs/manga-bubbles/dev_200_v7.parquet exists
#
# Usage:
#   cd /home/danny/Documents/personal/extension
#   bash backend/scripts/train/run_v7.sh            # real run
#   bash backend/scripts/train/run_v7.sh --dry      # dry-run (forward pass only)

set -euo pipefail
cd "$(dirname "$0")/../../.."

CFG="backend/training/configs/qwen3_4b_sft_v7.yaml"

if [[ "${1:-}" == "--dry" ]]; then
  echo "== DRY-RUN: verify tokenization + forward pass, no disk writes =="
  WANDB_DISABLED=1 uv run --project backend python \
    backend/scripts/train/sft_qwen3_unsloth.py --config "$CFG" --dry-run
  exit $?
fi

echo "== LAUNCHING v7 SFT =="
echo "  config: $CFG"
echo "  base:   Qwen/Qwen3-4B-Base"
echo "  data:   backend/training/runs/manga-bubbles/data_v7.parquet (144k rows)"
echo "  output: backend/training/runs/manga-bubbles/qwen3_4b_sft_v7/"
echo "  approx walltime on 5090: ~8 hours"

# Free GPU memory check — abort if <20 GB free
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [[ $free_mb -lt 20000 ]]; then
  echo "ABORT: only ${free_mb} MB GPU free. Need ≥20 GB for Qwen3-4B LoRA."
  exit 1
fi

WANDB_DISABLED=1 uv run --project backend python \
  backend/scripts/train/sft_qwen3_unsloth.py --config "$CFG"
