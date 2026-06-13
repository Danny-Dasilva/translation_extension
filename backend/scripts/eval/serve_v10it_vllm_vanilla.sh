#!/usr/bin/env bash
# Vanilla (no MTP) sister of serve_v10it_vllm.sh.
#
# Why this exists
# ---------------
# We need an apples-to-apples comparison: same vLLM engine, same merged
# weights, same chat template, same GPU util — but WITHOUT the MTP
# speculative-config. This isolates the MTP overhead/benefit from the
# rest of the vLLM stack.
#
# The base script (serve_v10it_vllm.sh) hardcodes --speculative-config
# in the exec line, so we fork rather than try to retrofit a flag.
#
# Usage
# -----
#   PORT=8000 SKIP_MERGE=1 MERGED_DIR=... \
#     bash backend/scripts/eval/serve_v10it_vllm_vanilla.sh
#
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/danny/Documents/personal/extension}"
ADAPTER_DIR="${ADAPTER_DIR:-${PROJECT_ROOT}/backend/training/runs/manga-bubbles/gemma4_e4b_v10it/final}"
MERGED_DIR="${MERGED_DIR:-${PROJECT_ROOT}/backend/training/runs/manga-bubbles/gemma4_e4b_v10it/merged}"
BASE_MODEL="${BASE_MODEL:-unsloth/gemma-4-E4B-it}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.78}"
VLLM_VENV="${VLLM_VENV:-/home/danny/.venvs/vllm}"
SKIP_MERGE="${SKIP_MERGE:-0}"

log() { printf '[serve_v10it_vllm_vanilla] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

[[ -x "${VLLM_VENV}/bin/vllm" ]] || die "vllm CLI not found at ${VLLM_VENV}/bin/vllm — run setup first"

if [[ ! -d "$ADAPTER_DIR" && "$SKIP_MERGE" != "1" ]]; then
  die "ADAPTER_DIR does not exist: $ADAPTER_DIR (training may not be finished yet)"
fi

if [[ "$SKIP_MERGE" != "1" ]]; then
  if [[ -f "${MERGED_DIR}/config.json" ]]; then
    log "skip merge: ${MERGED_DIR} already populated"
  else
    log "merging LoRA into ${BASE_MODEL} (CPU bf16)..."
    log "  adapter: $ADAPTER_DIR"
    log "  output:  $MERGED_DIR"
    mkdir -p "$(dirname "$MERGED_DIR")"
    cd "$PROJECT_ROOT"
    uv run --project backend python backend/scripts/eval/merge_gemma4_lora_clean.py \
      --adapter "$ADAPTER_DIR" \
      --output  "$MERGED_DIR" \
      --base    "$BASE_MODEL"
  fi
fi

[[ -f "${MERGED_DIR}/config.json" ]] || die "merge produced no config.json at $MERGED_DIR"

log "starting vLLM OpenAI-compatible server (VANILLA — no MTP)"
log "  model:        $MERGED_DIR"
log "  port:         $PORT"
log "  max_model_len: $MAX_LEN"
log "  gpu_util:     $GPU_UTIL"

# flashinfer JIT requires `ninja` on PATH
export PATH="${VLLM_VENV}/bin:${PATH}"

exec "${VLLM_VENV}/bin/vllm" serve "$MERGED_DIR" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --trust-remote-code \
  --served-model-name v10it \
  "$@"
