#!/usr/bin/env bash
# Path A deploy: v10-it (Gemma 4 E4B-it + LoRA, merged) + Google's official Gemma 4 MTP drafter.
#
# Why this script exists
# ----------------------
# The recommended deploy is the `vllm/vllm-openai:gemma4-0505-cu130` Docker image.
# Docker isn't installed on this host, so we use a non-Docker path:
#
#   * Isolated venv at /home/danny/.venvs/vllm built from vLLM nightly post-2026-05-06
#     (PR #41745 "Add Gemma4 MTP speculative decoding support" merged in commit
#     27e0057ae on 2026-05-06; nightly wheel built from main HEAD includes it).
#   * vLLM Gemma 4 LoRA load is broken (issue #41754 — adapters silently ignored),
#     so we merge the adapter into the base on CPU first, then serve the merged
#     weights with the official MTP drafter `google/gemma-4-E4B-it-assistant`.
#
# Expected speedup
# ----------------
#   * v9c llama.cpp Q8_0 + flash-attn baseline: 77.8 tok/s median (single-batch, 25 prompts)
#   * v10-it vLLM + MTP (γ=4): targeting ~250-380 tok/s per Path A research notes
#     (Google reports E4B 171 -> 304.7 tok/s on H100; RTX 5090 should be in range)
#
# Usage
# -----
#   bash backend/scripts/eval/serve_v10it_vllm.sh                # default port 8000
#   PORT=8765 bash backend/scripts/eval/serve_v10it_vllm.sh
#   ADAPTER_DIR=.../checkpoint-XXXX bash backend/scripts/eval/serve_v10it_vllm.sh
#
# Environment knobs (all optional):
#   ADAPTER_DIR     — LoRA adapter dir   (default: .../v10it/final)
#   MERGED_DIR      — merged output dir  (default: .../v10it/merged)
#   BASE_MODEL      — base model id      (default: unsloth/gemma-4-E4B-it)
#   DRAFTER         — MTP drafter id     (default: google/gemma-4-E4B-it-assistant)
#   PORT            — OpenAI API port    (default: 8000)
#   GAMMA           — speculative tokens (default: 2)
#   MAX_LEN         — max model length   (default: 4096)
#   GPU_UTIL        — gpu_memory_util    (default: 0.85)
#   VLLM_VENV       — venv path          (default: /home/danny/.venvs/vllm)
#   SKIP_MERGE      — set to 1 to skip the merge step (assumes MERGED_DIR is ready)
#
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/danny/Documents/personal/extension}"
ADAPTER_DIR="${ADAPTER_DIR:-${PROJECT_ROOT}/backend/training/runs/manga-bubbles/gemma4_e4b_v10it/final}"
MERGED_DIR="${MERGED_DIR:-${PROJECT_ROOT}/backend/training/runs/manga-bubbles/gemma4_e4b_v10it/merged}"
BASE_MODEL="${BASE_MODEL:-unsloth/gemma-4-E4B-it}"
DRAFTER="${DRAFTER:-google/gemma-4-E4B-it-assistant}"
PORT="${PORT:-8000}"
GAMMA="${GAMMA:-2}"  # γ=2 wins on this corpus: 109 tok/s vs γ=4's 104; per-pos accept falls off too steeply past pos 1
MAX_LEN="${MAX_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.85}"
VLLM_VENV="${VLLM_VENV:-/home/danny/.venvs/vllm}"
SKIP_MERGE="${SKIP_MERGE:-0}"

log() { printf '[serve_v10it_vllm] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# --- preflight ---------------------------------------------------------------

[[ -x "${VLLM_VENV}/bin/vllm" ]] || die "vllm CLI not found at ${VLLM_VENV}/bin/vllm — run setup first"

if [[ ! -d "$ADAPTER_DIR" && "$SKIP_MERGE" != "1" ]]; then
  die "ADAPTER_DIR does not exist: $ADAPTER_DIR (training may not be finished yet)"
fi

# Confirm drafter is on disk (already cached for this rig).
DRAFTER_HF_CACHE_HINT="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${DRAFTER//\//--}"
if [[ ! -d "$DRAFTER_HF_CACHE_HINT" ]]; then
  log "WARN: drafter cache not found at $DRAFTER_HF_CACHE_HINT — vLLM will download it on first start"
fi

# --- step 1: merge LoRA -> base (CPU, ~3-5 min) ------------------------------
# vLLM Gemma 4 LoRA load is silently broken (issue #41754) — we MUST merge first.
# merge_gemma4_lora_clean.py is base-agnostic: walks safetensors keys, no PEFT
# class checks. Works for unsloth/gemma-4-E4B-it the same as the -pt variant.

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

# --- step 2: serve with official Gemma 4 MTP drafter -------------------------
# Speculative config is a JSON blob; method=mtp triggers the Gemma4MTP path
# added in PR #41745. num_speculative_tokens=γ controls draft length.

# Serve with Google's official Gemma 4 MTP drafter (method=mtp).
SPEC_METHOD="mtp"
SPEC_DRAFTER="$DRAFTER"

log "starting vLLM OpenAI-compatible server"
log "  model:        $MERGED_DIR"
log "  drafter:      $SPEC_DRAFTER (method=$SPEC_METHOD, γ=$GAMMA)"
log "  port:         $PORT"
log "  max_model_len: $MAX_LEN"
log "  gpu_util:     $GPU_UTIL"

# Build speculative-config JSON without trusting shell quoting heuristics.
SPEC_CONFIG=$(printf '{"method":"%s","model":"%s","num_speculative_tokens":%s}' \
  "$SPEC_METHOD" "$SPEC_DRAFTER" "$GAMMA")

# flashinfer JIT requires `ninja` on PATH; the venv's ninja is at $VLLM_VENV/bin/ninja
# but the `vllm` shim doesn't activate the venv's PATH for child processes.
export PATH="${VLLM_VENV}/bin:${PATH}"

exec "${VLLM_VENV}/bin/vllm" serve "$MERGED_DIR" \
  --speculative-config "$SPEC_CONFIG" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --trust-remote-code \
  --served-model-name v10it \
  "$@"
