#!/usr/bin/env bash
# ============================================================================
# v12vision SERVE STUB — Gemma-4 E4B vision MERGED model on vLLM (multimodal).
# ============================================================================
# *** UNTESTED SCAFFOLDING — DO NOT ASSUME THIS WORKS AS-IS. ***
# No GPU has run this. It mirrors serve_v10it_vllm.sh but for the MULTIMODAL
# (image+text -> EN) path. Two things MUST be verified on the box first:
#
#   (1) vLLM on this rig actually supports `Gemma4ForConditionalGeneration`
#       MULTIMODAL (image) inputs. vLLM supports Gemma3 vision; gemma4 vision
#       support is UNCONFIRMED here. Check `vllm --help` / the model registry,
#       or just start the server and POST an image (see README curl).
#       If vLLM lacks gemma4 vision support, use the unsloth native inference
#       snippet in V12VISION_README.md instead — that path is guaranteed.
#
#   (2) The MERGED model dir must be a FULL multimodal checkpoint (vision tower
#       + projector intact + the vision-LoRA folded in). Merge with unsloth's
#       save_pretrained_merged (see README) — NOT the language-only clean-merge.
#
# DIFFERENCES vs the text serve script:
#   * NO MTP drafter (the official Gemma-4 MTP drafter is text-only; pairing it
#     with multimodal prompts is unverified — left OFF here).
#   * `--limit-mm-per-prompt` advertises 1 image/prompt.
#   * Larger default max-model-len: each page image costs ~280 soft tokens
#     (config vision_soft_tokens_per_image=280) ON TOP of OCR/context/output.
#
# Usage:
#   bash backend/scripts/eval/serve_v12vision_vllm.sh
#   PORT=8770 MERGED_DIR=/path/to/merged bash backend/scripts/eval/serve_v12vision_vllm.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/danny/Documents/personal/extension}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/backend/training/runs/manga-bubbles/gemma4_e4b_v12vision_poc}"
ADAPTER_DIR="${ADAPTER_DIR:-${RUN_DIR}/final}"
# MERGED multimodal checkpoint (vision tower + projector + folded LoRA).
MERGED_DIR="${MERGED_DIR:-${RUN_DIR}/merged}"
BASE_MODEL="${BASE_MODEL:-unsloth/gemma-4-E4B-it}"
PORT="${PORT:-8000}"
# Image soft-tokens (~280) + page OCR + context + output. Bump if pages are dense.
MAX_LEN="${MAX_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.65}"          # bf16 multimodal weights are heavier than text-only
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"   # safe default; vision graph capture is the
                                      # first thing to disable when debugging.
VLLM_VENV="${VLLM_VENV:-/home/danny/.venvs/vllm}"

log() { printf '[serve_v12vision_vllm] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

log "*** UNTESTED SCAFFOLDING — verify vLLM gemma4-vision support before trusting output ***"

[[ -x "${VLLM_VENV}/bin/vllm" ]] || die "vllm CLI not found at ${VLLM_VENV}/bin/vllm"

# --- merge the vision-LoRA into a FULL multimodal checkpoint, if not present ---
# NOTE: the repo's merge_gemma4_lora_clean.py was written for LANGUAGE-ONLY
# adapters. A vision-LoRA's safetensors also contain vision_tower(.linear) +
# (maybe) projector keys; that merge walker resolves `.linear` leaves too, BUT
# this is UNVERIFIED for the vision branch. The SAFE path is unsloth's native
# save_pretrained_merged at train time (see V12VISION_README.md). Here we just
# require a pre-merged dir and refuse to guess.
if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  die "merged multimodal model not found at ${MERGED_DIR}.
       Produce it with unsloth save_pretrained_merged (save_method=merged_16bit)
       — see V12VISION_README.md 'Merging the vision-LoRA'. Do NOT assume the
       language-only clean-merge covers the vision tower."
fi

# --- serve --------------------------------------------------------------------
export PATH="${VLLM_VENV}/bin:${PATH}"
# Per project notes, this rig needs these for serve stability on cu130/sm120:
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

EAGER_ARGS=()
[[ "$ENFORCE_EAGER" == "1" ]] && EAGER_ARGS+=(--enforce-eager)

log "starting vLLM multimodal server"
log "  model:        $MERGED_DIR"
log "  port:         $PORT  max_model_len: $MAX_LEN  gpu_util: $GPU_UTIL"
log "  images/prompt: $MAX_IMAGES_PER_PROMPT  (NO MTP drafter on the vision path)"

# ?? VERIFY the --limit-mm-per-prompt syntax for this vLLM build. Newer vLLM
# accepts `image=N` (key=val); older builds want a JSON blob '{"image": N}'.
exec "${VLLM_VENV}/bin/vllm" serve "$MERGED_DIR" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --limit-mm-per-prompt "image=${MAX_IMAGES_PER_PROMPT}" \
  "${EAGER_ARGS[@]}" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --trust-remote-code \
  --served-model-name v12vision \
  "$@"
