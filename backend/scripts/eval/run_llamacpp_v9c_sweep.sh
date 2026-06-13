#!/usr/bin/env bash
# Sweep llama.cpp benchmarks for v9c across {Q4_K_M,Q8_0,F16} x {fa,no-fa} x {vanilla,draft}.
#
# Designed to be re-runnable: each combo writes to its own subdir under bench_out/.
# Skip a combo by leaving its model/draft path empty.
#
# Usage:
#   bash backend/scripts/eval/run_llamacpp_v9c_sweep.sh
set -euo pipefail

ROOT="${ROOT:-/home/danny/Documents/personal/extension}"
LLAMA_BIN="${LLAMA_BIN:-/home/danny/llama.cpp/build/bin/llama-server}"
LORA_GGUF="${LORA_GGUF:-/home/danny/Documents/personal/extension/backend/training/runs/manga-bubbles/gemma4_e4b_v9c/v9c-adapter.gguf}"

MODEL_Q4="${MODEL_Q4:-/home/danny/llama.cpp/models/gemma-4-E4B-pt.Q4_K_M.gguf}"
MODEL_Q8="${MODEL_Q8:-/home/danny/llama.cpp/models/gemma-4-E4B-pt.Q8_0.gguf}"
MODEL_F16="${MODEL_F16:-/home/danny/llama.cpp/models/gemma-4-E4B-pt.f16.gguf}"

DRAFT_GGUF="${DRAFT_GGUF:-}"   # set to a small drafter GGUF path to enable spec-decode runs

PROMPTS="${PROMPTS:-${ROOT}/backend/scripts/eval/bench_prompts.jsonl}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/backend/scripts/eval/bench_out}"
PYTHON="${PYTHON:-/home/danny/Documents/personal/extension/backend/.venv/bin/python}"
LIMIT="${LIMIT:-25}"
CTX="${CTX:-2048}"
MAX_TOK="${MAX_TOK:-64}"

mkdir -p "$OUT_ROOT"

run_one () {
  local label="$1"
  local model="$2"
  local fa="$3"     # "fa" or "no-fa"
  local draft="$4"  # path or empty

  if [[ ! -f "$model" ]]; then
    echo "SKIP $label: model not found ($model)" >&2
    return 0
  fi

  local out_dir="${OUT_ROOT}/${label}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "SKIP $label: summary already exists at ${out_dir}/summary.json" >&2
    return 0
  fi

  local fa_flag=()
  if [[ "$fa" == "fa" ]]; then
    fa_flag=(--flash-attn)
  fi

  local draft_flag=()
  if [[ -n "$draft" ]]; then
    draft_flag=(--draft-model "$draft")
  fi

  echo "RUN  $label (fa=$fa draft=${draft:-none})" >&2
  $PYTHON "${ROOT}/backend/scripts/eval/bench_llamacpp_v9c.py" \
      --llama-bin "$LLAMA_BIN" \
      --model "$model" \
      --lora "$LORA_GGUF" \
      --prompts "$PROMPTS" \
      --out-dir "$out_dir" \
      --quant "$label" \
      --ctx "$CTX" \
      --max-tokens "$MAX_TOK" \
      --limit "$LIMIT" \
      "${fa_flag[@]}" \
      "${draft_flag[@]}"
  echo "DONE $label" >&2
}

# vanilla decode sweep
run_one "q4km-nofa-vanilla" "$MODEL_Q4" "no-fa" ""
run_one "q4km-fa-vanilla"   "$MODEL_Q4" "fa"    ""
run_one "q8_0-nofa-vanilla" "$MODEL_Q8" "no-fa" ""
run_one "q8_0-fa-vanilla"   "$MODEL_Q8" "fa"    ""
run_one "f16-fa-vanilla"    "$MODEL_F16" "fa"   ""

# spec-decode sweep (only if DRAFT_GGUF is set)
if [[ -n "$DRAFT_GGUF" && -f "$DRAFT_GGUF" ]]; then
  run_one "q4km-fa-draft"  "$MODEL_Q4" "fa" "$DRAFT_GGUF"
  run_one "q8_0-fa-draft"  "$MODEL_Q8" "fa" "$DRAFT_GGUF"
fi

echo "ALL SWEEP RUNS COMPLETE  -> $OUT_ROOT" >&2
