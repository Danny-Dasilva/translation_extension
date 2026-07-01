#!/usr/bin/env bash
# start_dev.sh — bring up the full local manga-translation stack for development.
#
# Canonical port layout
# ----------------------
#   * vLLM (OpenAI-compatible LLM server)   -> :8000   (the translation backend)
#   * FastAPI manga-translation backend     -> :8001   (OCR + detect + inpaint + render)
#   * Browser extension                     -> talks to :8001
#
# The FastAPI backend calls vLLM at http://127.0.0.1:8000/v1 (settings.vllm_base_url).
# This script starts vLLM on 8000, waits for it to report ready, then starts the
# FastAPI app on 8001. Ctrl-C tears both down.
#
# Usage:
#   bash backend/scripts/start_dev.sh
#   (run from anywhere — paths are resolved relative to this script)

set -euo pipefail

# --- resolve paths -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- native sm_120 (RTX 5090 / Blackwell) ONNX Runtime GPU -------------------
# onnxruntime-gpu 1.27.0.dev (CUDA-13 nightly) ships sm_120 + sm_100 cubins but
# DT_NEEDED-links CUDA-13 sonames (libcudart.so.13, libcublas.so.13, libcublasLt.so.13,
# libcufft.so.12, libcurand.so.10, libcudnn.so.9). PyPI only has stub cu13 runtime
# wheels, so the real CUDA 13.2 redistributable .so files were staged here:
#   .venv/cuda13-libs/
# Prepend that dir so ORT's CUDA EP resolves the cu13 libs at dlopen time. It is
# kept separate from .venv/.../site-packages/nvidia/*/lib (torch's CUDA-12 libs)
# to avoid the libcudnn.so.9 soname clash between the cu12 and cu13 cudnn builds.
CUDA13_LIBS="${BACKEND_DIR}/.venv/cuda13-libs"
export LD_LIBRARY_PATH="${CUDA13_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

VLLM_PORT="${VLLM_PORT:-8000}"
BACKEND_PORT="${BACKEND_PORT:-8001}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"   # seconds to wait for vLLM (model load is slow)

log() { printf '[start_dev] %s\n' "$*" >&2; }

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "stopping vLLM (pid ${VLLM_PID})"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# --- 1. start vLLM on :8000 in the background --------------------------------
log "starting vLLM on :${VLLM_PORT} (background)"
PORT="${VLLM_PORT}" bash "${BACKEND_DIR}/scripts/eval/serve_v10it_vllm.sh" &
VLLM_PID=$!

# --- 2. wait for vLLM /health (fall back to /v1/models) ----------------------
log "waiting for vLLM to become ready (timeout ${HEALTH_TIMEOUT}s)..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
until curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 \
   || curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "ERROR: vLLM exited before becoming ready — check its logs above"
    exit 1
  fi
  if [[ "$(date +%s)" -ge "${deadline}" ]]; then
    log "ERROR: vLLM did not become ready within ${HEALTH_TIMEOUT}s"
    exit 1
  fi
  sleep 2
done
log "vLLM is ready on :${VLLM_PORT}"

# --- 3. start FastAPI backend on :8001 (foreground) --------------------------
# Lean OCR config (co-located GPU): disable the hybrid AR-OCR retry and let the
# ONNX arena allocate only what it needs. With vLLM already holding ~13.5GB, the
# hybrid AR model + arena ballooning OOMs the OCR and triggers a retry storm
# (~10s/page). Off: ~1.4s/page, ~6GB freed, no OOM. Override to re-enable on a
# dedicated GPU: HYBRID_OCR_ENABLED=true bash start_dev.sh
export HYBRID_OCR_ENABLED="${HYBRID_OCR_ENABLED:-false}"
export ORT_ARENA_KSAME="${ORT_ARENA_KSAME:-1}"
log "starting FastAPI backend on :${BACKEND_PORT} (hybrid_ocr=${HYBRID_OCR_ENABLED}, foreground)"
cd "${BACKEND_DIR}"
PORT="${BACKEND_PORT}" exec "${BACKEND_DIR}/.venv/bin/python" main.py
