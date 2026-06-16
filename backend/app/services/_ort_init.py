"""Preload CUDA shared libs so onnxruntime-gpu binds the CUDA EP reliably.
Import this BEFORE any onnxruntime InferenceSession is created."""
import logging
import os
logger = logging.getLogger(__name__)


def cuda_provider_options(extra: "dict | None" = None) -> dict:
    """CUDA EP options with env-gated arena control.

    By default onnxruntime's CUDA arena uses kNextPowerOfTwo, which greedily
    doubles and fills all free VRAM — fatal when co-located with a vLLM server.
    Set ORT_ARENA_KSAME=1 to switch to kSameAsRequested (allocate only what each
    request needs). Optionally set ORT_GPU_MEM_LIMIT_BYTES to hard-cap the
    per-session arena. Both are no-ops unless the env vars are set, so normal
    serving is unchanged.
    """
    opts = dict(extra or {})
    if os.environ.get("ORT_ARENA_KSAME") == "1":
        opts["arena_extend_strategy"] = "kSameAsRequested"
    lim = os.environ.get("ORT_GPU_MEM_LIMIT_BYTES")
    if lim:
        try:
            opts["gpu_mem_limit"] = int(lim)
        except ValueError:
            logger.warning("Invalid ORT_GPU_MEM_LIMIT_BYTES=%r; ignoring", lim)
    return opts
try:
    import torch  # noqa: F401  (loads libcublas/libcudnn into the process)
except Exception as e:  # pragma: no cover
    logger.warning("torch import failed in _ort_init: %s", e)
try:
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
except Exception as e:  # pragma: no cover
    logger.warning("ort.preload_dlls failed: %s", e)
