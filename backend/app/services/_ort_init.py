"""Preload CUDA shared libs so onnxruntime-gpu binds the CUDA EP reliably.
Import this BEFORE any onnxruntime InferenceSession is created."""
import logging
logger = logging.getLogger(__name__)
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
