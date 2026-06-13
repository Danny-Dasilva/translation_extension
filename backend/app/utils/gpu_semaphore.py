"""Shared GPU semaphore to prevent concurrent CUDA operations across services.

Prevents CUDA error 900 (operation not permitted when stream is capturing)
caused by onnxruntime and the transformers translation model competing for
GPU streams.
"""

import asyncio

# Single semaphore shared by OCR (onnxruntime) and translation.
# Ensures only one GPU-bound operation runs at a time.
gpu_semaphore = asyncio.Semaphore(1)
