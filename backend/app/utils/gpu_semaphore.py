"""Shared GPU semaphore to prevent concurrent CUDA operations across services.

Prevents CUDA error 900 (operation not permitted when stream is capturing)
caused by onnxruntime and llama-cpp-python competing for GPU streams.
"""

import asyncio

# Single semaphore shared by OCR (onnxruntime) and translation (llama-cpp).
# Ensures only one GPU-bound operation runs at a time.
gpu_semaphore = asyncio.Semaphore(1)
