# AnimeText YOLO12s Benchmark Results

Generated: 2026-01-07T09:26:46.801775

## System Information

- GPU: NVIDIA GeForce RTX 5090
- GPU Memory: 32111 MB
- CUDA Version: 12.8
- PyTorch Version: 2.9.1+cu128

## Performance Summary

| Model | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | FPS | CV% | Speedup |
|-------|-----------|----------|----------|----------|-----|-----|---------|
| PyTorch (FP32) | 5.556 | 5.521 | 5.802 | 6.178 | 180.0 | 2.5% | 1.00x |
| ONNX FP32 (CUDA) | 3.982 | 3.977 | 4.148 | 4.506 | 251.1 | 4.3% | 1.40x |
| ONNX FP16 (CUDA) | 2.691 | 2.696 | 2.726 | 2.734 | 371.6 | 1.8% | 2.06x |
| TensorRT FP16 | 2.764 | 2.747 | 2.893 | 3.036 | 361.7 | 2.4% | 2.01x |

## Analysis

- **Best performer**: ONNX FP16 (CUDA) (371.6 FPS)
- **Speedup over PyTorch**: 2.06x
- **P99 latency**: 2.734 ms