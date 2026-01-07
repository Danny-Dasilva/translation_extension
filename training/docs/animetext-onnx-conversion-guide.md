# AnimeText YOLO12s ONNX Conversion & Optimization Guide

This document details the complete process of converting the AnimeText YOLO12s model from HuggingFace to optimized ONNX and TensorRT formats, including benchmarking methodology and results.

## Table of Contents

1. [Overview](#overview)
2. [Model Research](#model-research)
3. [Environment Setup](#environment-setup)
4. [Model Download](#model-download)
5. [ONNX Export](#onnx-export)
6. [TensorRT Export](#tensorrt-export)
7. [Benchmarking](#benchmarking)
8. [Results](#results)
9. [Recommendations](#recommendations)

---

## Overview

**Goal**: Convert the `yolotext12s` model from [deepghs/AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo) to optimized ONNX format with TensorRT acceleration for low-latency anime text detection.

**Source Model**:
- Repository: `deepghs/AnimeText_yolo`
- Variant: `yolo12s_animetext`
- Architecture: YOLO12 (attention-centric with R-ELAN, FlashAttention)
- Task: Single-class detection (`text_block`)

---

## Model Research

### Model Specifications

| Property | Value |
|----------|-------|
| Architecture | YOLO12 (YOLOv12) |
| Parameters | 9.25M |
| FLOPS | 21.5G |
| F1 Score | 0.88 |
| mAP50 | 0.9389 |
| mAP50-95 | 0.8652 |
| Classes | 1 (`text_block`) |
| License | GPL-3.0 |

### YOLO12 Architecture Highlights

YOLO12 introduces significant architectural changes from previous YOLO versions:

- **Attention-Centric Design**: Departure from traditional CNN-only approaches
- **Area Attention Mechanism**: Divides feature maps into equally-sized regions
- **R-ELAN**: Residual Efficient Layer Aggregation Networks
- **FlashAttention**: Reduced memory overhead for attention operations
- **7x7 Separable Convolution**: Implicit positional encoding

### Optimization Research Summary

| Optimization | Expected Speedup | Notes |
|--------------|------------------|-------|
| ONNX Runtime (CUDA) | 1.3-1.5x | Cross-platform, easy deployment |
| ONNX FP16 | 1.5-2x | Half memory, Tensor Core acceleration |
| TensorRT FP16 | 2-5x | NVIDIA-specific, kernel fusion |
| TensorRT INT8 | 3-6x | Requires calibration dataset |

---

## Environment Setup

### Dependencies

```bash
# Using uv for package management
uv pip install ultralytics>=8.3.0      # YOLO12 support
uv pip install onnx onnxruntime-gpu    # ONNX inference
uv pip install onnx-simplifier         # Graph optimization
uv pip install tensorrt-cu12           # TensorRT (auto-installed by Ultralytics)
uv pip install huggingface_hub         # Model download
```

### Version Compatibility

For this guide, we used:
- Python 3.11
- PyTorch 2.9.1+cu128
- CUDA 12.8
- TensorRT 10.14.1
- ONNX Runtime 1.20+
- Ultralytics 8.3.244

---

## Model Download

### Authentication

The AnimeText_yolo repository is gated. First, authenticate with HuggingFace:

```bash
# Check authentication status
huggingface-cli whoami
# Or: hf auth whoami (new command)

# If not logged in:
huggingface-cli login
```

### Download Model Files

```bash
# Create destination directory
mkdir -p models/animetext

# Download yolo12s_animetext variant
huggingface-cli download deepghs/AnimeText_yolo \
  --include "yolo12s_animetext/*" \
  --local-dir models/animetext
```

### Downloaded Files

```
models/animetext/yolo12s_animetext/
├── model.pt              # PyTorch weights (18.9 MB)
├── model.onnx            # Pre-exported ONNX (37.2 MB)
├── labels.json           # Class labels: ["text_block"]
├── threshold.json        # Detection threshold: 0.272
├── model_type.json       # Model metadata
├── results.csv           # Training metrics
├── results.png           # Training visualization
├── F1_curve.png          # F1 score curve
├── P_curve.png           # Precision curve
├── R_curve.png           # Recall curve
├── PR_curve.png          # Precision-Recall curve
├── confusion_matrix.png  # Confusion matrix
└── labels.jpg            # Label distribution
```

### Validation

```python
from ultralytics import YOLO
import torch

# Load model
model = YOLO('models/animetext/yolo12s_animetext/model.pt')

# Verify
print(f"Task: {model.task}")           # detect
print(f"Classes: {model.names}")       # {0: 'text_block'}
print(f"CUDA: {torch.cuda.is_available()}")

# Test inference
import numpy as np
dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
results = model(dummy, verbose=False)
print("Inference successful!")
```

---

## ONNX Export

### Why Re-export?

While the HuggingFace repo includes a pre-exported `model.onnx`, we re-export with specific settings for:
- Consistent opset version (17) for TensorRT compatibility
- Graph simplification with onnxslim
- Fixed input shape for optimal TensorRT optimization

### FP32 ONNX Export

```python
from ultralytics import YOLO

model = YOLO('models/animetext/yolo12s_animetext/model.pt')

model.export(
    format='onnx',
    imgsz=640,           # Input resolution
    dynamic=False,       # Fixed batch size (better for TensorRT)
    simplify=True,       # Graph optimization with onnxslim
    opset=17,            # ONNX opset (TensorRT 10.x compatible)
    half=False,          # FP32 precision
)
# Output: model.onnx (35.5 MB)
```

### FP16 ONNX Export

```python
model.export(
    format='onnx',
    imgsz=640,
    dynamic=False,
    simplify=True,
    opset=17,
    half=True,           # FP16 precision
)
# Rename to model_fp16.onnx (17.9 MB)
```

### Export Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `format` | `'onnx'` | Target format |
| `imgsz` | `640` | Input resolution (standard YOLO) |
| `dynamic` | `False` | Fixed batch=1 for TensorRT optimization |
| `simplify` | `True` | Run onnxslim for graph optimization |
| `opset` | `17` | ONNX operator set version |
| `half` | `True/False` | FP16 or FP32 precision |

### ONNX Graph Optimizations Applied

1. **onnxslim**: Automatic graph simplification
2. **Constant folding**: Pre-compute constant operations
3. **Dead code elimination**: Remove unused nodes
4. **Operator fusion**: Combine sequential operations

---

## TensorRT Export

### Direct Export via Ultralytics

Ultralytics handles the ONNX-to-TensorRT conversion automatically:

```python
from ultralytics import YOLO

model = YOLO('models/animetext/yolo12s_animetext/model.pt')

model.export(
    format='engine',     # TensorRT engine
    imgsz=640,
    half=True,           # FP16 for Tensor Cores
    dynamic=False,       # Fixed batch
    workspace=4,         # 4GB optimization workspace
    device=0,            # GPU device ID
)
# Output: model.engine (20.1 MB)
```

### Build Process

The TensorRT build process:
1. Exports to ONNX first (intermediate step)
2. Parses ONNX graph
3. Applies TensorRT optimizations (layer fusion, kernel selection)
4. Builds hardware-specific engine
5. Saves serialized engine

Build time: ~15-20 seconds on RTX 5090

### TensorRT Optimizations

- **Layer Fusion**: Conv+BN+ReLU → single fused layer
- **Kernel Auto-Tuning**: Selects optimal GPU kernels
- **FP16 Tensor Cores**: Leverages half-precision hardware
- **Memory Optimization**: Efficient activation memory allocation

### Important Notes

1. **Hardware-Specific**: TensorRT engines are NOT portable across different GPU architectures
2. **Version-Specific**: Engines may not work across TensorRT versions
3. **Rebuild Required**: Must rebuild engine for each deployment target

---

## Benchmarking

### Methodology

Following best practices for accurate GPU benchmarking:

#### 1. CUDA Event Timing (Preferred)

```python
import torch

# Create CUDA events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Record timing
start_event.record()
output = model(input_tensor)
end_event.record()

# Synchronize and get elapsed time
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

**Why CUDA Events?**
- Measures actual GPU execution time
- Hides kernel launch overhead
- More accurate than CPU timers with `synchronize()`

#### 2. Warmup Phase

```python
WARMUP_ITERATIONS = 100

for _ in range(WARMUP_ITERATIONS):
    _ = model(input_tensor)
    torch.cuda.synchronize()

# Discard warmup results
torch.cuda.reset_peak_memory_stats()
```

**Why Warmup?**
- JIT compilation overhead
- GPU driver initialization
- CUDA context setup
- Memory allocation optimization

#### 3. Statistical Significance

```python
BENCHMARK_ITERATIONS = 1000

times = []
for i in range(BENCHMARK_ITERATIONS):
    start_event.record()
    _ = model(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event))
```

**Metrics Collected:**
- Mean, Median (P50)
- P95, P99 percentiles
- Min, Max
- Standard deviation
- CV% (Coefficient of Variation) - target < 10%
- FPS (throughput)

### Benchmark Script

The complete benchmark script is located at:
```
scripts/benchmark_animetext.py
```

Usage:
```bash
python scripts/benchmark_animetext.py --warmup 100 --iterations 1000
```

---

## Results

### System Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 5090 |
| GPU Memory | 32,111 MB |
| CUDA Version | 12.8 |
| PyTorch | 2.9.1+cu128 |
| TensorRT | 10.14.1 |

### Performance Comparison

| Model | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | FPS | CV% | Speedup |
|-------|-----------|----------|----------|----------|-----|-----|---------|
| PyTorch (FP32) | 5.556 | 5.521 | 5.802 | 6.178 | 180.0 | 2.5% | 1.00x |
| ONNX FP32 (CUDA) | 3.982 | 3.977 | 4.148 | 4.506 | 251.1 | 4.3% | 1.40x |
| **ONNX FP16 (CUDA)** | **2.691** | **2.696** | **2.726** | **2.734** | **371.6** | **1.8%** | **2.06x** |
| TensorRT FP16 | 2.764 | 2.747 | 2.893 | 3.036 | 361.7 | 2.4% | 2.01x |

### Analysis

1. **Best Performer**: ONNX FP16 with CUDA EP (371.6 FPS, 2.06x speedup)

2. **TensorRT vs ONNX FP16**: Nearly identical performance
   - TensorRT slightly slower due to Ultralytics wrapper overhead
   - On RTX 5090 (Blackwell), ONNX Runtime CUDA EP is highly optimized

3. **Latency Stability**: All optimized variants have excellent CV% < 5%
   - ONNX FP16 has best stability (1.8% CV)

4. **P99 Latency**: Critical for real-time applications
   - ONNX FP16: 2.734 ms (best)
   - TensorRT FP16: 3.036 ms

### File Sizes

| Format | Size | Compression |
|--------|------|-------------|
| PyTorch (.pt) | 19 MB | Baseline |
| ONNX FP32 | 36 MB | 1.9x larger |
| ONNX FP16 | 18 MB | ~Same as PT |
| TensorRT FP16 | 21 MB | 1.1x larger |

---

## Recommendations

### For Production Deployment

**Recommended: ONNX FP16 with ONNX Runtime CUDA EP**

```python
import onnxruntime as ort

session = ort.InferenceSession(
    'model_fp16.onnx',
    providers=['CUDAExecutionProvider']
)

# Inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_tensor})
```

**Why ONNX FP16?**
- Best FPS (371.6)
- Lowest P99 latency (2.734 ms)
- Most stable (1.8% CV)
- Portable across CUDA versions
- No hardware-specific rebuild needed

### When to Use TensorRT

Use TensorRT engine when:
- Deploying to a fixed hardware target
- Need INT8 quantization (requires calibration)
- Running on older GPUs where TensorRT has better optimization

### Post-processing Note

YOLO ONNX exports do NOT include NMS. Implement post-processing:

```python
from ultralytics.utils.ops import non_max_suppression

# Raw model output shape: (1, 5, 8400)
# 5 = 4 (bbox) + 1 (class score)

# Transpose and apply NMS
predictions = output[0].transpose(0, 2, 1)  # (1, 8400, 5)
results = non_max_suppression(
    torch.from_numpy(predictions),
    conf_thres=0.272,  # From threshold.json
    iou_thres=0.45
)
```

---

## Quick Start

```bash
# 1. Download model
huggingface-cli download deepghs/AnimeText_yolo \
  --include "yolo12s_animetext/*" \
  --local-dir models/animetext

# 2. Export ONNX FP16
python -c "
from ultralytics import YOLO
model = YOLO('models/animetext/yolo12s_animetext/model.pt')
model.export(format='onnx', imgsz=640, half=True, simplify=True, opset=17)
"

# 3. Run benchmark
python scripts/benchmark_animetext.py

# 4. Use in production
python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models/animetext/yolo12s_animetext/model.onnx',
                               providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name

# Your image preprocessing here
image = np.random.rand(1, 3, 640, 640).astype(np.float32)
output = session.run(None, {input_name: image})
print(f'Output shape: {output[0].shape}')
"
```

---

## References

- [Ultralytics ONNX Export Documentation](https://docs.ultralytics.com/integrations/onnx/)
- [Ultralytics TensorRT Export Documentation](https://docs.ultralytics.com/integrations/tensorrt/)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- [AnimeText YOLO HuggingFace Repository](https://huggingface.co/deepghs/AnimeText_yolo)
- [YOLO12 Architecture Overview](https://docs.ultralytics.com/models/yolo12/)
