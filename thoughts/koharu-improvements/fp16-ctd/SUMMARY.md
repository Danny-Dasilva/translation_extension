# FP16 Comic Text Detector (Item #13)

Goal: cut CTD (detection + mask) latency by exporting the ONNX model to FP16
so it lands on the RTX 5090 tensor cores (~2x dense conv throughput).

## Conversion

Script: `backend/scripts/export_ctd_fp16.py`

- Input  : `backend/models/comictextdetector.onnx`
- Output : `backend/models/comictextdetector.fp16.onnx`
- Converter: `onnxruntime.transformers.float16.convert_float_to_float16`
  with `keep_io_types=True, force_fp16_initializers=True,
  disable_shape_infer=False`. ORT-safe converter (also used for PARSeq in
  `optimize_parseq_onnx.py`) - handles Div/LayerNorm/attention subgraphs
  correctly, unlike onnxconverter_common.

### File size

| Model | Size  |
|-------|-------|
| FP32  | 94.7 MB |
| FP16  | 47.4 MB |

50% on-disk reduction. Single file (no external data needed).

### Sanity (dummy [1,3,1024,1024])

```
out[blk] (1, 64512, 7)       nan=False inf=False
out[seg] (1, 1, 1024, 1024)  nan=False inf=False
out[det] (1, 2, 1024, 1024)  nan=False inf=False
```

## Benchmark

Script: `backend/scripts/benchmark_ctd_fp16.py`
Log   : `thoughts/koharu-improvements/fp16-ctd/benchmark.txt`

50 iters on `de.png`, 5 warmup.

### CUDA availability

CUDA was **unavailable during this bench run**. The venv has both
`onnxruntime` (CPU) and `onnxruntime-gpu` installed and the CPU package's
shared libs won the install race (providers exposed:
`AzureExecutionProvider, CPUExecutionProvider`). `optimize_parseq_onnx.py`
already comments on the same conflict. Numbers below are CPU; re-run after
cleaning up the venv (uninstall `onnxruntime`, keep `onnxruntime-gpu`) to
capture the real tensor-core speedup (~2x expected).

### Latency (CPU fallback)

| Model | mean      | p50    | p99     | min    |
|-------|-----------|--------|---------|--------|
| FP32  | 500.19 ms | 415.24 | 1199.29 | 321.08 |
| FP16  | 412.56 ms | 362.97 |  862.93 | 309.58 |

FP16 speedup (mean): **1.21x** on CPU. The big win is on CUDA tensor cores.

## Parity (FP32 vs FP16 on de.png)

| Output | Shape                 | max\|diff\| | mean\|diff\| | notes |
|--------|-----------------------|-------------|--------------|-------|
| blk    | (1, 64512, 7)         | 1.876       | 0.034        | raw box logits, pre-threshold |
| seg    | (1, 1, 1024, 1024)    | 0.0071      | 0.00001      | sigmoid mask <= 0.05 |
| det    | (1, 2, 1024, 1024)    | 0.0163      | 0.00032      | sigmoid mask <= 0.05 |

Sigmoid max abs diff: **0.0163** (<= 0.05). Detection/segmentation
thresholds will not shift meaningfully.

## Integration note

The `ctd_service.py` agent will add a preferred-load branch that tries
`comictextdetector.fp16.onnx` first and falls back to `.onnx`. No changes
to `ctd_service.py` in this task.

## Files (all new)

- `backend/scripts/export_ctd_fp16.py`
- `backend/scripts/benchmark_ctd_fp16.py`
- `backend/models/comictextdetector.fp16.onnx`
- `thoughts/koharu-improvements/fp16-ctd/benchmark.txt`
- `thoughts/koharu-improvements/fp16-ctd/SUMMARY.md`
