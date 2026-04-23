"""Optimize the exported PARSeq ONNX model for the local RTX 5090.

Steps:
  1. Load original ONNX (with external weights).
  2. Run ORT graph optimizations (`GraphOptimizationLevel.ORT_ENABLE_ALL`) and
     dump the optimized graph.
  3. Convert float32 tensors to float16 (keeping I/O as float32) for the GPU
     build, which nets ~2x throughput on modern Nvidia GPUs.
  4. Benchmark CPU-fp32 vs CUDA-fp32 vs CUDA-fp16.

Usage:
    backend/.venv/bin/python backend/scripts/optimize_parseq_onnx.py \
        --in backend/models/parseq_manga_large_5p16.onnx \
        --out-fp32 backend/models/parseq_manga_large_5p16.opt.onnx \
        --out-fp16 backend/models/parseq_manga_large_5p16.fp16.onnx
"""
import argparse
import time
from pathlib import Path

# torch first so its bundled CUDA libs are resolvable by onnxruntime-gpu
import torch  # noqa: F401
import numpy as np
import onnx
import onnxruntime as ort
# ORT ships its own fp16 converter which handles initializer/attribute
# propagation more completely than onnxconverter_common for transformer models.
from onnxruntime.transformers.float16 import convert_float_to_float16 as ort_convert_fp16


def save_optimized_graph(src: Path, dst: Path) -> None:
    """Persist a graph-optimized copy via ORT's SessionOptions."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(dst)
    # CPU provider is fine; we just want the optimized graph saved.
    ort.InferenceSession(str(src), sess_options=so, providers=["CPUExecutionProvider"])
    print(f"Optimized graph: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def convert_fp16(src: Path, dst: Path) -> None:
    model = onnx.load(str(src), load_external_data=True)
    # ORT's transformer-aware fp16 converter is more thorough at propagating
    # dtypes across attention subgraphs (initializers, scalar attrs, etc.),
    # avoiding the mixed-dtype bind errors onnxconverter_common leaves behind.
    fp16_model = ort_convert_fp16(
        model,
        keep_io_types=True,
        force_fp16_initializers=True,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        fp16_model,
        str(dst),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=dst.name + ".data",
    )
    print(f"FP16 model: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def bench(path: Path, providers, runs: int = 20, warmup: int = 3) -> float:
    sess = ort.InferenceSession(str(path), providers=providers)
    in_name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, 128, 512).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {in_name: x})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {in_name: x})
    dt = (time.perf_counter() - t0) / runs * 1000
    used = sess.get_providers()
    print(f"  {path.name:48s} {dt:7.2f} ms/run  providers={used}")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-fp32", required=True)
    ap.add_argument("--out-fp16", required=True)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    src = Path(args.inp)
    save_optimized_graph(src, Path(args.out_fp32))
    # Convert FP16 from the ORIGINAL (not the CPU-optimized graph, which
    # injects CPU-specific layouts like conv2d_nchwc that break CUDA load).
    convert_fp16(src, Path(args.out_fp16))

    if args.skip_bench:
        return

    print("\nBenchmarks (lower is better):")
    bench(src, [("CPUExecutionProvider", {})])
    bench(Path(args.out_fp32), [("CPUExecutionProvider", {})])
    try:
        bench(Path(args.out_fp32), [("CUDAExecutionProvider", {}), "CPUExecutionProvider"])
        bench(Path(args.out_fp16), [("CUDAExecutionProvider", {}), "CPUExecutionProvider"])
    except Exception as e:
        print(f"CUDA bench skipped: {e}")


if __name__ == "__main__":
    main()
