#!/usr/bin/env python3
"""
Comprehensive benchmark script for AnimeText YOLO12s models.

Compares performance across:
- PyTorch (model.pt)
- ONNX FP32 (model.onnx)
- ONNX FP16 (model_fp16.onnx)
- TensorRT FP16 (model.engine)
- ONNX Runtime with CUDA EP
- ONNX Runtime with TensorRT EP

Uses proper GPU timing with CUDA Events for accurate measurements.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Constants
MODEL_DIR = Path(__file__).parent.parent / "models" / "animetext" / "yolo12s_animetext"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "benchmark"
INPUT_SIZE = 640
WARMUP_ITERATIONS = 100
BENCHMARK_ITERATIONS = 1000


def get_device_info() -> dict:
    """Get GPU and system information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        info["cuda_version"] = torch.version.cuda
    return info


def create_dummy_input(batch_size: int = 1) -> np.ndarray:
    """Create dummy input tensor."""
    return np.random.rand(batch_size, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)


def calculate_statistics(times: list[float]) -> dict:
    """Calculate comprehensive latency statistics."""
    times_array = np.array(times)
    return {
        "mean_ms": float(np.mean(times_array)),
        "median_ms": float(np.median(times_array)),
        "std_ms": float(np.std(times_array)),
        "min_ms": float(np.min(times_array)),
        "max_ms": float(np.max(times_array)),
        "p50_ms": float(np.percentile(times_array, 50)),
        "p95_ms": float(np.percentile(times_array, 95)),
        "p99_ms": float(np.percentile(times_array, 99)),
        "cv_percent": float(np.std(times_array) / np.mean(times_array) * 100),
        "fps": float(1000.0 / np.mean(times_array)),
        "iterations": len(times),
    }


def benchmark_pytorch(model_path: Path, warmup: int, iterations: int) -> dict:
    """Benchmark PyTorch model using CUDA Events."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print("Benchmarking: PyTorch (model.pt)")
    print(f"{'='*60}")

    model = YOLO(str(model_path))
    model.to("cuda")

    # Create input tensor on GPU
    dummy_np = create_dummy_input()
    dummy_tensor = torch.from_numpy(dummy_np).cuda()

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.model(dummy_tensor)
        torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark with CUDA Events
    print(f"Benchmarking ({iterations} iterations)...")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        with torch.no_grad():
            _ = model.model(dummy_tensor)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Memory stats
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    stats = calculate_statistics(times)
    stats["peak_memory_mb"] = float(peak_memory_mb)
    stats["model_type"] = "pytorch"
    stats["model_path"] = str(model_path)

    print(f"Mean latency: {stats['mean_ms']:.3f} ms")
    print(f"P99 latency: {stats['p99_ms']:.3f} ms")
    print(f"Throughput: {stats['fps']:.1f} FPS")
    print(f"CV%: {stats['cv_percent']:.2f}%")

    return stats


def benchmark_onnx_runtime(model_path: Path, warmup: int, iterations: int,
                           use_tensorrt: bool = False) -> dict:
    """Benchmark ONNX model with ONNX Runtime."""
    import onnxruntime as ort

    ep_name = "TensorRT" if use_tensorrt else "CUDA"
    print(f"\n{'='*60}")
    print(f"Benchmarking: ONNX Runtime ({ep_name} EP) - {model_path.name}")
    print(f"{'='*60}")

    # Configure session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Configure providers
    if use_tensorrt:
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': str(RESULTS_DIR / 'trt_cache'),
            }),
            ('CUDAExecutionProvider', {'device_id': 0}),
        ]
    else:
        providers = [('CUDAExecutionProvider', {'device_id': 0})]

    print(f"Loading model with providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
    session = ort.InferenceSession(str(model_path), sess_options, providers=providers)

    # Get input/output names
    input_name = session.get_inputs()[0].name

    # Create input
    dummy_input = create_dummy_input()

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark with CUDA Events
    print(f"Benchmarking ({iterations} iterations)...")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        torch.cuda.synchronize()
        start_events[i].record()
        _ = session.run(None, {input_name: dummy_input})
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Memory stats
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    stats = calculate_statistics(times)
    stats["peak_memory_mb"] = float(peak_memory_mb)
    stats["model_type"] = f"onnx_runtime_{ep_name.lower()}"
    stats["model_path"] = str(model_path)
    stats["execution_provider"] = ep_name

    print(f"Mean latency: {stats['mean_ms']:.3f} ms")
    print(f"P99 latency: {stats['p99_ms']:.3f} ms")
    print(f"Throughput: {stats['fps']:.1f} FPS")
    print(f"CV%: {stats['cv_percent']:.2f}%")

    return stats


def benchmark_tensorrt_engine(model_path: Path, warmup: int, iterations: int) -> dict:
    """Benchmark TensorRT engine directly via Ultralytics."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print("Benchmarking: TensorRT Engine (model.engine)")
    print(f"{'='*60}")

    model = YOLO(str(model_path))

    # Create dummy input
    dummy_np = create_dummy_input()
    # Ultralytics expects HWC format for predict
    dummy_img = (dummy_np[0].transpose(1, 2, 0) * 255).astype(np.uint8)

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = model(dummy_img, verbose=False)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark with CUDA Events
    print(f"Benchmarking ({iterations} iterations)...")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        torch.cuda.synchronize()
        start_events[i].record()
        _ = model(dummy_img, verbose=False)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Memory stats
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    stats = calculate_statistics(times)
    stats["peak_memory_mb"] = float(peak_memory_mb)
    stats["model_type"] = "tensorrt_engine"
    stats["model_path"] = str(model_path)

    print(f"Mean latency: {stats['mean_ms']:.3f} ms")
    print(f"P99 latency: {stats['p99_ms']:.3f} ms")
    print(f"Throughput: {stats['fps']:.1f} FPS")
    print(f"CV%: {stats['cv_percent']:.2f}%")

    return stats


def generate_report(results: dict[str, Any], output_path: Path) -> None:
    """Generate markdown report."""
    report = []
    report.append("# AnimeText YOLO12s Benchmark Results\n")
    report.append(f"Generated: {datetime.now().isoformat()}\n")

    # Device info
    device_info = results.get("device_info", {})
    report.append("## System Information\n")
    report.append(f"- GPU: {device_info.get('gpu_name', 'N/A')}")
    report.append(f"- GPU Memory: {device_info.get('gpu_memory_total_mb', 'N/A')} MB")
    report.append(f"- CUDA Version: {device_info.get('cuda_version', 'N/A')}")
    report.append(f"- PyTorch Version: {device_info.get('torch_version', 'N/A')}")
    report.append("")

    # Summary table
    report.append("## Performance Summary\n")
    report.append("| Model | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | FPS | CV% | Speedup |")
    report.append("|-------|-----------|----------|----------|----------|-----|-----|---------|")

    # Get baseline (PyTorch)
    baseline_mean = None
    for name, stats in results.get("benchmarks", {}).items():
        if stats.get("model_type") == "pytorch":
            baseline_mean = stats["mean_ms"]
            break

    for name, stats in results.get("benchmarks", {}).items():
        speedup = baseline_mean / stats["mean_ms"] if baseline_mean else 1.0
        report.append(
            f"| {name} | {stats['mean_ms']:.3f} | {stats['p50_ms']:.3f} | "
            f"{stats['p95_ms']:.3f} | {stats['p99_ms']:.3f} | {stats['fps']:.1f} | "
            f"{stats['cv_percent']:.1f}% | {speedup:.2f}x |"
        )

    report.append("")
    report.append("## Analysis\n")

    # Find best performer
    best_name = None
    best_fps = 0
    for name, stats in results.get("benchmarks", {}).items():
        if stats["fps"] > best_fps:
            best_fps = stats["fps"]
            best_name = name

    if best_name and baseline_mean:
        best_stats = results["benchmarks"][best_name]
        report.append(f"- **Best performer**: {best_name} ({best_fps:.1f} FPS)")
        report.append(f"- **Speedup over PyTorch**: {baseline_mean / best_stats['mean_ms']:.2f}x")
        report.append(f"- **P99 latency**: {best_stats['p99_ms']:.3f} ms")

    # Write report
    report_path = output_path.with_suffix(".md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark AnimeText YOLO12s models")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS, help="Benchmark iterations")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"animetext_benchmark_{timestamp}.json"

    print("=" * 60)
    print("AnimeText YOLO12s Benchmark Suite")
    print("=" * 60)
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.iterations}")
    print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE}")

    # Get device info
    device_info = get_device_info()
    print(f"\nGPU: {device_info.get('gpu_name', 'N/A')}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "device_info": device_info,
        "config": {
            "warmup_iterations": args.warmup,
            "benchmark_iterations": args.iterations,
            "input_size": INPUT_SIZE,
        },
        "benchmarks": {},
    }

    # Model paths
    pt_path = MODEL_DIR / "model.pt"
    onnx_fp32_path = MODEL_DIR / "model.onnx"
    onnx_fp16_path = MODEL_DIR / "model_fp16.onnx"
    engine_path = MODEL_DIR / "model.engine"

    # Run benchmarks
    try:
        # 1. PyTorch baseline
        if pt_path.exists():
            results["benchmarks"]["PyTorch (FP32)"] = benchmark_pytorch(
                pt_path, args.warmup, args.iterations
            )

        # 2. ONNX FP32 with CUDA EP
        if onnx_fp32_path.exists():
            results["benchmarks"]["ONNX FP32 (CUDA)"] = benchmark_onnx_runtime(
                onnx_fp32_path, args.warmup, args.iterations, use_tensorrt=False
            )

        # 3. ONNX FP16 with CUDA EP
        if onnx_fp16_path.exists():
            results["benchmarks"]["ONNX FP16 (CUDA)"] = benchmark_onnx_runtime(
                onnx_fp16_path, args.warmup, args.iterations, use_tensorrt=False
            )

        # 4. ONNX with TensorRT EP (skip - known CUDA context issues)
        # if onnx_fp32_path.exists():
        #     try:
        #         results["benchmarks"]["ONNX (TensorRT EP)"] = benchmark_onnx_runtime(
        #             onnx_fp32_path, args.warmup, args.iterations, use_tensorrt=True
        #         )
        #     except Exception as e:
        #         print(f"TensorRT EP benchmark failed: {e}")
        print("\nSkipping ONNX TensorRT EP (known CUDA context issues with ORT)")

        # 5. TensorRT Engine
        if engine_path.exists():
            results["benchmarks"]["TensorRT FP16"] = benchmark_tensorrt_engine(
                engine_path, args.warmup, args.iterations
            )

    except Exception as e:
        print(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate report
    generate_report(results, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    baseline_mean = None
    for name, stats in results["benchmarks"].items():
        if "PyTorch" in name:
            baseline_mean = stats["mean_ms"]
            break

    for name, stats in results["benchmarks"].items():
        speedup = baseline_mean / stats["mean_ms"] if baseline_mean else 1.0
        print(f"{name:25} | {stats['mean_ms']:8.3f} ms | {stats['fps']:8.1f} FPS | {speedup:5.2f}x")


if __name__ == "__main__":
    main()
