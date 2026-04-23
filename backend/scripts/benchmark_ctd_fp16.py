"""Benchmark FP32 vs FP16 Comic Text Detector ONNX models.

Runs a real-image (de.png at repo root) inference loop and reports
mean/p50/p99 latency. Also computes an L1 parity diff on the `seg` (and
`det`) output tensors, which are sigmoid-activated pixel masks - max abs
diff > 0.05 there would mean detection thresholds might shift.

Falls back to CPU if CUDA is contended (other process holds the GPU).

Usage:
    cd backend && uv run python scripts/benchmark_ctd_fp16.py
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

# torch first so onnxruntime-gpu finds CUDA libs.
import torch  # noqa: F401
import numpy as np
import cv2
import onnxruntime as ort


INPUT_SIZE = 1024  # CTD's fixed input resolution.
WARMUP = 5
RUNS = 50


def preprocess(img_path: Path, size: int = INPUT_SIZE) -> np.ndarray:
    """Match ctd_service._preprocess: letterbox-resize to size, then NCHW/0..1."""
    img = cv2.imread(str(img_path))  # BGR
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    x = padded.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # NHWC -> NCHW
    return np.ascontiguousarray(x)


def make_session(path: Path, prefer_cuda: bool) -> tuple[ort.InferenceSession, str]:
    """Create an ORT session preferring CUDA, else fall back to CPU."""
    if prefer_cuda:
        try:
            sess = ort.InferenceSession(
                str(path),
                providers=[
                    ("CUDAExecutionProvider", {
                        # Heuristic conv algo picking - cheap & handles varying shapes
                        # without the long EXHAUSTIVE autotune.
                        "cudnn_conv_algo_search": "HEURISTIC",
                    }),
                    "CPUExecutionProvider",
                ],
            )
            actual = sess.get_providers()[0]
            if actual == "CUDAExecutionProvider":
                return sess, "CUDA"
            print(f"[warn] Requested CUDA but got {actual} for {path.name}")
        except Exception as e:
            print(f"[warn] CUDA session failed for {path.name}: {e}")

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess, "CPU"


def bench(sess: ort.InferenceSession, x: np.ndarray, runs: int, warmup: int) -> dict:
    """Time runs iterations after warmup. Returns stats in milliseconds."""
    in_name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {in_name: x})

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {in_name: x})
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    return {
        "mean": statistics.fmean(times_ms),
        "p50":  times_ms[len(times_ms) // 2],
        "p99":  times_ms[min(len(times_ms) - 1, int(len(times_ms) * 0.99))],
        "min":  times_ms[0],
        "max":  times_ms[-1],
    }


def parity(sess32: ort.InferenceSession, sess16: ort.InferenceSession,
           x: np.ndarray) -> list[dict]:
    """Compute per-output L1 diff stats between fp32 and fp16 sessions."""
    in32 = sess32.get_inputs()[0].name
    in16 = sess16.get_inputs()[0].name
    out32 = sess32.run(None, {in32: x})
    out16 = sess16.run(None, {in16: x})

    results = []
    for i, (a, b) in enumerate(zip(out32, out16)):
        name = sess32.get_outputs()[i].name
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        diff = np.abs(a - b)
        denom = np.maximum(np.abs(a), 1e-6)
        rel = diff / denom
        results.append({
            "name": name,
            "shape": a.shape,
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "max_rel": float(rel.max()),
            "mean_rel": float(rel.mean()),
        })
    return results


class Tee:
    """Write print() output to both stdout and a log file."""
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w")
        self._stdout = sys.stdout

    def write(self, s):
        self._stdout.write(s)
        self.f.write(s)

    def flush(self):
        self._stdout.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default="models/comictextdetector.onnx")
    ap.add_argument("--fp16", default="models/comictextdetector.fp16.onnx")
    ap.add_argument("--image", default="../de.png")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--log", default="../thoughts/koharu-improvements/fp16-ctd/benchmark.txt")
    args = ap.parse_args()

    fp32 = Path(args.fp32)
    fp16 = Path(args.fp16)
    img = Path(args.image)
    for p in (fp32, fp16, img):
        if not p.exists():
            raise FileNotFoundError(p)

    tee = Tee(Path(args.log))
    sys.stdout = tee
    try:
        print("=" * 72)
        print("CTD FP32 vs FP16 Benchmark")
        print("=" * 72)
        print(f"fp32 model : {fp32}  ({fp32.stat().st_size / 1e6:.1f} MB)")
        print(f"fp16 model : {fp16}  ({fp16.stat().st_size / 1e6:.1f} MB)")
        print(f"image      : {img}")
        print(f"runs       : {args.runs}  (warmup {args.warmup})")
        print(f"ort        : {ort.__version__}")
        print(f"available  : {ort.get_available_providers()}")
        print()

        x = preprocess(img)
        print(f"input shape: {x.shape} dtype={x.dtype}")
        print()

        # Try CUDA first; fall back to CPU once (together) if the GPU is busy.
        sess32, ep32 = make_session(fp32, prefer_cuda=True)
        sess16, ep16 = make_session(fp16, prefer_cuda=(ep32 == "CUDA"))
        if ep32 != ep16:
            # Re-benchmark on the same EP for fairness.
            print(f"[warn] EP mismatch ({ep32} vs {ep16}) - forcing both to CPU for fair comparison")
            sess32, ep32 = make_session(fp32, prefer_cuda=False)
            sess16, ep16 = make_session(fp16, prefer_cuda=False)

        if ep32 == "CPU":
            print("[note] CUDA unavailable during bench. (Running on CPU.)")
        print(f"[ep] using {ep32} for both models")
        print()

        # Parity first (cheap, 1 iter each) - reports whether FP16 is numerically safe.
        print("-- Parity (fp32 vs fp16 on real image) --")
        par = parity(sess32, sess16, x)
        worst_sigmoid_diff = 0.0
        for p in par:
            print(f"  out[{p['name']}] shape={p['shape']}"
                  f"  max_abs={p['max_abs']:.5f}"
                  f"  mean_abs={p['mean_abs']:.5f}"
                  f"  max_rel={p['max_rel']:.3f}")
            # seg and det are sigmoid masks - those are the safety-critical ones.
            if p["name"] in ("seg", "det"):
                worst_sigmoid_diff = max(worst_sigmoid_diff, p["max_abs"])
        if worst_sigmoid_diff > 0.05:
            print(f"[WARN] sigmoid-output max abs diff {worst_sigmoid_diff:.4f} > 0.05 "
                  "- detection thresholds may shift.")
        else:
            print(f"[ok] sigmoid-output max abs diff {worst_sigmoid_diff:.4f} <= 0.05")
        print()

        # Latency
        print(f"-- Latency ({args.runs} runs, {args.warmup} warmup) --")
        s32 = bench(sess32, x, args.runs, args.warmup)
        print(f"  fp32  mean={s32['mean']:7.2f} ms  p50={s32['p50']:7.2f}  p99={s32['p99']:7.2f}  "
              f"min={s32['min']:7.2f}  max={s32['max']:7.2f}")

        s16 = bench(sess16, x, args.runs, args.warmup)
        print(f"  fp16  mean={s16['mean']:7.2f} ms  p50={s16['p50']:7.2f}  p99={s16['p99']:7.2f}  "
              f"min={s16['min']:7.2f}  max={s16['max']:7.2f}")

        speedup = s32["mean"] / s16["mean"] if s16["mean"] > 0 else float("nan")
        print()
        print(f"fp16 speedup (mean): {speedup:.2f}x")
        print()
        print("=" * 72)
    finally:
        sys.stdout = tee._stdout
        tee.close()
    print(f"\nBenchmark log written to: {args.log}")


if __name__ == "__main__":
    main()
