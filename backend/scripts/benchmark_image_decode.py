"""
Benchmark: PIL-based base64 decode vs cv2.imdecode-based fast path.

Run from the `backend/` directory:

    uv run python scripts/benchmark_image_decode.py

Times 50 iterations per sample and writes results to
`thoughts/koharu-improvements/image-utils/benchmark.txt`.
"""
from __future__ import annotations

import base64
import io
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, List

from PIL import Image

# Make the `app` package importable when run from `backend/`
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.utils.image_processing import (  # noqa: E402
    decode_base64_to_numpy_fast,
    decode_base64_to_pil,
)

REPO_ROOT = BACKEND_DIR.parent
OUTPUT_DIR = REPO_ROOT / "thoughts" / "koharu-improvements" / "image-utils"
OUTPUT_FILE = OUTPUT_DIR / "benchmark.txt"

ITERATIONS = 50


def _pil_only_decode(b64: str):
    """
    Legacy path: pure PIL, no cv2. Mirrors the original implementation including
    a full decode (PIL.Image.open is lazy; downstream code always forces a full
    decode via convert/np.array, so we force it here too for a fair comparison).
    """
    data = b64
    if "," in data and data.startswith("data:image"):
        data = data.split(",", 1)[1]
    image_bytes = base64.b64decode(data)
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Force full decode (matches downstream np.array / .crop / etc. behavior)
    image.load()
    return image


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _time_it(fn: Callable[[str], object], b64: str, iterations: int) -> List[float]:
    times_ms: List[float] = []
    # Warmup
    for _ in range(3):
        fn(b64)
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn(b64)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return times_ms


def _encode_file_to_b64(path: Path) -> str:
    with open(path, "rb") as fh:
        raw = fh.read()
    ext = path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "png")
    return f"data:image/{mime};base64," + base64.b64encode(raw).decode("ascii")


def _format_stats(label: str, times_ms: List[float]) -> str:
    mean = statistics.mean(times_ms)
    p50 = _percentile(times_ms, 50)
    p99 = _percentile(times_ms, 99)
    return f"  {label:<32s} mean={mean:7.3f}ms  p50={p50:7.3f}ms  p99={p99:7.3f}ms"


def _pick_samples() -> List[Path]:
    samples: List[Path] = []
    de_png = REPO_ROOT / "de.png"
    if de_png.exists():
        samples.append(de_png)

    # Try to grab a sample JPG from runs/detect/ if one exists.
    runs_dir = REPO_ROOT / "runs" / "detect" / "runs"
    if runs_dir.exists():
        for jpg in runs_dir.rglob("*.jpg"):
            samples.append(jpg)
            break

    return samples


def main() -> int:
    samples = _pick_samples()
    if not samples:
        print("No sample images found (tried de.png and runs/detect/runs/*.jpg)")
        return 1

    lines: List[str] = []
    lines.append("Image decode benchmark")
    lines.append("=" * 72)
    lines.append(f"Iterations per variant: {ITERATIONS}")
    lines.append("")

    for sample in samples:
        b64 = _encode_file_to_b64(sample)
        size_kb = len(b64) * 3 / 4 / 1024  # decoded-bytes estimate
        header = f"Sample: {sample.relative_to(REPO_ROOT)}  ({size_kb:.1f} KB decoded)"
        print(header)
        lines.append(header)
        lines.append("-" * 72)

        old_times = _time_it(_pil_only_decode, b64, ITERATIONS)
        new_pil_times = _time_it(decode_base64_to_pil, b64, ITERATIONS)
        new_np_times = _time_it(decode_base64_to_numpy_fast, b64, ITERATIONS)

        for label, times in [
            ("PIL (old decode_base64_to_pil)", old_times),
            ("cv2->PIL (new decode_base64_to_pil)", new_pil_times),
            ("cv2 fast (decode_base64_to_numpy_fast)", new_np_times),
        ]:
            line = _format_stats(label, times)
            print(line)
            lines.append(line)

        # Speedup summary
        old_mean = statistics.mean(old_times)
        fast_mean = statistics.mean(new_np_times)
        speedup = old_mean / fast_mean if fast_mean > 0 else float("inf")
        summary = f"  -> cv2 fast path is {speedup:.2f}x faster than PIL-only"
        print(summary)
        lines.append(summary)
        lines.append("")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
