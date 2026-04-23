"""Export the Comic Text Detector (CTD) ONNX model to FP16.

The RTX 5090 tensor cores deliver ~2x throughput on FP16 vs FP32 for the
dense conv/attention layers CTD uses. We use the ORT-safe float16 converter
(`onnxruntime.transformers.float16.convert_float_to_float16`) instead of
`onnxconverter_common.float16`, because the ORT variant propagates dtypes
across attention/LayerNorm/Div subgraphs and avoids the mixed-dtype bind
errors the onnxconverter version produces on transformer-ish models.

Usage:
    cd backend && uv run python scripts/export_ctd_fp16.py \
        --in models/comictextdetector.onnx \
        --out models/comictextdetector.fp16.onnx
"""
import argparse
from pathlib import Path

# torch must be imported before onnxruntime-gpu so its bundled CUDA libs
# are on the resolver path.
import torch  # noqa: F401
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.transformers.float16 import convert_float_to_float16 as ort_convert_fp16


def convert_fp16(src: Path, dst: Path) -> None:
    """Load the FP32 CTD model and emit a FP16 copy.

    keep_io_types=True           : I/O stay float32; only internal compute is fp16.
                                   Lets us swap the file in without touching the
                                   preprocessing pipeline.
    force_fp16_initializers=True : Convert constant weights/initializers to fp16
                                   (otherwise they stay fp32 and you lose the
                                   memory / bandwidth win).
    disable_shape_infer=False    : Run shape inference first so the converter can
                                   correctly walk broadcast/Div ops.
    """
    print(f"[load] {src} ({src.stat().st_size / 1e6:.1f} MB)")
    model = onnx.load(str(src), load_external_data=True)

    print("[convert] running ORT transformer-aware fp16 conversion ...")
    fp16_model = ort_convert_fp16(
        model,
        keep_io_types=True,
        force_fp16_initializers=True,
        disable_shape_infer=False,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    # CTD is ~94 MB fp32, expected ~47 MB fp16 - small enough to skip external
    # data, which keeps deployment simpler (single file).
    onnx.save_model(fp16_model, str(dst))
    print(f"[save]  {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def sanity_check(fp32_path: Path, fp16_path: Path) -> None:
    """Run a dummy [1,3,1024,1024] tensor through both models, confirm shapes
    match and fp16 outputs are free of NaN/Inf.
    """
    print("\n[sanity] running dummy-input inference parity check ...")
    dummy = np.random.rand(1, 3, 1024, 1024).astype(np.float32)

    # Try CUDA, gracefully fall back to CPU.
    cuda_providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    try:
        sess_fp32 = ort.InferenceSession(str(fp32_path), providers=cuda_providers)
        sess_fp16 = ort.InferenceSession(str(fp16_path), providers=cuda_providers)
        ep = sess_fp16.get_providers()[0]
        print(f"[sanity] provider: {ep}")
    except Exception as e:
        print(f"[sanity] CUDA unavailable ({e}); using CPU.")
        sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        sess_fp16 = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])

    in_name_32 = sess_fp32.get_inputs()[0].name
    in_name_16 = sess_fp16.get_inputs()[0].name

    out32 = sess_fp32.run(None, {in_name_32: dummy})
    out16 = sess_fp16.run(None, {in_name_16: dummy})

    assert len(out32) == len(out16), f"Output count mismatch: {len(out32)} vs {len(out16)}"
    for i, (a, b) in enumerate(zip(out32, out16)):
        name = sess_fp32.get_outputs()[i].name
        assert a.shape == b.shape, f"Output '{name}' shape mismatch: {a.shape} vs {b.shape}"
        nan16 = np.isnan(b).any()
        inf16 = np.isinf(b).any()
        max_abs = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
        print(f"  out[{i}] {name:6s} shape={a.shape} max|fp32-fp16|={max_abs:.4f} "
              f"nan={nan16} inf={inf16}")
        assert not nan16, f"FP16 output '{name}' contains NaN"
        assert not inf16, f"FP16 output '{name}' contains Inf"

    print("[sanity] OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="models/comictextdetector.onnx")
    ap.add_argument("--out", default="models/comictextdetector.fp16.onnx")
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args()

    src = Path(args.inp)
    dst = Path(args.out)
    if not src.exists():
        raise FileNotFoundError(f"FP32 model not found: {src}")

    convert_fp16(src, dst)

    if not args.skip_sanity:
        sanity_check(src, dst)

    print("\nDone.")
    print(f"  fp32: {src}  ({src.stat().st_size / 1e6:.1f} MB)")
    print(f"  fp16: {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
