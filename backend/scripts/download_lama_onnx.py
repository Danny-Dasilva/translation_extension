#!/usr/bin/env python3
"""Download a LaMa ONNX checkpoint for manga inpainting.

Primary source: `Carve/LaMa-ONNX` on HuggingFace (community-standard LaMa
ONNX port, fp32, ~200 MB). We save the file to `backend/models/lama.onnx`.

Usage:
    uv run python scripts/download_lama_onnx.py

If HuggingFace is unreachable inside your sandbox, the script prints the
exact `huggingface-cli` command you can run manually on a network-enabled
terminal.

After download, the script inspects the ONNX graph and prints:
    - input / output node names
    - expected input shapes
    - opset / IR version
so the service can wire up the correct input ordering.

Observed IO (Carve/LaMa-ONNX/lama_fp32.onnx, opset 17):
    image  : (batch, 3, 512, 512)  float32, scaled to [0, 1]
    mask   : (batch, 1, 512, 512)  float32, binary {0, 1}
    output : (batch, 3, 512, 512)  float32, scaled to [0, 255]

Although the graph declares `batch` as dynamic, the spatial dims (512, 512)
are BAKED INTO the weights — the service must resize every crop to 512×512
before forward and resize the output back.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
MODELS_DIR = BACKEND / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Candidate LaMa ONNX artifacts, tried in order.
CANDIDATES = [
    # (repo_id, filename, out_basename)
    ("Carve/LaMa-ONNX", "lama_fp32.onnx", "lama.onnx"),
    # Fallbacks — kept here so future runs on a different mirror still work.
    ("anyisalin/big-lama-onnx", "big-lama.onnx", "lama.onnx"),
    ("Sanster/iopaint-models", "lama-manga.safetensors", "lama-manga.safetensors"),
]


def manual_instructions(target: Path) -> str:
    """Print a copy-pasteable command for manual fetches."""
    return (
        f"Manual download instructions (run on a machine with network access):\n\n"
        f"    huggingface-cli download Carve/LaMa-ONNX lama_fp32.onnx "
        f"--local-dir {target.parent}\n"
        f"    mv {target.parent / 'lama_fp32.onnx'} {target}\n\n"
        f"Expected destination: {target}\n"
    )


def inspect_onnx(path: Path) -> None:
    """Print the graph IO so we can confirm input ordering."""
    try:
        import onnx  # type: ignore[import-untyped]
    except ImportError:
        print("(onnx not installed — skipping graph inspection)")
        return

    model = onnx.load(str(path))
    print("\n=== ONNX graph summary ===")
    print(f"  ir_version : {model.ir_version}")
    print(f"  opset      : {[o.version for o in model.opset_import]}")
    print("  inputs:")
    for inp in model.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(d.dim_param or d.dim_value)
        print(f"    {inp.name}: {dims}")
    print("  outputs:")
    for out in model.graph.output:
        dims = []
        for d in out.type.tensor_type.shape.dim:
            dims.append(d.dim_param or d.dim_value)
        print(f"    {out.name}: {dims}")
    print("===========================\n")


def download_lama(force: bool = False) -> Path:
    target = MODELS_DIR / "lama.onnx"
    if target.exists() and not force:
        print(f"LaMa ONNX already present at {target} ({target.stat().st_size / 1e6:.1f} MB)")
        inspect_onnx(target)
        return target

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: huggingface_hub is not installed. `uv pip install huggingface-hub`.")
        print(manual_instructions(target))
        sys.exit(1)

    last_err: Exception | None = None
    for repo_id, filename, out_name in CANDIDATES:
        # Only the first candidate produces `lama.onnx`; subsequent entries are
        # listed for manual use, not for auto-renaming.
        if out_name != "lama.onnx":
            continue
        print(f"Trying {repo_id}/{filename} …")
        try:
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(MODELS_DIR))
            src = Path(downloaded)
            if src != target:
                src.replace(target)
            size_mb = target.stat().st_size / 1e6
            print(f"OK — saved to {target} ({size_mb:.1f} MB)")
            inspect_onnx(target)
            return target
        except Exception as exc:  # noqa: BLE001 — download failures are diverse
            print(f"  failed: {exc}")
            last_err = exc
            continue

    print("\nALL auto-downloads failed.")
    if last_err is not None:
        print(f"Last error: {last_err}\n")
    print(manual_instructions(target))
    sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LaMa ONNX for manga inpainting.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    args = parser.parse_args()
    download_lama(force=args.force)


if __name__ == "__main__":
    main()
