#!/usr/bin/env python3
"""
Export trained YOLOv10n model to ONNX format for production deployment.

Usage:
    uv run python export.py
    uv run python export.py --model runs/manga-bubbles/weights/best.pt
    uv run python export.py --format onnx --simplify

Supported formats:
- onnx: ONNX format (recommended for production)
- torchscript: TorchScript format
- engine: TensorRT format (NVIDIA GPUs)
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: uv sync")
    sys.exit(1)


def find_best_model(base_dir: Path) -> Path:
    """Find the best trained model."""
    candidates = [
        base_dir / "runs" / "manga-bubbles" / "weights" / "best.pt",
        base_dir.parent / "app" / "models" / "yolov10n_manga.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def export_model(
    model_path: Path,
    output_dir: Path,
    format: str = "onnx",
    simplify: bool = True,
    half: bool = False,
    imgsz: int = 640,
):
    """Export model to specified format."""
    print("\n" + "=" * 60)
    print("YOLOv10n Model Export")
    print("=" * 60)

    print(f"\nInput model: {model_path}")
    print(f"Output format: {format}")
    print(f"Image size: {imgsz}")
    print(f"Simplify: {simplify}")
    print(f"Half precision (FP16): {half}")
    print()

    # Load model
    print("Loading model...")
    model = YOLO(str(model_path))

    # Export
    print(f"Exporting to {format}...")
    print("-" * 60)

    export_args = {
        "format": format,
        "imgsz": imgsz,
        "half": half,
    }

    if format == "onnx":
        export_args["simplify"] = simplify
        export_args["opset"] = 13

    exported_path = model.export(**export_args)

    print("-" * 60)
    print(f"\nExported model: {exported_path}")

    # Copy to app/models
    if exported_path:
        exported_path = Path(exported_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine output filename
        suffix = exported_path.suffix
        output_path = output_dir / f"yolov10n_manga{suffix}"

        shutil.copy(exported_path, output_path)
        print(f"Copied to: {output_path}")

        # Get file sizes
        pt_size = model_path.stat().st_size / (1024 * 1024)
        export_size = output_path.stat().st_size / (1024 * 1024)

        print(f"\nFile sizes:")
        print(f"  PyTorch (.pt): {pt_size:.1f} MB")
        print(f"  {format.upper()}: {export_size:.1f} MB")

    # Verify export
    print("\nVerifying exported model...")
    try:
        test_model = YOLO(str(output_path))
        # Run a quick inference test
        import numpy as np

        test_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        results = test_model.predict(test_img, verbose=False)
        print("Verification: PASSED")
    except Exception as e:
        print(f"Verification: FAILED - {e}")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)

    print("\nUsage in production:")
    if format == "onnx":
        print("""
    import onnxruntime as ort

    session = ort.InferenceSession("yolov10n_manga.onnx")
    # ... run inference
        """)
    else:
        print(f"""
    from ultralytics import YOLO

    model = YOLO("yolov10n_manga{suffix}")
    results = model.predict(image)
        """)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export trained YOLOv10n model"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        help="Path to model weights (default: auto-detect best model)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: ../app/models/)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "engine", "coreml"],
        help="Export format (default: onnx)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify ONNX model (default: True)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_false",
        dest="simplify",
        help="Don't simplify ONNX model",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 half precision",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for export (default: 640)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # Find model
    model_path = args.model
    if not model_path:
        model_path = find_best_model(base_dir)
        if not model_path:
            print("Error: No trained model found.")
            print("\nTrain a model first:")
            print("  uv run python train.py")
            sys.exit(1)

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Set output directory
    output_dir = args.output
    if not output_dir:
        output_dir = base_dir.parent / "app" / "models"

    export_model(
        model_path=model_path,
        output_dir=output_dir,
        format=args.format,
        simplify=args.simplify,
        half=args.half,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
