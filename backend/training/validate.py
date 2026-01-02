#!/usr/bin/env python3
"""
Validate trained YOLOv10n model on test/validation set.

Usage:
    uv run python validate.py
    uv run python validate.py --model runs/manga-bubbles/weights/best.pt
    uv run python validate.py --model ../app/models/yolov10n_manga.pt

Reports:
- mAP@50, mAP@50-95
- Precision, Recall
- Confusion matrix
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: uv sync")
    sys.exit(1)


def find_best_model(base_dir: Path) -> Path:
    """Find the best trained model."""
    # Check common locations
    candidates = [
        base_dir / "runs" / "manga-bubbles" / "weights" / "best.pt",
        base_dir.parent / "app" / "models" / "yolov10n_manga.pt",
        base_dir / "weights" / "yolov10n.pt",  # Fallback to pretrained
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def find_dataset(base_dir: Path) -> Path:
    """Find the dataset config."""
    data_yaml = base_dir / "datasets" / "data.yaml"
    if data_yaml.exists():
        return data_yaml

    # Try to find any data.yaml in datasets dir
    datasets_dir = base_dir / "datasets"
    if datasets_dir.exists():
        for yaml_file in datasets_dir.glob("**/data.yaml"):
            return yaml_file

    return None


def validate(model_path: Path, data_path: Path, split: str = "val"):
    """Run validation on the model."""
    print("\n" + "=" * 60)
    print("YOLOv10n Model Validation")
    print("=" * 60)

    print(f"\nModel: {model_path}")
    print(f"Dataset: {data_path}")
    print(f"Split: {split}")
    print()

    # Load model
    print("Loading model...")
    model = YOLO(str(model_path))

    # Run validation
    print(f"Running validation on {split} set...")
    print("-" * 60)

    results = model.val(
        data=str(data_path),
        split=split,
        plots=True,
        save_json=True,
    )

    print("-" * 60)
    print("\nRESULTS:")
    print("=" * 60)

    # Extract metrics
    metrics = results.results_dict

    print(f"\nPrecision:     {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall:        {metrics.get('metrics/recall(B)', 0):.4f}")
    print(f"mAP@50:        {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP@50-95:     {metrics.get('metrics/mAP50-95(B)', 0):.4f}")

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION:")

    map50 = metrics.get("metrics/mAP50(B)", 0)
    if map50 >= 0.8:
        print("  Excellent! Model is ready for production.")
    elif map50 >= 0.6:
        print("  Good. Model should work well for most manga pages.")
    elif map50 >= 0.4:
        print("  Fair. Consider training longer or with more data.")
    else:
        print("  Poor. Model needs more training or better data.")

    precision = metrics.get("metrics/precision(B)", 0)
    recall = metrics.get("metrics/recall(B)", 0)

    if precision > recall + 0.2:
        print("  High precision, low recall: Model is conservative.")
        print("  May miss some speech bubbles.")
    elif recall > precision + 0.2:
        print("  High recall, low precision: Model is aggressive.")
        print("  May detect non-bubbles as bubbles.")

    print()
    print("Plots saved to validation output directory.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate trained YOLOv10n model"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        help="Path to model weights (default: auto-detect best model)",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        help="Path to data.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to validate on (default: val)",
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

    # Find dataset
    data_path = args.data
    if not data_path:
        data_path = find_dataset(base_dir)
        if not data_path:
            print("Error: No dataset found.")
            print("\nDownload dataset first:")
            print("  uv run python scripts/download_dataset.py")
            sys.exit(1)

    if not data_path.exists():
        print(f"Error: Dataset config not found: {data_path}")
        sys.exit(1)

    validate(model_path, data_path, args.split)


if __name__ == "__main__":
    main()
