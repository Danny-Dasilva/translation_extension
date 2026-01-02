#!/usr/bin/env python3
"""
Fine-tune YOLOv10n on manga speech bubble dataset.

Usage:
    uv run python train.py
    uv run python train.py --config configs/default.yaml
    uv run python train.py --epochs 100 --batch 8

Prerequisites:
    1. Download dataset: uv run python scripts/download_dataset.py
    2. Download model: uv run python scripts/download_model.py
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: uv sync")
    sys.exit(1)


def check_gpu():
    """Check GPU availability and VRAM."""
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Training will use CPU (very slow).")
        return "cpu"

    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s)")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)  # GB
        print(f"  GPU {i}: {props.name} ({total_memory:.1f} GB)")

        if total_memory < 4:
            print(f"  Warning: GPU {i} has low VRAM (<4GB). May need batch=4")
        elif total_memory < 6:
            print(f"  Warning: GPU {i} has moderate VRAM. May need batch=8")

    return 0  # Default to first GPU


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def validate_paths(config: dict, base_dir: Path):
    """Validate that required files exist."""
    # Check model weights
    model_path = base_dir / config.get("model", "weights/yolov10n.pt")
    if not model_path.exists():
        print(f"Error: Model weights not found: {model_path}")
        print("\nRun: uv run python scripts/download_model.py")
        sys.exit(1)

    # Check dataset
    data_path = base_dir / config.get("data", "datasets/data.yaml")
    if not data_path.exists():
        print(f"Error: Dataset config not found: {data_path}")
        print("\nRun: uv run python scripts/download_dataset.py")
        sys.exit(1)

    return model_path, data_path


def train(config_path: Path, overrides: dict = None):
    """Run training with given configuration."""
    base_dir = Path(__file__).parent

    # Load config
    config = load_config(config_path)

    # Apply command-line overrides
    if overrides:
        config.update(overrides)

    print("\n" + "=" * 60)
    print("YOLOv10n Fine-tuning for Manga Speech Bubble Detection")
    print("=" * 60)

    # Check GPU
    device = check_gpu()
    if "device" not in config:
        config["device"] = device

    # Validate paths
    model_path, data_path = validate_paths(config, base_dir)

    print(f"\nModel: {model_path}")
    print(f"Dataset: {data_path}")
    print(f"Device: {config['device']}")
    print(f"Epochs: {config.get('epochs', 50)}")
    print(f"Batch size: {config.get('batch', 16)}")
    print()

    # Load model
    print("Loading YOLOv10n model...")
    model = YOLO(str(model_path))

    # Prepare training arguments
    train_args = {
        "data": str(data_path),
        "epochs": config.get("epochs", 50),
        "batch": config.get("batch", 16),
        "imgsz": config.get("imgsz", 640),
        "patience": config.get("patience", 10),
        "save_period": config.get("save_period", 10),
        "device": config.get("device", 0),
        "project": str(base_dir / config.get("project", "runs")),
        "name": config.get("name", "manga-bubbles"),
        "exist_ok": True,
        # Optimizer
        "optimizer": config.get("optimizer", "AdamW"),
        "lr0": config.get("lr0", 0.02),
        "lrf": config.get("lrf", 0.01),
        "weight_decay": config.get("weight_decay", 0.0005),
        "warmup_epochs": config.get("warmup_epochs", 3),
        # Augmentation
        "hsv_h": config.get("hsv_h", 0.015),
        "hsv_s": config.get("hsv_s", 0.4),
        "hsv_v": config.get("hsv_v", 0.3),
        "degrees": config.get("degrees", 15),
        "translate": config.get("translate", 0.1),
        "scale": config.get("scale", 0.3),
        "perspective": config.get("perspective", 0.001),
        "flipud": config.get("flipud", 0.0),
        "fliplr": config.get("fliplr", 0.3),
        "mosaic": config.get("mosaic", 0.8),
        "mixup": config.get("mixup", 0.1),
        "copy_paste": config.get("copy_paste", 0.1),
        # Validation
        "val": config.get("val", True),
        "plots": config.get("plots", True),
    }

    print("Starting training...")
    print("-" * 60)

    # Train
    try:
        results = model.train(**train_args)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\nError: GPU out of memory!")
            print("Try reducing batch size:")
            print("  uv run python train.py --batch 8")
            print("  uv run python train.py --batch 4")
            sys.exit(1)
        raise

    print("-" * 60)
    print("Training complete!")

    # Copy best model to app/models
    best_model = base_dir / config.get("project", "runs") / config.get("name", "manga-bubbles") / "weights" / "best.pt"
    if best_model.exists():
        output_dir = base_dir.parent / "app" / "models"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "yolov10n_manga.pt"

        shutil.copy(best_model, output_path)
        print(f"\nBest model copied to: {output_path}")

    print("\nNext steps:")
    print("  1. Validate: uv run python validate.py")
    print("  2. Export: uv run python export.py")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv10n on manga speech bubbles"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path(__file__).parent / "configs" / "default.yaml",
        help="Training config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Override batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Override device (0, 1, cpu, etc.)",
    )
    args = parser.parse_args()

    # Build overrides from command-line arguments
    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch:
        overrides["batch"] = args.batch
    if args.device:
        overrides["device"] = args.device

    train(args.config, overrides)


if __name__ == "__main__":
    main()
