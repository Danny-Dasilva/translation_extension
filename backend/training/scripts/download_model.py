#!/usr/bin/env python3
"""
Download YOLOv10n pretrained weights from Ultralytics.

Usage:
    uv run python scripts/download_model.py
    uv run python scripts/download_model.py --model yolov10n
    uv run python scripts/download_model.py --model yolov10s  # small variant

YOLOv10 is NMS-free, providing deterministic inference time regardless of
the number of detections - ideal for manga pages with many speech bubbles.
"""

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

# YOLOv10 model URLs from official releases
# https://github.com/THU-MIG/yolov10/releases
MODELS = {
    "yolov10n": {
        "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
        "size_mb": 5.8,
        "description": "Nano - fastest, lowest accuracy (~100MB VRAM)",
    },
    "yolov10s": {
        "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
        "size_mb": 16.0,
        "description": "Small - balanced speed/accuracy",
    },
    "yolov10m": {
        "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt",
        "size_mb": 32.0,
        "description": "Medium - higher accuracy, slower",
    },
    "yolov10b": {
        "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt",
        "size_mb": 40.0,
        "description": "Base - high accuracy",
    },
    "yolov10l": {
        "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt",
        "size_mb": 50.0,
        "description": "Large - very high accuracy, requires more VRAM",
    },
    "yolov10x": {
        "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt",
        "size_mb": 62.0,
        "description": "Extra-large - highest accuracy, most VRAM",
    },
}


def download_progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
    sys.stdout.flush()


def download_model(model_name: str, output_dir: Path) -> Path:
    """Download a YOLO model."""
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print("\nAvailable models:")
        for name, info in MODELS.items():
            print(f"  {name}: {info['description']} (~{info['size_mb']} MB)")
        sys.exit(1)

    model_info = MODELS[model_name]
    output_path = output_dir / f"{model_name}.pt"

    # Check if already downloaded
    if output_path.exists():
        print(f"Model already exists: {output_path}")
        response = input("Re-download? [y/N]: ").strip().lower()
        if response != "y":
            print("Using existing model.")
            return output_path

    print(f"\nDownloading {model_name}...")
    print(f"  {model_info['description']}")
    print(f"  URL: {model_info['url']}")
    print(f"  Size: ~{model_info['size_mb']} MB")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    try:
        urlretrieve(model_info["url"], output_path, reporthook=download_progress)
        print()  # newline after progress
    except Exception as e:
        print(f"\nError downloading model: {e}")
        sys.exit(1)

    # Verify file exists and has reasonable size
    if not output_path.exists():
        print("Error: Download failed - file not created")
        sys.exit(1)

    actual_size = output_path.stat().st_size / (1024 * 1024)
    expected_size = model_info["size_mb"]

    if actual_size < expected_size * 0.5:
        print(f"Warning: Downloaded file seems too small ({actual_size:.1f} MB)")
        print("The download may have failed. Try again.")

    print(f"\nModel saved to: {output_path}")
    print(f"File size: {actual_size:.1f} MB")

    return output_path


def list_models():
    """Print available models."""
    print("\nAvailable YOLOv10 models (NMS-free):\n")
    print("  Model      Size    Description")
    print("  " + "-" * 60)
    for name, info in MODELS.items():
        print(f"  {name:<10} {info['size_mb']:>5.1f}MB  {info['description']}")
    print()
    print("Recommended: yolov10n (fastest) or yolov10s (balanced)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download YOLOv10 pretrained weights"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov10n",
        help="Model variant (default: yolov10n)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parent.parent / "weights",
        help="Output directory (default: weights/)",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    output_path = download_model(args.model, args.output)

    print("\nDone! Next steps:")
    print("  1. Ensure dataset is downloaded: uv run python scripts/download_dataset.py")
    print("  2. Start training: uv run python train.py")


if __name__ == "__main__":
    main()
