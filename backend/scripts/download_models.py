#!/usr/bin/env python3
"""
Download models for the local AI manga translation pipeline.

Usage:
    uv run python scripts/download_models.py [--all|--ocr|--translation]

Models:
    - PaddleOCR-VL-For-Manga (~2GB) - Vision-Language OCR for manga
    - HY-MT1.5-1.8B-GGUF (~2GB) - Quantized translation model

Storage:
    Models are saved to backend/app/weights/
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


# Model configurations
MODELS = {
    "ocr": {
        "repo_id": "jzhang533/PaddleOCR-VL-For-Manga",
        "description": "PaddleOCR-VL-For-Manga (Vision-Language OCR)",
        "type": "snapshot",  # Download entire repo
        "size": "~2GB",
    },
    "translation": {
        "repo_id": "Hongyao-Yu/HY-MT1.5-1.8B-GGUF",
        "filename": "HY-MT1.5-1.8B-Q8_0.gguf",
        "description": "HY-MT1.5 (Q8_0 quantized translation model)",
        "type": "file",  # Download single file
        "size": "~2GB",
    },
}


def get_weights_dir() -> Path:
    """Get the weights directory path."""
    # Script is in backend/scripts/, weights go in backend/app/weights/
    script_dir = Path(__file__).parent
    weights_dir = script_dir.parent / "app" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def download_ocr_model(weights_dir: Path) -> None:
    """Download PaddleOCR-VL-For-Manga model."""
    config = MODELS["ocr"]
    print(f"\n{'='*60}")
    print(f"Downloading: {config['description']}")
    print(f"Size: {config['size']}")
    print(f"{'='*60}\n")

    target_dir = weights_dir / "paddleocr-vl"

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Model already exists at {target_dir}")
        response = input("Re-download? [y/N]: ").strip().lower()
        if response != 'y':
            print("Skipping OCR model download.")
            return

    print(f"Downloading to: {target_dir}")
    snapshot_download(
        repo_id=config["repo_id"],
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    print(f"✓ OCR model downloaded to {target_dir}")


def download_translation_model(weights_dir: Path) -> None:
    """Download HY-MT1.5 GGUF model."""
    config = MODELS["translation"]
    print(f"\n{'='*60}")
    print(f"Downloading: {config['description']}")
    print(f"Size: {config['size']}")
    print(f"{'='*60}\n")

    target_path = weights_dir / config["filename"]

    if target_path.exists():
        print(f"Model already exists at {target_path}")
        response = input("Re-download? [y/N]: ").strip().lower()
        if response != 'y':
            print("Skipping translation model download.")
            return

    print(f"Downloading to: {target_path}")
    hf_hub_download(
        repo_id=config["repo_id"],
        filename=config["filename"],
        local_dir=str(weights_dir),
        local_dir_use_symlinks=False,
    )
    print(f"✓ Translation model downloaded to {target_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download models for local AI manga translation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all models
    uv run python scripts/download_models.py --all

    # Download only OCR model
    uv run python scripts/download_models.py --ocr

    # Download only translation model
    uv run python scripts/download_models.py --translation
        """
    )

    parser.add_argument(
        "--all", action="store_true",
        help="Download all models (OCR + Translation)"
    )
    parser.add_argument(
        "--ocr", action="store_true",
        help="Download PaddleOCR-VL-For-Manga model"
    )
    parser.add_argument(
        "--translation", action="store_true",
        help="Download HY-MT1.5 GGUF translation model"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models and their sizes"
    )

    args = parser.parse_args()

    # If --list, show models and exit
    if args.list:
        print("\nAvailable models:")
        print("-" * 60)
        for name, config in MODELS.items():
            print(f"  --{name}")
            print(f"      {config['description']}")
            print(f"      Repo: {config['repo_id']}")
            print(f"      Size: {config['size']}")
            print()
        return

    # Default to --all if no specific model selected
    if not (args.all or args.ocr or args.translation):
        print("No model specified. Use --all to download all models.")
        print("Run with --help for more options.")
        sys.exit(1)

    weights_dir = get_weights_dir()
    print(f"Weights directory: {weights_dir}")

    download_ocr = args.all or args.ocr
    download_translation = args.all or args.translation

    try:
        if download_ocr:
            download_ocr_model(weights_dir)

        if download_translation:
            download_translation_model(weights_dir)

        print("\n" + "="*60)
        print("✓ Download complete!")
        print("="*60)

        # Show what was downloaded
        print("\nDownloaded models:")
        for item in weights_dir.iterdir():
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                print(f"  📁 {item.name}/ ({size / 1e9:.2f} GB)")
            else:
                size = item.stat().st_size
                print(f"  📄 {item.name} ({size / 1e9:.2f} GB)")

    except KeyboardInterrupt:
        print("\n\nDownload cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
