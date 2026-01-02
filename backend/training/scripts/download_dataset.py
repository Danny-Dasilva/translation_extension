#!/usr/bin/env python3
"""
Download manga speech bubble datasets from Roboflow.

Usage:
    uv run python scripts/download_dataset.py
    uv run python scripts/download_dataset.py --dataset manga-bubble-detect
    uv run python scripts/download_dataset.py --list

Requires ROBOFLOW_API_KEY environment variable.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    print("Error: roboflow not installed. Run: uv sync")
    sys.exit(1)


# Known manga speech bubble datasets on Roboflow
KNOWN_DATASETS = {
    "manga-bubble-detect": {
        "workspace": "thesis-8cvtj",
        "project": "manga-bubble-detect",
        "version": 1,
        "description": "4,492 images - largest manga bubble dataset",
    },
    "speech-bubble-detector": {
        "workspace": "manga-translator",
        "project": "speech-bubble-detector",
        "version": 1,
        "description": "Speech bubble detection for manga translation",
    },
    "comic-text-segmentation": {
        "workspace": "roboflow-100",
        "project": "comic-text-segmentation",
        "version": 2,
        "description": "Comic text and bubble segmentation",
    },
}


def list_datasets():
    """Print available datasets."""
    print("\nAvailable manga speech bubble datasets:\n")
    for name, info in KNOWN_DATASETS.items():
        print(f"  {name}")
        print(f"    {info['description']}")
        print(f"    Workspace: {info['workspace']}/{info['project']} v{info['version']}")
        print()


def download_dataset(api_key: str, dataset_name: str, output_dir: Path):
    """Download a dataset from Roboflow."""
    if dataset_name not in KNOWN_DATASETS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print("Use --list to see available datasets")
        sys.exit(1)

    info = KNOWN_DATASETS[dataset_name]
    print(f"\nDownloading {dataset_name}...")
    print(f"  {info['description']}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(info["workspace"]).project(info["project"])
    version = project.version(info["version"])

    # Download in YOLOv8 format (compatible with YOLOv10)
    dataset = version.download("yolov8", location=str(output_dir))

    print(f"\nDataset downloaded to: {output_dir}")
    print(f"Data YAML: {output_dir / 'data.yaml'}")

    return dataset


def interactive_select(api_key: str, output_dir: Path):
    """Interactive dataset selection."""
    print("\nSelect a dataset to download:\n")
    datasets = list(KNOWN_DATASETS.items())

    for i, (name, info) in enumerate(datasets, 1):
        print(f"  [{i}] {name}")
        print(f"      {info['description']}")
        print()

    while True:
        try:
            choice = input("Enter choice (1-{}): ".format(len(datasets)))
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                dataset_name = datasets[idx][0]
                break
            print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    return download_dataset(api_key, dataset_name, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download manga speech bubble datasets from Roboflow"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Dataset name to download (see --list for options)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parent.parent / "datasets",
        help="Output directory (default: datasets/)",
    )
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    # Check API key
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY environment variable not set")
        print("\nTo get an API key:")
        print("  1. Go to https://app.roboflow.com/")
        print("  2. Sign up or log in")
        print("  3. Go to Settings > API Keys")
        print("  4. Copy your Private API Key")
        print("\nThen run:")
        print('  export ROBOFLOW_API_KEY="your_key_here"')
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        download_dataset(api_key, args.dataset, args.output)
    else:
        interactive_select(api_key, args.output)

    print("\nDone! Next steps:")
    print("  1. Run: uv run python scripts/validate_dataset.py")
    print("  2. Run: uv run python scripts/download_model.py")


if __name__ == "__main__":
    main()
