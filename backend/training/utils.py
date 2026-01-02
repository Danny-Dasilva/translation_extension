"""Shared utilities for training scripts."""
from pathlib import Path
from typing import Optional, Dict, Any
import argparse


def find_best_model(base_dir: Path) -> Optional[Path]:
    """Find trained or default model across common locations."""
    candidates = [
        base_dir / "runs" / "manga-bubbles" / "weights" / "best.pt",
        base_dir.parent / "app" / "models" / "yolov10n_manga.pt",
        base_dir / "weights" / "yolov10n.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_dataset(base_dir: Path) -> Optional[Path]:
    """Find dataset config yaml."""
    candidates = [
        base_dir / "datasets" / "data.yaml",
        base_dir / "datasets" / "manga-bubble-pqdou" / "data.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Fallback to glob search
    datasets_dir = base_dir / "datasets"
    if datasets_dir.exists():
        for yaml_file in datasets_dir.glob("**/data.yaml"):
            return yaml_file
    return None


class TaskFormatter:
    """Utilities for consistent CLI output formatting."""

    @staticmethod
    def header(title: str) -> None:
        print(f"\n{'='*60}")
        print(title)
        print('='*60 + "\n")

    @staticmethod
    def section(title: str) -> None:
        print(f"\n{title}")
        print("-" * 60)

    @staticmethod
    def config(config: Dict[str, Any], indent: int = 2) -> None:
        for key, value in config.items():
            print(f"{' '*indent}{key}: {value}")

    @staticmethod
    def complete(message: str = "Complete!") -> None:
        print("-" * 60)
        print(f"{message}\n")
