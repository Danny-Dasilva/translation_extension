#!/usr/bin/env python3
"""
Validate downloaded dataset quality and provide recommendations.

Usage:
    uv run python scripts/validate_dataset.py
    uv run python scripts/validate_dataset.py --dataset datasets/

Checks:
- Total instance count (warns if < 10,000)
- Class distribution
- Image dimensions
- Provides augmentation recommendations
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml


def count_instances(labels_dir: Path) -> tuple[int, Counter]:
    """Count total instances and class distribution in label files."""
    total_instances = 0
    class_counts = Counter()

    if not labels_dir.exists():
        return 0, class_counts

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_instances += 1

    return total_instances, class_counts


def count_images(images_dir: Path) -> int:
    """Count images in directory."""
    if not images_dir.exists():
        return 0

    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        count += len(list(images_dir.glob(ext)))
    return count


def validate_dataset(dataset_dir: Path):
    """Validate a YOLO-format dataset."""
    print(f"\nValidating dataset: {dataset_dir}\n")

    # Check for data.yaml
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        # Try common variations
        for alt in ["data.yml", "dataset.yaml", "dataset.yml"]:
            alt_path = dataset_dir / alt
            if alt_path.exists():
                data_yaml = alt_path
                break

    if not data_yaml.exists():
        print("Error: data.yaml not found")
        print("Make sure you downloaded the dataset in YOLO format")
        sys.exit(1)

    # Parse data.yaml
    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    print(f"Dataset config: {data_yaml}")
    print(f"Classes: {config.get('names', config.get('nc', 'unknown'))}")
    print()

    # Count instances in each split
    splits = ["train", "valid", "test"]
    total_instances = 0
    total_images = 0
    split_stats = {}

    for split in splits:
        # Try different path formats
        labels_dir = None
        images_dir = None

        for path_format in [
            dataset_dir / split / "labels",
            dataset_dir / f"{split}/labels",
            dataset_dir / "labels" / split,
        ]:
            if path_format.exists():
                labels_dir = path_format
                break

        for path_format in [
            dataset_dir / split / "images",
            dataset_dir / f"{split}/images",
            dataset_dir / "images" / split,
        ]:
            if path_format.exists():
                images_dir = path_format
                break

        if labels_dir:
            instances, class_counts = count_instances(labels_dir)
            images = count_images(images_dir) if images_dir else 0

            split_stats[split] = {
                "instances": instances,
                "images": images,
                "class_counts": class_counts,
            }
            total_instances += instances
            total_images += images

            print(f"{split.capitalize()}:")
            print(f"  Images: {images}")
            print(f"  Instances: {instances}")
            if images > 0:
                print(f"  Avg instances/image: {instances / images:.1f}")
            print()

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total images: {total_images}")
    print(f"Total instances: {total_instances}")
    print()

    # Warnings and recommendations
    warnings = []
    recommendations = []

    if total_instances < 10000:
        warnings.append(
            f"Low instance count ({total_instances} < 10,000 recommended)"
        )
        recommendations.append("Use aggressive augmentation (mosaic=0.8, mixup=0.1)")
        recommendations.append("Consider combining multiple datasets")
        recommendations.append("Use early stopping with patience=10")

    if total_instances < 5000:
        warnings.append("Very low instance count - high overfitting risk")
        recommendations.append("Consider using a larger model variant for transfer learning")
        recommendations.append("Use copy_paste augmentation (copy_paste=0.1)")

    if total_images < 1000:
        warnings.append(f"Low image count ({total_images} < 1,000)")
        recommendations.append("Increase epochs to 100")

    # Check class balance
    all_class_counts = Counter()
    for stats in split_stats.values():
        all_class_counts.update(stats["class_counts"])

    if len(all_class_counts) > 1:
        counts = list(all_class_counts.values())
        if max(counts) > 5 * min(counts):
            warnings.append("Significant class imbalance detected")
            recommendations.append("Use class weights or focal loss")

    # Print warnings
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
        print()

    # Print recommendations
    if recommendations:
        print("RECOMMENDATIONS:")
        for r in recommendations:
            print(f"  - {r}")
        print()

    # Suggested config values
    print("SUGGESTED CONFIG VALUES:")
    if total_instances < 5000:
        print("  mosaic: 0.9")
        print("  mixup: 0.15")
        print("  copy_paste: 0.15")
        print("  epochs: 100")
    elif total_instances < 10000:
        print("  mosaic: 0.8")
        print("  mixup: 0.1")
        print("  copy_paste: 0.1")
        print("  epochs: 50")
    else:
        print("  mosaic: 0.5")
        print("  mixup: 0.0")
        print("  epochs: 30")

    print()

    # Final status
    if total_instances >= 10000:
        print("STATUS: Dataset looks good for training")
    elif total_instances >= 5000:
        print("STATUS: Dataset is usable with recommended augmentations")
    else:
        print("STATUS: Dataset is small - expect some overfitting")

    return total_instances, total_images


def main():
    parser = argparse.ArgumentParser(
        description="Validate YOLO dataset quality"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=Path(__file__).parent.parent / "datasets",
        help="Dataset directory (default: datasets/)",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset directory not found: {args.dataset}")
        print("\nRun download script first:")
        print("  uv run python scripts/download_dataset.py")
        sys.exit(1)

    validate_dataset(args.dataset)


if __name__ == "__main__":
    main()
