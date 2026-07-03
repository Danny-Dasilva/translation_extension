#!/usr/bin/env python3
"""Build an INDEPENDENT detection-recall gold set from the AnimeText val split.

Why this exists (audit §5):
    Our POV / OCR gold sets are seeded from our OWN CTD detector output, so every
    gold bbox has IoU==1.0 against a detection by construction. A detector
    false-negative (a bubble the detector never proposes) is therefore invisible
    to every existing harness. To measure *recall* we need ground-truth bubble
    boxes drawn INDEPENDENTLY of our detector.

Source (independent, NOT detector-seeded):
    training/comic-text-detector/data/yolo_blocks/val/ — the block-detection
    split converted by scripts/prepare_animetext.py from the AnimeText dataset
    on HuggingFace. Labels are the dataset authors' human/dataset bbox
    annotations (class 0 = text_block), in YOLO normalized
    [x_center, y_center, w, h] form. They are the *training/eval labels* for the
    CTD model, produced upstream of and independently of our production
    ComicTextDetectorService — exactly the independence property the recall
    measurement requires.

This script selects a small, diverse subset (~20 pages) spread across bubble-count
buckets and bakes each page's boxes (in pixel xyxy) + image dimensions into a
self-contained gold JSON. Images themselves are gitignored and stay on disk; the
harness reads them at run time via --images-dir.

Usage:
    python scripts/eval/build_recall_gold.py \
        --src /path/to/training/comic-text-detector/data/yolo_blocks/val \
        --out scripts/eval/detection_recall_gold.json \
        --n 20
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, iw: int, ih: int) -> list[int]:
    x1 = int(round((cx - w / 2) * iw))
    y1 = int(round((cy - h / 2) * ih))
    x2 = int(round((cx + w / 2) * iw))
    y2 = int(round((cy + h / 2) * ih))
    x1 = max(0, min(x1, iw - 1))
    y1 = max(0, min(y1, ih - 1))
    x2 = max(0, min(x2, iw))
    y2 = max(0, min(y2, ih))
    return [x1, y1, x2, y2]


def parse_label(txt_path: Path, iw: int, ih: int) -> list[list[int]]:
    boxes: list[list[int]] = []
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _cls, cx, cy, w, h = parts[:5]
        box = yolo_to_xyxy(float(cx), float(cy), float(w), float(h), iw, ih)
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="yolo_blocks/val dir (images/ + labels/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    src = Path(args.src)
    labels_dir = src / "labels"
    images_dir = src / "images"
    all_labels = sorted(labels_dir.glob("*.txt"))

    # Pull box counts for a sample, then bucket for diversity (sparse..dense).
    rng = random.Random(args.seed)
    pool = all_labels if len(all_labels) < 2000 else rng.sample(all_labels, 2000)

    scored: list[tuple[int, Path]] = []
    for lp in pool:
        img = images_dir / (lp.stem + ".jpg")
        if not img.exists():
            continue
        n = sum(1 for line in lp.read_text().splitlines() if len(line.split()) >= 5)
        if n == 0:
            continue
        scored.append((n, lp))

    scored.sort(key=lambda t: t[0])
    # 4 buckets by density; pick evenly.
    picks: list[Path] = []
    if scored:
        buckets = 4
        per = max(1, args.n // buckets)
        size = max(1, len(scored) // buckets)
        for b in range(buckets):
            chunk = scored[b * size : (b + 1) * size] if b < buckets - 1 else scored[b * size :]
            if not chunk:
                continue
            for _n, lp in rng.sample(chunk, min(per, len(chunk))):
                picks.append(lp)
    picks = picks[: args.n]

    pages = []
    for lp in picks:
        img_path = images_dir / (lp.stem + ".jpg")
        with Image.open(img_path) as im:
            iw, ih = im.size
        boxes = parse_label(lp, iw, ih)
        if not boxes:
            continue
        pages.append(
            {
                "image": img_path.name,
                "width": iw,
                "height": ih,
                "num_boxes": len(boxes),
                "boxes_xyxy": boxes,
            }
        )

    gold = {
        "schema": "detection_recall_gold/v1",
        "provenance": {
            "source": "AnimeText (HuggingFace) via prepare_animetext.py -> yolo_blocks/val",
            "independent_of_our_detector": True,
            "label_semantics": "class 0 = text_block; YOLO norm cxcywh -> pixel xyxy",
            "note": (
                "Boxes are the dataset's own annotations, NOT produced by "
                "ComicTextDetectorService. Safe for recall/false-negative measurement."
            ),
            "src_dir": str(src),
        },
        "pages": pages,
    }
    out = Path(args.out)
    out.write_text(json.dumps(gold, indent=2))
    total = sum(p["num_boxes"] for p in pages)
    print(f"Wrote {len(pages)} pages / {total} gold boxes -> {out}")


if __name__ == "__main__":
    main()
