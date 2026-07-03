#!/usr/bin/env python3
"""Detection-recall harness against an INDEPENDENT (non-detector-seeded) gold set.

Audit §5 context:
    Recall is currently unmeasurable because every POV/OCR gold bbox is seeded
    from our own CTD output (IoU==1.0 by construction), so a false-negative (a
    bubble the detector never proposes) is invisible. This harness scores CTD
    against gold boxes drawn independently of our detector (see
    build_recall_gold.py / detection_recall_gold.json), so misses are counted.

What it reports, per IoU threshold (default 0.5 and 0.75):
    recall     = matched_gold / total_gold          (1 - miss rate)
    precision  = matched_det  / total_det
    fn         = gold boxes with no detector match   (the previously-invisible misses)
    fp         = detections matching no gold box

Two layers, deliberately separated so the math is testable without a GPU:
    * Pure scoring functions (iou, match_boxes, score_page, aggregate) — depend
      only on stdlib; unit-tested on a synthetic fixture.
    * run() — loads images + invokes ComicTextDetectorService (needs onnxruntime
      + the CTD model + ideally a GPU). Imported lazily so `import
      detection_recall_eval` never drags in the detector stack.

Usage (real run, needs model + deps):
    cd backend && uv run python scripts/eval/detection_recall_eval.py \
        --gold scripts/eval/detection_recall_gold.json \
        --images-dir /path/to/yolo_blocks/val/images
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

Box = Sequence[float]  # [x1, y1, x2, y2]


# --------------------------------------------------------------------------- #
# Pure scoring math (no detector / no heavy deps) — unit-tested.
# --------------------------------------------------------------------------- #
def iou(a: Box, b: Box) -> float:
    """IoU of two axis-aligned boxes in xyxy pixel coords."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    threshold: float
    matched: list[tuple[int, int, float]]  # (gold_idx, det_idx, iou)
    false_negatives: list[int]             # gold indices with no match
    false_positives: list[int]             # det indices with no match
    n_gold: int
    n_det: int

    @property
    def recall(self) -> float:
        return len(self.matched) / self.n_gold if self.n_gold else 1.0

    @property
    def precision(self) -> float:
        return len(self.matched) / self.n_det if self.n_det else 1.0


def match_boxes(gold: list[Box], det: list[Box], threshold: float) -> MatchResult:
    """Greedy 1-1 matching by descending IoU above `threshold`.

    Each gold box matches at most one detection and vice-versa. Unmatched gold
    boxes are false-negatives (detector misses); unmatched detections are
    false-positives.
    """
    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gold):
        for di, d in enumerate(det):
            v = iou(g, d)
            if v >= threshold:
                pairs.append((v, gi, di))
    pairs.sort(reverse=True)  # highest IoU first

    used_gold: set[int] = set()
    used_det: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for v, gi, di in pairs:
        if gi in used_gold or di in used_det:
            continue
        used_gold.add(gi)
        used_det.add(di)
        matched.append((gi, di, v))

    fns = [gi for gi in range(len(gold)) if gi not in used_gold]
    fps = [di for di in range(len(det)) if di not in used_det]
    return MatchResult(threshold, matched, fns, fps, len(gold), len(det))


@dataclass
class PageScore:
    image: str
    n_gold: int
    n_det: int
    by_threshold: dict[float, MatchResult] = field(default_factory=dict)


def score_page(image: str, gold: list[Box], det: list[Box], thresholds: Sequence[float]) -> PageScore:
    ps = PageScore(image=image, n_gold=len(gold), n_det=len(det))
    for t in thresholds:
        ps.by_threshold[t] = match_boxes(gold, det, t)
    return ps


def aggregate(pages: list[PageScore], thresholds: Sequence[float]) -> dict:
    out: dict = {"n_pages": len(pages), "thresholds": {}}
    for t in thresholds:
        tot_gold = sum(p.by_threshold[t].n_gold for p in pages)
        tot_det = sum(p.by_threshold[t].n_det for p in pages)
        tot_match = sum(len(p.by_threshold[t].matched) for p in pages)
        tot_fn = sum(len(p.by_threshold[t].false_negatives) for p in pages)
        tot_fp = sum(len(p.by_threshold[t].false_positives) for p in pages)
        out["thresholds"][t] = {
            "recall": tot_match / tot_gold if tot_gold else 1.0,
            "precision": tot_match / tot_det if tot_det else 1.0,
            "gold": tot_gold,
            "det": tot_det,
            "matched": tot_match,
            "false_negatives": tot_fn,
            "false_positives": tot_fp,
        }
    return out


# --------------------------------------------------------------------------- #
# Detector-backed run (needs onnxruntime + CTD model; lazy import).
# --------------------------------------------------------------------------- #
def _detect_boxes(image_path: Path, service) -> list[list[int]]:
    """Run CTD on one image, return block boxes as pixel xyxy.

    Mirrors the production invocation: cv2.imread (BGR) -> detect(input_is_bgr=True).
    Uses the CTD `blocks` output (bubble-level text regions), which is what the
    gold set annotates.
    """
    import asyncio

    import cv2  # type: ignore

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    result = asyncio.run(service.detect(img, input_is_bgr=True))
    boxes = []
    for b in result.get("blocks", []):
        boxes.append([b["minX"], b["minY"], b["maxX"], b["maxY"]])
    return boxes


def run(gold_path: Path, images_dir: Path, thresholds: Sequence[float], model_path: str | None = None) -> dict:
    """Full harness: load gold, run detector on each page, score. Needs GPU/model."""
    from app.services.ctd_service import ComicTextDetectorService  # lazy

    gold = json.loads(gold_path.read_text())
    service = ComicTextDetectorService(model_path)

    pages: list[PageScore] = []
    per_page_report = []
    for entry in gold["pages"]:
        img_path = images_dir / entry["image"]
        det = _detect_boxes(img_path, service)
        gboxes = entry["boxes_xyxy"]
        ps = score_page(entry["image"], gboxes, det, thresholds)
        pages.append(ps)
        per_page_report.append(
            {
                "image": entry["image"],
                "gold": ps.n_gold,
                "det": ps.n_det,
                "fn@0.5": len(ps.by_threshold[thresholds[0]].false_negatives),
            }
        )

    summary = aggregate(pages, thresholds)
    summary["provenance"] = gold.get("provenance", {})
    summary["per_page"] = per_page_report
    return summary


def _print_summary(summary: dict) -> None:
    print(f"\nDetection recall over {summary['n_pages']} independent pages")
    print("=" * 60)
    for t, s in summary["thresholds"].items():
        print(
            f"IoU>={t}:  recall={s['recall']:.3f}  precision={s['precision']:.3f}  "
            f"FN={s['false_negatives']}  FP={s['false_positives']}  "
            f"(gold={s['gold']} det={s['det']})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default=str(Path(__file__).with_name("detection_recall_gold.json")))
    ap.add_argument("--images-dir", required=True, help="dir containing the gold page images")
    ap.add_argument("--model-path", default=None, help="override CTD model path")
    ap.add_argument("--thresholds", default="0.5,0.75")
    ap.add_argument("--out", default=None, help="optional path to write JSON summary")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    summary = run(Path(args.gold), Path(args.images_dir), thresholds, args.model_path)
    _print_summary(summary)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nWrote summary -> {args.out}")


if __name__ == "__main__":
    main()
