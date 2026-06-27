#!/usr/bin/env python
"""One-time vision-OCR pass over the human GT scanlation images.

PURPOSE (read this first)
-------------------------
The frozen ``data/ikenie4/gold.jsonl`` produced by ``build_ikenie4_gold.py`` is
seeded from the 24-agent LLM-judge comparison and therefore only covers the
~77 bubbles the judge flagged.  To grow the gold set to ~300 rows we need the
human English for the REST of the bubbles -- the ones the judge did not flag,
including the many *correct* ones (a regression set needs correct rows too, so a
change that breaks a currently-correct bubble is caught).

The only source of that human English is the typeset text baked into the GT
scanlation webp images.  This script is the ONE-TIME human_en recovery step: it
runs a vision model over each GT page, reads the rendered English in each
bubble, and aligns it to our OCR'd ``jp`` bubbles (by page + reading-order /
bbox overlap), then appends the new (jp, human_en) rows to the gold set.

This is a *one-time* pass: once the extended gold.jsonl is committed and
reviewed, it is frozen exactly like the judge-seeded rows.  You re-run this only
to re-derive the gold from scratch, not on every eval.

THE p41 OFFSET (baked in permanently)
-------------------------------------
Our bench pipeline emitted 134 pages (001..134).  The GT directory has 133 webp
(001..133).  The comparison reported ``missing_gt_page: 41`` -- GT page 41 does
not exist, so from bench page 41 onward the GT image index is shifted by +1.

    gt_webp(our_page) = our_page          if our_page < 41
                      = our_page - 1       if our_page >= 41

``resolve_gt_image_path()`` below encodes this so the vision pass reads the
CORRECT GT image for every bench bubble.  This is the single source of truth for
the offset; do not re-derive it ad hoc elsewhere.

RUNNABLE SHAPE
--------------
The script is runnable-shaped but the actual vision call is stubbed
(``_vision_transcribe_page`` raises NotImplementedError with a clear TODO).  You
do NOT need to run the vision model to use the rest of the harness; this is the
documented extension path.  When you do wire a model, fill in that one function
and the alignment in ``align_page``.

Usage (when the vision model is wired)
--------------------------------------
    PYTHONPATH=. python backend/scripts/eval/transcribe_gt_vision.py \
        --bubbles-root /home/danny/Documents/personal/extension/backend/.bench/ikenie4_final_insp \
        --gt-images-root "/mnt/nas/drive_2/onlyfans/external_content/nhentai/616137_Ikenie no Haha 4" \
        --existing-gold backend/scripts/eval/data/ikenie4/gold.jsonl \
        --out backend/scripts/eval/data/ikenie4/gold_vision_extended.jsonl \
        --pages 1-134
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Single source of truth for the offset, shared with the gold builder.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_ikenie4_gold import (  # noqa: E402
    MISSING_GT_PAGE,
    classify_ocr_clean,
    our_page_to_gt_page,
)


# ---------------------------------------------------------------------------
# GT image path resolution (p41 offset)
# ---------------------------------------------------------------------------


def resolve_gt_image_path(
    gt_images_root: Path,
    our_page: int,
    *,
    missing_gt_page: int = MISSING_GT_PAGE,
    ext: str = "webp",
    width: int = 3,
) -> Path | None:
    """Return the GT webp for a bench/our page, applying the p41 offset.

    Returns None if the resolved GT page would be the missing page or out of
    range on disk.
    """
    if our_page == missing_gt_page:
        # The bench page that has no GT counterpart at all.
        return None
    gt_page = our_page_to_gt_page(our_page, missing_gt_page=missing_gt_page)
    if gt_page < 1:
        return None
    p = gt_images_root / f"{gt_page:0{width}d}.{ext}"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Vision transcription (STUB)
# ---------------------------------------------------------------------------


@dataclass
class VisionBubble:
    """A single bubble of English text read off the GT image by the vision model."""

    text: str
    bbox: dict[str, int] | None = None  # {minX,minY,maxX,maxY} if the model returns one
    reading_order: int | None = None


def _vision_transcribe_page(image_path: Path) -> list[VisionBubble]:
    """Run the vision model on one GT image and return the English per bubble.

    STUB -- not implemented on purpose.  Wire your vision model here.  A working
    implementation should:

      1. Load ``image_path`` (a typeset English scanlation page).
      2. Prompt a vision-language model (e.g. Qwen2.5-VL / InternVL / a local
         served endpoint) to transcribe each speech bubble's English text in
         reading order, returning bbox if available.
      3. Return one VisionBubble per detected bubble.

    Keep the prompt deterministic (temperature 0, fixed seed) so the one-time
    pass is reproducible.
    """
    raise NotImplementedError(
        "Vision transcription is a stub. Wire a vision-language model here "
        "(transcribe each English bubble on the GT page). See module docstring."
    )


# ---------------------------------------------------------------------------
# Alignment: our JP bubbles <-> GT English bubbles
# ---------------------------------------------------------------------------


def _bbox_center(b: dict[str, Any]) -> tuple[float, float]:
    return ((b["minX"] + b["maxX"]) / 2.0, (b["minY"] + b["maxY"]) / 2.0)


def _bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0, ay0, ax1, ay1 = a["minX"], a["minY"], a["maxX"], a["maxY"]
    bx0, by0, bx1, by1 = b["minX"], b["minY"], b["maxX"], b["maxY"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter)


def align_page(
    our_bubbles: list[dict[str, Any]],
    gt_bubbles: list[VisionBubble],
) -> list[tuple[dict[str, Any], VisionBubble | None]]:
    """Pair each of our JP bubbles with a GT English bubble.

    Strategy (when GT bboxes are available): greedy max-IoU match.  Falls back
    to reading-order zip when bboxes are missing.  Unmatched our-bubbles get
    ``None`` (the human had no text there / vision missed it).
    """
    pairs: list[tuple[dict[str, Any], VisionBubble | None]] = []

    have_bbox = all(g.bbox is not None for g in gt_bubbles) and bool(gt_bubbles)
    if have_bbox:
        used: set[int] = set()
        for ob in our_bubbles:
            obx = ob.get("bbox")
            best_j, best_iou = None, 0.0
            if obx:
                for j, gb in enumerate(gt_bubbles):
                    if j in used:
                        continue
                    iou = _bbox_iou(obx, gb.bbox)  # type: ignore[arg-type]
                    if iou > best_iou:
                        best_iou, best_j = iou, j
            if best_j is not None and best_iou > 0.1:
                used.add(best_j)
                pairs.append((ob, gt_bubbles[best_j]))
            else:
                pairs.append((ob, None))
    else:
        # Reading-order zip fallback.
        for i, ob in enumerate(our_bubbles):
            gb = gt_bubbles[i] if i < len(gt_bubbles) else None
            pairs.append((ob, gb))
    return pairs


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _load_existing_jp_keys(existing_gold: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not existing_gold.exists():
        return keys
    for line in existing_gold.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        keys.add((r.get("src", ""), r.get("jp", "")))
    return keys


def _load_page_bubbles(bubbles_root: Path, page: int) -> list[dict[str, Any]]:
    p = bubbles_root / f"{page:03d}" / "bubbles.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _parse_pages(spec: str, default_max: int = 134) -> list[int]:
    if not spec:
        return list(range(1, default_max + 1))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_BUBBLES_ROOT = (
    "/home/danny/Documents/personal/extension/backend/.bench/ikenie4_final_insp"
)
DEFAULT_GT_ROOT = (
    "/mnt/nas/drive_2/onlyfans/external_content/nhentai/616137_Ikenie no Haha 4"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bubbles-root", default=DEFAULT_BUBBLES_ROOT)
    ap.add_argument("--gt-images-root", default=DEFAULT_GT_ROOT)
    ap.add_argument(
        "--existing-gold",
        default=str(SCRIPT_DIR / "data" / "ikenie4" / "gold.jsonl"),
        help="Judge-seeded gold; rows already present are skipped.",
    )
    ap.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "data" / "ikenie4" / "gold_vision_extended.jsonl"),
    )
    ap.add_argument("--pages", default="1-134", help="e.g. '1-134' or '1,2,5-9'")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve GT image paths + count bubbles WITHOUT calling the vision "
        "model (verifies the p41 offset wiring end-to-end).",
    )
    args = ap.parse_args(argv)

    bubbles_root = Path(args.bubbles_root)
    gt_root = Path(args.gt_images_root)
    existing_keys = _load_existing_jp_keys(Path(args.existing_gold))
    pages = _parse_pages(args.pages)

    new_rows: list[dict[str, Any]] = []
    resolved, missing = 0, 0

    for page in pages:
        our_bubbles = _load_page_bubbles(bubbles_root, page)
        if not our_bubbles:
            continue
        gt_path = resolve_gt_image_path(gt_root, page)
        if gt_path is None:
            missing += 1
            if page == MISSING_GT_PAGE:
                print(f"  p{page:03d}: no GT page (missing_gt_page) -- skipped")
            else:
                print(f"  p{page:03d}: GT image not found on disk -- skipped")
            continue
        resolved += 1

        if args.dry_run:
            print(
                f"  p{page:03d} -> GT {gt_path.name}  "
                f"(our_bubbles={len(our_bubbles)})"
            )
            continue

        # --- real path (requires the vision stub to be implemented) ---
        gt_bubbles = _vision_transcribe_page(gt_path)
        for ob, gb in align_page(our_bubbles, gt_bubbles):
            if gb is None or not gb.text.strip():
                continue
            idx = ob.get("idx")
            src = f"ikenie4:p{page:02d}:idx{idx}"
            if (src, ob.get("ocr_jp", "")) in existing_keys:
                continue
            jp = (ob.get("ocr_jp") or "").strip()
            if not jp:
                continue
            # No judge note for vision rows; default ocr_clean from category
            # heuristic (mistranslation-class) -> treat as clean unless the
            # bubble was gate-dropped / flagged garble by the pipeline.
            note = "vision-recovered; no judge note"
            new_rows.append(
                {
                    "jp": jp,
                    "en": gb.text.strip(),
                    "src": src,
                    "register_tag": "manga_nsfw",
                    "category": "vision_recovered",
                    "severity": 0,
                    "ocr_clean": classify_ocr_clean("mistranslation", note),
                    "ocr_conf": ob.get("ocr_conf"),
                    "bbox": ob.get("bbox"),
                    "our_en": (ob.get("translation_en") or "").strip(),
                    "source_field": "vision",
                    "judge_note": note,
                    "matched_bubble": True,
                }
            )

    print(f"\nGT image resolution: {resolved} resolved, {missing} skipped")
    if args.dry_run:
        print("dry-run: vision model NOT called; offset wiring verified above.")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in new_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(new_rows)} vision-recovered rows -> {out_path}")
    print("NOTE: review + merge into gold.jsonl, then FREEZE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
