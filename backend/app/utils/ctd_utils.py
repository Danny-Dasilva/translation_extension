"""Helpers for mapping CTD outputs to text regions."""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np


def match_blocks_to_bubbles(
    blocks: List[Dict],
    bubbles: List[Dict],
    min_expand: float = 1.15,
) -> List[Optional[Dict]]:
    """For each text block, return the speech-bubble rect it sits inside (so
    translated text can be typeset to the *bubble* interior instead of the
    tight text column), or None.

    A bubble is only returned when it genuinely contains the block's center AND
    is meaningfully larger than the block (area ≥ ``min_expand``×) — otherwise
    fitting to it buys nothing. Among qualifying bubbles, the smallest is chosen
    (tightest enclosure), so a block lands in its own bubble rather than a big
    panel-spanning detection. Blocks with no bubble (SFX over art) get None and
    the caller should fall back to the block's own bbox.
    """
    # Pass 1: best (smallest qualifying) bubble per block.
    cand: List[Optional[int]] = []
    areas: List[float] = []
    for b in blocks:
        bx0, by0, bx1, by1 = b["minX"], b["minY"], b["maxX"], b["maxY"]
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        b_area = max(1, (bx1 - bx0) * (by1 - by0))
        areas.append(b_area)
        best_i = None
        best_area = None
        for i, bub in enumerate(bubbles or []):
            ux0, uy0, ux1, uy1 = bub["minX"], bub["minY"], bub["maxX"], bub["maxY"]
            if not (ux0 <= cx <= ux1 and uy0 <= cy <= uy1):
                continue
            u_area = (ux1 - ux0) * (uy1 - uy0)
            if u_area < b_area * min_expand:
                continue
            if best_area is None or u_area < best_area:
                best_i, best_area = i, u_area
        cand.append(best_i)

    # Pass 2: a bubble serves at most ONE block — the largest (the main
    # dialogue). Other blocks in the same bubble (e.g. orphan-paragraph
    # fragments of the same balloon) fall back to their own bbox so two
    # full-bubble renders don't overlap.
    winner: Dict[int, int] = {}
    for bi, bub_i in enumerate(cand):
        if bub_i is None:
            continue
        if bub_i not in winner or areas[bi] > areas[winner[bub_i]]:
            winner[bub_i] = bi
    out: List[Optional[Dict]] = []
    for bi, bub_i in enumerate(cand):
        if bub_i is not None and winner.get(bub_i) == bi:
            out.append(bubbles[bub_i])
        else:
            out.append(None)
    return out


def build_inpaint_mask(
    image_shape,
    blocks: List[Dict],
    text_lines: List[Dict],
    detector_mask: Optional[np.ndarray],
    erase_blocks: Optional[List[Dict]] = None,
) -> np.ndarray:
    """Binary 0/255 LaMa mask covering ONLY regions that will be re-rendered.

    `blocks` must be the post-filter ("kept") blocks — the ones whose OCR text
    passed the Japanese filter and will receive a rendered translation. Erasing
    anything else produces text that is inpainted away (often with ghosting on
    large regions) but never replaced, which reads as corruption on the final
    page. Text outside kept blocks is left untouched instead.

    `erase_blocks` are regions that were DROPPED by the OCR-confidence gate but
    are real Japanese ink (e.g. stylized SFX) — they get no translation but must
    still be erased so raw Japanese doesn't survive into the final render. We do
    NOT draw their full bbox into the mask (that paints rectangular patches over
    art); instead we extend the detector seg-mask clip area so the tight ink
    pixels over them are retained. Only when there is no detector mask do we fall
    back to a bbox fill, and only for SMALL erase_blocks (area<=9000) to avoid
    large rectangular ghosting.

    Sources, in priority order per block:
      * text_lines whose center falls inside the block (tight strokes)
      * the block bbox itself when no line is assigned to it
    The detector's pixel mask is OR-ed in only where it intersects a kept
    block's bbox OR an erase_block's bbox (the raw mask covers every detection on
    the page, including dropped ones).
    """
    erase_blocks = erase_blocks or []
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not blocks and not erase_blocks:
        return mask

    # Same center-containment rule recognize_blocks_with_lines uses, so the
    # erased area always matches the OCR'd area.
    blocks_with_lines = set()
    for ln in text_lines or []:
        cx = (ln["minX"] + ln["maxX"]) / 2
        cy = (ln["minY"] + ln["maxY"]) / 2
        for bi, b in enumerate(blocks):
            if b["minX"] <= cx <= b["maxX"] and b["minY"] <= cy <= b["maxY"]:
                # Detector line bboxes routinely clip ascenders/descenders and
                # the first/last glyph; a fixed dilate can't cover that on
                # large fonts. Pad proportionally to glyph size (≈ the line's
                # short side) so the erase mask swallows the whole stroke.
                lw = ln["maxX"] - ln["minX"]
                lh = ln["maxY"] - ln["minY"]
                pad = max(4, int(0.35 * min(lw, lh)))
                x0 = max(0, int(ln["minX"]) - pad); y0 = max(0, int(ln["minY"]) - pad)
                x1 = min(w, int(ln["maxX"]) + pad); y1 = min(h, int(ln["maxY"]) + pad)
                if x1 > x0 and y1 > y0:
                    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
                blocks_with_lines.add(bi)
                break

    block_area = np.zeros((h, w), dtype=np.uint8)
    for bi, b in enumerate(blocks):
        x0 = max(0, int(b["minX"])); y0 = max(0, int(b["minY"]))
        x1 = min(w, int(b["maxX"])); y1 = min(h, int(b["maxY"]))
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(block_area, (x0, y0), (x1, y1), 255, thickness=-1)
        if bi not in blocks_with_lines:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    # Extend block_area to cover erase-only (dropped-but-real-JP) blocks so the
    # detector seg-mask clip below RETAINS their ink instead of clipping it away.
    # We do NOT draw their bbox into `mask` (avoids rectangular patches over art);
    # the tight detector ink is what gets erased. Only fall back to a bbox fill
    # for SMALL erase_blocks when there is no detector mask to rely on.
    has_detector_mask = detector_mask is not None and detector_mask.size
    for b in erase_blocks:
        x0 = max(0, int(b["minX"])); y0 = max(0, int(b["minY"]))
        x1 = min(w, int(b["maxX"])); y1 = min(h, int(b["maxY"]))
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(block_area, (x0, y0), (x1, y1), 255, thickness=-1)
        if not has_detector_mask and (x1 - x0) * (y1 - y0) <= 9000:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    if has_detector_mask:
        dm = detector_mask
        if dm.shape[:2] != (h, w):
            dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_NEAREST)
        _, dm_bin = cv2.threshold(dm, 127, 255, cv2.THRESH_BINARY)
        # Clip to kept-block + erase-block area so unrelated dropped detections
        # stay untouched while real-JP erase regions keep their ink.
        dm_bin = cv2.bitwise_and(dm_bin.astype(np.uint8), block_area)
        mask = np.maximum(mask, dm_bin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # 2 iterations (~4-5 px): glyph anti-aliasing left outside a thin mask
    # seeds LaMa into reconstructing text-shaped artifacts ("ghosting").
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def build_text_regions(
    blocks: List[Dict],
    text_lines: List[Dict] = None,
    inset_percent: float = 0.01  # 1% inset on each side = 98% final size
) -> List[List[Dict]]:
    """
    Build per-block text regions with inset for white box masking.

    Args:
        blocks: List of detected text blocks with minX, minY, maxX, maxY
        text_lines: Unused (kept for API compatibility)
        inset_percent: Percentage to shrink on each side (0.01 = 1% = 98% final size)

    Returns:
        List of text regions, each slightly smaller than the original block
    """
    regions: List[List[Dict]] = []

    for block in blocks:
        w = block["maxX"] - block["minX"]
        h = block["maxY"] - block["minY"]

        # Apply inset (shrink box by inset_percent on each side)
        inset_x = int(w * inset_percent)
        inset_y = int(h * inset_percent)

        regions.append([{
            "minX": block["minX"] + inset_x,
            "minY": block["minY"] + inset_y,
            "maxX": block["maxX"] - inset_x,
            "maxY": block["maxY"] - inset_y,
        }])

    return regions
