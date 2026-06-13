"""Helpers for mapping CTD outputs to text regions."""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np


def build_inpaint_mask(
    image_shape,
    blocks: List[Dict],
    text_lines: List[Dict],
    detector_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Binary 0/255 LaMa mask covering ONLY regions that will be re-rendered.

    `blocks` must be the post-filter ("kept") blocks — the ones whose OCR text
    passed the Japanese filter and will receive a rendered translation. Erasing
    anything else produces text that is inpainted away (often with ghosting on
    large regions) but never replaced, which reads as corruption on the final
    page. Text outside kept blocks is left untouched instead.

    Sources, in priority order per block:
      * text_lines whose center falls inside the block (tight strokes)
      * the block bbox itself when no line is assigned to it
    The detector's pixel mask is OR-ed in only where it intersects a kept
    block's bbox (the raw mask covers every detection on the page, including
    dropped ones).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not blocks:
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

    if detector_mask is not None and detector_mask.size:
        dm = detector_mask
        if dm.shape[:2] != (h, w):
            dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_NEAREST)
        _, dm_bin = cv2.threshold(dm, 127, 255, cv2.THRESH_BINARY)
        # Clip to kept-block area so dropped detections stay untouched.
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
