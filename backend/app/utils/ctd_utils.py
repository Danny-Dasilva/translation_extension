"""Helpers for mapping CTD outputs to text regions."""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np


# --- Erase-mask tuning (module-level by design; NOT config fields) ----------
# Seg threshold used to build the ERASE mask, DECOUPLED from and lower than the
# detection threshold (``settings.ctd_text_threshold`` ~0.8). Detection needs
# high precision for block/line geometry, but erasing must also catch faint
# glyph tails / anti-aliased stroke edges — otherwise LaMa reseeds text-shaped
# ghosts from the un-erased remnant. ``ComicTextDetectorService._process_mask``
# thresholds the erase seg mask at this value (detection still uses 0.8).
ERASE_SEG_THRESHOLD: float = 0.45

# Over-broad (per-component) art guard: a single detected text region (line or
# block) whose bbox covers more than this fraction of the page is almost never
# text — it is a panel-spanning false detection or seg bleed over artwork. Such
# a region is dropped from the erase mask so the median/NS fill never paints
# over art. (A second, ink-ratio + page-fraction clamp runs downstream in
# ``lama_inpaint_service.inpaint`` per connected component; this is the cheap
# build-time guard.)
OVER_BROAD_AREA_FRAC: float = 0.30

# Proportional padding for detected line rects (detector line bboxes routinely
# clip ascenders/descenders and the first/last glyph).
LINE_PAD_RATIO: float = 0.25
LINE_PAD_MIN: int = 6

# Final dilation pad, proportional to glyph short-side: ``max(6, 0.25*short)``.
# A fixed ~4-5 px pad is too small for large fonts, leaving stroke halos that
# reseed LaMa ghosting.
DILATE_PAD_MIN: int = 6
DILATE_PAD_MAX: int = 48


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
    fit_rects: Optional[List[Optional[Dict]]] = None,
    leave_intact_blocks: Optional[List[Dict]] = None,
) -> np.ndarray:
    """Binary 0/255 LaMa erase mask covering ALL detected text ink.

    MASK-RECALL FIX (2026-06-30): the #1 visual defect is residual Japanese — JP
    ink the detector found but the pipeline DROPPED before re-render (jp-filter /
    english early-exit / OCR garble-gate) and therefore never erased. The old
    mask was gated to "what gets re-rendered" (``kept ∪ erase`` blocks): any
    detection dropped earlier lost its seg ink and survived onto the page. This
    builds the erase mask from EVERY detected text region instead:

      * every detected ``text_lines`` bbox (padded) — the detector found strokes
        there, so it is text and must be erased whether or not its OCR was kept;
      * every kept ``blocks`` bbox — re-rendered dialogue, SOLID-filled (now ALL
        kept blocks, not only bubble-matched ones; fills are median/NS so the
        downstream clamp + the over-broad guard protect art);
      * ``erase_blocks`` (gate-dropped real JP) — UNCHANGED: their tight detector
        ink is retained via the seg-mask clip; a full-bbox fill is used only for
        SMALL erase blocks when no detector mask is available (avoids painting
        rectangles over art);
      * the detector seg mask (thresholded LOW upstream, see ``ERASE_SEG_THRESHOLD``)
        OR-ed in wherever it fires inside a detected region (tight stroke pixels
        that spill outside a line bbox).

    GUARDS kept intact:
      * Over-broad per-region clamp (``OVER_BROAD_AREA_FRAC``): any single line /
        block bbox larger than a sane page fraction is dropped — a panel-spanning
        false detection must not erase artwork. (A second ink-ratio/page-fraction
        clamp runs per connected component downstream in ``lama_inpaint_service``.)
      * Leave-intact labels (``leave_intact_blocks`` — 表紙用イラスト / 奥付 /
        editorial margin): punched OUT of the mask LAST (after fills, seg ink and
        dilation) so the human-reference-kept labels are never erased even though
        the detector also found their text lines.

    ``fit_rects`` is retained for signature compatibility (callers pass the
    bubble match); it no longer gates the fills.
    """
    erase_blocks = erase_blocks or []
    leave_intact_blocks = leave_intact_blocks or []
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not blocks and not erase_blocks and not (text_lines or []):
        return mask

    page_area = float(h * w)
    over_broad_area = OVER_BROAD_AREA_FRAC * page_area

    def _clip(b: Dict):
        x0 = max(0, int(b["minX"])); y0 = max(0, int(b["minY"]))
        x1 = min(w, int(b["maxX"])); y1 = min(h, int(b["maxY"]))
        return x0, y0, x1, y1

    # Region within which detector seg ink may be erased: the UNION of every
    # detected text region (all lines + kept + erase blocks). Retaining seg ink
    # here — rather than clipping it to only the re-rendered kept∪erase set — is
    # the recall fix. Over-broad (art-sized) regions are excluded entirely so
    # neither their bbox fill NOR their seg ink reaches the mask.
    detected_area = np.zeros((h, w), dtype=np.uint8)
    short_sides: List[int] = []

    # (1) Every detected text LINE -> solid padded box. Core recall fix: catches
    #     residual JP whose OCR was dropped (the detector still found the strokes).
    for ln in text_lines or []:
        x0, y0, x1, y1 = _clip(ln)
        if x1 <= x0 or y1 <= y0:
            continue
        lw, lh = x1 - x0, y1 - y0
        if lw * lh > over_broad_area:  # over-broad clamp (art guard)
            continue
        short_sides.append(min(lw, lh))
        pad = max(LINE_PAD_MIN, int(LINE_PAD_RATIO * min(lw, lh)))
        px0 = max(0, x0 - pad); py0 = max(0, y0 - pad)
        px1 = min(w, x1 + pad); py1 = min(h, y1 + pad)
        cv2.rectangle(mask, (px0, py0), (px1, py1), 255, thickness=-1)
        cv2.rectangle(detected_area, (px0, py0), (px1, py1), 255, thickness=-1)

    # (2) Every KEPT block -> solid fill (re-rendered dialogue). fix #3: ALL kept
    #     blocks, not only bubble-matched ones. The over-broad clamp drops any
    #     pathologically large block so a runaway detection never paints art.
    for b in blocks:
        x0, y0, x1, y1 = _clip(b)
        if x1 <= x0 or y1 <= y0:
            continue
        if (x1 - x0) * (y1 - y0) > over_broad_area:  # over-broad clamp (art guard)
            continue
        cv2.rectangle(detected_area, (x0, y0), (x1, y1), 255, thickness=-1)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    # (3) erase-only (dropped-but-real-JP) blocks — UNCHANGED behaviour: extend
    #     the seg-clip area so their TIGHT detector ink is retained, but do NOT
    #     paint their full bbox when a detector mask exists (avoids rectangles
    #     over art). Fall back to a bbox fill only for SMALL erase blocks with no
    #     detector mask to rely on.
    has_detector_mask = detector_mask is not None and getattr(detector_mask, "size", 0)
    for b in erase_blocks:
        x0, y0, x1, y1 = _clip(b)
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(detected_area, (x0, y0), (x1, y1), 255, thickness=-1)
        if not has_detector_mask and (x1 - x0) * (y1 - y0) <= 9000:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    # (4) Detector seg mask (tight strokes, thresholded LOW upstream) OR-ed in
    #     wherever it fires INSIDE a detected region. Catches stroke pixels that
    #     spill outside a line bbox; clipping to detected_area keeps stray seg
    #     over art out of the mask.
    if has_detector_mask:
        dm = detector_mask
        if dm.shape[:2] != (h, w):
            dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_NEAREST)
        _, dm_bin = cv2.threshold(dm.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        dm_bin = cv2.bitwise_and(dm_bin, detected_area)
        mask = np.maximum(mask, dm_bin)

    # (5) Final dilation, proportional to glyph size: max(6, ~0.25 * line short
    #     side). Anti-aliasing left outside a thin mask reseeds LaMa into
    #     text-shaped ghosts; a fixed ~4 px pad is too small for large fonts.
    if mask.any():
        rep = int(np.median(short_sides)) if short_sides else 0
        pad = int(max(DILATE_PAD_MIN, min(DILATE_PAD_MAX, round(0.25 * rep))))
        k = 2 * pad + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # (6) Leave-intact labels are kept as ORIGINAL art — punch them OUT LAST so
    #     neither the fills, the seg ink, nor the dilation erase them.
    for b in leave_intact_blocks:
        x0, y0, x1, y1 = _clip(b)
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(mask, (x0, y0), (x1, y1), 0, thickness=-1)

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
