"""CV-based fallback region proposer for horizontal chat/UI text (BEST-EFFORT
mitigation for detection-recall gap #1: the v26 OBB text-line head scores
horizontal chat/phone-UI text at 0.05-0.32, straddling and often falling
below the 0.3 op-point, so whole SMS/chat bubbles never produce a single
``text_line`` and are invisible to every downstream stage (block-derivation,
orphan-line recovery, OCR, translation, render) -- the reader sees raw
Japanese.

FLAG-GATED, OFF by default (``settings.ctd_cv_region_fallback``). This module
has ZERO model dependency: it is a classical CV proposer (Otsu binarize ->
morphological line-merge -> contour + glyph-count + ink-density gating) that
looks ONLY in page area the model detector did not already claim, and hands
its output back to ``ComicTextDetectorService.detect()`` as extra
``text_lines`` entries. Those flow through the EXISTING orphan-line-recovery
pipeline (``app.utils.orphan_lines.find_orphan_lines`` /
``cluster_orphan_lines``, called by the production router) completely
unmodified -- a CV-proposed line looks just like any other orphaned
``text_line`` to that code.

Precision over recall: manga art (screentone, hatching, halftone) produces
MANY small dark blobs that a naive text-line heuristic would happily
misread as "text lines". Multiple independent gates keep this conservative:
  * only searches area NOT already covered by a real detector block/line
    (dilated exclusion mask) -- never duplicates a real detection,
  * horizontal-only (aspect ratio gate) -- this fallback targets gap #1
    (horizontal chat/UI text) specifically, not vertical narration columns,
  * a glyph-count-in-bbox floor (>=3 small ink components) rejects single
    stray marks / panel-border fragments,
  * an ink-density band (not near-empty, not near-solid-black) rejects both
    noise specks and large solid-fill shapes (panel gutters, silhouettes),
  * a RAW-candidate-count circuit breaker: if the cheap Otsu+contour pass
    turns up an implausibly large number of raw candidates before any of the
    above gates run, the page is almost certainly busy line-art/screentone
    that would swamp the heuristic, so the whole page is skipped (returns
    empty) rather than flooding it with false regions,
  * a hard cap on the number of candidates returned per page.

Even with all of the above, this is a PROTOTYPE: it is a coarse geometric
heuristic, not a learned text detector, and it will occasionally miss part
of a bubble or (rarely) pick up a genuinely text-like non-text mark. See
``ctd_service.py``'s wiring for the exact call site and the eval notes in
the handoff for measured false-positive rates on non-chat pages.

KNOWN CEILING (measured on furube ch1 p010/p015, the motivating chat-bubble
pages): this recovers isolated/short lines, captions, and SFX the OBB head
missed reasonably well, but it does NOT reliably recover the single worst
sub-case -- a DENSE multi-line stacked chat bubble (many JP lines with
near-zero visual gap between them). The row-projection an obvious "split
into per-line bands" fix would use does not show clean zero-ink troughs
between lines at that density/resolution (checked directly), so rather than
guess at a split and risk a worse (misaligned) box, this returns the
merged blob's rejection as-is -- i.e. it silently gives up on that specific
case instead of proposing something low-confidence. A real fix for THIS
sub-case needs either a learned line-splitter or the OBB retrain, not more
CV heuristics.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


def _to_gray(img: np.ndarray, input_is_bgr: bool) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        code = cv2.COLOR_BGRA2GRAY if input_is_bgr else cv2.COLOR_RGBA2GRAY
        return cv2.cvtColor(img, code)
    code = cv2.COLOR_BGR2GRAY if input_is_bgr else cv2.COLOR_RGB2GRAY
    return cv2.cvtColor(img, code)


def _build_exclusion_mask(
    image_size: Tuple[int, int],
    existing: Sequence[Dict],
    margin_frac: float = 0.15,
    min_margin_px: int = 4,
) -> np.ndarray:
    """Rasterize a padded union of already-detected boxes to skip.

    Padding is proportional to each box's shorter side (roughly font-size
    scaled), matching the padding style ``_build_block_bounds_mask`` uses,
    so the CV pass never proposes a line that just clips the edge of a real
    detection.
    """
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in existing:
        bw = b["maxX"] - b["minX"]
        bh = b["maxY"] - b["minY"]
        if bw <= 0 or bh <= 0:
            continue
        pad = max(min_margin_px, int(margin_frac * min(bw, bh)))
        x1 = max(0, int(b["minX"]) - pad)
        y1 = max(0, int(b["minY"]) - pad)
        x2 = min(w, int(b["maxX"]) + pad)
        y2 = min(h, int(b["maxY"]) + pad)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def propose_horizontal_text_regions(
    img: np.ndarray,
    existing_blocks: Sequence[Dict],
    existing_text_lines: Sequence[Dict],
    *,
    input_is_bgr: bool = True,
    min_line_width_frac: float = 0.03,
    min_line_height_px: int = 8,
    max_line_height_frac: float = 0.05,
    min_aspect_ratio: float = 2.2,
    min_ink_glyphs: int = 3,
    ink_frac_range: Tuple[float, float] = (0.04, 0.6),
    max_candidates: int = 10,
    max_raw_candidates: int = 300,
) -> List[Dict]:
    """Propose horizontal text-line boxes the model's OBB head likely missed.

    Returns a list of dicts with the SAME schema as
    ``ComicTextDetectorService._extract_text_lines`` output (``minX/minY/
    maxX/maxY/area/confidence``), plus ``cv_fallback: True`` so callers/tests
    can tell these apart from model detections. Returns ``[]`` when nothing
    plausible is found, the page looks too busy to trust (raw-candidate
    circuit breaker), or the image is degenerate.
    """
    if img is None or img.size == 0:
        return []
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return []

    gray = _to_gray(img, input_is_bgr)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # INV+OTSU: text is assumed darker than its local bubble/UI fill, which
    # holds for the target case (dark JP glyphs on a white/pastel chat
    # bubble). Light-on-dark UI text is out of scope for this prototype.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    exclude = _build_exclusion_mask((w, h), list(existing_blocks) + list(existing_text_lines))
    binary[exclude > 0] = 0

    line_h = max(3, int(0.012 * h))

    # GLYPH-SIZE PRE-GATE (critical): manga line-art (hair/silhouette fills,
    # panel borders, screentone blobs) survives the same INV+OTSU threshold
    # as text ink and, left in, merges with real glyphs into one page-spanning
    # blob once dilated -- exactly the "flooded with false regions" failure
    # this module exists to avoid. Individual JP/EN glyphs at any normal page
    # resolution are small (bounded by roughly one line-height per side), so
    # drop any RAW ink component whose bbox exceeds ~2.5 line-heights on
    # either side BEFORE dilating -- this keeps glyph-scale ink and discards
    # large solid art regions, so the line-merge dilation below only bridges
    # actual glyphs into line-blobs instead of art into page-blobs.
    max_glyph_side = max(6, int(2.5 * line_h))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    glyph_only = np.zeros_like(binary)
    for i in range(1, n_labels):
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if cw <= max_glyph_side and ch <= max_glyph_side:
            glyph_only[labels == i] = 255

    # Merge glyphs along a row into line-blobs with a wide-short kernel sized
    # off page height (so it scales across page resolutions).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, line_h * 3), max(1, line_h // 2)))
    dilated = cv2.dilate(glyph_only, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > max_raw_candidates:
        # Too many raw blobs to trust on this page (busy art/screentone) --
        # bail rather than risk flooding it with false regions.
        return []

    min_w = min_line_width_frac * w
    max_h = max_line_height_frac * h
    ink_lo, ink_hi = ink_frac_range

    candidates: List[Dict] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < min_w or bh < min_line_height_px or bh > max_h:
            continue
        if bh <= 0 or bw / bh < min_aspect_ratio:
            continue

        # Measure ink density / glyph count on the GLYPH-ONLY mask (not the
        # raw ``binary``), so a candidate line-blob sitting next to a large
        # excluded art region doesn't get credit for that art's ink.
        roi = glyph_only[y:y + bh, x:x + bw]
        if roi.size == 0:
            continue
        ink_frac = float(np.count_nonzero(roi)) / float(roi.size)
        if ink_frac < ink_lo or ink_frac > ink_hi:
            continue

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        glyphs = sum(1 for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 3)
        if glyphs < min_ink_glyphs:
            continue

        candidates.append({
            "minX": int(x),
            "minY": int(y),
            "maxX": int(x + bw),
            "maxY": int(y + bh),
            "area": int(bw * bh),
            "confidence": 0.35,
            "cv_fallback": True,
        })

    candidates.sort(key=lambda c: c["area"], reverse=True)
    return candidates[:max_candidates]
