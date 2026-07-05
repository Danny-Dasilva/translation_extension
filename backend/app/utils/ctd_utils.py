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

# --- Ono (SFX) channel tuning (2026-07-04 partial-coverage audit) -----------
# The v26 ono/ch1 channel routinely fires on only PART of a stylized glyph's
# strokes — a render audit found ~60-70% of touched SFX end up partially
# erased (a legible fragment survives next to a blur remnant), and the
# largest/most-stylized glyphs survive almost entirely. Raw-heatmap
# inspection (see thoughts audit) showed two distinct failure modes:
#   1. Some glyphs get near-zero ch1 activation across the ENTIRE character —
#      a genuine detector recall miss. No post-processing threshold or
#      morphology can recover signal that was never emitted; this needs a
#      model/training fix, not a mask tweak.
#   2. Other glyphs DO fire, but only on a subset of strokes, leaving small
#      (~10-30px) gaps between detected fragments of the SAME glyph. This is
#      the fixable case: a modest threshold drop surfaces marginal fragments,
#      and a morphological close bridges the inter-fragment gaps into one
#      solid glyph blob, followed by a small dilate to cover the
#      anti-aliased glyph edge.
# ``ONO_ERASE_SEG_THRESHOLD`` is DELIBERATELY SEPARATE from
# ``ERASE_SEG_THRESHOLD`` (used for the ch0 text channel and the combined
# ch0-max-ch1 mask in ``_process_mask``) so this tuning touches ONLY the
# unclipped ono/SFX path — the text-channel erase mask is unaffected.
ONO_ERASE_SEG_THRESHOLD: float = 0.30

# Close kernel (ellipse, diameter px): bridges gaps BETWEEN nearby stroke
# fragments of the same glyph. Sized from measured inter-fragment gaps on a
# densest-SFX sample (~10-30px) — large enough to merge those, small enough
# to leave genuinely separate glyphs/words (50px+ apart) unmerged.
ONO_CLOSE_KERNEL_SIZE: int = 25

# Final dilate kernel (ellipse, diameter px) after the close: grows the
# merged blob out to the glyph's anti-aliased edge, same spirit as the
# proportional dilate ``build_inpaint_mask`` applies to the combined mask.
ONO_DILATE_KERNEL_SIZE: int = 9

# Safety-net art guard: since the ono mask has NO block-bounds clip (it
# deliberately lives outside detected text regions — see
# ``_process_ono_mask``), an unlucky close/dilate could in principle bridge
# across unrelated background texture (screentone/hatching) into one runaway
# blob. Real hand-drawn SFX is localized; any single connected component that
# ends up covering more than this fraction of the page after the morphology
# is dropped rather than erased, so a bridging failure cannot smear art.
ONO_MAX_COMPONENT_AREA_FRAC: float = 0.05

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
#
# ``LINE_PAD_RATIO`` is the CROSS-axis pad (perpendicular to the reading
# direction) — it catches stroke anti-aliasing / ascenders that spill out the
# *side* of a tight line bbox.
#
# ``TAIL_PAD_RATIO`` is the END-OF-LINE pad, applied ALONG the reading axis
# (top/bottom of a vertical column, left/right of a horizontal line). Detector
# line bboxes frequently clip — or entirely drop — the FIRST/LAST glyph of a
# column (e.g. the p005 "に" at the tail of a vertical column), which then
# survives un-erased onto the page. One glyph ≈ the line's short side, so the
# two ends of the long axis are padded by ~a full glyph (symmetrically), not by
# the much smaller cross pad. (audit §4.5)
LINE_PAD_RATIO: float = 0.25
LINE_PAD_MIN: int = 6
TAIL_PAD_RATIO: float = 1.0

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
    ono_mask: Optional[np.ndarray] = None,
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
      * every kept ``blocks`` bbox — re-rendered dialogue, SOLID-filled ONLY
        when the block matched a speech bubble (``fit_rects[bi] is not None``)
        so the rectangle lands inside the balloon, not on artwork (audit §4.4).
        Un-bubbled kept blocks rely on their tight seg ink (below) instead;
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

    ``fit_rects`` (per-block bubble match, index-aligned with ``blocks``) gates
    the step-(2) solid block fill: only bubble-matched blocks are solid-filled;
    ``None`` entries / a ``None`` list fall back to tight line + seg ink.

    ``ono_mask`` (v26 detector only, flag-gated by ``settings.inpaint_ono_mask``
    in the caller — see ``app.routers.translate``) is the UNCLIPPED ch1
    onomatopoeia/SFX seg-mask from ``ComicTextDetectorService._process_ono_mask``.
    It is OR-ed into the erase mask WITHOUT clipping to ``detected_area`` —
    free-floating hand-drawn SFX ink rarely sits inside any detected text
    line/block, so clipping it (as the detector-seg OR above does) would erase
    nothing. When ``None`` (flag off, or a legacy single-channel detector),
    this is a no-op and the returned mask is BYTE-IDENTICAL to before this
    parameter existed.
    """
    erase_blocks = erase_blocks or []
    leave_intact_blocks = leave_intact_blocks or []
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    _has_ono = ono_mask is not None and getattr(ono_mask, "size", 0)
    if not blocks and not erase_blocks and not (text_lines or []) and not _has_ono:
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
        short = min(lw, lh)
        short_sides.append(short)
        # Asymmetric-by-axis padding: a small CROSS pad (perpendicular to the
        # reading direction) for stroke spill, and a full-glyph TAIL pad along
        # the reading axis so an end-of-column glyph the detector bbox clipped
        # (p005 "に", audit §4.5) is still covered — SYMMETRICALLY on both ends
        # of that axis. Applied to the long axis: vertical column -> top/bottom,
        # horizontal line -> left/right.
        cross_pad = max(LINE_PAD_MIN, int(LINE_PAD_RATIO * short))
        tail_pad = max(LINE_PAD_MIN, int(TAIL_PAD_RATIO * short))
        if lh >= lw:  # vertical column: reading axis is vertical
            px0 = max(0, x0 - cross_pad); px1 = min(w, x1 + cross_pad)
            py0 = max(0, y0 - tail_pad); py1 = min(h, y1 + tail_pad)
        else:  # horizontal line: reading axis is horizontal
            px0 = max(0, x0 - tail_pad); px1 = min(w, x1 + tail_pad)
            py0 = max(0, y0 - cross_pad); py1 = min(h, y1 + cross_pad)
        cv2.rectangle(mask, (px0, py0), (px1, py1), 255, thickness=-1)
        cv2.rectangle(detected_area, (px0, py0), (px1, py1), 255, thickness=-1)

    # (2) KEPT block -> solid bbox fill (re-rendered dialogue), BUBBLE-GATED.
    #     A block is solid-filled ONLY when it matched a speech bubble
    #     (``fit_rects[bi] is not None``): the rectangle then lands INSIDE the
    #     balloon interior, never on bare artwork. Un-bubbled kept blocks (SFX /
    #     narration over art) are added to ``detected_area`` only — their TIGHT
    #     detector seg ink is still erased for recall by step (4), without
    #     painting a rectangular scar over the art.
    #
    #     REGRESSION FIX (audit §4.4): 22fd106 dropped this gate ("ALL kept
    #     blocks, not only bubble-matched ones") and scarred artwork with
    #     rectangles; the gate is restored here. Line-level recall (step 1) and
    #     seg-ink recall (step 4) still catch dropped/residual JP over art.
    #     When no bubble detector is available (``fit_rects`` None) NO block is
    #     solid-filled — everything falls back to tight line rects + seg ink,
    #     the historically safe behaviour. The over-broad clamp still drops any
    #     pathologically large block first.
    for bi, b in enumerate(blocks):
        x0, y0, x1, y1 = _clip(b)
        if x1 <= x0 or y1 <= y0:
            continue
        if (x1 - x0) * (y1 - y0) > over_broad_area:  # over-broad clamp (art guard)
            continue
        cv2.rectangle(detected_area, (x0, y0), (x1, y1), 255, thickness=-1)
        in_bubble = (
            fit_rects is not None
            and bi < len(fit_rects)
            and fit_rects[bi] is not None
        )
        if in_bubble:
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

    # (4b) OR in the flag-gated v26 ono/SFX channel mask UNCLIPPED — free-
    # floating hand-drawn SFX lives outside any detected text block/line, so
    # clipping to ``detected_area`` (as step 4 does for the combined detector
    # seg mask) would discard it entirely, defeating the purpose. ``ono_mask``
    # is None unless the caller passes the v26 ch1 channel AND
    # ``settings.inpaint_ono_mask`` is True, so this is a no-op (byte-identical
    # output) on the default flag-off path.
    if _has_ono:
        om = ono_mask
        if om.shape[:2] != (h, w):
            om = cv2.resize(om, (w, h), interpolation=cv2.INTER_NEAREST)
        _, om_bin = cv2.threshold(om.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        mask = np.maximum(mask, om_bin)

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
