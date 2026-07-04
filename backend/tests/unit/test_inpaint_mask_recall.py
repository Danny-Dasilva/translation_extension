"""Unit tests for the MASK-RECALL fix (residual-Japanese #1 visual defect).

The erase mask used to be gated to "what gets re-rendered" (``kept ∪ erase``
blocks), so any region the detector found but the pipeline DROPPED earlier
(jp-filter / english early-exit / OCR garble-gate) lost its ink and survived
onto the page as raw Japanese. ``build_inpaint_mask`` now erases ALL detected
text ink, with two guards kept intact:

  * an over-broad per-region clamp (art protection), and
  * a leave-intact punch-out (editorial labels the human reference keeps).

These also pin the decoupled erase seg-threshold (lower than detection) that
``ComicTextDetectorService._process_mask`` applies for inpainting only.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from app.utils.ctd_utils import (
    ERASE_SEG_THRESHOLD,
    OVER_BROAD_AREA_FRAC,
    build_inpaint_mask,
)


def _block(x0, y0, x1, y1):
    return {"minX": x0, "minY": y0, "maxX": x1, "maxY": y1}


# --------------------------------------------------------------------------- #
# fix #1 — a detected-but-DROPPED text region is now in the erase mask
# --------------------------------------------------------------------------- #


def test_dropped_text_line_is_now_erased():
    """A detector text line that is NOT a kept/erase block (it was dropped before
    render) must still be erased — the detector found strokes there, so it is
    text. No detector mask is supplied, proving the recall is via the line bbox,
    not seg ink."""
    h, w = 300, 300
    shape = (h, w, 3)
    kept = _block(10, 10, 60, 60)
    dropped_line = _block(200, 200, 240, 240)  # detected, but its OCR was dropped

    mask = build_inpaint_mask(shape, [kept], [dropped_line], None)
    assert mask[220, 220] == 255, "dropped detected text line was not erased"

    # The OLD behaviour: with the dropped line absent from the inputs, that
    # region stays untouched — this is exactly the residual-JP bug.
    mask_old = build_inpaint_mask(shape, [kept], [], None)
    assert mask_old[220, 220] == 0


def test_dropped_line_seg_ink_retained_but_stray_seg_clipped():
    """Detector seg ink over a detected line is retained (recall), while stray
    seg over un-detected art stays out of the mask (art protection)."""
    h, w = 300, 300
    shape = (h, w, 3)
    kept = _block(10, 10, 60, 60)
    dropped_line = _block(200, 200, 240, 240)

    detector_mask = np.zeros((h, w), dtype=np.uint8)
    detector_mask[210:230, 210:230] = 255  # ink inside the dropped line
    detector_mask[285:295, 285:295] = 255  # stray ink over art (no detection)

    mask = build_inpaint_mask(shape, [kept], [dropped_line], detector_mask)
    assert mask[220, 220] == 255, "seg ink over a detected line must be retained"
    assert mask[289, 289] == 0, "stray seg over un-detected art must be clipped"


# --------------------------------------------------------------------------- #
# fix #1 guard — leave-intact labels are NOT erased
# --------------------------------------------------------------------------- #


def test_leave_intact_label_is_not_erased():
    """An editorial label the detector found (so it is a text line) but which is
    marked leave-intact must be punched OUT of the mask — even though ALL other
    detected text ink is now erased."""
    h, w = 300, 300
    shape = (h, w, 3)
    label = _block(200, 200, 260, 260)

    # Same region is BOTH a detected line AND a leave-intact label.
    masked = build_inpaint_mask(
        shape, [], [label], None, leave_intact_blocks=[label]
    )
    assert masked[230, 230] == 0, "leave-intact label was erased"

    # Without the leave-intact guard the same region WOULD be erased — proving
    # the punch-out is what protects it.
    masked_no_guard = build_inpaint_mask(shape, [], [label], None)
    assert masked_no_guard[230, 230] == 255


def test_leave_intact_punch_survives_dilation():
    """The punch-out runs AFTER dilation, so a kept block adjacent to a label
    cannot regrow into and erase the label."""
    h, w = 300, 300
    shape = (h, w, 3)
    label = _block(150, 150, 200, 200)
    neighbour = _block(120, 150, 149, 200)  # touches the label's left edge

    mask = build_inpaint_mask(
        shape, [neighbour], [neighbour, label], None, leave_intact_blocks=[label]
    )
    assert mask[175, 175] == 0, "label centre erased despite leave-intact guard"


# --------------------------------------------------------------------------- #
# fix #1 guard — over-broad per-region clamp rejects a giant box (art guard)
# --------------------------------------------------------------------------- #


def test_over_broad_block_is_rejected():
    """A single detected block covering far more than a sane page fraction is a
    panel-spanning false detection — it must NOT be filled (would smear art)."""
    h, w = 100, 100
    shape = (h, w, 3)
    # over_broad threshold = 0.30 * 10000 = 3000 px.
    giant = _block(0, 0, 100, 100)   # 10000 px -> rejected
    mask = build_inpaint_mask(shape, [giant], [], None)
    assert mask.sum() == 0, "over-broad block was not rejected"


def test_over_broad_clamp_boundary():
    """Just over the page-fraction cap is rejected; a normal block is kept."""
    h, w = 100, 100
    shape = (h, w, 3)
    assert OVER_BROAD_AREA_FRAC * h * w == 3000

    big = _block(0, 0, 60, 60)    # 3600 > 3000 -> rejected
    small = _block(0, 0, 50, 50)  # 2500 < 3000 -> kept (filled)
    bub = _block(0, 0, 100, 100)  # enclosing bubble so the block IS solid-filled

    # The over-broad clamp fires BEFORE the bubble gate, so a giant block is
    # rejected even when bubble-matched.
    assert build_inpaint_mask(shape, [big], [], None, fit_rects=[bub]).sum() == 0
    # A small, bubble-matched block is solid-filled (clamp does not reject it).
    assert build_inpaint_mask(shape, [small], [], None, fit_rects=[bub])[25, 25] == 255


def test_over_broad_giant_line_rejected():
    """The clamp applies to detected LINES too (not only blocks)."""
    h, w = 100, 100
    shape = (h, w, 3)
    giant_line = _block(0, 0, 100, 100)
    assert build_inpaint_mask(shape, [], [giant_line], None).sum() == 0


# --------------------------------------------------------------------------- #
# audit §4.4 — the solid BLOCK fill is BUBBLE-GATED (regression from 22fd106
# restored): a solid rectangle is painted only INSIDE a matched bubble; an
# un-bubbled block relies on tight seg ink so no scar lands on artwork.
# --------------------------------------------------------------------------- #


def test_bubbled_block_is_solid_filled():
    """A kept block matched to a speech bubble (fit_rects[i] not None) gets a
    SOLID bbox fill — the fill lands inside the balloon interior."""
    h, w = 200, 200
    shape = (h, w, 3)
    kept = _block(40, 40, 90, 90)
    bubble = _block(20, 20, 120, 120)  # encloses the block
    mask = build_inpaint_mask(shape, [kept], [], None, fit_rects=[bubble])
    assert mask[65, 65] == 255, "bubble-matched block was not solid-filled"
    # A corner well inside the block bbox (but with no text there) is filled too
    # because the whole balloon-interior rectangle is safe to erase.
    assert mask[45, 45] == 255


def test_unbubbled_block_not_solid_filled_no_scar():
    """A kept block with NO bubble match and NO detector mask must NOT be
    solid-filled — painting its full bbox would scar the artwork underneath
    (audit §4.4). Without seg ink there is nothing to erase for it."""
    h, w = 200, 200
    shape = (h, w, 3)
    kept = _block(40, 40, 90, 90)
    # fit_rects present but this block matched no bubble.
    mask = build_inpaint_mask(shape, [kept], [], None, fit_rects=[None])
    assert mask[65, 65] == 0, "un-bubbled block was solid-filled (art scar)"
    assert mask.sum() == 0

    # fit_rects entirely absent -> same safe fallback (no bubble detector).
    mask_none = build_inpaint_mask(shape, [kept], [], None, fit_rects=None)
    assert mask_none.sum() == 0, "no-bubble fallback must not solid-fill blocks"


def test_unbubbled_block_seg_ink_still_erased():
    """The recall path is preserved: an un-bubbled kept block is NOT solid-
    filled, but its TIGHT detector seg ink is still erased (so residual JP over
    art is removed without a rectangular scar)."""
    h, w = 200, 200
    shape = (h, w, 3)
    kept = _block(40, 40, 90, 90)

    detector_mask = np.zeros((h, w), dtype=np.uint8)
    detector_mask[55:75, 55:75] = 255   # ink inside the block
    detector_mask[150:170, 150:170] = 255  # stray ink over art (no detection)

    mask = build_inpaint_mask(
        shape, [kept], [], detector_mask, fit_rects=[None]
    )
    assert mask[65, 65] == 255, "seg ink over an un-bubbled block must be erased"
    # But the block CORNER with no ink is NOT filled (no solid rectangle scar).
    assert mask[42, 42] == 0, "un-bubbled block painted a full-bbox scar"
    # Stray seg over un-detected art stays clipped out.
    assert mask[160, 160] == 0, "stray seg over art must be clipped"


# --------------------------------------------------------------------------- #
# audit §4.5 — end-of-column tail glyph is covered by the erase mask (the
# reading-axis pad is a FULL glyph, symmetric on both ends; the cross axis
# stays tight so the mask does not smear sideways into art).
# --------------------------------------------------------------------------- #


def test_tail_glyph_covered_both_ends_vertical():
    """A tall-narrow VERTICAL column: the detector bbox routinely clips the
    first/last glyph of the column. The end-of-line pad (~one glyph ≈ short
    side) must cover a tail glyph just past BOTH ends of the reading axis."""
    h, w = 400, 400
    shape = (h, w, 3)
    line = _block(100, 100, 130, 300)  # lw=30, lh=200 -> short=30 ≈ one glyph
    mask = build_inpaint_mask(shape, [], [line], None)
    # Bottom (end-of-column) tail glyph, ~one glyph past the bbox end.
    assert mask[322, 115] == 255, "end-of-column tail glyph not covered"
    # Top (start-of-column) glyph, symmetric on the other end.
    assert mask[82, 115] == 255, "start-of-column glyph not covered symmetrically"
    # Cross axis stays TIGHT: it is NOT padded by a full glyph, so the mask does
    # not smear sideways into neighbouring artwork.
    assert mask[200, 72] == 0, "cross axis was padded like the reading axis"


def test_tail_glyph_covered_horizontal_line():
    """The same full-glyph end pad applies along the reading axis of a wide
    HORIZONTAL line (left/right ends)."""
    h, w = 400, 400
    shape = (h, w, 3)
    line = _block(100, 100, 300, 140)  # lw=200, lh=40 -> short=40 ≈ one glyph
    mask = build_inpaint_mask(shape, [], [line], None)
    # Right-end glyph, ~one glyph past the bbox end.
    assert mask[120, 322] == 255, "end-of-line glyph not covered"
    assert mask[120, 82] == 255, "start-of-line glyph not covered symmetrically"
    # Cross (vertical) axis stays tight.
    assert mask[200, 200] == 0, "cross axis was padded like the reading axis"


# --------------------------------------------------------------------------- #
# backward compatibility — existing erase-block contract is unchanged
# --------------------------------------------------------------------------- #


def test_empty_inputs_return_blank_mask():
    mask = build_inpaint_mask((50, 50, 3), [], [], None)
    assert mask.shape == (50, 50)
    assert mask.sum() == 0


# --------------------------------------------------------------------------- #
# fix #2 — erase seg threshold is decoupled from (and lower than) detection
# --------------------------------------------------------------------------- #


def test_erase_seg_threshold_is_lower_than_detection():
    assert 0.3 <= ERASE_SEG_THRESHOLD <= 0.5


def test_process_mask_lower_threshold_erases_faint_ink():
    """``_process_mask`` with an erase_threshold captures mid-confidence ink that
    the detection threshold (0.8) discards. Exercised via the unbound method with
    a lightweight stand-in for ``self`` so no ONNX model is loaded."""
    from app.services.ctd_service import ComicTextDetectorService

    fake_self = types.SimpleNamespace(text_threshold=0.8)
    prob = np.zeros((20, 20), dtype=np.float32)
    prob[5:15, 5:15] = 0.6  # > 0.45 erase thr, < 0.8 detection thr

    detect = ComicTextDetectorService._process_mask(
        fake_self, prob.copy(), (20, 20), (20, 20), blocks=None, legacy=True
    )
    erase = ComicTextDetectorService._process_mask(
        fake_self, prob.copy(), (20, 20), (20, 20),
        blocks=None, legacy=True, erase_threshold=ERASE_SEG_THRESHOLD,
    )
    assert detect.max() == 0, "0.6 ink should be below the 0.8 detection threshold"
    assert erase.max() == 255, "0.6 ink should be erased at the lower threshold"
    assert (erase > 0).sum() > (detect > 0).sum()


def test_process_mask_erase_threshold_never_exceeds_detection():
    """The erase threshold is clamped to be no higher than detection, so a low
    detection threshold is never made *less* aggressive for erasing."""
    from app.services.ctd_service import ComicTextDetectorService

    fake_self = types.SimpleNamespace(text_threshold=0.3)
    prob = np.zeros((20, 20), dtype=np.float32)
    prob[5:15, 5:15] = 0.35  # above the 0.3 detection thr, below 0.45 erase const

    erase = ComicTextDetectorService._process_mask(
        fake_self, prob.copy(), (20, 20), (20, 20),
        blocks=None, legacy=True, erase_threshold=ERASE_SEG_THRESHOLD,
    )
    assert erase.max() == 255, "min(erase, detect) must keep the lower 0.3 threshold"
