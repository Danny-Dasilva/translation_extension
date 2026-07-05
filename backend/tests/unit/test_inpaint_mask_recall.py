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
    ONO_CLOSE_KERNEL_SIZE,
    ONO_DILATE_KERNEL_SIZE,
    ONO_ERASE_SEG_THRESHOLD,
    ONO_MAX_COMPONENT_AREA_FRAC,
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


# --------------------------------------------------------------------------- #
# SFX ono-mask port (flag-gated inpaint_ono_mask): build_inpaint_mask's
# unclipped OR-in of the v26 ch1 onomatopoeia/SFX seg-mask.
# --------------------------------------------------------------------------- #


def test_ono_mask_none_is_byte_identical_to_no_param():
    """ono_mask=None (the flag-off default) must produce EXACTLY the same mask
    as not passing the parameter at all — the no-op guard must not perturb any
    existing pixel, guard, or the dilation pass."""
    h, w = 300, 300
    shape = (h, w, 3)
    kept = _block(10, 10, 60, 60)
    dropped_line = _block(200, 200, 240, 240)
    detector_mask = np.zeros((h, w), dtype=np.uint8)
    detector_mask[20:50, 20:50] = 255

    baseline = build_inpaint_mask(shape, [kept], [dropped_line], detector_mask)
    with_none = build_inpaint_mask(
        shape, [kept], [dropped_line], detector_mask, ono_mask=None
    )
    assert np.array_equal(baseline, with_none), (
        "passing ono_mask=None must be byte-identical to omitting the param"
    )


def test_ono_mask_none_matches_pre_existing_erase_and_leave_intact_behavior():
    """With the flag off (ono_mask=None), the pre-existing erase_blocks /
    leave_intact_blocks / fit_rects behaviour is untouched pixel-for-pixel."""
    h, w = 200, 200
    shape = (h, w, 3)
    kept = _block(10, 10, 60, 60)
    bubble = _block(0, 0, 100, 100)
    erase_only = _block(120, 120, 160, 160)

    a = build_inpaint_mask(
        shape, [kept], [], None,
        erase_blocks=[erase_only], fit_rects=[bubble],
    )
    b = build_inpaint_mask(
        shape, [kept], [], None,
        erase_blocks=[erase_only], fit_rects=[bubble], ono_mask=None,
    )
    assert np.array_equal(a, b)


def test_ono_mask_erases_ink_outside_every_detected_region():
    """The core SFX fix: ink in ``ono_mask`` that sits OUTSIDE any kept block,
    detected text line, or erase-only block (i.e. free-floating hand-drawn SFX
    over bare artwork) is NOT discarded — proving the OR-in is unclipped,
    unlike the detector-seg OR (step 4) which clips to ``detected_area``."""
    h, w = 300, 300
    shape = (h, w, 3)
    kept = _block(10, 10, 60, 60)  # far from the SFX region below

    ono_mask = np.zeros((h, w), dtype=np.uint8)
    ono_mask[220:260, 220:260] = 255  # SFX ink with NO text-line/block box

    # Flag off (ono_mask=None): the SFX region is untouched — this is the
    # documented bug (round9's ono channel fires, but nothing consumes it).
    mask_off = build_inpaint_mask(shape, [kept], [], None)
    assert mask_off[240, 240] == 0, "SFX region erased even without ono_mask"

    # Flag on (ono_mask passed): the SAME region is now in the erase mask,
    # despite being outside every detected block/line/erase region.
    mask_on = build_inpaint_mask(shape, [kept], [], None, ono_mask=ono_mask)
    assert mask_on[240, 240] == 255, "unclipped ono ink was not OR-ed into the mask"
    # The pre-existing kept-block behaviour elsewhere is unaffected.
    assert mask_on[35, 35] == 0, "unrelated pixels perturbed by the ono OR-in"


def test_ono_mask_still_clipped_by_leave_intact_punch_out():
    """The leave-intact punch-out (step 6) runs AFTER the ono OR-in, so a
    human-reference-kept label still wins even if it overlaps SFX-flagged ono
    ink (defence in depth — the punch-out is unconditional, not ono-aware)."""
    h, w = 300, 300
    shape = (h, w, 3)
    label = _block(200, 200, 260, 260)

    ono_mask = np.zeros((h, w), dtype=np.uint8)
    ono_mask[210:250, 210:250] = 255  # overlaps the leave-intact label

    mask = build_inpaint_mask(
        shape, [], [], None, leave_intact_blocks=[label], ono_mask=ono_mask
    )
    assert mask[230, 230] == 0, "leave-intact label erased via the ono OR-in"


def test_ono_mask_resized_when_shape_mismatch():
    """An ono_mask produced at a different resolution (e.g. a stale caller) is
    resized (nearest-neighbor, matching the detector-seg OR) to the page size
    before being OR-ed in."""
    h, w = 400, 400
    shape = (h, w, 3)
    ono_small = np.zeros((200, 200), dtype=np.uint8)
    ono_small[120:180, 120:180] = 255  # maps to ~(240:360, 240:360) at full res

    mask = build_inpaint_mask(shape, [], [], None, ono_mask=ono_small)
    assert mask[300, 300] == 255, "resized ono mask did not erase the mapped region"


def test_ono_mask_only_input_still_produces_nonempty_mask():
    """A page with NO blocks/lines/erase_blocks but a non-empty ono_mask must
    not early-return a blank mask — pure-SFX pages are the whole point."""
    h, w = 200, 200
    shape = (h, w, 3)
    ono_mask = np.zeros((h, w), dtype=np.uint8)
    ono_mask[80:120, 80:120] = 255

    mask = build_inpaint_mask(shape, [], [], None, ono_mask=ono_mask)
    assert mask[100, 100] == 255
    assert mask.sum() > 0


# --------------------------------------------------------------------------- #
# ComicTextDetectorService._process_ono_mask — ch1 extraction, unclipped by
# _build_block_bounds_mask (the block-bounds clip _process_mask applies).
# --------------------------------------------------------------------------- #


def test_process_ono_mask_extracts_ch1_only():
    """``_process_ono_mask`` reads channel 1 (SFX) ALONE, ignoring channel 0
    (text) entirely — even where ch0 fires strongly and ch1 does not. Regions
    are placed far enough apart (a 100x100 canvas) that the partial-coverage
    close/dilate tuning cannot bridge one into the other."""
    from app.services.ctd_service import ComicTextDetectorService

    fake_self = types.SimpleNamespace()
    raw = np.zeros((1, 2, 100, 100), dtype=np.float32)
    raw[0, 0, 5:15, 5:15] = 0.9    # ch0 (text) fires strongly
    raw[0, 1, 60:70, 60:70] = 0.9  # ch1 (ono) fires in a DIFFERENT, far region

    ono = ComicTextDetectorService._process_ono_mask(
        fake_self, raw, (100, 100), (100, 100)
    )
    assert ono is not None
    assert ono[65, 65] == 255, "ch1 (ono) ink was not extracted"
    assert ono[10, 10] == 0, "ch0 (text) ink leaked into the ono-only mask"


def test_process_ono_mask_none_when_single_channel():
    """A legacy single-channel CTD export has no ono channel; the extractor
    must return None (not raise), so the caller safely no-ops."""
    from app.services.ctd_service import ComicTextDetectorService

    fake_self = types.SimpleNamespace()
    raw = np.zeros((1, 1, 20, 20), dtype=np.float32)
    raw[0, 0, 5:15, 5:15] = 0.9

    assert ComicTextDetectorService._process_ono_mask(
        fake_self, raw, (20, 20), (20, 20)
    ) is None


def test_process_ono_mask_thresholded_at_ono_erase_seg_threshold():
    """Ink below ``ONO_ERASE_SEG_THRESHOLD`` (0.30 — deliberately lower than,
    and separate from, the shared ``ERASE_SEG_THRESHOLD`` used by the text
    channel) is not erased; ink above it is. Isolated ink far from any other
    ink (a 100x100 canvas, blobs 40+px apart) so the gap-bridging close/dilate
    cannot manufacture coverage where the raw channel emitted none."""
    from app.services.ctd_service import ComicTextDetectorService

    assert ONO_ERASE_SEG_THRESHOLD < ERASE_SEG_THRESHOLD, (
        "ono threshold must stay decoupled from (and lower than) the shared "
        "text-channel erase threshold"
    )

    fake_self = types.SimpleNamespace()
    raw = np.zeros((2, 100, 100), dtype=np.float32)  # (C, H, W), no batch dim
    raw[1, 10:14, 10:14] = ONO_ERASE_SEG_THRESHOLD - 0.10  # below -> not erased
    raw[1, 60:64, 60:64] = ONO_ERASE_SEG_THRESHOLD + 0.10  # above -> erased

    ono = ComicTextDetectorService._process_ono_mask(
        fake_self, raw, (100, 100), (100, 100)
    )
    assert ono is not None
    assert ono[11, 11] == 0, "sub-threshold ono ink was erased"
    assert ono[61, 61] == 255, "above-threshold ono ink was not erased"


def test_process_ono_mask_unletterboxes_and_resizes_like_process_mask():
    """The padded-region crop + resize-to-original-size must mirror
    ``_process_mask``'s alignment so ono_mask lines up with the page (a
    misaligned mask would erase the wrong pixels). The close/dilate tuning
    means the two paths are no longer byte-identical (ono grows beyond a
    plain threshold), so this checks GEOMETRIC alignment instead: the ono
    ink must land inside the region ``_process_mask``'s plain-threshold path
    (no morphology) places it, and padding-region junk must never survive
    the crop regardless of which path processes it.

    Uses a page-scale (400x560) canvas rather than a tiny one: the tuned
    morphology kernels (``ONO_CLOSE_KERNEL_SIZE``/``ONO_DILATE_KERNEL_SIZE``,
    sized for real ~1280px-wide pages) would dwarf a toy 20x28 image (and a
    naively-scaled-up single ink blob can still trip the area guard below —
    see the sizing note inline). A small canvas is an artifact of the test,
    not a real bleed — ``test_process_ono_mask_drops_oversized_component``
    covers the actual area-guard behavior directly.
    """
    from app.services.ctd_service import ComicTextDetectorService

    fake_self = types.SimpleNamespace(text_threshold=0.8)
    # Simulate a letterboxed 400x400 model input where only the top-left
    # 280x200 (padded_size) holds real content; the rest is black padding.
    raw = np.zeros((1, 2, 400, 400), dtype=np.float32)
    raw[0, 1, 20:40, 20:40] = 0.9  # ono ink inside the valid (unpadded) region
    raw[0, 1, 300:380, 300:380] = 0.9  # ink IN THE PADDING region — must be cropped out

    padded_size = (280, 200)  # (pw, ph) — the valid region before letterbox pad
    orig_size = (560, 400)    # (w, h) — final page size (2x upscale)

    ono = ComicTextDetectorService._process_ono_mask(
        fake_self, raw, padded_size, orig_size
    )
    assert ono is not None
    assert ono.shape == (400, 560)  # (h, w)

    # The 2x-upscaled ink footprint (raw cols/rows 20:40 -> resized 40:80)
    # must be covered; the padding-region ink (raw rows/cols 300:380, OUTSIDE
    # the 280x200 valid padded_size) must never appear post-crop.
    assert ono[60, 60] == 255, "ono ink not aligned to the expected upscaled position"
    assert ono[399, 559] == 0, (
        "padding-region ink survived the crop (should have been discarded "
        "before resize, same as _process_mask's letterbox handling)"
    )

    ch1_only = np.zeros((1, 1, 400, 400), dtype=np.float32)
    ch1_only[0, 0] = raw[0, 1]
    text_like = ComicTextDetectorService._process_mask(
        fake_self, ch1_only, padded_size, orig_size, blocks=None,
        legacy=True, erase_threshold=ERASE_SEG_THRESHOLD,
    )
    # Every pixel the plain-threshold path erases must be a SUBSET of what the
    # (lower-threshold, close/dilate-grown) ono path erases — the tuning only
    # ever adds coverage, on the same aligned geometry, never removes it.
    assert np.array_equal(np.maximum(ono, text_like), ono), (
        "ono_mask does not superset the plain-threshold path's aligned ink "
        "— crop/resize geometry may have diverged"
    )


# --------------------------------------------------------------------------- #
# PARTIAL-COVERAGE FIX (2026-07-04): close/dilate bridges same-glyph stroke
# gaps; a per-component area guard drops runaway (art-bleed) blobs instead of
# erasing them. ono_mask has no block-bounds clip to fall back on, so this
# guard is the ono path's ONLY defense against a bridging failure.
# --------------------------------------------------------------------------- #


def test_process_ono_mask_bridges_small_gap_between_stroke_fragments():
    """Core partial-coverage fix: two small, separately-detected stroke
    fragments of the SAME glyph (a realistic ~12px gap, per the render audit)
    must merge into ONE solid connected component after the close/dilate —
    a plain threshold with no morphology would leave them as two disjoint
    islands with the gap still showing the un-erased page underneath."""
    from app.services.ctd_service import ComicTextDetectorService
    import cv2

    fake_self = types.SimpleNamespace()
    raw = np.zeros((2, 100, 100), dtype=np.float32)
    # Two 6x6 stroke fragments of "one glyph", ~12px gap between their edges.
    raw[1, 40:46, 40:46] = 0.9
    raw[1, 40:46, 58:64] = 0.9

    ono = ComicTextDetectorService._process_ono_mask(
        fake_self, raw, (100, 100), (100, 100)
    )
    assert ono is not None
    # The gap midpoint (un-detected in the raw channel) must now be bridged.
    assert ono[43, 52] == 255, (
        "small inter-fragment gap was not bridged — partial-coverage residual "
        "would survive erase"
    )
    n_labels, _ = cv2.connectedComponentsWithStats(ono, connectivity=8)[:2]
    assert n_labels - 1 == 1, "bridged fragments must form a SINGLE component"


def test_process_ono_mask_does_not_bridge_distant_unrelated_glyphs():
    """Two fragments far apart (~40px, beyond the close+dilate reach) must
    stay separate — the tuning connects strokes WITHIN a glyph, it must not
    flood-merge across genuinely distinct glyphs/words."""
    from app.services.ctd_service import ComicTextDetectorService
    import cv2

    fake_self = types.SimpleNamespace()
    raw = np.zeros((2, 100, 100), dtype=np.float32)
    raw[1, 20:26, 20:26] = 0.9
    raw[1, 20:26, 70:76] = 0.9  # ~44px gap — a different glyph/word

    ono = ComicTextDetectorService._process_ono_mask(
        fake_self, raw, (100, 100), (100, 100)
    )
    assert ono is not None
    assert ono[23, 48] == 0, "morphology bled across two unrelated glyphs"
    n_labels, _ = cv2.connectedComponentsWithStats(ono, connectivity=8)[:2]
    assert n_labels - 1 == 2, "distant fragments must remain two components"


def test_process_ono_mask_drops_oversized_component_as_art_guard():
    """A component that (after close/dilate) balloons past
    ``ONO_MAX_COMPONENT_AREA_FRAC`` of the page is DROPPED rather than
    erased — the safety net for a bridging failure into background texture,
    since this path has no block-bounds clip to fall back on."""
    from app.services.ctd_service import ComicTextDetectorService

    fake_self = types.SimpleNamespace()
    h = w = 100
    raw = np.zeros((2, h, w), dtype=np.float32)
    # A single blob covering ~25% of the page — far past the 5% guard.
    raw[1, 10:60, 10:60] = 0.9

    ono = ComicTextDetectorService._process_ono_mask(fake_self, raw, (w, h), (w, h))
    assert ono is not None
    assert ono.sum() == 0, (
        "oversized component should be dropped by the art-bleed safety net, "
        "not erased"
    )


def test_process_ono_mask_keeps_normal_sized_component_under_guard():
    """Sanity check for the guard above: a normal glyph-sized component
    (well under the area cap) must still survive the drop-if-oversized
    check — the guard must not be so tight it eats real SFX."""
    from app.services.ctd_service import ComicTextDetectorService

    fake_self = types.SimpleNamespace()
    h = w = 100
    raw = np.zeros((2, h, w), dtype=np.float32)
    raw[1, 40:50, 40:50] = 0.9  # a 10x10 glyph-sized blob (~1% of page)

    assert 10 * 10 < ONO_MAX_COMPONENT_AREA_FRAC * h * w

    ono = ComicTextDetectorService._process_ono_mask(fake_self, raw, (w, h), (w, h))
    assert ono is not None
    assert ono[45, 45] == 255, "normal-sized SFX component was dropped"


def test_process_ono_mask_close_and_dilate_kernel_sizes_are_bounded():
    """Sanity guard on the tuned constants themselves — regression check so a
    future edit can't silently make the kernels huge (uncontrolled art bleed)
    or zero (no-op, regressing back to the partial-coverage bug)."""
    assert 0 < ONO_CLOSE_KERNEL_SIZE <= 41
    assert 0 < ONO_DILATE_KERNEL_SIZE <= 21
    assert 0 < ONO_MAX_COMPONENT_AREA_FRAC <= 0.15
