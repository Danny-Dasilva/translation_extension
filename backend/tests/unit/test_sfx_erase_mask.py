"""Unit tests for FIX 1 (SFX erase).

Low-confidence SFX is dropped by the OCR-confidence gate before translation. The
dropped block must NOT be translated/rendered, but if it is real Japanese ink it
must still be ERASED (inpainted away) so raw Japanese doesn't survive into the
final page. These tests cover:

  * should_erase_dropped — which gate-dropped regions are worth erasing
  * build_inpaint_mask(..., erase_blocks=...) — erase regions keep their
    detector seg-mask ink in the mask, while the same call WITHOUT erase_blocks
    leaves those pixels at 0.
"""
from __future__ import annotations

import numpy as np

from app.utils.ctd_utils import build_inpaint_mask
from app.utils.ocr_confidence_gate import should_erase_dropped


# --- should_erase_dropped --------------------------------------------------
def test_erase_empty_text():
    # Empty low-conf crop = real ink the recognizer gave up on -> erase.
    assert should_erase_dropped("") is True
    assert should_erase_dropped("   ") is True


def test_erase_japanese_sfx():
    # Mostly-Japanese SFX scrawl -> erase.
    assert should_erase_dropped("ドドド") is True
    assert should_erase_dropped("ばーん") is True


def test_erase_japanese_with_garble():
    # Garble char present AND a JP glyph -> erase.
    assert should_erase_dropped("ド]]") is True


def test_no_erase_latin_only():
    # No Japanese glyph -> leave untouched (conservative).
    assert should_erase_dropped("abc") is False
    assert should_erase_dropped("[[ //") is False


def test_no_erase_low_jp_ratio():
    # Mostly non-JP, ratio <= 0.5, no garble char -> not worth erasing.
    assert should_erase_dropped("abcdefgあ") is False


# --- build_inpaint_mask with erase_blocks ----------------------------------
def _block(x0, y0, x1, y1):
    return {"minX": x0, "minY": y0, "maxX": x1, "maxY": y1}


def test_erase_block_retains_detector_ink():
    """Detector ink over an erase-only block is kept ONLY when erase_blocks is
    passed; without it those pixels are clipped to 0."""
    h, w = 200, 200
    image_shape = (h, w, 3)

    kept = _block(10, 10, 60, 60)        # normal kept (translated) block
    erase = _block(120, 120, 160, 160)   # dropped-but-real-JP SFX block

    # Detector seg-mask: ink inside BOTH the kept block and the erase block.
    detector_mask = np.zeros((h, w), dtype=np.uint8)
    detector_mask[20:50, 20:50] = 255    # ink over kept block
    detector_mask[130:150, 130:150] = 255  # ink over erase block

    # WITHOUT erase_blocks: erase-region ink must be clipped away.
    mask_no_erase = build_inpaint_mask(
        image_shape, [kept], text_lines=[], detector_mask=detector_mask
    )
    assert mask_no_erase[140, 140] == 0, "erase ink leaked without erase_blocks"
    assert mask_no_erase[35, 35] == 255, "kept-block ink should be erased"

    # WITH erase_blocks: erase-region ink must now be in the mask.
    mask_erase = build_inpaint_mask(
        image_shape, [kept], text_lines=[], detector_mask=detector_mask,
        erase_blocks=[erase],
    )
    assert mask_erase[140, 140] == 255, "erase ink not retained with erase_blocks"
    assert mask_erase[35, 35] == 255, "kept-block ink should still be erased"


def test_erase_block_no_full_bbox_with_detector_mask():
    """With a detector mask present, an erase block must NOT paint its whole
    bbox into the mask (only the tight ink) — corners with no ink stay 0."""
    h, w = 200, 200
    image_shape = (h, w, 3)
    kept = _block(10, 10, 60, 60)
    erase = _block(120, 120, 160, 160)

    detector_mask = np.zeros((h, w), dtype=np.uint8)
    detector_mask[20:50, 20:50] = 255
    detector_mask[130:150, 130:150] = 255  # ink only in part of erase bbox

    mask = build_inpaint_mask(
        image_shape, [kept], text_lines=[], detector_mask=detector_mask,
        erase_blocks=[erase],
    )
    # A corner of the erase bbox with no detector ink stays unmasked.
    assert mask[122, 122] == 0, "erase block painted a full-bbox rectangle"


def test_small_erase_block_bbox_fallback_no_detector_mask():
    """No detector mask + SMALL erase block -> fall back to bbox fill so the
    SFX is still erased."""
    h, w = 200, 200
    image_shape = (h, w, 3)
    kept = _block(10, 10, 60, 60)
    erase = _block(120, 120, 150, 150)  # area 900 <= 9000 (small)

    mask = build_inpaint_mask(
        image_shape, [kept], text_lines=[], detector_mask=None,
        erase_blocks=[erase],
    )
    assert mask[135, 135] == 255, "small erase block not bbox-filled"


def test_large_erase_block_no_bbox_fallback_no_detector_mask():
    """No detector mask + LARGE erase block -> NO bbox fill (avoid rectangular
    ghosting over art)."""
    h, w = 400, 400
    image_shape = (h, w, 3)
    kept = _block(10, 10, 60, 60)
    erase = _block(100, 100, 300, 300)  # area 40000 > 9000 (large)

    mask = build_inpaint_mask(
        image_shape, [kept], text_lines=[], detector_mask=None,
        erase_blocks=[erase],
    )
    assert mask[200, 200] == 0, "large erase block should not be bbox-filled"


def test_erase_blocks_default_behavior_unchanged():
    """Calling without erase_blocks behaves exactly as before."""
    h, w = 200, 200
    image_shape = (h, w, 3)
    kept = _block(10, 10, 60, 60)
    detector_mask = np.zeros((h, w), dtype=np.uint8)
    detector_mask[20:50, 20:50] = 255

    a = build_inpaint_mask(image_shape, [kept], [], detector_mask)
    b = build_inpaint_mask(image_shape, [kept], [], detector_mask, erase_blocks=[])
    assert np.array_equal(a, b)
