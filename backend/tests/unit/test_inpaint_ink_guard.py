"""Unit tests for the contrast/ink mask refinement (Phase-0 over-erase fix).

The Phase-0 mask-recall change (``ctd_utils.build_inpaint_mask`` +
``ERASE_SEG_THRESHOLD`` = 0.45) builds the erase mask from ALL detected text ink
— every detected line bbox (padded + glyph-proportional dilation) plus the
low-threshold seg. On DARK / TEXTURED pages the detector + low seg fire on
false-positive "text" in screentone, sweat-lines, panel gutters and shadows, and
the padded/dilated boxes sweep in the dark ARTWORK around the strokes. The
AREA-only ``_mask_box_too_broad`` clamp cannot catch this — the bad pixels live
INSIDE plausible boxes, and components merge bright text with dark art so a
per-component test passes the whole blob. LaMa then smears the art (confirmed on
Furube p020 dark screentone panel, p035 torso/hair/gutters).

``_refine_mask_to_text_ink`` is a LOCAL (per-pixel) refinement: a masked pixel
is kept only where its neighbourhood actually looks like text ink — a BRIGHT
background/glyph fill is present AND there is enough local dynamic range. Real
glyphs (black-on-white, white-on-dark, white-haloed over art) sit next to a
bright reference and are kept, so the genuinely-dropped JP the recall targets
stays erased; the flat/dark false positives are dropped and preserved as art.

Tests pin the heuristic on the helper (both ink polarities, every reject class)
and the end-to-end effect through ``inpaint()`` with the non-neural service (no
ONNX model is loaded): a dark/smooth region is NOT erased; genuine glyph ink IS.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.services.lama_inpaint_service import (
    _INK_BRIGHT_MIN,
    _INK_CONTRAST_MIN,
    _INK_REFINE_WIN,
    LamaInpaintService,
    _refine_mask_to_text_ink,
)


# --------------------------------------------------------------------------- #
# helpers — synthetic regions placed on a canvas
# --------------------------------------------------------------------------- #


def _canvas(size, val):
    return np.full((size, size, 3), val, dtype=np.uint8)


def _glyph(bg, ink, size=60):
    """A text-ink-like patch: ``bg`` background with thin ``ink`` strokes (sharp
    high-contrast edges). Works for black-on-white (bg=250, ink=15) and
    white-on-dark (bg=20, ink=235)."""
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    c = (int(ink), int(ink), int(ink))
    for x in range(8, size - 6, 10):
        cv2.line(img, (x, 8), (x, size - 8), c, 2)
    for y in range(14, size - 6, 16):
        cv2.line(img, (8, y), (size - 8, y), c, 1)
    return img


def _dark_textured(size=60, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(15, 70, size=(size, size), dtype=np.int16).astype(np.uint8)
    return np.stack([g, g, g], axis=2)


def _place(canvas, patch, x0, y0):
    h, w = patch.shape[:2]
    canvas[y0:y0 + h, x0:x0 + w] = patch


def _solid_mask(shape, box):
    m = np.zeros(shape[:2], dtype=np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 255
    return m


# --------------------------------------------------------------------------- #
# _refine_mask_to_text_ink — keep glyph ink, drop flat/dark art
# --------------------------------------------------------------------------- #


def test_refine_keeps_black_on_white_glyph():
    """Black strokes on white = canonical dialogue. The stroke pixels stay
    masked (still erased); nothing in the glyph region is fully dropped."""
    img = _canvas(220, 250)
    _place(img, _glyph(250, 15, 60), 40, 40)
    mask = _solid_mask(img.shape, (40, 40, 100, 100))

    refined = _refine_mask_to_text_ink(img, mask)
    assert refined.any(), "glyph region was wholly dropped"
    # A pixel sitting on a stroke is retained.
    ys, xs = np.where(cv2.cvtColor(img[40:100, 40:100], cv2.COLOR_RGB2GRAY) < 60)
    sy, sx = int(ys[len(ys) // 2]) + 40, int(xs[len(xs) // 2]) + 40
    assert refined[sy, sx] == 255, "a glyph stroke pixel was dropped"


def test_refine_keeps_white_on_dark_glyph():
    """White strokes on dark = laughter / SFX (the p035 recall class). The bright
    GLYPH FILL satisfies the bright-reference gate though the bg is dark."""
    img = _canvas(220, 20)
    _place(img, _glyph(20, 235, 60), 40, 40)
    mask = _solid_mask(img.shape, (40, 40, 100, 100))

    refined = _refine_mask_to_text_ink(img, mask)
    assert refined.any(), "white-on-dark glyph region was wholly dropped"
    ys, xs = np.where(cv2.cvtColor(img[40:100, 40:100], cv2.COLOR_RGB2GRAY) > 180)
    sy, sx = int(ys[len(ys) // 2]) + 40, int(xs[len(xs) // 2]) + 40
    assert refined[sy, sx] == 255, "a white stroke pixel was dropped"


def test_refine_drops_flat_dark_region():
    """A uniform dark gutter/shadow has no bright reference → fully dropped (the
    confirmed smear class)."""
    img = _canvas(220, 40)  # whole page dark
    mask = _solid_mask(img.shape, (60, 60, 160, 160))
    assert _refine_mask_to_text_ink(img, mask).sum() == 0


def test_refine_drops_dark_textured_screentone():
    """A dark, TEXTURED screentone/shadow (has local contrast from noise, but no
    bright reference) → dropped on the brightness gate."""
    img = _canvas(220, 30)
    _place(img, _dark_textured(120, seed=3), 50, 50)
    mask = _solid_mask(img.shape, (70, 70, 150, 150))  # interior of the texture
    assert _refine_mask_to_text_ink(img, mask).sum() == 0


def test_refine_drops_flat_bright_region():
    """A featureless near-white wash (no strokes) lacks dynamic range → dropped
    (harmless: nothing to erase, and skipping it avoids needless inpaint)."""
    img = _canvas(220, 250)
    mask = _solid_mask(img.shape, (60, 60, 160, 160))
    assert _refine_mask_to_text_ink(img, mask).sum() == 0


def test_refine_empty_mask_is_noop():
    img = _canvas(60, 128)
    mask = np.zeros((60, 60), dtype=np.uint8)
    assert _refine_mask_to_text_ink(img, mask).sum() == 0


def test_refine_constants_separate_dark_art_from_grey_text():
    """The brightness floor sits above the dark-screentone ceiling and below a
    grey dialogue background, so black-on-grey text is kept but deep-dark art is
    dropped; the window spans bold strokes."""
    assert 100 <= _INK_BRIGHT_MIN <= 130
    assert 50 <= _INK_CONTRAST_MIN <= 90
    assert _INK_REFINE_WIN >= 15 and _INK_REFINE_WIN % 2 == 1


# --------------------------------------------------------------------------- #
# end-to-end through inpaint() — non-neural service (no ONNX model loaded)
# --------------------------------------------------------------------------- #


def _service():
    # use_neural=False -> __init__ returns early, loads nothing (no GPU / ONNX).
    return LamaInpaintService(use_neural=False)


def test_inpaint_preserves_dark_art_component():
    """A mask over a deep dark, textured region (no text ink) is left INTACT:
    the refinement drops it from the mask so LaMa never smears the art. The
    region is byte-identical to the input after inpaint()."""
    svc = _service()
    img = _canvas(260, 30)  # dark page so the masked centre has no bright nearby
    _place(img, _dark_textured(160, seed=4), 50, 50)
    before = img.copy()
    mask = _solid_mask(img.shape, (90, 90, 170, 170))  # deep interior

    out = svc.inpaint(img, mask)
    assert np.array_equal(out[50:210, 50:210], before[50:210, 50:210])


def test_inpaint_erases_black_on_white_glyph():
    """Genuine black-on-white text IS erased (recall preserved): the masked
    strokes are replaced by the bright background."""
    svc = _service()
    img = _canvas(220, 250)
    _place(img, _glyph(250, 15, 60), 40, 40)
    before = img.copy()
    mask = _solid_mask(img.shape, (40, 40, 100, 100))

    out = svc.inpaint(img, mask)
    assert out[40:100, 40:100].mean() > before[40:100, 40:100].mean()
    assert float(out[40:100, 40:100].min()) > 200, "ink strokes were not erased"


def test_inpaint_erases_white_on_dark_glyph():
    """White-on-dark text (the p035 laughter recall class) IS erased: the bright
    glyph fill passes the refinement, then the dark-aware fill removes it."""
    svc = _service()
    img = _canvas(220, 25)
    _place(img, _glyph(25, 235, 60), 40, 40)
    before_white = int((cv2.cvtColor(img[40:100, 40:100], cv2.COLOR_RGB2GRAY) > 180).sum())
    assert before_white > 0
    mask = _solid_mask(img.shape, (40, 40, 100, 100))

    out = svc.inpaint(img, mask)
    after_white = int((cv2.cvtColor(out[40:100, 40:100], cv2.COLOR_RGB2GRAY) > 180).sum())
    assert after_white < before_white, "white strokes were not erased"


def test_inpaint_mixed_page_erases_text_keeps_art():
    """A page with BOTH a genuine glyph region and a deep dark art region: only
    the text is erased; the dark art is preserved. Precision without losing
    recall — the invariant the fix targets."""
    svc = _service()
    img = _canvas(320, 250)
    _place(img, _glyph(250, 15, 60), 30, 30)            # real text (bright page)
    _place(img, _canvas(120, 22), 180, 180)            # deep dark art block
    _place(img, _dark_textured(120, seed=5), 180, 180)
    art_before = img[180:300, 180:300].copy()

    mask = np.zeros((320, 320), dtype=np.uint8)
    mask[30:90, 30:90] = 255       # over the text
    mask[210:270, 210:270] = 255   # deep inside the dark art

    out = svc.inpaint(img, mask)
    assert float(out[30:90, 30:90].min()) > 200, "text not erased"
    assert np.array_equal(out[180:300, 180:300], art_before), "art was smeared"
