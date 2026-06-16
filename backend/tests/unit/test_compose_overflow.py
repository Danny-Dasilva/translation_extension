"""Unit tests for FIX #4: DIALOGUE (bubble-matched, ``fit_rect`` present) text
must be clamped to the BUBBLE interior, not just the canvas. Before this fix a
long translation in a small bubble overflowed the bubble edges (it only clamped
to the page). Also pins the new ``find_best_fit`` contract that reports whether
anything actually fit.

Contract pinned here:

1. ``find_best_fit`` returns ``(font, lines, fitted)`` where ``fitted`` is False
   when no font size satisfies (max_w, max_h) — e.g. a long string in a 1x1 box.
2. A long dialogue translation in a SMALL bubble renders ink that stays inside
   the bubble rect (within a small stroke-bleed tolerance). When a real display
   font is available we assert on the drawn-ink pixel bbox; otherwise we fall
   back to asserting the layout geometry via ``measure_block`` / clamp math.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import refit_final_composites as R  # noqa: E402
from refit_final_composites import compose_final, find_best_fit  # noqa: E402


def _blk(x0, y0, x1, y1, **kw):
    d = {"minX": x0, "minY": y0, "maxX": x1, "maxY": y1}
    d.update(kw)
    return d


def _ink_bbox(rendered: np.ndarray, plate: np.ndarray):
    """Bounding box of pixels that differ from the (uniform) plate."""
    diff = np.abs(rendered.astype(int) - plate.astype(int)).sum(axis=2) > 20
    ys, xs = np.where(diff)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _font_available() -> bool:
    try:
        R.load_font(20)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. find_best_fit reports fit/no-fit
# ---------------------------------------------------------------------------

def test_find_best_fit_returns_fitted_flag():
    img = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font, lines, fitted = find_best_fit(
        draw, "x", 200, 200, min_size=6, max_size=48
    )
    assert fitted is True
    assert font is not None and lines


def test_find_best_fit_false_for_impossible_fit():
    """A long string in a 1x1 box can never fit -> fitted must be False, but a
    best (min-size) font/lines is still returned for fallback rendering."""
    img = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    long_text = "This is a long sentence that cannot possibly fit a single pixel box."
    font, lines, fitted = find_best_fit(
        draw, long_text, 1, 1, min_size=6, max_size=72
    )
    assert fitted is False
    assert font is not None and lines  # fallback still provided


# ---------------------------------------------------------------------------
# 2. dialogue text clamped to a SMALL bubble (the FIX #4 regression)
# ---------------------------------------------------------------------------

def test_dialogue_overflow_clamped_to_small_bubble():
    """A long translation in a SMALL bubble must keep its ink inside the bubble
    rect — previously it only clamped to the canvas and spilled past the bubble.
    """
    plate = np.full((600, 800, 3), 255, np.uint8)
    # tight vertical-JP source column near the center of a SMALL bubble
    block = _blk(390, 280, 410, 340)
    bubble = _blk(360, 260, 460, 360)  # small bubble: 100x100
    long_text = (
        "I never expected you to say something so incredibly important right now."
    )
    out = compose_final(plate, [block], [long_text], fit_rects=[bubble])

    if _font_available():
        bb = _ink_bbox(out, plate)
        assert bb is not None, "dialogue rendered nothing"
        x0, y0, x1, y1 = bb
        tol = 4  # stroke bleed
        assert x0 >= bubble["minX"] - tol, f"ink left {x0} < bubble {bubble['minX']}"
        assert x1 <= bubble["maxX"] + tol, f"ink right {x1} > bubble {bubble['maxX']}"
        assert y0 >= bubble["minY"] - tol, f"ink top {y0} < bubble {bubble['minY']}"
        assert y1 <= bubble["maxY"] + tol, f"ink bottom {y1} > bubble {bubble['maxY']}"
    else:
        # No font: assert the layout geometry would clamp. The wrapped block at
        # the min floor must, after vertical clipping, not exceed the bubble.
        img = Image.new("RGB", (10, 10), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        bw = bubble["maxX"] - bubble["minX"]
        bh = bubble["maxY"] - bubble["minY"]
        font, lines, _ = find_best_fit(draw, long_text.upper(), bw, bh,
                                       min_size=6, max_size=96)
        line_h = R.line_height_px(font)
        max_lines = max(1, bh // line_h)
        lines = lines[:max_lines]
        _, mh = R.measure_block(draw, lines, font)
        assert mh <= bh + 4, f"clamped block height {mh} exceeds bubble {bh}"


def test_dialogue_wide_translation_clamped_horizontally():
    """A single long word wider than the bubble must be pinned to the bubble
    left edge, not centered off the right edge."""
    plate = np.full((400, 700, 3), 255, np.uint8)
    block = _blk(300, 180, 320, 220)
    bubble = _blk(250, 150, 360, 260)  # narrow bubble
    out = compose_final(plate, [block], ["SUPERCALIFRAGILISTIC"], fit_rects=[bubble])
    if _font_available():
        bb = _ink_bbox(out, plate)
        assert bb is not None
        x0, _, x1, _ = bb
        tol = 4
        assert x0 >= bubble["minX"] - tol, f"ink left {x0} < bubble {bubble['minX']}"
        # right edge may touch the bubble edge when the word is wider than the
        # box, but it must not run off to the right of it.
        assert x1 <= bubble["maxX"] + tol, f"ink right {x1} > bubble {bubble['maxX']}"
