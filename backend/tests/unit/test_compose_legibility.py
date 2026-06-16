"""Unit tests for FIX #5: caption legibility — minimum font floor, robust
luminance-based contrast, and a heavier stroke.

These pin the renderer contract in
``scripts/refit_final_composites.py``:

1. ``sample_bg_luminance`` returns ``(median_luma, dark_fraction)``. A
   predominantly-dark crop (even with a few bright specks) yields a low median
   and/or high dark_fraction, which the color-selection logic must resolve to
   WHITE text; a light crop yields BLACK text.
2. A clamped (no-bubble) caption of normal length is never rendered below the
   hard font floor (9px).
3. The stroke-width formula produces the heavier FIX #5 value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import refit_final_composites as R  # noqa: E402
from refit_final_composites import sample_bg_luminance  # noqa: E402


# --- mirrors of the in-renderer decision logic (kept in lockstep) ----------
HARD_FLOOR = 9
MIN_FLOOR = 14


def _pick_color(bg_median: float, dark_fraction: float):
    """Replicates the FIX #5 auto-contrast branch in compose_final."""
    if bg_median < 140 or dark_fraction > 0.35:
        return (255, 255, 255), (0, 0, 0)  # WHITE fill, black stroke
    return (0, 0, 0), (255, 255, 255)  # BLACK fill, white stroke


def _stroke_w(font_size: int) -> int:
    """Replicates the FIX #5 stroke-width formula."""
    return max(3, min(8, round(font_size * 0.14)))


# ---------------------------------------------------------------------------
# 1. luminance stats + color selection
# ---------------------------------------------------------------------------

def test_dark_bg_picks_white_text():
    """Wet/dark art: mostly-dark crop with a few bright specks must still
    yield a dark median / high dark_fraction → WHITE text."""
    arr = np.full((40, 40, 3), 20, dtype=np.uint8)  # near-black
    # sprinkle a few bright specks (the case mean-luminance got wrong)
    arr[0:3, 0:3] = 255
    median, dark_frac = sample_bg_luminance(arr, 0, 0, 40, 40)
    assert median < 140, f"expected dark median, got {median}"
    assert dark_frac > 0.35, f"expected high dark_fraction, got {dark_frac}"
    fill, stroke = _pick_color(median, dark_frac)
    assert fill == (255, 255, 255), "dark bg must use WHITE text"
    assert stroke == (0, 0, 0), "dark bg must use black stroke"


def test_light_bg_picks_black_text():
    """A bright background yields a high median / low dark_fraction → BLACK text."""
    arr = np.full((40, 40, 3), 235, dtype=np.uint8)  # near-white
    median, dark_frac = sample_bg_luminance(arr, 0, 0, 40, 40)
    assert median >= 140, f"expected light median, got {median}"
    assert dark_frac <= 0.35, f"expected low dark_fraction, got {dark_frac}"
    fill, stroke = _pick_color(median, dark_frac)
    assert fill == (0, 0, 0), "light bg must use BLACK text"
    assert stroke == (255, 255, 255), "light bg must use white stroke"


def test_sample_bg_luminance_degenerate_returns_tuple():
    """Empty/degenerate rect returns the (255.0, 0.0) sentinel tuple."""
    arr = np.full((10, 10, 3), 128, dtype=np.uint8)
    assert sample_bg_luminance(arr, 5, 5, 5, 5) == (255.0, 0.0)
    assert sample_bg_luminance(arr, 8, 8, 2, 2) == (255.0, 0.0)


# ---------------------------------------------------------------------------
# 2. clamped caption font floor
# ---------------------------------------------------------------------------

def test_clamped_caption_never_below_hard_floor():
    """A normal-length caption fit into a clamped block must not render below
    the hard floor (9px), even when the box is small."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    text = "A short caption."

    # A modest box: small enough to exercise the floor, large enough that the
    # text doesn't need sub-9px to fit. find_best_fit falls back to min_size
    # when nothing fits, so retrying at hard_floor never goes below 9px.
    font, lines, _fitted = R.find_best_fit(
        draw, text, 120, 80, R.DEFAULT_FONT_PATH,
        min_size=HARD_FLOOR, max_size=96,
    )
    assert font.size >= HARD_FLOOR, (
        f"clamped caption font {font.size}px below hard floor {HARD_FLOOR}px"
    )


def test_floor_constants_match_renderer():
    """Guard the floor values used in compose_final against regression."""
    src = (_SCRIPTS / "refit_final_composites.py").read_text()
    assert "min_floor = 14" in src, "soft floor must be 14px"
    assert "hard_floor = 9" in src, "hard floor must be 9px"


# ---------------------------------------------------------------------------
# 3. stroke width
# ---------------------------------------------------------------------------

def test_stroke_formula_is_heavier():
    """The FIX #5 formula (max(3, min(8, round(size*0.14)))) yields a heavier
    stroke than the old max(2, min(5, round(size*0.10)))."""
    def old(size):
        return max(2, min(5, round(size * 0.10)))

    for size in (14, 20, 30, 48, 96):
        new = _stroke_w(size)
        assert new >= old(size), f"size {size}: new {new} not >= old {old(size)}"
    # spot-check exact values
    assert _stroke_w(14) == 3   # round(1.96)=2 -> floored to 3
    assert _stroke_w(30) == 4   # round(4.2)=4
    assert _stroke_w(48) == 7   # round(6.72)=7
    assert _stroke_w(96) == 8   # round(13.44)=13 -> capped to 8


def test_stroke_formula_matches_renderer():
    """Guard the stroke formula in compose_final against regression."""
    src = (_SCRIPTS / "refit_final_composites.py").read_text()
    assert "max(3, min(8, round(font.size * 0.14)))" in src
