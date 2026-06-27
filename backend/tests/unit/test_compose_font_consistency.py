"""Unit tests for the readability FLOOR + page-level font CONSISTENCY policy in
``scripts/refit_final_composites.py``.

Contract pinned here:

1. Resolution-aware readability floor: ``resolution_font_floor(img_h)`` returns
   ``max(ABS_FLOOR, round(img_h * FLOOR_FRAC))`` and is monotonic in image
   height. On a full-resolution manga page (~1791px) the floor is well above the
   absolute floor; on a tiny page it falls back to the absolute floor.

2. No dialogue bubble on a page renders below the resolution-aware floor. A long
   translation in a small bubble is wrapped / allowed modest overflow instead of
   shrinking to illegibility.

3. Page-level consistency: ``page_dialogue_target(...)`` drives same-class
   (dialogue) blocks toward ONE shared size — a low percentile of the per-bubble
   max-fit sizes, never below the floor — so the spread of rendered dialogue font
   sizes tightens. With consistency ON the rendered dialogue sizes on a page span
   a much narrower range than with it OFF.

4. SFX/caption (clamped, no-bubble) blocks stay on their own independent track
   (not pulled to the dialogue target).
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
from refit_final_composites import compose_final  # noqa: E402


def _blk(x0, y0, x1, y1, **kw):
    d = {"minX": x0, "minY": y0, "maxX": x1, "maxY": y1}
    d.update(kw)
    return d


def _font_available() -> bool:
    try:
        R.load_font(20)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. resolution-aware floor
# ---------------------------------------------------------------------------

def test_resolution_font_floor_scales_with_height():
    floor_small = R.resolution_font_floor(400)
    floor_big = R.resolution_font_floor(1791)
    # absolute floor dominates on tiny pages, fraction dominates at manga res
    assert floor_small == R.ABS_FONT_FLOOR
    assert floor_big > floor_small
    assert floor_big == max(R.ABS_FONT_FLOOR, round(1791 * R.FONT_FLOOR_FRAC))


def test_resolution_font_floor_monotonic():
    heights = [200, 500, 1000, 1500, 2000]
    floors = [R.resolution_font_floor(h) for h in heights]
    assert floors == sorted(floors), f"floor not monotonic: {floors}"


def test_floor_constants_are_sane():
    assert 12 <= R.ABS_FONT_FLOOR <= 30
    assert 0.005 <= R.FONT_FLOOR_FRAC <= 0.03


# ---------------------------------------------------------------------------
# 2. dialogue never rendered below the floor
# ---------------------------------------------------------------------------

def test_dialogue_never_below_resolution_floor():
    """Several dialogue bubbles of varying tightness on one tall page: none of
    the rendered dialogue font sizes may fall below the resolution-aware floor.
    """
    if not _font_available():
        return
    img_h = 1600
    floor = R.resolution_font_floor(img_h)
    plate = np.full((img_h, 1200, 3), 255, np.uint8)
    # mix of roomy and cramped bubbles with long-ish translations
    blocks, fits, texts = [], [], []
    specs = [
        ((100, 100, 360, 320), "This is a normal length line of dialogue."),
        ((500, 120, 640, 360), "A cramped narrow bubble with a long sentence here."),
        ((700, 200, 1100, 520), "Plenty of room for this one to breathe nicely."),
    ]
    for (x0, y0, x1, y1), t in specs:
        b = _blk(x0, y0, x1, y1)
        blocks.append(b)
        fits.append(_blk(x0, y0, x1, y1))
        texts.append(t)
    sizes = R.compose_final(plate, blocks, texts, fit_rects=fits,
                            _debug_sizes=True)
    dialogue_sizes = [s for s in sizes if s is not None]
    assert dialogue_sizes, "no dialogue rendered"
    assert min(dialogue_sizes) >= floor, (
        f"a dialogue bubble rendered at {min(dialogue_sizes)}px below floor {floor}px"
    )


# ---------------------------------------------------------------------------
# 3. page-level consistency tightens the spread
# ---------------------------------------------------------------------------

def test_page_dialogue_target_is_low_percentile_not_min():
    maxfits = [40, 42, 44, 20, 46, 48, 50]
    floor = 18
    target = R.page_dialogue_target(maxfits, floor,
                                    percentile=R.CONSISTENT_FONT_PERCENTILE)
    assert target >= floor
    # a low percentile sits below the median but ABOVE the single tiny outlier
    assert target > min(maxfits), "target collapsed to the min outlier"
    assert target <= sorted(maxfits)[len(maxfits) // 2], "target above the median"


def test_page_dialogue_target_respects_floor():
    maxfits = [10, 11, 9, 12]  # all below floor (tiny cramped bubbles)
    floor = 20
    target = R.page_dialogue_target(maxfits, floor, percentile=35)
    assert target == floor


def test_consistency_tightens_dialogue_spread():
    """Same page rendered with consistency ON vs OFF: the spread (max-min) of
    rendered DIALOGUE font sizes must be smaller (or equal) with it ON."""
    if not _font_available():
        return
    img_h = 1600
    plate = np.full((img_h, 1400, 3), 255, np.uint8)
    # bubbles whose independent max-fit sizes would vary a lot:
    #  - a huge bubble with a tiny word would fit huge
    #  - cramped bubbles fit small
    specs = [
        ((100, 100, 700, 600), "HI"),                       # huge box, tiny text
        ((100, 700, 300, 1000), "A longer sentence here please."),
        ((800, 100, 1000, 400), "Another fairly long dialogue line for size."),
        ((800, 500, 1300, 1100), "Yes."),                   # big box short text
    ]
    blocks = [_blk(*s[0]) for s in specs]
    fits = [_blk(*s[0]) for s in specs]
    texts = [s[1] for s in specs]

    sizes_off = [s for s in R.compose_final(
        plate, blocks, texts, fit_rects=fits,
        consistent_font=False, _debug_sizes=True) if s is not None]
    sizes_on = [s for s in R.compose_final(
        plate, blocks, texts, fit_rects=fits,
        consistent_font=True, _debug_sizes=True) if s is not None]

    spread_off = max(sizes_off) - min(sizes_off)
    spread_on = max(sizes_on) - min(sizes_on)
    assert spread_on <= spread_off, (
        f"consistency did not tighten spread: on={spread_on} off={spread_off} "
        f"(sizes_on={sizes_on}, sizes_off={sizes_off})"
    )
    # and it should actually be a meaningful tightening for this adversarial page
    assert spread_on < spread_off, "expected a strictly tighter spread"


# ---------------------------------------------------------------------------
# 4. clamped (SFX/caption) blocks stay on their own track
# ---------------------------------------------------------------------------

def test_clamped_blocks_not_pulled_to_dialogue_target():
    """A clamped caption keeps its independently-fit size; turning page
    consistency on/off must not change a clamped block's size."""
    if not _font_available():
        return
    img_h = 1600
    plate = np.full((img_h, 1200, 3), 255, np.uint8)
    blocks = [
        _blk(100, 100, 360, 320),                       # dialogue
        _blk(800, 800, 1000, 1100),                     # clamped caption
    ]
    fits = [_blk(100, 100, 360, 320), None]
    texts = ["A normal dialogue line goes here.", "A clamped side caption."]

    sizes_on = R.compose_final(plate, blocks, texts, fit_rects=fits,
                               consistent_font=True, _debug_sizes=True)
    sizes_off = R.compose_final(plate, blocks, texts, fit_rects=fits,
                                consistent_font=False, _debug_sizes=True)
    # index 1 is the clamped caption — its size is independent of the policy
    assert sizes_on[1] == sizes_off[1], (
        f"clamped block size changed with consistency policy: "
        f"on={sizes_on[1]} off={sizes_off[1]}"
    )
