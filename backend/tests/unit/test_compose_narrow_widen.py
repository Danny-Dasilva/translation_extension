"""Unit tests for the NARROW-NARRATION bounded horizontal widen in
``scripts/refit_final_composites.py::compose_final``.

Tall-narrow CLAMPED (no-bubble) *caption* columns come from vertical-JP text and
otherwise cram horizontal EN one-word-per-line down to the hard floor. The fix
grants a BOUNDED horizontal widen onto the clean inpaint plate for these (never
SFX/orphan boxes), bounded by image edges + already-placed rects, and allows
modest vertical overflow rather than shrinking below the readable floor.

Contract pinned here:

1. A tall-narrow non-SFX caption renders ink WIDER than its original column
   (the widen engaged) while staying inside the image bounds.
2. Such a caption renders at/above the resolution-aware readability floor
   (no longer collapses to the clamped hard floor).
3. SFX-sized / orphan tall-narrow boxes are NOT widened (still clamped to bbox).
4. A non-narrow (wide/normal-aspect) caption is unaffected (no widen).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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


def _ink_bbox(rendered: np.ndarray, plate: np.ndarray):
    diff = np.abs(rendered.astype(int) - plate.astype(int)).sum(axis=2) > 20
    ys, xs = np.where(diff)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# A tall-narrow non-SFX caption column: short side 100px (> 48) and area 50k
# (> 9k) so it is NOT SFX-sized; aspect h/w = 5.0 (> NARROW_WIDEN_TRIGGER).
_CAPTION = (560, 200, 660, 700)
_CAPTION_TEXT = "She had finally understood the weight of his quiet resolve."


def test_narrow_caption_ink_widens_beyond_column():
    if not _font_available():
        return
    img_h, img_w = 1000, 1200
    plate = np.full((img_h, img_w, 3), 255, np.uint8)
    block = _blk(*_CAPTION)
    out = compose_final(plate, [block], [_CAPTION_TEXT], fit_rects=[None])
    bb = _ink_bbox(out, plate)
    assert bb is not None, "caption rendered nothing"
    x0, y0, x1, y1 = bb
    col_w = _CAPTION[2] - _CAPTION[0]
    # the widen must produce ink wider than the original tight column
    assert (x1 - x0) > col_w, (
        f"ink width {x1 - x0} did not exceed original column {col_w} (no widen)"
    )
    # ...but never past the image bounds
    assert x0 >= 0 and x1 <= img_w, f"widened ink {x0}-{x1} left image bounds"


def test_narrow_caption_renders_at_or_above_readability_floor():
    if not _font_available():
        return
    img_h, img_w = 1000, 1200
    floor = R.resolution_font_floor(img_h)
    plate = np.full((img_h, img_w, 3), 255, np.uint8)
    block = _blk(*_CAPTION)
    sizes = compose_final(plate, [block], [_CAPTION_TEXT], fit_rects=[None],
                          _debug_sizes=True)
    assert sizes[0] is not None, "caption suppressed unexpectedly"
    assert sizes[0] >= floor, (
        f"widened narrow caption rendered at {sizes[0]}px below floor {floor}px"
    )


def test_sfx_sized_tall_narrow_box_not_widened():
    """A tall-narrow but SFX-sized (short side <= 48) box is excluded from the
    widen and stays clamped inside its bbox in x."""
    if not _font_available():
        return
    img_h, img_w = 1000, 1200
    plate = np.full((img_h, img_w, 3), 255, np.uint8)
    sfx = _blk(560, 200, 600, 600, orphan=True)  # w=40 (<=48) -> SFX-sized
    out = compose_final(plate, [sfx], ["GASP"], fit_rects=[None])
    bb = _ink_bbox(out, plate)
    assert bb is not None
    x0, _, x1, _ = bb
    assert x0 >= sfx["minX"] - 3 and x1 <= sfx["maxX"] + 3, (
        f"SFX box widened: ink x[{x0}-{x1}] vs box x[{sfx['minX']}-{sfx['maxX']}]"
    )


def test_wide_aspect_caption_not_widened():
    """A normal/wide-aspect caption (aspect below the trigger) is untouched —
    its ink stays inside the bbox in x as before."""
    if not _font_available():
        return
    img_h, img_w = 600, 800
    plate = np.full((img_h, img_w, 3), 255, np.uint8)
    # w=240 h=160 -> aspect 0.67, well below NARROW_WIDEN_TRIGGER
    wide = _blk(60, 60, 300, 220)
    out = compose_final(plate, [wide], ["He had become nothing more than a nuisance."],
                        fit_rects=[None])
    bb = _ink_bbox(out, plate)
    assert bb is not None
    x0, _, x1, _ = bb
    assert x0 >= wide["minX"] - 3 and x1 <= wide["maxX"] + 3, (
        f"wide caption widened unexpectedly: ink x[{x0}-{x1}]"
    )


def test_widen_constants_are_conservative():
    assert R.NARROW_WIDEN_TRIGGER >= 2.0, "trigger too aggressive (would catch normal columns)"
    assert R.NARROW_WIDEN_MAX_GROWTH <= 3.0, "max growth too large to be conservative"
    assert R.NARROW_WIDEN_TARGET_ASPECT >= 1.5, "target aspect too wide"
