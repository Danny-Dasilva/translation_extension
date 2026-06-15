"""Unit tests for FIX A: compose_final must keep ``fit_rect is None`` blocks
(captions / orphan / SFX over art) strictly inside their own bbox and must not
overlap an already-placed block.

These pin the renderer contract added to
``scripts/refit_final_composites.py::compose_final``:

1. A ``fit_rect is None`` block's rendered text is clamped to the BLOCK bbox
   (x0..x1, y0..y1), never the canvas. A long multi-word caption that overflows
   its box is shrunk-to-fit / wrapped to the bbox width, and if it still does
   not fit at the min floor it is clipped/truncated rather than spilling.
2. Inter-block overlap avoidance: two captions whose naive renders would
   collide do not produce overlapping ink rects; an orphan/SFX block that would
   land on top of a DIALOGUE block is suppressed (not rendered) instead.
3. Verbose SFX glosses in tiny boxes are truncated to onomatopoeia length.
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


def _ink_bbox(rendered: np.ndarray, plate: np.ndarray):
    """Bounding box of pixels that differ from the (uniform) plate."""
    diff = np.abs(rendered.astype(int) - plate.astype(int)).sum(axis=2) > 20
    ys, xs = np.where(diff)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ---------------------------------------------------------------------------
# 1. caption clamp to bbox
# ---------------------------------------------------------------------------

def test_caption_text_stays_within_block_bbox_x():
    """A long caption in a narrow box must not render ink outside [x0..x1]."""
    plate = np.full((400, 600, 3), 255, np.uint8)
    block = _blk(40, 40, 200, 360)  # narrow tall caption column
    text = ["That sense of superiority led to a great deal of confidence."]
    out = compose_final(plate, [block], text, fit_rects=[None])
    bb = _ink_bbox(out, plate)
    assert bb is not None, "caption rendered nothing"
    x0, y0, x1, y1 = bb
    # allow a tiny stroke bleed margin
    assert x0 >= block["minX"] - 3, f"ink left {x0} < bbox left {block['minX']}"
    assert x1 <= block["maxX"] + 3, f"ink right {x1} > bbox right {block['maxX']}"


def test_tiny_box_verbose_text_does_not_overflow_bbox():
    """The hard case (052 idx7): a verbose gloss in a tiny SFX box must NOT
    render ink outside the box in x or y — current behaviour overflows both."""
    plate = np.full((500, 600, 3), 255, np.uint8)
    block = _blk(100, 100, 140, 260, orphan=True)
    text = ["SFX FOR A MOMENTARY SHOCK OR SURPRISE LIKE A GASP OF AIR"]
    out = compose_final(plate, [block], text, fit_rects=[None])
    bb = _ink_bbox(out, plate)
    assert bb is not None
    x0, y0, x1, y1 = bb
    assert x0 >= block["minX"] - 3 and x1 <= block["maxX"] + 3, (
        f"x overflow: ink x[{x0}-{x1}] vs box x[{block['minX']}-{block['maxX']}]"
    )
    assert y0 >= block["minY"] - 3 and y1 <= block["maxY"] + 3, (
        f"y overflow: ink y[{y0}-{y1}] vs box y[{block['minY']}-{block['maxY']}]"
    )


def test_caption_text_stays_within_block_bbox_y():
    plate = np.full((500, 600, 3), 255, np.uint8)
    block = _blk(60, 60, 260, 220)
    text = ["He had become nothing more than a nuisance who tried to steal her man."]
    out = compose_final(plate, [block], text, fit_rects=[None])
    bb = _ink_bbox(out, plate)
    assert bb is not None
    _, y0, _, y1 = bb
    assert y0 >= block["minY"] - 3, f"ink top {y0} above bbox {block['minY']}"
    assert y1 <= block["maxY"] + 3, f"ink bottom {y1} below bbox {block['maxY']}"


# ---------------------------------------------------------------------------
# 2. inter-block overlap avoidance
# ---------------------------------------------------------------------------

def test_two_captions_do_not_overlap():
    """Two adjacent caption blocks must render to disjoint ink rects."""
    plate = np.full((400, 800, 3), 255, np.uint8)
    a = _blk(20, 40, 200, 360)
    b = _blk(210, 40, 390, 360)
    long = "This is a fairly long caption that would naturally overflow its box."
    out_a = compose_final(plate, [a], [long], fit_rects=[None])
    out_b = compose_final(plate, [b], [long], fit_rects=[None])
    bba = _ink_bbox(out_a, plate)
    bbb = _ink_bbox(out_b, plate)
    assert bba and bbb
    # rendered separately each stays in its own column -> combined no overlap
    assert bba[2] <= b["minX"] + 3
    assert bbb[0] >= a["maxX"] - 3


def test_orphan_sfx_suppressed_when_overlapping_dialogue():
    """An orphan/SFX block whose render would land on a dialogue block is
    suppressed rather than overlapping the dialogue."""
    plate = np.full((400, 600, 3), 255, np.uint8)
    # dialogue bubble (has a fit_rect -> typeset to bubble interior)
    dialogue = _blk(100, 100, 400, 300)
    dia_rect = _blk(100, 100, 400, 300)
    # orphan SFX with a tiny bbox sitting INSIDE the dialogue region
    sfx = _blk(150, 150, 200, 260, orphan=True)
    blocks = [dialogue, sfx]
    fits = [dia_rect, None]
    texts = ["Don't look at me while I'm doing this!", "GAH"]
    out = compose_final(plate, blocks, texts, fit_rects=fits)
    # render dialogue alone for comparison
    out_dialogue_only = compose_final(plate, [dialogue], [texts[0]], fit_rects=[dia_rect])
    # The SFX must not have added ink inside the dialogue's tiny bbox region
    # beyond what the dialogue itself draws. Compare ink pixel count in the SFX
    # box: with suppression it should equal the dialogue-only count there.
    def ink_count(img):
        reg = np.abs(img[150:260, 150:200].astype(int) - 255).sum(axis=2) > 20
        return int(reg.sum())
    assert ink_count(out) <= ink_count(out_dialogue_only) + 5, (
        "orphan SFX added ink on top of dialogue (not suppressed)"
    )


# ---------------------------------------------------------------------------
# 3. verbose SFX gloss truncation
# ---------------------------------------------------------------------------

def test_verbose_sfx_gloss_truncated_for_tiny_box():
    fn = getattr(R, "_truncate_sfx_text", None)
    assert fn is not None, "_truncate_sfx_text helper missing"
    # tiny SFX box -> a verbose gloss must be shortened
    tiny = _blk(925, 138, 952, 247, orphan=True)
    long_gloss = "SFX FOR A MOMENTARY SHOCK OR SURPRISE, LIKE A GASP"
    short = fn(long_gloss, tiny)
    assert len(short) < len(long_gloss)
    assert len(short) <= 16


def test_truncate_leaves_short_sfx_alone():
    fn = R._truncate_sfx_text
    tiny = _blk(925, 138, 952, 247, orphan=True)
    assert fn("GAH", tiny) == "GAH"


# ---------------------------------------------------------------------------
# regression: dialogue (fit_rect present) still centers in bubble as before
# ---------------------------------------------------------------------------

def test_dialogue_with_fit_rect_renders_inside_bubble():
    plate = np.full((400, 600, 3), 255, np.uint8)
    block = _blk(280, 180, 320, 240)  # tight vertical JP column
    bubble = _blk(150, 100, 450, 320)  # wide bubble
    out = compose_final(plate, [block], ["Hello there friend"], fit_rects=[bubble])
    bb = _ink_bbox(out, plate)
    assert bb is not None
    x0, y0, x1, y1 = bb
    # should use the bubble interior, i.e. ink wider than the tight block column
    assert (x1 - x0) > (block["maxX"] - block["minX"])
    assert x0 >= bubble["minX"] - 4 and x1 <= bubble["maxX"] + 4
