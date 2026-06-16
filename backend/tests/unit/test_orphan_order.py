"""Unit tests for FIX P1-3: vertical-column reading order in OCR line join.

Dense vertical-Japanese bubbles are OCR'd as separate line fragments then
joined. The OLD ordering (``sorted(key=(-minX, minY))``) scrambles the join
when columns overlap in X or a column wraps: fragments of the SAME column with
jittered ``minX`` get split across the right-to-left grouping, so a left column
reads bottom-to-top and the whole bubble comes out in the wrong order. The
translator then "smooths" the scrambled Japanese into fluent-but-wrong English
(QA pages 023/024).

These tests build synthetic line fragments from the manga reading-order
convention (right-to-left columns, top-to-bottom within a column) and assert
``order_cluster_lines`` yields the correct order. Coordinates are scaled to the
real page bubbles in ``.bench/.../023|024/bubbles.json`` (glyph ~60-90px, column
heights ~150-330px).
"""
from __future__ import annotations

from app.utils.orphan_lines import order_cluster_lines, merge_orphans_into_blocks


def _frag(text, minX, minY, maxX, maxY):
    return {"text": text, "minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY}


def _texts(ordered):
    return [f["text"] for f in ordered]


# The OLD algorithm, kept here verbatim so we can prove a regression: the
# wrapped-column case orders differently under old vs new.
def _old_order(cluster):
    horiz = sum(
        1 for ln in cluster
        if (ln["maxX"] - ln["minX"]) > (ln["maxY"] - ln["minY"])
    )
    if horiz >= len(cluster) / 2:
        return sorted(cluster, key=lambda ln: (ln["minY"], ln["minX"]))
    return sorted(cluster, key=lambda ln: (-ln["minX"], ln["minY"]))


# ---- (a) clean 2-column right-to-left vertical case -------------------------

def test_clean_two_column_right_to_left():
    # Right column (higher X) read first, top-to-bottom; then left column.
    frags = [
        _frag("R1", 600, 120, 660, 260),
        _frag("R2", 600, 270, 660, 440),
        _frag("L1", 520, 120, 580, 260),
        _frag("L2", 520, 270, 580, 440),
    ]
    assert _texts(order_cluster_lines(frags)) == ["R1", "R2", "L1", "L2"]


# ---- (b) 3-column vertical case ---------------------------------------------

def test_three_column_right_to_left():
    frags = [
        _frag("C1a", 700, 120, 760, 250),
        _frag("C1b", 700, 260, 760, 400),
        _frag("C2a", 600, 120, 660, 250),
        _frag("C2b", 600, 260, 660, 400),
        _frag("C3a", 500, 120, 560, 250),
        _frag("C3b", 500, 260, 560, 400),
    ]
    assert _texts(order_cluster_lines(frags)) == [
        "C1a", "C1b", "C2a", "C2b", "C3a", "C3b"
    ]


# ---- (c) horizontal case MUST stay top-to-bottom ----------------------------

def test_horizontal_stays_top_to_bottom():
    # Wide lines (w > h): stacked rows read top-to-bottom (a chat balloon).
    frags = [
        _frag("row1", 100, 40, 400, 80),
        _frag("row2", 100, 100, 400, 140),
        _frag("row3", 100, 160, 400, 200),
    ]
    assert _texts(order_cluster_lines(frags)) == ["row1", "row2", "row3"]


# ---- (d) wrapped / overlapping-X column case: OLD mis-orders, NEW correct ----

def test_wrapped_overlapping_x_column_regression():
    # Same two columns as (a) but each column's fragments jitter in minX (a
    # wrapped column / slanted speech bubble). The left column's lower fragment
    # sits a hair RIGHT of its upper fragment, so the OLD (-minX, minY) sort
    # ranks it first => left column reads bottom-to-top.
    frags = [
        _frag("R1", 600, 120, 660, 260),
        _frag("R2", 595, 270, 655, 440),   # slightly left of R1
        _frag("L1", 520, 120, 580, 260),
        _frag("L2", 525, 270, 585, 440),   # slightly RIGHT of L1
    ]
    correct = ["R1", "R2", "L1", "L2"]

    old = _texts(_old_order(frags))
    new = _texts(order_cluster_lines(frags))

    # Document the regression the fix targets.
    assert old == ["R1", "R2", "L2", "L1"], old   # OLD: left column reversed
    assert new == correct, new                     # NEW: correct manga order
    assert old != new


def test_wrapped_single_column_does_not_split():
    # A single wrapping column with X jitter must stay ONE column, read
    # strictly top-to-bottom (not split into phantom columns).
    frags = [
        _frag("a", 600, 100, 660, 200),
        _frag("b", 605, 210, 665, 310),
        _frag("c", 598, 320, 658, 420),
        _frag("d", 602, 430, 662, 530),
    ]
    assert _texts(order_cluster_lines(frags)) == ["a", "b", "c", "d"]


# ---- merge_orphans_into_blocks uses the corrected order ----------------------

def test_merge_two_box_vertical_right_to_left():
    # Two stacked vertical pseudo-boxes in the SAME column: the merge must read
    # top-to-bottom (upper text leads), not be flipped by raw -minX.
    # Boxes overlap (center of each lies in the other) so the merge path runs;
    # they share one column, so the higher box (UPPER) must lead.
    blocks = [{"minX": 600, "minY": 240, "maxX": 660, "maxY": 440}]
    texts = ["LOWER"]
    synth = [{"minX": 602, "minY": 120, "maxX": 662, "maxY": 320, "orphan": True}]
    synth_texts = ["UPPER"]
    out_blocks, out_texts = merge_orphans_into_blocks(
        blocks, texts, synth, synth_texts, iou_thresh=0.0
    )
    # Same column (centers within a glyph), so top (UPPER) leads.
    assert out_texts[0] == "UPPERLOWER", out_texts


def test_merge_two_box_distinct_columns_right_to_left():
    # Two side-by-side columns that overlap enough to merge (center of LEFT
    # falls inside the union region): the right column must lead.
    blocks = [{"minX": 500, "minY": 120, "maxX": 640, "maxY": 440}]  # LEFT-ish
    texts = ["LEFT"]
    synth = [{"minX": 560, "minY": 120, "maxX": 700, "maxY": 440, "orphan": True}]
    synth_texts = ["RIGHT"]
    out_blocks, out_texts = merge_orphans_into_blocks(
        blocks, texts, synth, synth_texts, iou_thresh=0.1
    )
    # RIGHT center (630) > LEFT center (570) => right column leads.
    assert out_texts[0] == "RIGHTLEFT", out_texts
