"""Unit tests for the shared PAGE-LEVEL block reading-order sort.

The v11 LoRA was trained with `scripts/data/v11/build_v11_dataset.py`'s
`manga_reading_order(rows)` — a COLUMN-MAJOR right-to-left, top-to-bottom block
order (tol = max(40, page_w*0.06)). At serve time the detector emitted blocks
with a NAIVE `sorted(key=(-minX, minY))` key, which interleaves columns and
splits wrapped columns, so the "Page:" context the model saw did NOT match
training. `reading_order_sort(blocks)` must reproduce the training algorithm
exactly (on minX/minY/maxX/maxY-keyed blocks).

These tests pin: (a) clean multi-column RTL, (b) wrapped/jittered column does
not split, (c) byte-for-byte agreement with the training `manga_reading_order`
on randomized boxes, (d) orphans land in reading position (not appended).
"""
from __future__ import annotations

import random

from app.utils.orphan_lines import reading_order_sort


def _b(tag, minX, minY, maxX, maxY, **extra):
    d = {"minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY, "tag": tag}
    d.update(extra)
    return d


def _tags(ordered):
    return [b["tag"] for b in ordered]


# Reference implementation: a verbatim copy of build_v11_dataset.manga_reading_order
# translated to the minX/minY/maxX/maxY block schema (xmin->minX etc.). The shared
# helper MUST agree with this on every input.
def _training_order(blocks):
    rows = [dict(b) for b in blocks]
    if len(rows) <= 1:
        return rows
    for r in rows:
        r["_cx"] = (r["minX"] + r["maxX"]) / 2.0
    page_w = max(r["maxX"] for r in rows) - min(r["minX"] for r in rows)
    tol = max(40.0, page_w * 0.06)
    by_x = sorted(rows, key=lambda r: -r["_cx"])
    columns = []
    for r in by_x:
        placed = False
        for col in columns:
            col_cx = sum(c["_cx"] for c in col) / len(col)
            if abs(r["_cx"] - col_cx) <= tol:
                col.append(r)
                placed = True
                break
        if not placed:
            columns.append([r])
    columns.sort(key=lambda col: -sum(c["_cx"] for c in col) / len(col))
    ordered = []
    for col in columns:
        col.sort(key=lambda r: r["minY"])
        ordered.extend(col)
    for r in ordered:
        r.pop("_cx", None)
    return ordered


def test_single_block_passthrough():
    blocks = [_b("A", 100, 100, 200, 200)]
    assert _tags(reading_order_sort(blocks)) == ["A"]


def test_empty_passthrough():
    assert reading_order_sort([]) == []


def test_clean_two_column_right_to_left():
    # page width ~ 660-100 = 560 -> tol = max(40, 33.6) = 40. Columns ~80 apart.
    blocks = [
        _b("R1", 600, 120, 660, 260),
        _b("R2", 600, 270, 660, 440),
        _b("L1", 480, 120, 540, 260),
        _b("L2", 480, 270, 540, 440),
    ]
    # Right column (higher cx) first, top-to-bottom; then left column.
    assert _tags(reading_order_sort(blocks)) == ["R1", "R2", "L1", "L2"]


def test_three_column_right_to_left():
    blocks = [
        _b("C1a", 700, 120, 780, 250),
        _b("C1b", 700, 260, 780, 400),
        _b("C2a", 500, 120, 580, 250),
        _b("C2b", 500, 260, 580, 400),
        _b("C3a", 300, 120, 380, 250),
        _b("C3b", 300, 260, 380, 400),
    ]
    assert _tags(reading_order_sort(blocks)) == [
        "C1a", "C1b", "C2a", "C2b", "C3a", "C3b"
    ]


def test_wrapped_column_jitter_does_not_split():
    # One vertical column whose blocks jitter in X by a few px stays ONE column,
    # read strictly top-to-bottom — the naive (-minX,minY) sort reverses it.
    blocks = [
        _b("a", 600, 100, 660, 200),
        _b("b", 605, 210, 665, 310),
        _b("c", 598, 320, 658, 420),
        _b("d", 602, 430, 662, 530),
    ]
    assert _tags(reading_order_sort(blocks)) == ["a", "b", "c", "d"]


def test_naive_sort_regression_documented():
    # Two columns; the LEFT column's lower box sits a hair RIGHT of its upper box.
    # The naive (-minX, minY) key ranks the lower-left box before the upper-left,
    # reversing the left column. reading_order_sort buckets by column and reads
    # each top-to-bottom, fixing it.
    blocks = [
        _b("R1", 600, 120, 660, 260),
        _b("R2", 595, 270, 655, 440),
        _b("L1", 480, 120, 540, 260),
        _b("L2", 485, 270, 545, 440),
    ]
    naive = sorted(blocks, key=lambda b: (-b["minX"], b["minY"]))
    assert _tags(naive) == ["R1", "R2", "L2", "L1"]  # left column reversed
    assert _tags(reading_order_sort(blocks)) == ["R1", "R2", "L1", "L2"]


def test_orphan_lands_in_reading_position_not_appended():
    # A page with two real blocks (right + left columns) and an orphan block
    # whose X-center sits in the RIGHT column below R1. After sort, the orphan
    # must read in the right column (position 2), NOT appended at the end.
    blocks = [
        _b("R1", 600, 100, 660, 200),
        _b("L1", 480, 100, 540, 200),
        _b("ORPH", 600, 220, 660, 360, orphan=True),
    ]
    assert _tags(reading_order_sort(blocks)) == ["R1", "ORPH", "L1"]


def test_matches_training_manga_reading_order_randomized():
    rng = random.Random(1234)
    for trial in range(200):
        n = rng.randint(2, 14)
        blocks = []
        for i in range(n):
            x0 = rng.randint(0, 1400)
            y0 = rng.randint(0, 2000)
            w = rng.randint(30, 160)
            h = rng.randint(30, 300)
            blocks.append(_b(f"t{i}", x0, y0, x0 + w, y0 + h))
        expect = _tags(_training_order(blocks))
        got = _tags(reading_order_sort(blocks))
        assert got == expect, f"trial {trial}: {got} != {expect}"


def test_sort_does_not_mutate_input_or_drop_keys():
    blocks = [
        _b("A", 600, 100, 660, 200, confidence=0.9, orphan=True),
        _b("B", 480, 100, 540, 200, confidence=0.8),
    ]
    before = [dict(b) for b in blocks]
    out = reading_order_sort(blocks)
    # original list untouched (same objects, same keys), helper must not leak _cx
    assert blocks == before
    for b in out:
        assert "_cx" not in b
        assert "_w" not in b
    # confidence / orphan markers preserved
    a_out = next(b for b in out if b["tag"] == "A")
    assert a_out["confidence"] == 0.9 and a_out["orphan"] is True
