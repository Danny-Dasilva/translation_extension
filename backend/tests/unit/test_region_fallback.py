"""Unit tests for ``app.utils.region_fallback`` -- the flag-gated CV fallback
region proposer for horizontal chat/UI text the OBB detector missed entirely.

Pure OpenCV/numpy against synthetic images: no ONNX model, no GPU, no real
manga pages -- deterministic and fast. Real-page validation (furube chat
pages + AnimeText false-positive sampling) was done manually during this
investigation; see the handoff notes for those measured numbers. These tests
pin the GEOMETRIC CONTRACT: what shapes get proposed, what gets excluded, and
the safety gates that keep this conservative.

Synthetic pages are sized close to a real manga page (tall, ~1200px) because
several of the module's internal gates (glyph-size pre-gate, max line height)
are derived as FRACTIONS of the page height -- a tiny toy image would make
even normal-proportioned "glyphs" look like oversized art blobs and get
rejected by the wrong gate.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.utils.region_fallback import (
    _build_exclusion_mask,
    propose_horizontal_text_regions,
)


def _blank_page(w=800, h=1200):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _draw_text_line(img, x, y, n_glyphs=8, glyph_w=14, glyph_h=20, gap=6):
    """Draw a row of small filled black rectangles simulating glyphs of one
    horizontal text line (dark ink on a light background)."""
    for i in range(n_glyphs):
        gx = x + i * (glyph_w + gap)
        cv2.rectangle(img, (gx, y), (gx + glyph_w, y + glyph_h), (0, 0, 0), -1)
    return img


def _draw_vertical_column(img, x, y, n_glyphs=10, glyph_w=20, glyph_h=14, gap=6):
    """Draw a vertical stack of glyphs (a JP vertical-text column shape)."""
    for i in range(n_glyphs):
        gy = y + i * (glyph_h + gap)
        cv2.rectangle(img, (x, gy), (x + glyph_w, gy + glyph_h), (0, 0, 0), -1)
    return img


# --------------------------------------------------------------------------- #
# _build_exclusion_mask
# --------------------------------------------------------------------------- #

def test_exclusion_mask_covers_padded_box():
    boxes = [{"minX": 50, "minY": 50, "maxX": 100, "maxY": 80}]
    mask = _build_exclusion_mask((400, 300), boxes)
    # Inside the box: excluded.
    assert mask[65, 75] == 255
    # Comfortably outside (well beyond any plausible padding): not excluded.
    assert mask[10, 10] == 0


def test_exclusion_mask_empty_for_no_boxes():
    mask = _build_exclusion_mask((400, 300), [])
    assert mask.max() == 0


# --------------------------------------------------------------------------- #
# propose_horizontal_text_regions -- positive case
# --------------------------------------------------------------------------- #

def test_finds_a_synthetic_horizontal_text_line():
    img = _blank_page()
    _draw_text_line(img, x=60, y=300)
    cands = propose_horizontal_text_regions(img, existing_blocks=[], existing_text_lines=[], input_is_bgr=True)
    assert len(cands) >= 1
    c = cands[0]
    # The candidate should roughly bound the drawn line (some slack for
    # dilation-driven bbox growth).
    assert c["minY"] <= 300 + 8
    assert c["maxY"] >= 320 - 8
    assert c["maxX"] - c["minX"] > (c["maxY"] - c["minY"])  # horizontal, not tall
    assert c["cv_fallback"] is True


def test_multiple_lines_produce_multiple_candidates():
    img = _blank_page()
    _draw_text_line(img, x=60, y=200)
    _draw_text_line(img, x=60, y=700)
    cands = propose_horizontal_text_regions(img, existing_blocks=[], existing_text_lines=[], input_is_bgr=True)
    assert len(cands) == 2


# --------------------------------------------------------------------------- #
# propose_horizontal_text_regions -- exclusion / negative gates
# --------------------------------------------------------------------------- #

def test_does_not_duplicate_area_already_covered_by_a_block():
    img = _blank_page()
    _draw_text_line(img, x=60, y=300)
    # The whole line sits inside this "already detected" block.
    existing_blocks = [{"minX": 0, "minY": 260, "maxX": 800, "maxY": 360, "confidence": 0.9}]
    cands = propose_horizontal_text_regions(img, existing_blocks, [], input_is_bgr=True)
    assert cands == []


def test_does_not_duplicate_area_already_covered_by_a_text_line():
    img = _blank_page()
    _draw_text_line(img, x=60, y=300)
    existing_lines = [{"minX": 0, "minY": 260, "maxX": 800, "maxY": 360}]
    cands = propose_horizontal_text_regions(img, [], existing_lines, input_is_bgr=True)
    assert cands == []


def test_rejects_a_large_solid_art_blob():
    """A big solid-fill rectangle (simulating hair/silhouette art, not text)
    must NOT be proposed -- the glyph-size pre-gate should drop it before the
    line-merge dilation ever sees it."""
    img = _blank_page()
    cv2.rectangle(img, (100, 100), (500, 500), (0, 0, 0), -1)  # 400x400 solid blob
    cands = propose_horizontal_text_regions(img, [], [], input_is_bgr=True)
    assert cands == []


def test_rejects_vertical_glyph_columns():
    """A vertical JP-style glyph column (tall, narrow) fails the horizontal
    aspect-ratio gate -- this fallback targets gap #1 (horizontal chat/UI
    text) specifically, not vertical narration columns."""
    img = _blank_page()
    _draw_vertical_column(img, x=200, y=200)
    cands = propose_horizontal_text_regions(img, [], [], input_is_bgr=True)
    assert cands == []


def test_rejects_a_single_stray_mark_below_glyph_count_floor():
    """One or two isolated marks (not a real text line) should not pass the
    minimum glyph-count gate."""
    img = _blank_page()
    _draw_text_line(img, x=60, y=300, n_glyphs=2)
    cands = propose_horizontal_text_regions(img, [], [], input_is_bgr=True, min_ink_glyphs=3)
    assert cands == []


def test_blank_page_yields_nothing():
    img = _blank_page()
    cands = propose_horizontal_text_regions(img, [], [], input_is_bgr=True)
    assert cands == []


# --------------------------------------------------------------------------- #
# Safety valves
# --------------------------------------------------------------------------- #

def test_raw_candidate_circuit_breaker_bails_on_busy_pages():
    """When the cheap raw-contour pass turns up more candidates than
    ``max_raw_candidates``, the whole page is skipped rather than risking a
    flood of false regions -- verified with an artificially low threshold on
    an otherwise-normal single-line image (which would return >=1 candidate
    without the breaker)."""
    img = _blank_page()
    _draw_text_line(img, x=60, y=300)
    baseline = propose_horizontal_text_regions(img, [], [], input_is_bgr=True)
    assert len(baseline) >= 1  # sanity: this image normally proposes something

    tripped = propose_horizontal_text_regions(img, [], [], input_is_bgr=True, max_raw_candidates=0)
    assert tripped == []


def test_max_candidates_cap_is_enforced():
    img = _blank_page(w=1200, h=1800)
    y = 80
    while y < 1750:
        _draw_text_line(img, x=60, y=y)
        y += 90
    cands = propose_horizontal_text_regions(img, [], [], input_is_bgr=True, max_candidates=3)
    assert len(cands) <= 3


def test_degenerate_image_returns_empty():
    assert propose_horizontal_text_regions(np.zeros((0, 0, 3), dtype=np.uint8), [], []) == []
    assert propose_horizontal_text_regions(None, [], []) == []
