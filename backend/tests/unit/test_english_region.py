"""Unit tests for the English early-exit utility (app/utils/english_region.py).

Goal: a detected region that is horizontal/Latin English text (not vertical
Japanese) must be SKIPPED — left as original pixels, never OCR-translated or
inpainted. The guard is two-pronged:

  geometry  -> horizontal (per-line `direction`, fallback aspect ratio)
  content   -> the OCR text is NOT recognized as Japanese

A horizontal *Japanese* SFX block must NOT be skipped (geometry alone is not
enough), so we AND the geometry test with `not is_japanese_fn(ocr_text)`.
"""
from __future__ import annotations

from app.utils.english_region import is_horizontal_region, should_skip_as_english
from app.utils.japanese_text_filter import is_japanese_text


def _block(min_x, min_y, max_x, max_y, conf=0.9):
    return {"minX": min_x, "minY": min_y, "maxX": max_x, "maxY": max_y, "confidence": conf}


def _line(min_x, min_y, max_x, max_y, direction=None):
    d = {"minX": min_x, "minY": min_y, "maxX": max_x, "maxY": max_y}
    if direction is not None:
        d["direction"] = direction
    return d


# --------------------------------------------------------------------------
# is_horizontal_region
# --------------------------------------------------------------------------
def test_horizontal_when_all_contained_lines_horizontal():
    block = _block(0, 0, 200, 40)
    lines = [
        _line(10, 5, 90, 35, direction="horizontal"),
        _line(100, 5, 190, 35, direction="horizontal"),
    ]
    assert is_horizontal_region(block, lines) is True


def test_vertical_when_contained_lines_vertical():
    # Tall narrow block, vertical JP columns inside.
    block = _block(0, 0, 60, 300)
    lines = [
        _line(5, 10, 25, 290, direction="vertical"),
        _line(30, 10, 55, 290, direction="vertical"),
    ]
    assert is_horizontal_region(block, lines) is False


def test_threshold_fraction_of_horizontal_lines():
    # 2 of 3 contained lines horizontal => 0.66 >= 0.6 default => horizontal.
    block = _block(0, 0, 300, 40)
    lines = [
        _line(0, 5, 90, 35, direction="horizontal"),
        _line(100, 5, 190, 35, direction="horizontal"),
        _line(200, 5, 290, 35, direction="vertical"),
    ]
    assert is_horizontal_region(block, lines) is True
    # Raise the bar above the actual fraction => no longer horizontal.
    assert is_horizontal_region(block, lines, threshold=0.7) is False


def test_only_lines_with_center_inside_block_count():
    block = _block(0, 0, 100, 40)
    lines = [
        _line(10, 5, 90, 35, direction="horizontal"),   # center inside
        _line(500, 5, 590, 35, direction="vertical"),   # center far outside -> ignored
    ]
    assert is_horizontal_region(block, lines) is True


def test_direction_fallback_from_line_geometry_when_field_absent():
    # No `direction` key -> infer from (maxX-minX) > (maxY-minY).
    block = _block(0, 0, 200, 40)
    lines = [_line(10, 5, 190, 35)]  # wide -> horizontal
    assert is_horizontal_region(block, lines) is True


def test_no_contained_lines_falls_back_to_block_aspect_horizontal():
    # Wide block, aspect (w/h) = 200/40 = 5.0 > 1.5 -> horizontal.
    block = _block(0, 0, 200, 40)
    assert is_horizontal_region(block, []) is True


def test_no_contained_lines_falls_back_to_block_aspect_vertical():
    # Tall block, aspect (w/h) = 40/200 = 0.2 < 1.5 -> not horizontal.
    block = _block(0, 0, 40, 200)
    assert is_horizontal_region(block, []) is False


def test_no_contained_lines_none_text_lines():
    block = _block(0, 0, 200, 40)
    assert is_horizontal_region(block, None) is True


def test_degenerate_zero_size_block_not_horizontal():
    block = _block(10, 10, 10, 10)
    assert is_horizontal_region(block, []) is False


# --------------------------------------------------------------------------
# should_skip_as_english
# --------------------------------------------------------------------------
def test_skip_horizontal_english_ascii():
    # Horizontal geometry + ASCII English OCR -> skip.
    block = _block(0, 0, 200, 40)
    lines = [_line(10, 5, 190, 35, direction="horizontal")]
    assert should_skip_as_english(block, lines, "HELLO WORLD", is_japanese_text) is True


def test_do_not_skip_vertical_japanese():
    # Vertical JP dialogue -> never skipped.
    block = _block(0, 0, 60, 300)
    lines = [_line(5, 10, 55, 290, direction="vertical")]
    assert should_skip_as_english(block, lines, "こんにちは", is_japanese_text) is False


def test_do_not_skip_horizontal_japanese_sfx():
    # Horizontal geometry BUT the OCR text is Japanese (SFX laid out
    # horizontally) -> is_japanese_fn True -> must NOT be skipped.
    block = _block(0, 0, 200, 40)
    lines = [_line(10, 5, 190, 35, direction="horizontal")]
    assert should_skip_as_english(block, lines, "ドドドド", is_japanese_text) is False


def test_skip_horizontal_english_block_no_lines():
    # No text_lines: wide block aspect -> horizontal; ASCII text -> skip.
    block = _block(0, 0, 200, 40)
    assert should_skip_as_english(block, [], "SOUND FX", is_japanese_text) is True


def test_do_not_skip_when_geometry_vertical_even_if_english():
    # Vertical geometry overrides: a vertical column of stray Latin is left to
    # the existing JP filter, not this early-exit.
    block = _block(0, 0, 40, 200)
    assert should_skip_as_english(block, [], "ABC", is_japanese_text) is False


def test_empty_ocr_text_horizontal_block():
    # Empty OCR on a horizontal block: not Japanese -> treated as English skip.
    block = _block(0, 0, 200, 40)
    lines = [_line(10, 5, 190, 35, direction="horizontal")]
    assert should_skip_as_english(block, lines, "", is_japanese_text) is True


def test_mixed_but_majority_japanese_horizontal_not_skipped():
    # Mostly-Japanese horizontal line should be recognized as JP -> not skipped.
    block = _block(0, 0, 200, 40)
    lines = [_line(10, 5, 190, 35, direction="horizontal")]
    assert should_skip_as_english(block, lines, "これはテスト", is_japanese_text) is False
