"""Unit tests for the bbox-IoU spatial join in build_predictions_for_gold.

The eval harness must join run predictions to gold rows by a STABLE spatial
key (bbox IoU on the same page), NOT by the Japanese OCR text -- because OCR
text changes between runs, making a jp-join score different gold subsets per
run (apples-to-oranges).  These tests pin the join semantics:

  * an overlapping bubble matches its gold row (highest IoU wins),
  * a non-overlapping bubble does NOT match,
  * a gold row with no overlapping bubble on its page is unmatched (en="").
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "eval"
    / "build_predictions_for_gold.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_predictions_for_gold", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_predictions_for_gold"] = mod
    spec.loader.exec_module(mod)
    return mod


bp = _load_module()


# --------------------------------------------------------------------------- #
# iou
# --------------------------------------------------------------------------- #
def test_iou_identical_boxes_is_one():
    a = {"minX": 0, "minY": 0, "maxX": 10, "maxY": 10}
    assert bp.iou(a, dict(a)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = {"minX": 0, "minY": 0, "maxX": 10, "maxY": 10}
    b = {"minX": 100, "minY": 100, "maxX": 110, "maxY": 110}
    assert bp.iou(a, b) == 0.0


def test_iou_half_overlap():
    # a = [0,0,10,10] (area 100); b = [5,0,15,10] (area 100)
    # intersection = [5,0,10,10] area 50; union = 150; iou = 1/3
    a = {"minX": 0, "minY": 0, "maxX": 10, "maxY": 10}
    b = {"minX": 5, "minY": 0, "maxX": 15, "maxY": 10}
    assert bp.iou(a, b) == pytest.approx(50.0 / 150.0)


def test_iou_touching_edges_is_zero():
    a = {"minX": 0, "minY": 0, "maxX": 10, "maxY": 10}
    b = {"minX": 10, "minY": 0, "maxX": 20, "maxY": 10}
    assert bp.iou(a, b) == 0.0


# --------------------------------------------------------------------------- #
# page parsing
# --------------------------------------------------------------------------- #
def test_parse_page_from_src():
    assert bp.parse_page("ikenie4:p05:idx0") == 5
    assert bp.parse_page("ikenie4:p99:idx13") == 99


def test_parse_page_bad_src_returns_none():
    assert bp.parse_page("garbage") is None


# --------------------------------------------------------------------------- #
# best_match_bubble: the core join
# --------------------------------------------------------------------------- #
def _bub(idx, box, jp, en):
    return {"idx": idx, "bbox": box, "ocr_jp": jp, "translation_en": en}


def test_overlapping_bubble_matches_highest_iou():
    gold_box = {"minX": 0, "minY": 0, "maxX": 100, "maxY": 100}
    bubbles = [
        _bub(0, {"minX": 500, "minY": 500, "maxX": 600, "maxY": 600}, "X", "far"),
        _bub(1, {"minX": 5, "minY": 5, "maxX": 95, "maxY": 95}, "Y", "overlap"),
        _bub(2, {"minX": 50, "minY": 50, "maxX": 150, "maxY": 150}, "Z", "partial"),
    ]
    bub, score = bp.best_match_bubble(gold_box, bubbles)
    assert bub is not None
    assert bub["idx"] == 1  # highest IoU
    assert bub["translation_en"] == "overlap"
    assert score > 0.3


def test_non_overlapping_bubble_does_not_match_above_threshold():
    gold_box = {"minX": 0, "minY": 0, "maxX": 100, "maxY": 100}
    bubbles = [
        _bub(0, {"minX": 500, "minY": 500, "maxX": 600, "maxY": 600}, "X", "far"),
    ]
    bub, score = bp.best_match_bubble(gold_box, bubbles)
    # best bubble is returned, but its IoU is 0 -> caller treats as unmatched
    assert score == 0.0


def test_no_bubbles_returns_none():
    gold_box = {"minX": 0, "minY": 0, "maxX": 100, "maxY": 100}
    bub, score = bp.best_match_bubble(gold_box, [])
    assert bub is None
    assert score == 0.0


# --------------------------------------------------------------------------- #
# build_rows: end-to-end join keyed by gold src
# --------------------------------------------------------------------------- #
def test_build_rows_matched_and_unmatched(tmp_path):
    gold = [
        {
            "src": "ikenie4:p05:idx0",
            "jp": "GOLDJP_A",
            "en": "GOLD_EN_A",
            "bbox": {"minX": 0, "minY": 0, "maxX": 100, "maxY": 100},
        },
        {
            "src": "ikenie4:p05:idx9",
            "jp": "GOLDJP_B",
            "en": "GOLD_EN_B",
            # no bubble overlaps this -> unmatched
            "bbox": {"minX": 900, "minY": 900, "maxX": 950, "maxY": 950},
        },
        {
            "src": "ikenie4:p06:idx0",
            "jp": "GOLDJP_C",
            "en": "GOLD_EN_C",
            # page 6 has no inspect dir -> unmatched
            "bbox": {"minX": 0, "minY": 0, "maxX": 100, "maxY": 100},
        },
    ]
    # inspect dir: page 5 has one overlapping bubble (different ocr text!)
    insp = tmp_path / "insp"
    (insp / "005").mkdir(parents=True)
    (insp / "005" / "bubbles.json").write_text(
        '[{"idx": 0, "bbox": {"minX": 2, "minY": 2, "maxX": 98, "maxY": 98}, '
        '"ocr_jp": "RUNJP_A", "translation_en": "RUN_EN_A"}]'
    )

    rows = bp.build_rows(gold, insp, iou_threshold=0.3)

    by_src = {r["src"]: r for r in rows}
    # 1:1 with gold by src
    assert set(by_src) == {row["src"] for row in gold}

    a = by_src["ikenie4:p05:idx0"]
    assert a["matched"] is True
    assert a["en"] == "RUN_EN_A"  # our run's translation
    assert a["jp"] == "RUNJP_A"  # our run's OCR (NOT gold jp)
    assert a["gold_en"] == "GOLD_EN_A"
    assert a["gold_jp"] == "GOLDJP_A"
    assert a["iou"] > 0.3

    b = by_src["ikenie4:p05:idx9"]
    assert b["matched"] is False
    assert b["en"] == ""
    assert b["gold_en"] == "GOLD_EN_B"

    c = by_src["ikenie4:p06:idx0"]
    assert c["matched"] is False
    assert c["en"] == ""
