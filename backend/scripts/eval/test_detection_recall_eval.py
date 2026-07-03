"""Unit tests for the detection-recall scoring math.

These exercise the pure scoring layer (iou / match_boxes / score_page /
aggregate) on a tiny synthetic fixture — 2 fake detections vs 2 fake gold boxes
— proving the IoU / recall / false-negative logic WITHOUT loading the detector,
onnxruntime, or a GPU. The detector-backed run() is deferred (needs the model).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load the sibling module by path (scripts/ is not an importable package).
_SPEC = importlib.util.spec_from_file_location(
    "detection_recall_eval", Path(__file__).with_name("detection_recall_eval.py")
)
dre = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
# Register before exec so @dataclass introspection (py3.14) can resolve the module.
sys.modules["detection_recall_eval"] = dre
_SPEC.loader.exec_module(dre)


def test_iou_identity_and_disjoint():
    assert dre.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert dre.iou([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: inter=50, union=150 -> 1/3.
    assert abs(dre.iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9


def test_perfect_match_two_boxes():
    gold = [[0, 0, 10, 10], [100, 100, 120, 120]]
    det = [[0, 0, 10, 10], [100, 100, 120, 120]]
    m = dre.match_boxes(gold, det, 0.5)
    assert len(m.matched) == 2
    assert m.false_negatives == []
    assert m.false_positives == []
    assert m.recall == 1.0
    assert m.precision == 1.0


def test_one_miss_one_false_positive():
    # gold[0] is detected; gold[1] is MISSED (detector proposes a spurious box
    # far away instead) -> this is exactly the false-negative the audit says is
    # currently invisible.
    gold = [[0, 0, 10, 10], [100, 100, 120, 120]]
    det = [[0, 0, 10, 10], [500, 500, 520, 520]]
    m = dre.match_boxes(gold, det, 0.5)
    assert len(m.matched) == 1
    assert m.false_negatives == [1]      # gold box 1 missed
    assert m.false_positives == [1]      # det box 1 spurious
    assert m.recall == 0.5
    assert m.precision == 0.5


def test_threshold_gates_loose_match():
    # ~1/3 IoU pair: counts at 0.5? no. At 0.3? yes.
    gold = [[0, 0, 10, 10]]
    det = [[5, 0, 15, 10]]
    assert dre.match_boxes(gold, det, 0.5).recall == 0.0
    assert dre.match_boxes(gold, det, 0.30).recall == 1.0


def test_greedy_prefers_higher_iou():
    # One gold box, two candidate detections; the tighter one must win the match.
    gold = [[0, 0, 10, 10]]
    det = [[2, 0, 12, 10], [0, 0, 10, 10]]  # second is exact
    m = dre.match_boxes(gold, det, 0.5)
    assert len(m.matched) == 1
    gi, di, v = m.matched[0]
    assert di == 1 and v == 1.0
    assert m.false_positives == [0]


def test_aggregate_across_pages():
    p1 = dre.score_page("a", [[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.5, 0.75])
    p2 = dre.score_page("b", [[0, 0, 10, 10], [50, 50, 60, 60]], [[0, 0, 10, 10]], [0.5, 0.75])
    agg = dre.aggregate([p1, p2], [0.5, 0.75])
    # 3 gold total, 2 matched -> recall 2/3; 1 false-negative.
    assert agg["thresholds"][0.5]["gold"] == 3
    assert agg["thresholds"][0.5]["matched"] == 2
    assert agg["thresholds"][0.5]["false_negatives"] == 1
    assert abs(agg["thresholds"][0.5]["recall"] - 2 / 3) < 1e-9


def test_gold_file_is_wellformed_and_independent():
    """The shipped gold set must exist, be non-empty, and declare independence."""
    gold_path = Path(__file__).with_name("detection_recall_gold.json")
    if not gold_path.exists():
        import pytest

        pytest.skip("gold file not generated in this checkout")
    import json

    data = json.loads(gold_path.read_text())
    assert data["provenance"]["independent_of_our_detector"] is True
    assert len(data["pages"]) >= 15
    for p in data["pages"]:
        assert p["num_boxes"] == len(p["boxes_xyxy"]) >= 1
        for x1, y1, x2, y2 in p["boxes_xyxy"]:
            assert 0 <= x1 < x2 <= p["width"]
            assert 0 <= y1 < y2 <= p["height"]
