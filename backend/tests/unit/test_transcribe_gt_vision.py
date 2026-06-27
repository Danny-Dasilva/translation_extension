"""Unit tests for the one-time vision-gold transcription pass.

Covers the pure logic that does NOT require the live box VLM:
  * bbox normalisation + IoU on normalised boxes (different pixel spaces)
  * greedy max-IoU alignment of OUR bubbles <-> GT vision bubbles
  * reading-order zip fallback when IoU is too low
  * VLM JSON parsing (``[x0,y0,x1,y1]`` arrays + ```json fences``)
  * gold-row emission schema (jp/en/src/bbox/ocr_clean/iou/source_field)

The live VLM call (``_vision_transcribe_page``) is exercised by a separate,
network-gated smoke run, not here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "eval"
sys.path.insert(0, str(SCRIPT_DIR))

_spec = importlib.util.spec_from_file_location(
    "transcribe_gt_vision", SCRIPT_DIR / "transcribe_gt_vision.py"
)
assert _spec and _spec.loader
tgv = importlib.util.module_from_spec(_spec)
sys.modules["transcribe_gt_vision"] = tgv
_spec.loader.exec_module(tgv)


# ---------------------------------------------------------------------------
# VLM response parsing
# ---------------------------------------------------------------------------


def test_parse_vlm_array_basic():
    raw = '[{"text": "HELLO", "bbox": [10, 20, 30, 40]}]'
    out = tgv._parse_vision_response(raw, img_w=100, img_h=100)
    assert len(out) == 1
    assert out[0].text == "HELLO"
    assert out[0].bbox == {"minX": 10, "minY": 20, "maxX": 30, "maxY": 40}


def test_parse_vlm_strips_json_fence():
    raw = '```json\n[{"text": "A", "bbox": [0,0,5,5]}]\n```'
    out = tgv._parse_vision_response(raw, img_w=100, img_h=100)
    assert len(out) == 1
    assert out[0].text == "A"


def test_parse_vlm_strips_plain_fence():
    raw = '```\n[{"text": "B", "bbox": [1,2,3,4]}]\n```'
    out = tgv._parse_vision_response(raw, img_w=100, img_h=100)
    assert out[0].text == "B"


def test_parse_vlm_skips_empty_text():
    raw = '[{"text": "", "bbox": [0,0,5,5]}, {"text": "OK", "bbox": [1,1,9,9]}]'
    out = tgv._parse_vision_response(raw, img_w=100, img_h=100)
    assert [b.text for b in out] == ["OK"]


def test_parse_vlm_handles_missing_bbox():
    raw = '[{"text": "NOBOX"}]'
    out = tgv._parse_vision_response(raw, img_w=100, img_h=100)
    assert len(out) == 1
    assert out[0].text == "NOBOX"
    assert out[0].bbox is None


def test_parse_vlm_bad_json_returns_empty():
    assert tgv._parse_vision_response("not json at all", img_w=10, img_h=10) == []


def test_parse_vlm_salvages_truncated_array():
    # VLM looped on an SFX bubble and ran out of tokens mid-array; the good
    # bubbles before the runaway must still be recovered.
    runaway = "RY" + "Y" * 400  # no closing quote/brace after this
    raw = (
        '```json\n['
        '{"text": "AH...!!", "bbox": [142, 223, 205, 267]},\n'
        '{"text": "OK", "bbox": [10, 10, 20, 20]},\n'
        '{"text": "' + runaway
    )
    out = tgv._parse_vision_response(raw, img_w=1280, img_h=1791)
    texts = [b.text for b in out]
    assert "AH...!!" in texts
    assert "OK" in texts


def test_parse_vlm_squashes_runaway_in_text():
    raw = '[{"text": "R' + "Y" * 200 + '", "bbox": [0,0,5,5]}]'
    out = tgv._parse_vision_response(raw, img_w=100, img_h=100)
    assert len(out) == 1
    # runaway collapsed to a short string, not 200 chars
    assert len(out[0].text) < 10


# ---------------------------------------------------------------------------
# Normalised-bbox IoU (different pixel spaces must still overlap)
# ---------------------------------------------------------------------------


def test_norm_iou_identical_boxes_diff_spaces():
    # Same relative box in a 100x100 space and a 200x200 space -> IoU 1.0.
    a = {"minX": 10, "minY": 10, "maxX": 50, "maxY": 50}
    b = {"minX": 20, "minY": 20, "maxX": 100, "maxY": 100}
    iou = tgv._norm_iou(a, 100, 100, b, 200, 200)
    assert iou == pytest.approx(1.0, abs=1e-6)


def test_norm_iou_disjoint_is_zero():
    a = {"minX": 0, "minY": 0, "maxX": 10, "maxY": 10}
    b = {"minX": 90, "minY": 90, "maxX": 100, "maxY": 100}
    assert tgv._norm_iou(a, 100, 100, b, 100, 100) == 0.0


def test_norm_iou_partial_overlap():
    a = {"minX": 0, "minY": 0, "maxX": 100, "maxY": 100}
    b = {"minX": 50, "minY": 0, "maxX": 150, "maxY": 100}
    # same space (200 wide); intersection 50x100, union 150x100 -> 1/3
    iou = tgv._norm_iou(a, 200, 100, b, 200, 100)
    assert iou == pytest.approx(1.0 / 3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Alignment: OUR bubbles <-> GT vision bubbles (normalised IoU)
# ---------------------------------------------------------------------------


def _ob(idx, box, jp="あ", conf=0.9):
    return {"idx": idx, "bbox": box, "ocr_jp": jp, "ocr_conf": conf}


def test_align_greedy_max_iou_matches():
    our = [
        _ob(0, {"minX": 0, "minY": 0, "maxX": 50, "maxY": 50}),
        _ob(1, {"minX": 50, "minY": 50, "maxX": 100, "maxY": 100}),
    ]
    gt = [
        tgv.VisionBubble("TOP", {"minX": 2, "minY": 2, "maxX": 52, "maxY": 52}),
        tgv.VisionBubble("BOT", {"minX": 48, "minY": 48, "maxX": 98, "maxY": 98}),
    ]
    pairs = tgv.align_page(our, gt, our_w=100, our_h=100, gt_w=100, gt_h=100, iou_threshold=0.2)
    by_idx = {ob["idx"]: (gb.text if gb else None) for ob, gb, _ in pairs}
    assert by_idx[0] == "TOP"
    assert by_idx[1] == "BOT"


def test_align_distant_bubbles_stay_unmatched():
    # Far-apart bubbles must NOT be blindly zipped together -- a positional zip
    # would poison the gold (our idx order != VLM reading order). No spatial
    # overlap -> None (caller drops the row).
    our = [
        _ob(0, {"minX": 0, "minY": 0, "maxX": 10, "maxY": 10}),
        _ob(1, {"minX": 0, "minY": 0, "maxX": 10, "maxY": 10}),
    ]
    gt = [
        tgv.VisionBubble("FIRST", {"minX": 900, "minY": 900, "maxX": 999, "maxY": 999}),
        tgv.VisionBubble("SECOND", {"minX": 900, "minY": 900, "maxX": 999, "maxY": 999}),
    ]
    pairs = tgv.align_page(our, gt, our_w=1000, our_h=1000, gt_w=1000, gt_h=1000, iou_threshold=0.2)
    texts = [gb.text if gb else None for _, gb, _ in pairs]
    assert texts == [None, None]


def test_align_aspect_ratio_mismatch_matches_by_center():
    # Same bubble, orthogonal aspect ratios: OUR tall narrow vertical-JP box vs
    # GT wide short horizontal-EN box, sharing a centre. Plain IoU is tiny but
    # the centre-aware score must still match them.
    our = [_ob(0, {"minX": 480, "minY": 200, "maxX": 520, "maxY": 600})]  # tall
    gt = [tgv.VisionBubble("HELLO", {"minX": 300, "minY": 380, "maxX": 700, "maxY": 420})]  # wide
    pairs = tgv.align_page(our, gt, our_w=1000, our_h=1000, gt_w=1000, gt_h=1000, iou_threshold=0.2)
    _, gb, score = pairs[0]
    assert gb is not None and gb.text == "HELLO"
    assert score >= 0.2


def test_align_unmatched_our_bubble_when_zip_exhausted():
    # Two our-bubbles, ONE GT bubble: exactly one our-bubble gets the GT, the
    # other is None (greedy consumes the single GT; zip has nothing left).
    our = [
        _ob(0, {"minX": 0, "minY": 0, "maxX": 10, "maxY": 10}),
        _ob(1, {"minX": 0, "minY": 0, "maxX": 10, "maxY": 10}),
    ]
    gt = [tgv.VisionBubble("ONLY", {"minX": 0, "minY": 0, "maxX": 10, "maxY": 10})]
    pairs = tgv.align_page(our, gt, our_w=100, our_h=100, gt_w=100, gt_h=100, iou_threshold=0.2)
    matched = [gb for _, gb, _ in pairs]
    assert sum(1 for gb in matched if gb is not None) == 1
    assert sum(1 for gb in matched if gb is None) == 1
    got = next(gb for gb in matched if gb is not None)
    assert got.text == "ONLY"


def test_align_reports_iou_value():
    our = [_ob(0, {"minX": 0, "minY": 0, "maxX": 100, "maxY": 100})]
    gt = [tgv.VisionBubble("X", {"minX": 0, "minY": 0, "maxX": 100, "maxY": 100})]
    pairs = tgv.align_page(our, gt, our_w=100, our_h=100, gt_w=100, gt_h=100, iou_threshold=0.2)
    _, _, iou = pairs[0]
    assert iou == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Gold-row emission schema
# ---------------------------------------------------------------------------


def test_build_gold_row_schema_clean():
    ob = _ob(0, {"minX": 1, "minY": 2, "maxX": 3, "maxY": 4}, jp="昨日あんな事をしていた", conf=0.93)
    gb = tgv.VisionBubble("EVEN THOUGH SHE...", {"minX": 0, "minY": 0, "maxX": 9, "maxY": 9})
    row = tgv.build_gold_row(ob, gb, page=5, iou=0.81)
    assert row["jp"] == "昨日あんな事をしていた"
    assert row["en"] == "EVEN THOUGH SHE..."
    assert row["src"] == "ikenie4:p05:idx0"
    assert row["bbox"] == {"minX": 1, "minY": 2, "maxX": 3, "maxY": 4}
    assert row["ocr_clean"] is True
    assert row["category"] == ""
    assert row["severity"] == 1
    assert row["source_field"] == "vision_gt"
    assert row["iou"] == pytest.approx(0.81)


def test_build_gold_row_dirty_when_low_conf():
    ob = _ob(0, {"minX": 1, "minY": 2, "maxX": 3, "maxY": 4}, jp="普通の文", conf=0.40)
    gb = tgv.VisionBubble("HELLO", {"minX": 0, "minY": 0, "maxX": 9, "maxY": 9})
    row = tgv.build_gold_row(ob, gb, page=5, iou=0.5)
    assert row["ocr_clean"] is False  # conf < 0.85


def test_build_gold_row_dirty_when_garbled():
    # Latin-intrusion garble is caught by is_implausible_japanese even at high conf.
    ob = _ob(0, {"minX": 1, "minY": 2, "maxX": 3, "maxY": 4}, jp="ABCの日本語XYZ123text", conf=0.95)
    gb = tgv.VisionBubble("HELLO", {"minX": 0, "minY": 0, "maxX": 9, "maxY": 9})
    row = tgv.build_gold_row(ob, gb, page=5, iou=0.5)
    assert row["ocr_clean"] is False
