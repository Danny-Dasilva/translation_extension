"""Unit tests for FIX 3 (OCR line-join + dedup).

Covers three regressions:
  (i)   merging a duplicate orphan must NOT produce "X X".
  (ii)  a block with [high, low] line confidences is NOT fully dropped
        (per-line gating replaced the blanket min()).
  (iii) the ``ocr_line_join_newline`` flag toggles the line separator "" / "\n".
"""
from __future__ import annotations

import asyncio

import numpy as np

from app.config import settings
from app.services.parseq_ocr_service import ParseqOCRService
from app.utils.orphan_lines import merge_orphans_into_blocks


def _box(x0, y0, x1, y1, **extra):
    b = {"minX": x0, "minY": y0, "maxX": x1, "maxY": y1}
    b.update(extra)
    return b


# ---- (i) dedup: a duplicate orphan must not double the text -----------------

def test_duplicate_orphan_not_doubled():
    blocks = [_box(0, 0, 100, 100)]
    texts = ["こんにちは"]
    # Synthetic orphan overlaps and re-OCRs the SAME text.
    synth = [_box(10, 10, 90, 90)]
    synth_texts = ["こんにちは"]
    out_blocks, out_texts = merge_orphans_into_blocks(
        blocks, texts, synth, synth_texts, iou_thresh=0.1
    )
    assert out_texts[0] == "こんにちは"  # not "こんにちはこんにちは"


def test_substring_orphan_keeps_longer():
    blocks = [_box(0, 0, 100, 100)]
    texts = ["こんにちは皆さん"]
    synth = [_box(10, 10, 90, 90)]
    synth_texts = ["こんにちは"]  # substring of the original
    out_blocks, out_texts = merge_orphans_into_blocks(
        blocks, texts, synth, synth_texts, iou_thresh=0.1
    )
    assert out_texts[0] == "こんにちは皆さん"  # longer/containing one kept


def test_distinct_orphan_still_concatenated():
    blocks = [_box(0, 0, 100, 100)]
    texts = ["A"]
    synth = [_box(10, 10, 90, 90)]
    synth_texts = ["B"]
    out_blocks, out_texts = merge_orphans_into_blocks(
        blocks, texts, synth, synth_texts, iou_thresh=0.1
    )
    assert set(out_texts[0]) == {"A", "B"}  # both present, order by geometry


# ---- (ii) per-line confidence gating: one bad line must not drop the block --

class _StubOCR(ParseqOCRService):
    """Bypass __init__/ONNX; drive _recognize_batch_with_conf from a queue."""

    def __init__(self, tc):  # noqa: D401 - test shim
        self._tc = list(tc)

    async def _recognize_batch_with_conf(self, crops, batch_size=24):
        # Return the pre-baked (text, conf) pairs in crop order.
        return [self._tc.pop(0) for _ in crops]


def test_block_with_one_low_line_not_dropped():
    # One block, two lines: high conf + low conf. Old code: min() -> low -> the
    # garble gate would drop the whole bubble. New code: max over kept lines.
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    blocks = [_box(0, 0, 100, 100)]
    lines = [_box(10, 10, 90, 40), _box(10, 50, 90, 90)]
    ocr = _StubOCR([("good", 0.95), ("bad", 0.40)])
    gate = settings.ocr_confidence_gate_threshold
    texts, confs = asyncio.run(
        ocr.recognize_blocks_with_lines(img, blocks, lines, return_confidence=True)
    )
    assert texts[0]  # text retained
    # block conf == max of lines clearing the gate (0.95), not min (0.40).
    assert confs[0] >= gate
    assert confs[0] == 0.95


def test_block_with_all_low_lines_reports_low_conf():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    blocks = [_box(0, 0, 100, 100)]
    lines = [_box(10, 10, 90, 40), _box(10, 50, 90, 90)]
    ocr = _StubOCR([("aa", 0.30), ("bb", 0.20)])
    gate = settings.ocr_confidence_gate_threshold
    texts, confs = asyncio.run(
        ocr.recognize_blocks_with_lines(img, blocks, lines, return_confidence=True)
    )
    assert texts[0]  # text kept (downstream garble gate decides)
    assert confs[0] < gate  # but reported low (max of all == 0.30)
    assert confs[0] == 0.30


# ---- (iii) the newline-join flag toggles the separator ----------------------

def test_join_flag_toggles_separator(monkeypatch):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    blocks = [_box(0, 0, 100, 100)]
    # Two lines; right-to-left manga order -> the higher-minX line leads.
    lines = [_box(60, 10, 90, 90), _box(10, 10, 40, 90)]

    monkeypatch.setattr(settings, "ocr_line_join_newline", False, raising=False)
    ocr = _StubOCR([("R", 0.9), ("L", 0.9)])
    texts, _ = asyncio.run(
        ocr.recognize_blocks_with_lines(img, blocks, lines, return_confidence=True)
    )
    assert "\n" not in texts[0]
    assert texts[0] == "RL"

    monkeypatch.setattr(settings, "ocr_line_join_newline", True, raising=False)
    ocr = _StubOCR([("R", 0.9), ("L", 0.9)])
    texts, _ = asyncio.run(
        ocr.recognize_blocks_with_lines(img, blocks, lines, return_confidence=True)
    )
    assert texts[0] == "R\nL"


def test_default_flag_is_false():
    assert settings.ocr_line_join_newline is False
