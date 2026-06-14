"""Unit test for PARSeq OCR recognition-confidence decode (FIX 2).

``_decode_with_conf`` must return mean softmax max-prob over the DECODED tokens
(before EOS). We build a tiny fake instance (no ONNX session) and drive it with
hand-crafted logits so a crisp high-margin crop scores ~high and a flat/uncertain
crop scores ~low.
"""
from __future__ import annotations

import numpy as np

from app.services.parseq_ocr_service import ParseqOCRService


class _Fake(ParseqOCRService):
    """Bypass __init__/ONNX load; set just what _decode_with_conf needs."""

    def __init__(self):  # noqa: D401 - test shim
        self.charset = "ab"
        # head index 0 = EOS, 1 = 'a', 2 = 'b'
        self.eos_id = 0
        self._itos = ["[E]"] + list(self.charset)


def _logits(rows):
    return np.array([rows], dtype=np.float32)  # (B=1, L, C=3)


def test_high_margin_high_confidence():
    f = _Fake()
    # step0 strongly 'a', step1 strongly 'b', step2 EOS.
    logits = _logits([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0]])
    (text, conf), = f._decode_with_conf(logits)
    assert text == "ab"
    assert conf > 0.99


def test_flat_logits_low_confidence():
    f = _Fake()
    # near-uniform but argmax non-EOS at both steps => low max-prob on the
    # decoded tokens; step2 is EOS. (idx1='a' wins narrowly twice.)
    logits = _logits([[0.0, 0.20, 0.15], [0.0, 0.20, 0.15], [9.0, 0.0, 0.0]])
    (text, conf), = f._decode_with_conf(logits)
    assert len(text) == 2  # two tokens before EOS
    assert conf < 0.45


def test_immediate_eos_zero_confidence():
    f = _Fake()
    logits = _logits([[10.0, 0.0, 0.0]])
    (text, conf), = f._decode_with_conf(logits)
    assert text == ""
    assert conf == 0.0
