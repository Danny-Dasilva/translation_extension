"""Unit tests for the independent OBB oriented-text-line confidence gate
(``settings.ctd_obb_line_confidence`` / ``ComicTextDetectorService.
_extract_text_lines``).

Detection-recall investigation (2026-07-04): the v26 OBB text-line head
under-scores horizontal chat/phone-UI text at 0.05-0.32, straddling the
previously-hardcoded ``min(block_confidence, 0.3)`` op-point. This setting
decouples the OBB line-confidence gate from ``block_confidence`` so it can be
tuned independently, with ``None`` (default) preserving the EXACT prior
behavior byte-for-byte.

These tests exercise ``_extract_text_lines`` directly via a bare instance
(``object.__new__``) with only the few attributes that method reads set by
hand -- NOT a real ``ComicTextDetectorService(...)``, so no ONNX session or
model file is touched (keeps this fast, GPU-free, and independent of the
model artifact being present).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.services.ctd_service import ComicTextDetectorService


def _bare_service(*, block_confidence: float, min_area: int, obb_line_confidence):
    svc = object.__new__(ComicTextDetectorService)
    svc.block_confidence = block_confidence
    svc.min_area = min_area
    svc.text_threshold = 0.8
    svc.obb_line_confidence = obb_line_confidence
    return svc


def _obb_tensor(rows):
    """Build a [1, N, 7] OBB tensor: cx, cy, bw, bh, cls0, cls1, angle."""
    return np.array([rows], dtype=np.float32)


# A low-confidence row (0.20) and a comfortably-high-confidence row (0.35),
# well clear of every threshold under test (0.15 / 0.3 / 0.5) so each test's
# intent is unambiguous.
LOW_CONF_ROW = [100.0, 100.0, 50.0, 20.0, 0.20, 0.0, 0.0]
HIGH_CONF_ROW = [400.0, 400.0, 50.0, 20.0, 0.35, 0.0, 0.0]


def test_settings_default_is_none():
    """The setting is a no-op until explicitly configured."""
    assert Settings().ctd_obb_line_confidence is None


def test_default_none_reproduces_prior_min_block_confidence_030_behavior():
    """obb_line_confidence=None -> line_conf falls back to min(block_confidence, 0.3).

    block_confidence=0.4 -> line_conf=0.3: the 0.20-conf row is dropped, the
    0.35-conf row survives. This pins the EXACT prior hardcoded behavior.
    """
    svc = _bare_service(block_confidence=0.4, min_area=1, obb_line_confidence=None)
    lines_map = _obb_tensor([LOW_CONF_ROW, HIGH_CONF_ROW])
    out = svc._extract_text_lines(lines_map, scale=1.0, padded_size=(1000, 1000), orig_size=(1000, 1000))
    assert len(out) == 1
    assert out[0]["confidence"] == pytest.approx(0.35)


def test_lowered_override_recovers_the_low_confidence_line():
    """obb_line_confidence=0.15 keeps the 0.20-conf row too (independent of
    block_confidence, which stays at its normal 0.4)."""
    svc = _bare_service(block_confidence=0.4, min_area=1, obb_line_confidence=0.15)
    lines_map = _obb_tensor([LOW_CONF_ROW, HIGH_CONF_ROW])
    out = svc._extract_text_lines(lines_map, scale=1.0, padded_size=(1000, 1000), orig_size=(1000, 1000))
    assert len(out) == 2
    confs = sorted(o["confidence"] for o in out)
    assert confs == pytest.approx([0.20, 0.35])


def test_raised_override_is_independent_of_block_confidence():
    """A stricter override (0.5) drops the 0.35-conf row too, even though
    block_confidence (0.4) alone would not have implied that -- proves the
    two knobs are decoupled, not just re-deriving one from the other."""
    svc = _bare_service(block_confidence=0.4, min_area=1, obb_line_confidence=0.5)
    lines_map = _obb_tensor([LOW_CONF_ROW, HIGH_CONF_ROW])
    out = svc._extract_text_lines(lines_map, scale=1.0, padded_size=(1000, 1000), orig_size=(1000, 1000))
    assert out == []


def test_init_wires_setting_from_config(monkeypatch, tmp_path):
    """``__init__`` reads ``settings.ctd_obb_line_confidence`` into
    ``self.obb_line_confidence`` -- with the ONNX session creation mocked out
    so this never touches onnxruntime/GPU (other agents own the GPU in this
    environment)."""
    from app.config import settings as global_settings
    from pathlib import Path

    if not Path(global_settings.ctd_model_path).exists():
        pytest.skip("CTD model artifact not present in this checkout (gitignored weights)")

    class _FakeSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return []

    monkeypatch.setattr(ComicTextDetectorService, "_select_providers", lambda self: ["CPUExecutionProvider"])
    monkeypatch.setattr(
        ComicTextDetectorService, "_create_session", lambda self, model_path, providers: _FakeSession()
    )
    monkeypatch.setattr(global_settings, "ctd_obb_line_confidence", 0.22)

    svc = ComicTextDetectorService(model_path=global_settings.ctd_model_path)
    assert svc.obb_line_confidence == 0.22
