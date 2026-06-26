"""WS / pipelined OCR branch must thread REAL per-bubble OCR confidence.

Regression guard for the live-path name-invention bug: the pipelined
(WebSocket / streaming) branch of `process_single_image` used to hard-code
``kept_ocr_confs = [None] * len(ocr_texts)``, so the #3-fix low-confidence
name-invention suppressor (``canonicalize_names``/``postedit_one`` via
``ocr_conf``) NEVER fired on the LIVE extension path — only the batch path
(`scripts/batch_translate_chapter.py`) threaded real confidence. The result was
hallucinated proper names (Lona-san, Torachance, ...) from garbled low-conf OCR.

These tests drive the REAL pipelined branch end-to-end (real detector/OCR are
faked; the real japanese-filter + real `postedit_one`/`canonicalize_names`
run) and assert that a low-confidence generic-kana bubble is suppressed to the
safe generic gloss instead of keeping the model-invented proper name — mirroring
the batch-path test in ``test_translation_postedit.py``.
"""
import asyncio

import cv2
import numpy as np
import pytest

import app.routers.translate as tr


# Two 4x4 white tiles encoded once; the detector fake returns two blocks so the
# pipelined branch (use_pipeline_overlap and len(crops) > 1) is taken.
_IMG = np.full((8, 8, 3), 255, np.uint8)
_B64 = "data:image/png;base64," + __import__("base64").b64encode(
    cv2.imencode(".png", _IMG)[1].tobytes()
).decode()


class _FakeDetector:
    """Returns two text blocks + matching text_lines so crop count > 1."""

    def __init__(self, n_blocks: int = 2):
        self._n = n_blocks

    async def detect(self, image_np):
        blocks = [
            {"bbox": [0, 0, 4, 4], "minX": 0, "minY": 0, "maxX": 4, "maxY": 4},
            {"bbox": [4, 0, 8, 4], "minX": 4, "minY": 0, "maxX": 8, "maxY": 4},
        ][: self._n]
        return {"blocks": blocks, "text_lines": [], "mask": None}

    def crop_regions(self, image_np, blocks):
        return [np.full((4, 4, 3), 255, np.uint8) for _ in blocks]


class _FakeOCR:
    """Per-crop OCR with controllable text + confidence.

    NOT a ParseqOCRService, so the prefetched text_lines path is skipped and the
    pipelined branch calls ``recognize_text_batch_with_conf`` per mini-batch.
    """

    def __init__(self, results):
        # results: list of (text, conf), one per crop, in crop order.
        self._results = results

    async def recognize_text_batch_with_conf(self, crops):
        # Pop the next len(crops) results (mini-batches are sequential).
        n = len(crops)
        out, self._results = self._results[:n], self._results[n:]
        return out


def _patch_pipeline(monkeypatch, ocr_results, translations):
    """Wire fakes so the real pipelined branch runs deterministically.

    The REAL japanese-filter + REAL postedit_one/canonicalize_names execute; only
    detection, OCR, translation and inpaint are stubbed.
    """
    monkeypatch.setattr(tr, "detector_service", _FakeDetector(len(ocr_results)))
    monkeypatch.setattr(tr, "ocr_service", _FakeOCR(list(ocr_results)))
    monkeypatch.setattr(tr, "bubble_detector", None, raising=False)

    async def _fake_translation(texts, target_language, page_context_lines=None, target_positions=None):
        # Map each kept source line to its scripted translation (by source text).
        return [translations[t] for t in texts]

    monkeypatch.setattr(tr, "_run_translation", _fake_translation)

    # Force the pipelined (WS/streaming) branch; disable the garble gate so the
    # low-conf generic-kana bubble REACHES translation (the gate would otherwise
    # drop a 4-char kana line at low conf, masking the suppressor under test).
    monkeypatch.setattr(tr.settings, "use_pipeline_overlap", True)
    monkeypatch.setattr(tr.settings, "ocr_confidence_gate_enabled", False)
    monkeypatch.setattr(tr.settings, "japanese_filter_enabled", True)
    monkeypatch.setattr(tr.settings, "english_early_exit_enabled", False)
    monkeypatch.setattr(tr.settings, "orphan_line_recovery", False)
    monkeypatch.setattr(tr.settings, "enable_inpainting", False)
    monkeypatch.setattr(tr.settings, "overlap_inpaint", False)


def _run(target="English"):
    sem = asyncio.Semaphore(1)
    return asyncio.run(tr.process_single_image(0, _B64, target, sem))


def test_ws_pipelined_low_conf_suppresses_invented_name(monkeypatch):
    """A low-conf generic-kana bubble on the WS path must NOT keep an invented
    proper name — the real ocr_conf threads through to the suppressor.

    Bubble 0: high-conf real dialogue (decoy, keeps its translation).
    Bubble 1: 'おばさん' at conf 0.30 -> model invented 'Sue' -> must become the
    generic gloss ('auntie'), proving kept_ocr_confs carried the real 0.30.
    """
    _patch_pipeline(
        monkeypatch,
        ocr_results=[("こんにちは", 0.95), ("おばさん", 0.30)],
        translations={"こんにちは": "Hello", "おばさん": "Sue"},
    )

    idx, boxes, _plate = _run()

    by_src = {b.ocrText: b.translatedText for b in boxes}
    assert by_src["こんにちは"] == "Hello", "high-conf decoy unchanged"
    # THE BUG: with kept_ocr_confs hard-coded to None this stayed 'Sue'.
    assert by_src["おばさん"] != "Sue", "low-conf invented name must be suppressed"
    assert by_src["おばさん"] == "auntie", "suppressed to the generic gloss"


def test_ws_pipelined_high_conf_keeps_name(monkeypatch):
    """Symmetry guard: at HIGH confidence the same bubble is NOT suppressed — the
    real conf (not a blanket None/low) flows through, so genuine names survive.
    """
    _patch_pipeline(
        monkeypatch,
        ocr_results=[("こんにちは", 0.95), ("おばさん", 0.95)],
        translations={"こんにちは": "Hello", "おばさん": "Sue"},
    )

    idx, boxes, _plate = _run()
    by_src = {b.ocrText: b.translatedText for b in boxes}
    assert by_src["おばさん"] == "Sue", "high-conf name must NOT be suppressed"


def test_ws_pipelined_kept_ocr_confs_not_all_none(monkeypatch):
    """Direct contract guard: the pipelined branch must build kept_ocr_confs from
    real recognition confidence, not a list of None (the original bug).

    We capture the ocr_conf actually handed to postedit_one.
    """
    seen = []
    real_postedit = tr.postedit_one

    def _spy_postedit(en, jp, ocr_conf=None):
        seen.append((jp, ocr_conf))
        return real_postedit(en, jp, ocr_conf=ocr_conf)

    _patch_pipeline(
        monkeypatch,
        ocr_results=[("こんにちは", 0.95), ("おばさん", 0.30)],
        translations={"こんにちは": "Hello", "おばさん": "Sue"},
    )
    monkeypatch.setattr(tr, "postedit_one", _spy_postedit)

    _run()

    confs = {jp: c for jp, c in seen}
    assert confs["おばさん"] == pytest.approx(0.30), "real low conf threaded, not None"
    assert confs["こんにちは"] == pytest.approx(0.95), "real high conf threaded, not None"
    assert not all(c is None for c in confs.values()), "kept_ocr_confs must not be all None"
