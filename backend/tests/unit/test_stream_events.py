"""Per-bubble stream emission (WS event-frame protocol) tests.

Drives the REAL pipelined branch of ``process_single_image`` with faked
detector/OCR/translation (the pattern from ``test_ws_ocr_conf_suppression.py``)
and asserts the server->client event contract in ``src/types/stream.ts``:

  detections (first, geometry+seed) -> tl (per bubble, any order, idempotent by
  index) -> revise (supersedes a prior tl on a post-pass change) -> exactly one
  terminal done|error.

Key guarantees under test:
  * ``on_event=None`` => ZERO frames and a byte-identical result vs. today.
  * frame SEQUENCE: detections first, a tl per kept bubble, a revise when a
    page-level post-pass changes a bubble's final text, exactly one terminal.
  * ERROR path emits a terminal ``error`` frame.
  * as-completed tl callback is ORDERING-INDEPENDENT (a slow bubble does not
    hold back a fast one, and the final assembled text is correct regardless).
"""
import asyncio

import cv2
import numpy as np
import pytest

import app.routers.translate as tr


_IMG = np.full((8, 8, 3), 255, np.uint8)
_B64 = "data:image/png;base64," + __import__("base64").b64encode(
    cv2.imencode(".png", _IMG)[1].tobytes()
).decode()


class _FakeDetector:
    """Two text blocks (crop count > 1 -> the pipelined/WS branch)."""

    def __init__(self, n_blocks: int = 2):
        self._n = n_blocks

    async def detect(self, image_np):
        blocks = [
            {"bbox": [0, 0, 4, 4], "minX": 0, "minY": 0, "maxX": 4, "maxY": 4,
             "confidence": 0.9},
            {"bbox": [4, 0, 8, 4], "minX": 4, "minY": 0, "maxX": 8, "maxY": 4,
             "confidence": 0.8},
        ][: self._n]
        return {"blocks": blocks, "text_lines": [], "mask": None}

    def crop_regions(self, image_np, blocks):
        return [np.full((4, 4, 3), 255, np.uint8) for _ in blocks]


class _FakeOCR:
    def __init__(self, results):
        self._results = results

    async def recognize_text_batch_with_conf(self, crops):
        n = len(crops)
        out, self._results = self._results[:n], self._results[n:]
        return out


def _patch(monkeypatch, ocr_results, run_translation):
    monkeypatch.setattr(tr, "detector_service", _FakeDetector(len(ocr_results)))
    monkeypatch.setattr(tr, "ocr_service", _FakeOCR(list(ocr_results)))
    monkeypatch.setattr(tr, "bubble_detector", None, raising=False)
    monkeypatch.setattr(tr, "_run_translation", run_translation)
    monkeypatch.setattr(tr.settings, "use_pipeline_overlap", True)
    monkeypatch.setattr(tr.settings, "ocr_confidence_gate_enabled", False)
    monkeypatch.setattr(tr.settings, "japanese_filter_enabled", True)
    monkeypatch.setattr(tr.settings, "english_early_exit_enabled", False)
    monkeypatch.setattr(tr.settings, "orphan_line_recovery", False)
    monkeypatch.setattr(tr.settings, "enable_inpainting", False)
    monkeypatch.setattr(tr.settings, "overlap_inpaint", False)
    # No speech-bubble detector -> the bubble-keyed dedup path is skipped.
    monkeypatch.setattr(tr.settings, "translation_empty_bubble_backfill", False)


def _collector():
    frames = []

    async def on_event(frame):
        # Copy so later mutation of shared dicts can't rewrite history.
        frames.append(dict(frame))

    return frames, on_event


def _simple_translation(mapping):
    async def _run(texts, target_language, page_context_lines=None,
                   target_positions=None, merge_req=None,
                   page_image_data_url=None, on_marked_result=None):
        out = []
        for j, t in enumerate(texts):
            en = mapping[t]
            if on_marked_result is not None:
                await on_marked_result(j, en)
            out.append(en)
        return out

    return _run


def _run(on_event=None):
    sem = asyncio.Semaphore(1)
    return asyncio.run(
        tr.process_single_image(
            0, _B64, "English", sem, on_event=on_event, session_id="sess1"
        )
    )


# --------------------------------------------------------------------------- #
# 1. callback-off == byte-identical, NO frames
# --------------------------------------------------------------------------- #
def test_callback_off_emits_no_frames_and_same_result(monkeypatch):
    _patch(
        monkeypatch,
        ocr_results=[("こんにちは", 0.95), ("おはよう", 0.95)],
        run_translation=_simple_translation({"こんにちは": "Hello", "おはよう": "Morning"}),
    )
    # OFF: no on_event.
    idx_off, boxes_off, plate_off = _run(on_event=None)

    # ON: identical mapping, collect frames.
    _patch(
        monkeypatch,
        ocr_results=[("こんにちは", 0.95), ("おはよう", 0.95)],
        run_translation=_simple_translation({"こんにちは": "Hello", "おはよう": "Morning"}),
    )
    frames, on_event = _collector()
    idx_on, boxes_on, plate_on = _run(on_event=on_event)

    # The RESULT tuple is identical (frames are additive, not a behavior change).
    # ocr/translate TimeMs are wall-clock-derived jitter, not behavior -> drop.
    def _stable(box):
        d = box.model_dump()
        d.pop("ocrTimeMs", None)
        d.pop("translateTimeMs", None)
        return d

    assert idx_off == idx_on
    assert plate_off == plate_on
    assert [_stable(b) for b in boxes_off] == [_stable(b) for b in boxes_on]
    # OFF path emitted nothing (frames only exist on the ON run).
    assert frames, "ON run should emit frames"


# --------------------------------------------------------------------------- #
# 2. frame sequence: detections first, tl per bubble, single terminal
# --------------------------------------------------------------------------- #
def test_frame_sequence_detections_then_tl_then_done(monkeypatch):
    _patch(
        monkeypatch,
        ocr_results=[("こんにちは", 0.95), ("おはよう", 0.95)],
        run_translation=_simple_translation({"こんにちは": "Hello", "おはよう": "Morning"}),
    )
    frames, on_event = _collector()
    _run(on_event=on_event)

    types = [f["type"] for f in frames]
    assert types[0] == "detections", "detections must be FIRST"
    assert types[-1] == "done", "exactly one terminal, last"
    assert types.count("done") == 1 and types.count("error") == 0

    # Every frame carries the versioned envelope + session + image_index.
    for f in frames:
        assert f["v"] == 1
        assert f["session_id"] == "sess1"
        assert f["image_index"] == 0

    det = frames[0]
    assert [b["index"] for b in det["boxes"]] == [0, 1]
    b0 = det["boxes"][0]
    # Geometry + seed colors present; fontHeightPx OMITTED (post-translation).
    assert {"minX", "minY", "maxX", "maxY", "fontColor", "fontStrokeColor",
            "zIndex", "confidence", "originalLanguage"} <= set(b0)
    assert "fontHeightPx" not in b0

    # One tl per kept bubble, idempotent-by-index, carrying ocrText.
    tls = [f for f in frames if f["type"] == "tl"]
    by_index = {f["index"]: f for f in tls}
    assert set(by_index) == {0, 1}
    assert by_index[0]["translatedText"] == "Hello"
    assert by_index[0]["ocrText"] == "こんにちは"
    assert by_index[1]["translatedText"] == "Morning"


# --------------------------------------------------------------------------- #
# 3. revise supersedes a prior tl when a page-level post-pass changes text
# --------------------------------------------------------------------------- #
def test_revise_emitted_on_post_pass_change(monkeypatch):
    # tl emits the raw per-bubble EN; the run then rewrites bubble 1's final text
    # (simulating a post-pass e.g. dedup/glossary) so a revise must follow.
    async def _run_tl_then_change(texts, target_language, page_context_lines=None,
                                  target_positions=None, merge_req=None,
                                  page_image_data_url=None, on_marked_result=None):
        raw = {"こんにちは": "Hello", "おはよう": "Morning"}
        for j, t in enumerate(texts):
            if on_marked_result is not None:
                await on_marked_result(j, raw[t])
        # Final translations DIFFER for bubble 1 (post-pass correction).
        return ["Hello", "Good morning"]

    _patch(
        monkeypatch,
        ocr_results=[("こんにちは", 0.95), ("おはよう", 0.95)],
        run_translation=_run_tl_then_change,
    )
    frames, on_event = _collector()
    _run(on_event=on_event)

    revises = {f["index"]: f["translatedText"] for f in frames if f["type"] == "revise"}
    # Bubble 0 unchanged (Hello==Hello) -> NO revise; bubble 1 changed -> revise.
    assert 0 not in revises, "unchanged bubble must NOT be revised"
    assert revises.get(1) == "Good morning", "changed bubble must be revised to final"
    # Ordering: the revise for index 1 comes AFTER its tl.
    order = [(f["type"], f.get("index")) for f in frames
             if f["type"] in ("tl", "revise") and f.get("index") == 1]
    assert order == [("tl", 1), ("revise", 1)]


# --------------------------------------------------------------------------- #
# 4. error path emits a terminal error frame
# --------------------------------------------------------------------------- #
def test_error_path_emits_error_frame(monkeypatch):
    async def _boom(*a, **k):
        raise RuntimeError("translate exploded")

    _patch(
        monkeypatch,
        ocr_results=[("こんにちは", 0.95), ("おはよう", 0.95)],
        run_translation=_boom,
    )
    frames, on_event = _collector()
    idx, boxes, plate = _run(on_event=on_event)

    # Pipeline swallows + returns empty (unchanged HTTP-batch contract) ...
    assert boxes == [] and plate is None
    # ... but the stream got a terminal error frame (and detections still first).
    assert frames[0]["type"] == "detections"
    assert frames[-1]["type"] == "error"
    assert "translate exploded" in frames[-1]["error"]
    assert sum(1 for f in frames if f["type"] in ("done", "error")) == 1


# --------------------------------------------------------------------------- #
# 5. as-completed tl callback is ordering-independent
# --------------------------------------------------------------------------- #
def test_tl_callback_ordering_independent(monkeypatch):
    # Bubble 0 "translates" slowly, bubble 1 quickly: the FAST one's tl must be
    # emitted before the SLOW one's, and both final texts are correct.
    async def _run_out_of_order(texts, target_language, page_context_lines=None,
                                target_positions=None, merge_req=None,
                                page_image_data_url=None, on_marked_result=None):
        en = {"こんにちは": "Hello", "おはよう": "Morning"}
        delays = {0: 0.02, 1: 0.0}  # bubble 0 slow, bubble 1 fast

        async def _one(j, t):
            await asyncio.sleep(delays[j])
            if on_marked_result is not None:
                await on_marked_result(j, en[t])
            return j, en[t]

        pairs = await asyncio.gather(*(_one(j, t) for j, t in enumerate(texts)))
        pairs.sort(key=lambda p: p[0])
        return [e for _j, e in pairs]

    _patch(
        monkeypatch,
        ocr_results=[("こんにちは", 0.95), ("おはよう", 0.95)],
        run_translation=_run_out_of_order,
    )
    frames, on_event = _collector()
    _run(on_event=on_event)

    tl_order = [f["index"] for f in frames if f["type"] == "tl"]
    assert tl_order == [1, 0], "fast bubble 1's tl must precede slow bubble 0's"
    # Final assembled text is correct regardless of emission order (no revise
    # needed because raw == final for both).
    final = {f["index"]: f["translatedText"]
             for f in frames if f["type"] in ("tl", "revise")}
    assert final == {0: "Hello", 1: "Morning"}
