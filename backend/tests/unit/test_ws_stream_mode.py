"""WebSocket layer: stream-mode gating + terminal-frame wiring.

Verifies the ``translation_stream_events`` flag on ``_process_image``:
  * OFF (default): returns the legacy monolithic dict (caller sends it) and does
    NOT push any event frames itself.
  * ON: returns None (frames already sent) and pushes the frames produced by a
    (patched) ``process_single_image`` straight to ``websocket.send_json``.
  * ON + undecodable bytes: a terminal ``error`` frame is sent, no monolithic
    reply.
"""
import asyncio

import cv2
import numpy as np

import app.routers.websocket_upload as wsu


_IMG = np.full((8, 8, 3), 255, np.uint8)
_PNG = cv2.imencode(".png", _IMG)[1].tobytes()


class _FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, obj):
        self.sent.append(obj)


def test_legacy_mode_returns_monolithic_dict(monkeypatch):
    monkeypatch.setattr(wsu.settings, "translation_stream_events", False)

    async def _fake_psi(idx, b64, lang, sem, job_id=None, **kw):
        # Legacy path must NOT pass on_event.
        assert "on_event" not in kw or kw["on_event"] is None
        return (idx, [], None)

    monkeypatch.setattr(wsu, "process_single_image", _fake_psi)
    ws = _FakeWS()

    result = asyncio.run(wsu._process_image(_PNG, "English", "cid", ws))

    assert result is not None and result["success"] is True
    assert result["images"] == [[]]
    assert ws.sent == [], "legacy mode: the LAYER sends nothing itself"


def test_stream_mode_pushes_frames_and_returns_none(monkeypatch):
    monkeypatch.setattr(wsu.settings, "translation_stream_events", True)

    async def _fake_psi(idx, b64, lang, sem, job_id=None, *, on_event=None,
                        session_id=None, image_index=0):
        assert on_event is not None
        await on_event({"v": 1, "type": "detections", "session_id": session_id,
                        "image_index": image_index, "boxes": []})
        await on_event({"v": 1, "type": "done", "session_id": session_id,
                        "image_index": image_index})
        return (idx, [], None)

    monkeypatch.setattr(wsu, "process_single_image", _fake_psi)
    ws = _FakeWS()

    result = asyncio.run(wsu._process_image(_PNG, "English", "cid", ws))

    assert result is None, "stream mode: frames already sent, nothing to send after"
    types = [f["type"] for f in ws.sent]
    assert types == ["detections", "done"]
    assert all(f["v"] == 1 for f in ws.sent)


def test_stream_mode_invalid_image_emits_error_frame(monkeypatch):
    monkeypatch.setattr(wsu.settings, "translation_stream_events", True)
    # process_single_image must never be reached for undecodable bytes.
    monkeypatch.setattr(wsu, "process_single_image", None)
    ws = _FakeWS()

    result = asyncio.run(wsu._process_image(b"not-an-image", "English", "cid", ws))

    assert result is None
    assert len(ws.sent) == 1
    err = ws.sent[0]
    assert err["type"] == "error" and err["v"] == 1
    assert "decode" in err["error"]
