"""WebSocket path must delegate to the canonical HTTP pipeline.

Regression guard for the pipeline-divergence bug: the WS `_process_image`
used to reimplement detect->ocr->translate->build with the LEGACY translate
format and no postedit / no English early-exit / no inpaint. It now delegates
to `process_single_image` (the HTTP pipeline) and serializes TextBox objects
with `model_dump()`, so the WS JSON is field-identical to the HTTP response.
"""
import asyncio
import inspect

import cv2
import numpy as np

from app.models.response import TextBox, TextRegion

# Small valid white PNG used to exercise the cv2 decode path in _process_image.
_PNG_1X1 = cv2.imencode(".png", np.full((4, 4, 3), 255, np.uint8))[1].tobytes()


# ---------------------------------------------------------------------------
# Static guards: WS module imports the canonical pipeline, both endpoints call
# the shared _process_image, and the legacy reimplementation is gone.
# ---------------------------------------------------------------------------
def test_ws_imports_canonical_pipeline():
    import app.routers.websocket_upload as ws

    # Delegates to the HTTP pipeline + shares the GPU semaphore.
    assert hasattr(ws, "process_single_image"), "WS must import process_single_image"
    assert hasattr(ws, "_gpu_semaphore"), "WS must import the shared GPU semaphore"


def test_ws_process_image_delegates_not_reimplements():
    import app.routers.websocket_upload as ws

    src = inspect.getsource(ws._process_image)
    # It must call the canonical pipeline...
    assert "process_single_image(" in src
    # ...and NOT reimplement the legacy translate path.
    assert "translate_batched" not in src
    assert "translate_single" not in src


def test_both_ws_endpoints_use_process_image():
    import app.routers.websocket_upload as ws

    s1 = inspect.getsource(ws.websocket_translate)
    s2 = inspect.getsource(ws.websocket_translate_with_language)
    assert "_process_image(" in s1
    assert "_process_image(" in s2


# ---------------------------------------------------------------------------
# Response-contract guard: WS serialization is field-identical to HTTP.
# ---------------------------------------------------------------------------
def _make_textbox() -> TextBox:
    return TextBox(
        ocrText="こんにちは",
        originalLanguage="ja",
        minX=10, minY=20, maxX=110, maxY=70,
        fontHeightPx=18,
        fontColor="#000000",
        fontStrokeColor="#FFFFFF",
        translatedText="Hello",
        textRegions=[TextRegion(minX=10, minY=20, maxX=110, maxY=70)],
        bubbleRect=TextRegion(minX=5, minY=15, maxX=120, maxY=80),
        confidence=0.9,
    )


def test_ws_response_contract_matches_http(monkeypatch):
    """_process_image returns the WS contract with TextBox dicts == model_dump()."""
    import app.routers.websocket_upload as ws

    tb = _make_textbox()
    inpaint_plate = "data:image/png;base64,AAAA"

    async def fake_process_single_image(idx, base64_image, target_language, semaphore, job_id=None):
        # Sanity: WS must pass idx=0 and a data-URI-or-base64 string + job_id=None
        assert idx == 0
        assert isinstance(base64_image, str) and len(base64_image) > 0
        assert job_id is None
        return (0, [tb], inpaint_plate)

    monkeypatch.setattr(ws, "process_single_image", fake_process_single_image)

    result = asyncio.run(ws._process_image(_PNG_1X1, "English", "testcli"))

    assert result["success"] is True
    assert "session_id" in result
    # images is a list with one image, that image is a list of serialized boxes
    assert result["images"] == [[tb.model_dump()]]
    # inpaint plate flows through (HTTP returns it; WS used to drop it)
    assert result["inpainted_image_base64"] == [inpaint_plate]
    # timing/debug block preserved
    assert "debug" in result and "timing" in result["debug"]
    assert "request_total_ms" in result["debug"]["timing"]

    # Field-identical to the HTTP path: the box dict keys == TextBox model fields.
    ws_box = result["images"][0][0]
    assert sorted(ws_box.keys()) == sorted(tb.model_dump().keys())


def test_ws_empty_blocks_returns_empty_image(monkeypatch):
    """No detected blocks -> process_single_image returns [], WS returns images:[[]]."""
    import app.routers.websocket_upload as ws

    async def fake_empty(idx, base64_image, target_language, semaphore, job_id=None):
        return (0, [], None)

    monkeypatch.setattr(ws, "process_single_image", fake_empty)

    result = asyncio.run(ws._process_image(_PNG_1X1, "English", "testcli"))
    assert result["success"] is True
    assert result["images"] == [[]]
    assert result["inpainted_image_base64"] == [None]


def test_ws_invalid_image_returns_error(monkeypatch):
    """Undecodable bytes -> error response (never reaches the pipeline)."""
    import app.routers.websocket_upload as ws

    async def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("pipeline should not run on invalid image")

    monkeypatch.setattr(ws, "process_single_image", boom)

    result = asyncio.run(ws._process_image(b"not-an-image", "English", "testcli"))
    assert result["success"] is False
    assert "error" in result
