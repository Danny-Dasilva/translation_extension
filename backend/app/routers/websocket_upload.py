"""WebSocket binary upload router for faster image transfer."""

import base64
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import cv2

# Delegate to the canonical HTTP pipeline so the WS path produces IDENTICAL
# output to /translate (v11 page-context translate, postedit glossaries,
# English early-exit, classical inpaint). Reusing process_single_image + its
# GPU semaphore keeps the two paths from diverging again.
from app.routers.translate import (
    process_single_image,
    _gpu_semaphore,
)
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/translate")
async def websocket_translate(websocket: WebSocket):
    """
    WebSocket endpoint for binary image upload and translation.

    Performance advantage: WebSocket bypasses HTTP flow control entirely.
    - HTTP: 10 x 2MB images = ~1500ms overhead (flow control pause/resume cycles)
    - WebSocket: 10 x 2MB images = ~250ms overhead (persistent connection, minimal framing)

    Protocol:
    1. Client connects and sends binary image data
    2. Server processes and returns JSON result
    3. Connection stays open for multiple images

    Expected: 87% faster for batch uploads, 30-40ms faster per single image
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8]
    logger.info(f"WebSocket client connected: {client_id}")

    try:
        while True:
            # Receive binary image data (blocks until message arrives)
            # WebSocket handles frame assembly automatically - we get complete messages
            # NOTE: Don't measure time here - receive_bytes() blocks waiting for client
            image_bytes = await websocket.receive_bytes()

            logger.info(f"[WS:{client_id}] Received {len(image_bytes)} bytes")

            # Process image through translation pipeline
            result = await _process_image(image_bytes, "English", client_id, websocket)

            # Send JSON response. In STREAM mode `_process_image` already sent the
            # event frames (incl. the terminal done/error) and returns None.
            if result is not None:
                await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "success": False,
                "error": str(e)
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.websocket("/translate/{target_language}")
async def websocket_translate_with_language(websocket: WebSocket, target_language: str):
    """
    WebSocket endpoint with configurable target language.

    Usage: ws://localhost:8000/ws/translate/English
           ws://localhost:8000/ws/translate/Spanish
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8]
    logger.info(f"WebSocket client connected: {client_id} (target: {target_language})")

    try:
        while True:
            # Receive binary image data (blocks until message arrives)
            # NOTE: Don't measure time here - receive_bytes() blocks waiting for client
            image_bytes = await websocket.receive_bytes()

            logger.info(f"[WS:{client_id}] Received {len(image_bytes)} bytes")

            result = await _process_image(image_bytes, target_language, client_id, websocket)
            if result is not None:
                await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"success": False, "error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


async def _process_image(
    image_bytes: bytes,
    target_language: str,
    client_id: str,
    websocket: Optional[WebSocket] = None,
) -> Optional[dict]:
    """
    Process a binary image through the CANONICAL HTTP translation pipeline.

    STREAM MODE (settings.translation_stream_events True): each server->client
    event frame (detections/tl/revise/plate + terminal done/error, see
    src/types/stream.ts) is sent via `websocket.send_json` as
    `process_single_image` produces it, and this returns None (the `done` frame
    IS the terminal — no monolithic reply follows). The frontend assembles the
    legacy shape from the frames. There is NO client opt-in beyond the flag: the
    extension just connects; whether the socket streams is a pure server setting.

    LEGACY MODE (flag False, the default): unchanged — returns the single
    monolithic dict below and the caller sends it as one JSON message.

    This delegates to `process_single_image` (the same function the HTTP
    /translate endpoint uses) so the WS path produces byte-identical output:
    v11 page-context translation, postedit glossaries (SFX/name/register),
    English early-exit, and classical inpaint. It used to reimplement the
    pipeline with the legacy translate format and diverged badly.

    Returns a dict matching the extension's TranslateResponse contract:
        { success, session_id, images: [[<textbox dicts>]],
          inpainted_image_base64: [<plate or null>], debug: { timing: {...} } }

    Note: WebSocket doesn't report "receive time" since the connection is
    persistent and receive_bytes() blocks until a message arrives (would
    include user idle time).
    """
    session_id = str(uuid.uuid4())[:8]
    processing_start = time.time()
    frame_size = len(image_bytes)
    stream = bool(settings.translation_stream_events)

    try:
        # Validate decodability up-front so malformed frames return a clean
        # error instead of surfacing as a pipeline exception. cv2 here mirrors
        # the decode the pipeline does internally (PNG/JPEG/WebP auto-detect).
        if cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED) is None:
            if stream:
                # Terminal error frame (no monolithic reply follows).
                await websocket.send_json({
                    "v": 1, "type": "error", "session_id": session_id,
                    "image_index": 0,
                    "error": "Invalid image data - could not decode",
                })
                return None
            return {
                "success": False,
                "error": "Invalid image data - could not decode",
                "session_id": session_id,
            }

        # process_single_image expects a base64 string (data-URI prefix
        # optional — decode_base64_to_numpy strips it). The HTTP path receives
        # data URLs in `base64Images`, so build the same `data:image/...;base64,`
        # form here for parity.
        base64_image = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")

        if stream:
            # STREAM MODE: process_single_image emits every event frame (incl. the
            # terminal done/error) through this callback. reuse session_id per image.
            async def _on_event(frame: dict) -> None:
                await websocket.send_json(frame)

            await process_single_image(
                0, base64_image, target_language, _gpu_semaphore, job_id=None,
                on_event=_on_event, session_id=session_id, image_index=0,
            )
            total_time = (time.time() - processing_start) * 1000
            logger.info(
                f"[WS:{client_id}] {frame_size} bytes -> streamed, TOTAL={total_time:.1f}ms"
            )
            return None

        # Single-image WS request: idx=0, share the HTTP GPU semaphore, no
        # progress job (WS streams its own debug block instead).
        _idx, text_boxes, inpainted_b64 = await process_single_image(
            0, base64_image, target_language, _gpu_semaphore, job_id=None
        )

        # Serialize TextBox pydantic models to dicts EXACTLY as the HTTP
        # /translate endpoint does (FastAPI runs model_dump() via the
        # TranslateResponse response_model). Same keys -> the extension
        # renderer treats WS and HTTP boxes identically.
        serialized_boxes = [tb.model_dump() for tb in text_boxes]

        total_time = (time.time() - processing_start) * 1000
        logger.info(
            f"[WS:{client_id}] {frame_size} bytes -> {len(serialized_boxes)} boxes, "
            f"inpaint={'yes' if inpainted_b64 else 'no'}, TOTAL={total_time:.1f}ms"
        )

        return {
            "success": True,
            "session_id": session_id,
            # One image per WS frame -> single-element image list.
            "images": [serialized_boxes],
            # Inpaint plate aligned with images[] (HTTP returns this too; the
            # old WS path dropped it, so the extension fell back to white-boxing).
            "inpainted_image_base64": [inpainted_b64],
            "debug": {
                "timing": {
                    "ws_frame_bytes": frame_size,
                    # Per-stage timings live inside process_single_image's logs;
                    # the WS debug block only needs the wall-clock total.
                    "request_total_ms": round(total_time, 2),
                },
                "total_ms": round(total_time, 2),
            },
        }

    except Exception as e:
        logger.error(f"[WS:{client_id}] Processing error: {e}", exc_info=True)
        if stream:
            # process_single_image already emits its own terminal error frame for
            # in-pipeline failures; this covers failures OUTSIDE it (e.g. the
            # base64 encode). Best-effort — a dead socket must not re-raise here.
            try:
                await websocket.send_json({
                    "v": 1, "type": "error", "session_id": session_id,
                    "image_index": 0, "error": str(e),
                })
            except Exception:
                pass
            return None
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }
