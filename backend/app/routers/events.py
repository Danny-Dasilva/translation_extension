"""SSE endpoint for per-job pipeline progress streaming.

Clients connect to `GET /events/{job_id}` to receive
`event: progress` messages as the backend emits them via
`app.utils.progress_bus.bus`. When the pipeline calls
`bus.finish(job_id)`, an `event: done` frame is sent and the
connection closes.

Message format (SSE):

    event: progress
    data: {"job_id": "...", "stage": "ocr", "index": 1, "total": 4,
           "percent": 25.0, "note": null, "seq": 2}
    id: 2

    event: done
    data: {"status": "ok"}

Reconnect behavior:
    The client MAY send `Last-Event-ID: <seq>` when reconnecting.
    Because this bus is in-process and has no ring buffer, we cannot
    replay missed events — a reconnecting subscriber will only see
    events emitted after the subscription time. The header is accepted
    (and logged) but effectively ignored. For resilient replay we'd
    need a bounded ring buffer per job (see koharu-rpc/src/events.rs).

Heartbeat:
    An SSE comment (`: keepalive`) is emitted every 15 seconds while
    the stream is idle, so proxies / browsers don't drop the
    connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse

from app.utils.progress_bus import ProgressEvent, bus

logger = logging.getLogger(__name__)

router = APIRouter()

# How long to wait for the next event before emitting a keepalive comment.
_HEARTBEAT_SECONDS = 15.0


def _format_progress(event: ProgressEvent) -> str:
    payload = json.dumps(event.to_dict(), separators=(",", ":"))
    return f"event: progress\ndata: {payload}\nid: {event.seq}\n\n"


def _format_done(status: str = "ok") -> str:
    payload = json.dumps({"status": status}, separators=(",", ":"))
    return f"event: done\ndata: {payload}\n\n"


def _format_heartbeat() -> str:
    return ": keepalive\n\n"


async def _event_stream(
    job_id: str,
    request: Request,
    last_event_id: str | None,
) -> AsyncIterator[str]:
    """Yield SSE-formatted strings for `job_id` until the bus signals done."""
    if last_event_id:
        # Accepted but cannot replay — see module docstring.
        logger.info(
            "events.stream job=%s Last-Event-ID=%s (replay unsupported; continuing forward-only)",
            job_id, last_event_id,
        )

    subscription = bus.subscribe(job_id).__aiter__()

    try:
        while True:
            if await request.is_disconnected():
                logger.debug("events.stream job=%s client disconnected", job_id)
                return

            next_task = asyncio.create_task(subscription.__anext__())
            try:
                done, _pending = await asyncio.wait(
                    {next_task},
                    timeout=_HEARTBEAT_SECONDS,
                )
            except asyncio.CancelledError:
                next_task.cancel()
                raise

            if not done:
                # Timeout -> heartbeat and loop.
                next_task.cancel()
                try:
                    await next_task
                except (asyncio.CancelledError, BaseException):
                    pass
                yield _format_heartbeat()
                continue

            try:
                event = next_task.result()
            except StopAsyncIteration:
                # Bus signalled finish.
                yield _format_done("ok")
                return
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("events.stream job=%s error: %s", job_id, exc)
                yield _format_done("error")
                return

            yield _format_progress(event)
    finally:
        # Ensure the async generator is closed so ProgressBus can clean up.
        try:
            await subscription.aclose()  # type: ignore[attr-defined]
        except Exception:
            pass


@router.get("/events/{job_id}")
async def events(
    job_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Server-Sent Events stream for a single pipeline job."""
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # disable nginx buffering if present
    }
    return StreamingResponse(
        _event_stream(job_id, request, last_event_id),
        media_type="text/event-stream",
        headers=headers,
    )
