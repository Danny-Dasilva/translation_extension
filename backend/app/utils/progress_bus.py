"""In-process progress event bus for SSE streaming.

Provides a single-process, broker-less pub/sub channel for pipeline progress
events keyed by job_id. Designed for per-stage progress reporting
(detect -> ocr -> translate -> render) that the frontend can consume via the
SSE endpoint in `app.routers.events`.

Integration (future):

    from app.utils.progress_bus import bus

    async def process_single_image(req, job_id: str | None = None):
        await bus.emit(job_id, "detect", 0, 4)
        # ... detect ...
        await bus.emit(job_id, "ocr", 1, 4)
        # ... ocr ...
        await bus.emit(job_id, "translate", 2, 4)
        # ... translate ...
        await bus.emit(job_id, "render", 3, 4)
        # ... render ...
        await bus.finish(job_id)

The router is responsible for generating the job_id (e.g. uuid4) and handing
it back to the client so the client can connect to `/events/{job_id}`.

Reference: koharu-rpc/src/events.rs:55-102 (uses tokio broadcast + ring buffer
for replay-on-reconnect). Our version is simpler: per-job asyncio.Queue,
no ring buffer — reconnects only see future events.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import AsyncIterator

logger = logging.getLogger(__name__)


# Sentinel used internally to signal end-of-stream to subscribers.
_FINISH_SENTINEL = object()


@dataclass
class ProgressEvent:
    """A single progress event emitted by the pipeline.

    Attributes:
        job_id: Opaque identifier tying events to one pipeline run.
        stage: Human-readable stage name (e.g. "detect", "ocr").
        index: Zero-based index of the current stage.
        total: Total number of stages in the pipeline.
        percent: Derived progress percentage in [0.0, 100.0].
        note: Optional free-form note (e.g. "12 regions detected").
        seq: Monotonic per-job sequence number assigned by the bus.
    """

    job_id: str
    stage: str
    index: int
    total: int
    percent: float
    note: str | None = None
    seq: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _FinishMarker:
    """Internal marker carried on the queue to signal end-of-stream."""

    status: str = "ok"


class ProgressBus:
    """Per-job in-process pub/sub for progress events.

    One queue per job_id. `emit` is fire-and-forget from the producer's
    point of view (it awaits `queue.put` which is bounded by asyncio's
    default unbounded queue semantics — i.e. never blocks in practice).

    This bus is intentionally minimal:
      * No ring buffer — a subscriber that connects after events were
        emitted will only receive future events. `Last-Event-ID`
        reconnect replay is therefore a no-op in this implementation.
      * No broker — everything is in-process. Horizontal scaling would
        require swapping in Redis pub/sub or similar.
      * One subscriber per job is the common case. If multiple subscribe
        to the same job, only the first call will receive events (the
        queue is shared, events are consumed once). Extend with a
        broadcast fan-out if needed.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}
        self._seq: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    def _get_or_create_queue(self, job_id: str) -> asyncio.Queue:
        q = self._queues.get(job_id)
        if q is None:
            q = asyncio.Queue()
            self._queues[job_id] = q
        return q

    async def emit(
        self,
        job_id: str,
        stage: str,
        index: int,
        total: int,
        note: str | None = None,
    ) -> ProgressEvent:
        """Emit a progress event for `job_id`.

        Returns the emitted ProgressEvent (useful for tests/logging).
        """
        async with self._lock:
            self._seq[job_id] += 1
            seq = self._seq[job_id]
            q = self._get_or_create_queue(job_id)

        percent = 0.0
        if total > 0:
            # Stage `index` means "stage `index` has started"; use index/total
            # as the fraction completed BEFORE this stage. Callers that want
            # "completed fraction" can emit index=total at the end.
            percent = max(0.0, min(100.0, (index / total) * 100.0))

        event = ProgressEvent(
            job_id=job_id,
            stage=stage,
            index=index,
            total=total,
            percent=percent,
            note=note,
            seq=seq,
        )
        await q.put(event)
        logger.debug(
            "progress_bus.emit job=%s stage=%s index=%d/%d seq=%d",
            job_id, stage, index, total, seq,
        )
        return event

    async def subscribe(self, job_id: str) -> AsyncIterator[ProgressEvent]:
        """Async iterator over progress events for `job_id`.

        Yields ProgressEvent instances until a finish sentinel is received,
        at which point the iterator stops. If the job has not yet emitted
        any events, this will create a queue and wait for the first event.
        """
        async with self._lock:
            q = self._get_or_create_queue(job_id)

        try:
            while True:
                item = await q.get()
                if isinstance(item, _FinishMarker):
                    logger.debug(
                        "progress_bus.subscribe job=%s finished status=%s",
                        job_id, item.status,
                    )
                    return
                yield item
        finally:
            # Best-effort cleanup: only drop the queue if nothing else is
            # waiting and no more events are buffered.
            await self._maybe_cleanup(job_id)

    async def finish(self, job_id: str, status: str = "ok") -> None:
        """Signal that no more events will be emitted for `job_id`.

        This places a finish sentinel on the queue so any subscriber
        exits its loop. Safe to call multiple times (subsequent calls
        are a no-op if the queue has already been cleaned up).
        """
        async with self._lock:
            q = self._queues.get(job_id)
        if q is None:
            logger.debug("progress_bus.finish job=%s had no queue", job_id)
            return
        await q.put(_FinishMarker(status=status))

    async def _maybe_cleanup(self, job_id: str) -> None:
        async with self._lock:
            q = self._queues.get(job_id)
            if q is not None and q.empty():
                self._queues.pop(job_id, None)
                self._seq.pop(job_id, None)
                logger.debug("progress_bus.cleanup job=%s", job_id)


# Module-level singleton. Import as `from app.utils.progress_bus import bus`.
bus = ProgressBus()


__all__ = ["ProgressEvent", "ProgressBus", "bus"]
