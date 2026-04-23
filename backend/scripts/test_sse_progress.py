"""End-to-end test of the in-process ProgressBus.

Runs two concurrent coroutines against a single `ProgressBus`:

  * A producer that emits 5 progress events (sleeping 300ms between
    each) and then calls `bus.finish()`.
  * A consumer that subscribes and prints every event as it arrives.

Demonstrates:
  - Events arrive in-order.
  - `seq` is monotonically incrementing per job_id.
  - The consumer exits cleanly when finish() is emitted.

Usage (from backend/):

    uv run python scripts/test_sse_progress.py

Output is mirrored to:

    thoughts/koharu-improvements/sse-progress/test.log
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# Make `app` importable when run from backend/ directly.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.utils.progress_bus import ProgressBus  # noqa: E402

REPO_ROOT = BACKEND_DIR.parent
LOG_PATH = REPO_ROOT / "thoughts" / "koharu-improvements" / "sse-progress" / "test.log"


def _setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Truncate previous log for a clean run.
    LOG_PATH.write_text("", encoding="utf-8")

    logger = logging.getLogger("sse_progress_test")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


async def producer(bus: ProgressBus, job_id: str, logger: logging.Logger) -> None:
    stages = [
        ("detect", 0, 4, "starting YOLO detection"),
        ("ocr", 1, 4, "running manga-ocr on 12 regions"),
        ("translate", 2, 4, "HY-MT1.5 JA->EN"),
        ("render", 3, 4, "compositing overlay"),
        ("render", 4, 4, "done"),
    ]
    for stage, index, total, note in stages:
        ev = await bus.emit(job_id, stage, index, total, note=note)
        logger.info(
            "producer emitted: stage=%s index=%d/%d percent=%.1f seq=%d",
            ev.stage, ev.index, ev.total, ev.percent, ev.seq,
        )
        await asyncio.sleep(0.3)
    await bus.finish(job_id)
    logger.info("producer finished job=%s", job_id)


async def consumer(bus: ProgressBus, job_id: str, logger: logging.Logger) -> list:
    received: list = []
    async for ev in bus.subscribe(job_id):
        received.append(ev)
        logger.info(
            "consumer received: seq=%d stage=%s index=%d/%d percent=%.1f note=%s",
            ev.seq, ev.stage, ev.index, ev.total, ev.percent, ev.note,
        )
    logger.info("consumer stream closed for job=%s (got %d events)", job_id, len(received))
    return received


async def main() -> int:
    logger = _setup_logging()
    logger.info("=== SSE ProgressBus smoke test ===")

    bus = ProgressBus()
    job_id = f"test-job-{int(time.time())}"
    logger.info("job_id=%s", job_id)

    started = time.monotonic()
    # Run producer + consumer concurrently so the consumer sees events
    # as they arrive (streaming), not as one batch.
    consumer_task = asyncio.create_task(consumer(bus, job_id, logger))
    producer_task = asyncio.create_task(producer(bus, job_id, logger))

    received, _ = await asyncio.gather(consumer_task, producer_task)
    elapsed = time.monotonic() - started

    logger.info("elapsed=%.2fs received=%d events", elapsed, len(received))

    # Assertions
    assert len(received) == 5, f"expected 5 events, got {len(received)}"
    seqs = [ev.seq for ev in received]
    assert seqs == sorted(seqs), f"seq not monotonic: {seqs}"
    assert seqs == [1, 2, 3, 4, 5], f"seq mismatch: {seqs}"
    stages = [ev.stage for ev in received]
    assert stages == ["detect", "ocr", "translate", "render", "render"], stages
    assert received[-1].percent == 100.0, received[-1].percent

    logger.info("all assertions passed ✓")
    logger.info("log written to %s", LOG_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
