"""Unit tests for the async, non-blocking logging config.

The server logs requests/translations/flags WITHOUT blocking the request path:
a QueueHandler enqueues records on the hot path while a QueueListener drains
them on a background thread into the real handlers (console + rotating JSONL
file). These tests assert the wiring and the structured translation log.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import time
from pathlib import Path

import pytest


def test_setup_returns_queue_listener(tmp_path):
    from app.logging_config import setup_logging

    listener = setup_logging(log_dir=str(tmp_path), level=logging.INFO)
    try:
        assert isinstance(listener, logging.handlers.QueueListener)
        # The root logger must route through a QueueHandler (non-blocking).
        root = logging.getLogger()
        assert any(
            isinstance(h, logging.handlers.QueueHandler) for h in root.handlers
        ), "root logger has no QueueHandler"
    finally:
        listener.stop()


def test_translation_log_writes_json_line(tmp_path):
    from app.logging_config import setup_logging, log_translation

    listener = setup_logging(log_dir=str(tmp_path), level=logging.INFO)
    try:
        log_translation(
            request_id="req-123",
            num_images=2,
            num_boxes=7,
            ocr_ms=12.3,
            translate_ms=45.6,
            inpaint_ms=7.8,
            total_ms=70.0,
            model="v10it",
        )
        # The listener drains on a background thread; give it a beat to flush.
        listener.stop()  # stop() flushes all queued records before returning

        jsonl = Path(tmp_path) / "translations.jsonl"
        assert jsonl.exists(), "translations.jsonl not created"
        lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["request_id"] == "req-123"
        assert rec["num_images"] == 2
        assert rec["num_boxes"] == 7
        assert rec["ocr_ms"] == 12.3
        assert rec["translate_ms"] == 45.6
        assert rec["inpaint_ms"] == 7.8
        assert rec["total_ms"] == 70.0
        assert rec["model"] == "v10it"
        assert "timestamp" in rec
    finally:
        # stop() already called above; calling again is a no-op-safe guard.
        try:
            listener.stop()
        except Exception:
            pass


def test_logging_burst_is_non_blocking(tmp_path):
    """A burst of log_translation calls must enqueue quickly (no disk wait)."""
    from app.logging_config import setup_logging, log_translation

    listener = setup_logging(log_dir=str(tmp_path), level=logging.INFO)
    try:
        n = 500
        start = time.perf_counter()
        for i in range(n):
            log_translation(
                request_id=f"req-{i}",
                num_images=1,
                num_boxes=3,
                ocr_ms=1.0,
                translate_ms=2.0,
                inpaint_ms=0.0,
                total_ms=3.0,
                model="v10it",
            )
        enqueue_elapsed = time.perf_counter() - start

        # Enqueuing 500 records should be far faster than synchronously writing
        # 500 lines to a rotating file. Generous bound to avoid CI flakiness.
        assert enqueue_elapsed < 0.5, f"enqueue too slow: {enqueue_elapsed:.3f}s"

        listener.stop()  # flush
        jsonl = Path(tmp_path) / "translations.jsonl"
        lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) == n
    finally:
        try:
            listener.stop()
        except Exception:
            pass
