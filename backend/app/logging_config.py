"""Async, non-blocking logging configuration.

The request path must never block on log I/O. We achieve that with the stdlib
``QueueHandler`` + ``QueueListener`` pattern:

  * The hot path (request handlers, OCR/translate/inpaint stages) logs through a
    ``QueueHandler`` attached to the root logger. ``QueueHandler.emit`` only
    pushes the record onto an in-memory ``queue.Queue`` — it does NOT touch the
    console or the disk, so it returns in microseconds.
  * A single ``QueueListener`` runs on a background thread and drains the queue
    into the real (potentially slow) handlers: a console ``StreamHandler`` and a
    ``RotatingFileHandler`` for the structured translation log. All blocking
    formatting / disk writes happen on that thread, off the request path.

This composes with uvicorn's own logging: uvicorn's access/error loggers
propagate to the root logger, so their records flow through the same queue.
We attach the QueueHandler to the root (not to uvicorn's handlers) and clear
pre-existing root handlers once, so console output isn't duplicated.

Public API:
  setup_logging(log_dir=..., level=...) -> QueueListener
  log_translation(...)  # structured JSON line for /translate + WS requests
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import queue
import time
from pathlib import Path
from typing import Optional

# Name of the dedicated logger whose records become structured JSONL lines in
# translations.jsonl. Kept separate from app/uvicorn loggers so its handler
# (a JSON-only RotatingFileHandler) doesn't also receive their human messages.
TRANSLATION_LOGGER_NAME = "manga.translations"

# Module-level handle so a second setup_logging() call (e.g. uvicorn reload)
# tears the previous listener down instead of leaking a thread.
_listener: Optional[logging.handlers.QueueListener] = None
_translation_logger: Optional[logging.Logger] = None


class _DictQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler that preserves a dict ``record.msg`` across the queue.

    The stock ``QueueHandler.prepare`` calls ``self.format(record)`` and
    overwrites ``record.msg`` with the formatted STRING (to drop unpicklable
    args before queuing). That would stringify the structured-log dict that
    ``log_translation`` passes as the message, breaking JSONL serialization on
    the listener side. Our queue is in-process (no pickling), so we keep the
    record as-is and only zap the unpicklable bits defensively.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        # Leave record.msg / record.args intact so the listener's formatter
        # still sees the original dict (or string) message.
        return record


class _JsonLineFormatter(logging.Formatter):
    """Formats a record's ``.msg`` dict as a single compact JSON line.

    ``log_translation`` passes a dict as the log message; we emit it verbatim
    (plus a timestamp) so each line in translations.jsonl is a self-contained
    JSON object — the fine-tune / analytics seed format.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = record.msg
        if not isinstance(payload, dict):
            # Fallback: wrap any stray string message so the file stays valid JSONL.
            payload = {"message": record.getMessage()}
        if "timestamp" not in payload:
            payload = {"timestamp": _iso_now(record.created), **payload}
        return json.dumps(payload, ensure_ascii=False)


def _iso_now(epoch: Optional[float] = None) -> str:
    t = epoch if epoch is not None else time.time()
    lt = time.localtime(t)
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", lt) + f".{ms:03d}"


def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    *,
    console: bool = True,
    translations_filename: str = "translations.jsonl",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.handlers.QueueListener:
    """Configure non-blocking queue-based logging and return the QueueListener.

    Idempotent: calling it again stops the previous listener and rewires fresh
    handlers (so uvicorn's reloader / repeated app construction is safe).

    Args:
        log_dir: directory for the rotating translation log (created if absent).
        level: root log level.
        console: also mirror human-readable logs to the console.
        translations_filename: structured JSONL file name under ``log_dir``.
        max_bytes / backup_count: RotatingFileHandler rotation policy.

    Returns:
        The running QueueListener. Caller should keep a reference and call
        ``.stop()`` on shutdown to flush queued records.
    """
    global _listener, _translation_logger

    # Tear down a prior listener (idempotency for reloads).
    if _listener is not None:
        try:
            _listener.stop()
        except Exception:
            pass
        _listener = None

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # --- Real (blocking) handlers that run on the listener's background thread.
    real_handlers: list[logging.Handler] = []

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        # The console handler must ignore the dict-message translation records
        # (they belong only in the JSONL file). Filter them out by logger name.
        console_handler.addFilter(
            lambda r: r.name != TRANSLATION_LOGGER_NAME
        )
        real_handlers.append(console_handler)

    # Rotating JSONL file: receives ONLY the structured translation records.
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / translations_filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(_JsonLineFormatter())
    file_handler.addFilter(lambda r: r.name == TRANSLATION_LOGGER_NAME)
    real_handlers.append(file_handler)

    # --- Queue + QueueHandler on the hot path. ---------------------------------
    log_queue: "queue.Queue" = queue.Queue(-1)  # unbounded; never blocks emit()
    queue_handler = logging.handlers.QueueHandler(log_queue)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any handlers a prior basicConfig / setup installed so console output
    # isn't duplicated and disk writes only happen on the listener thread.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(queue_handler)

    # The translation logger propagates to root -> queue handler, so its records
    # ride the same non-blocking queue. Don't give it its own handler.
    _translation_logger = logging.getLogger(TRANSLATION_LOGGER_NAME)
    _translation_logger.setLevel(logging.INFO)
    _translation_logger.propagate = True

    # Start the background drain thread.
    listener = logging.handlers.QueueListener(
        log_queue, *real_handlers, respect_handler_level=True
    )
    listener.start()
    _listener = listener
    return listener


def log_translation(
    *,
    request_id: str,
    num_images: int,
    num_boxes: int,
    ocr_ms: float,
    translate_ms: float,
    inpaint_ms: float,
    total_ms: float,
    model: str,
    transport: str = "http",
    **extra,
) -> None:
    """Emit one structured translation record (non-blocking).

    The record is pushed onto the logging queue and written to
    translations.jsonl by the background QueueListener. Safe to call even if
    setup_logging() hasn't run (it logs through the default logger as a dict).
    """
    record = {
        "timestamp": _iso_now(),
        "request_id": request_id,
        "transport": transport,
        "num_images": num_images,
        "num_boxes": num_boxes,
        "ocr_ms": round(float(ocr_ms), 2),
        "translate_ms": round(float(translate_ms), 2),
        "inpaint_ms": round(float(inpaint_ms), 2),
        "total_ms": round(float(total_ms), 2),
        "model": model,
    }
    if extra:
        record.update(extra)
    logger = _translation_logger or logging.getLogger(TRANSLATION_LOGGER_NAME)
    # Pass the dict as the message; _JsonLineFormatter serializes it.
    logger.info(record)


def shutdown_logging() -> None:
    """Flush and stop the background listener (call on app shutdown)."""
    global _listener
    if _listener is not None:
        try:
            _listener.stop()
        finally:
            _listener = None
