"""Lazy per-engine model registry.

Ported conceptually from koharu's `Registry` at
`/tmp/koharu/koharu-app/src/pipeline/engine.rs:94-131`. Koharu keeps an
`RwLock<HashMap<&str, Arc<dyn Engine>>>` and lazy-loads engines on first
`get()`. We do the equivalent with an ``asyncio.Lock`` + ``dict[str, Any]``.

Today, `backend/app/routers/translate.py` *eagerly* constructs:
    - a detector (CTD or AnimeText)
    - an OCR service (PARSeq or MangaOCR)
    - a translation service (vLLM client or transformers Hy-MT model)
...at module-import time. That's ~10 GB baseline VRAM and ~30+ s cold start
just to reply to `GET /health`. The registry below changes nothing by itself
(it is purely additive) but lets a follow-up integration PR move those
loads to first-use behind an ``asyncio.Lock``.

---------------------------------------------------------------------------
INTEGRATION TODO  (DO NOT PERFORM HERE — separate PR)
---------------------------------------------------------------------------

In ``backend/app/routers/translate.py`` replace the eager init block with:

    from app.registry import registry

    # remove:  detector_service = create_detector()
    # remove:  ocr_service = ParseqOCRService(...) / MangaOCRService(...)
    # remove:  translation_service = VLLMOpenAITranslationService(...) / HyMTTransformersService()

    async def process_single_image(...):
        detector = await registry.get_detector()
        ocr = await registry.get_ocr()
        translator = await registry.get_translation()
        ...

In ``backend/app/main.py`` lifespan warmup (optional — only warm what's
configured for this deployment, or skip warmup entirely to get fast start):

    async def lifespan(app):
        if settings.warmup_on_start:
            await registry.get_detector()
            await registry.get_ocr()
            await registry.get_translation()
            # warmup calls go here (same as today)
        yield
        await registry.unload_all()

Add an admin endpoint in a new router (or reuse translate.py):

    @router.post("/unload")
    async def unload(name: str | None = None):
        if name: await registry.unload(name)
        else:    await registry.unload_all()
        return {"loaded": registry.loaded()}

    @router.get("/loaded")
    async def loaded():
        return {"engines": registry.loaded()}

Once that integration lands we expect:
- ``main.py`` lifespan shrinks from ~35 lines of warmup to ~5
- ``translate.py`` loses ~20 lines of module-level init + globals
- Baseline VRAM on idle: ~10 GB → ~0.3 GB (only CUDA context)
- Cold start time: ~30 s → <2 s (FastAPI ready before models load)
---------------------------------------------------------------------------
"""
from __future__ import annotations

import asyncio
import gc
import logging
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


# Canonical service IDs. Match these strings when calling `get()` / `unload()`.
ID_DETECTOR = "detector"
ID_OCR = "ocr"
ID_TRANSLATION = "translation"
ID_INPAINT = "inpaint"


class ServiceRegistry:
    """Lazy singleton registry for heavy model services.

    Thread-safety: a single ``asyncio.Lock`` guards the cache. When two
    coroutines race on the same cache miss, the first acquires the lock and
    loads; the second re-checks the cache after acquiring and returns the
    already-loaded instance (double-checked locking).
    """

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def get(self, name: str) -> Any:
        """Return a cached service by id, loading it lazily on miss.

        Dispatches to the typed helper methods below based on ``name``. Raises
        ``KeyError`` for unknown ids.
        """
        # Fast path — no lock if already loaded.
        svc = self._services.get(name)
        if svc is not None:
            return svc

        if name == ID_DETECTOR:
            return await self.get_detector()
        if name == ID_OCR:
            return await self.get_ocr()
        if name == ID_TRANSLATION:
            return await self.get_translation()
        if name == ID_INPAINT:
            return await self.get_inpaint()

        raise KeyError(
            f"Unknown service id: {name!r}. Known: "
            f"{ID_DETECTOR, ID_OCR, ID_TRANSLATION, ID_INPAINT}"
        )

    def loaded(self) -> list[str]:
        """Return the list of currently-loaded service ids."""
        return sorted(self._services.keys())

    async def unload(self, name: str) -> None:
        """Drop a service, call ``.close()`` if available, then ``gc.collect()``.

        Koharu's ``Registry::clear`` (engine.rs:129-131) simply empties the
        map and lets Rust's Drop handle GPU buffers. In Python we have to be
        explicit — services that allocate ONNX sessions or CUDA model
        contexts should expose ``close()`` for best-effort release; otherwise we rely
        on refcount + ``gc.collect()``.
        """
        async with self._lock:
            svc = self._services.pop(name, None)
            if svc is None:
                logger.debug("unload(%s): not loaded, nothing to do", name)
                return

            close = getattr(svc, "close", None)
            if callable(close):
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:  # noqa: BLE001 — best effort
                    logger.warning("Error calling %s.close(): %s", name, exc)

        # gc outside the lock — it can be slow and doesn't touch state.
        del svc
        gc.collect()
        logger.info("Unloaded service %r", name)

    async def unload_all(self) -> None:
        """Unload every cached service. Useful on shutdown / tests."""
        # Snapshot names under lock to avoid mutating-while-iterating.
        async with self._lock:
            names = list(self._services.keys())
        for name in names:
            await self.unload(name)

    # ------------------------------------------------------------------
    # Typed helpers — each does a *lazy import* so that merely importing
    # this module doesn't drag in torch/onnxruntime/httpx.
    # ------------------------------------------------------------------

    async def get_detector(self) -> Any:
        """Return the configured detector (CTD or AnimeText)."""
        svc = self._services.get(ID_DETECTOR)
        if svc is not None:
            return svc

        async with self._lock:
            svc = self._services.get(ID_DETECTOR)  # re-check
            if svc is not None:
                return svc

            detector_type = settings.detector_type.lower()
            logger.info("Lazy-loading detector (type=%s)", detector_type)

            if detector_type == "animetext":
                from app.services.animetext_service import AnimeTextDetectorService
                svc = AnimeTextDetectorService()
            elif detector_type == "ctd":
                from app.services.ctd_service import ComicTextDetectorService
                svc = ComicTextDetectorService()
            else:
                raise ValueError(
                    f"Unknown detector_type {detector_type!r}. "
                    "Use 'animetext' or 'ctd'."
                )

            self._services[ID_DETECTOR] = svc
            return svc

    async def get_ocr(self) -> Any:
        """Return the configured OCR service (PARSeq or MangaOCR)."""
        svc = self._services.get(ID_OCR)
        if svc is not None:
            return svc

        async with self._lock:
            svc = self._services.get(ID_OCR)
            if svc is not None:
                return svc

            backend = settings.ocr_backend.lower()
            logger.info("Lazy-loading OCR (backend=%s)", backend)

            if backend == "parseq":
                from app.services.parseq_ocr_service import ParseqOCRService
                svc = ParseqOCRService(
                    model_path=settings.parseq_model_path,
                    hybrid_enabled=settings.hybrid_ocr_enabled,
                    ar_model_path=settings.parseq_ar_model_path,
                    hybrid_conf_threshold=settings.ocr_confidence_gate_threshold,
                    vertical_ar_default=getattr(
                        settings, "ocr_vertical_ar_default", True
                    ),
                    vertical_ar_aspect=getattr(
                        settings, "ocr_vertical_ar_aspect", 1.5
                    ),
                )
            elif backend in ("manga-ocr", "manga_ocr", "mangaocr"):
                from app.services.manga_ocr_service import MangaOCRService
                svc = MangaOCRService()
            else:
                raise ValueError(
                    f"Unknown ocr_backend {backend!r}. "
                    "Use 'parseq' or 'manga-ocr'."
                )

            self._services[ID_OCR] = svc
            return svc

    async def get_translation(self) -> Any:
        """Return the configured translation service.

        Backend is selected by ``settings.translation_backend``:
          - ``"vllm-openai"`` (default): ``VLLMOpenAITranslationService``
            talking to a local vLLM + MTP server.
          - ``"transformers"``: ``HyMTTransformersService`` (Hy-MT1.5-2bit).
        """
        svc = self._services.get(ID_TRANSLATION)
        if svc is not None:
            return svc

        async with self._lock:
            svc = self._services.get(ID_TRANSLATION)
            if svc is not None:
                return svc

            backend = settings.translation_backend.lower()
            logger.info("Lazy-loading translation (backend=%s)", backend)

            if backend == "vllm-openai":
                from app.services.vllm_openai_translation_service import (
                    VLLMOpenAITranslationService,
                )
                svc = VLLMOpenAITranslationService(
                    base_url=settings.vllm_base_url,
                    model_name=settings.vllm_model_name,
                )
            elif backend == "transformers":
                from app.services.hymt_transformers_service import (
                    HyMTTransformersService,
                )
                svc = HyMTTransformersService()
            else:
                raise ValueError(
                    f"Unknown translation_backend {backend!r}. "
                    "Use 'vllm-openai' or 'transformers'."
                )

            self._services[ID_TRANSLATION] = svc
            return svc

    async def get_inpaint(self) -> Optional[Any]:
        """Return the LaMa inpaint service if implemented, else ``None``.

        Item #1 in KOHARU_COMPARISON.md adds ``LamaInpaintService``. Until
        that lands this helper returns ``None`` so callers can do an optional
        step: ``if (inp := await registry.get_inpaint()): mask = inp.run(...)``.
        """
        svc = self._services.get(ID_INPAINT)
        if svc is not None:
            return svc

        async with self._lock:
            svc = self._services.get(ID_INPAINT)
            if svc is not None:
                return svc

            try:
                # Module may not exist yet — that's fine, return None.
                from app.services.lama_inpaint_service import LamaInpaintService  # type: ignore[import-not-found]
            except ImportError:
                logger.debug("LaMa inpaint service not installed — returning None")
                return None

            logger.info("Lazy-loading LamaInpaintService")
            svc = LamaInpaintService()
            self._services[ID_INPAINT] = svc
            return svc


# Module-level singleton — import and use directly:
#     from app.registry import registry
#     detector = await registry.get_detector()
registry = ServiceRegistry()


__all__ = [
    "ServiceRegistry",
    "registry",
    "ID_DETECTOR",
    "ID_OCR",
    "ID_TRANSLATION",
    "ID_INPAINT",
]
