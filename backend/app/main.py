"""FastAPI application entry point"""
import asyncio
import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np

from app.services import _ort_init  # noqa: F401  (preload CUDA before ORT sessions)
from app.config import settings
from app.routers import translate
from app.routers import test_page
from app.routers import websocket_upload
from app.routers import inpaint

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration even if uvicorn already set up handlers
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Manga Translation API")
    logger.info(f"CORS origins: {settings.get_cors_origins()}")

    # Warmup models with dummy data to avoid cold start latency
    logger.info("Warming up AI models...")
    warmup_start = time.time()

    try:
        # Create dummy image for warmup
        dummy_size = settings.ctd_input_size
        dummy_image = np.zeros((dummy_size, dummy_size, 3), dtype=np.uint8)

        # Warmup detector (CTD text)
        detector_start = time.time()
        await translate.detector_service.detect(dummy_image)
        logger.info(f"Detector warmup: {(time.time() - detector_start)*1000:.1f}ms")

        # Warmup YOLOv10n speech-bubble detector (separate CUDA forward) so the
        # first real page doesn't pay its cold-start cost in the detect stage.
        if translate.bubble_detector is not None:
            try:
                yolo_start = time.time()
                await translate.bubble_detector.detect_bubbles(dummy_image)
                logger.info(f"Bubble detector warmup: {(time.time() - yolo_start)*1000:.1f}ms")
            except Exception as e:
                logger.warning(f"Bubble detector warmup failed (non-fatal): {e}")

        # Warmup LaMa inpaint with a small synthetic masked region (the ONNX
        # session is warmed at load, but this also primes the cv2/numpy router
        # paths and a real forward at crop scale).
        if translate.inpaint_service is not None:
            try:
                lama_start = time.time()
                lama_dummy = np.zeros((256, 256, 3), dtype=np.uint8)
                lama_mask = np.zeros((256, 256), dtype=np.uint8)
                lama_mask[100:150, 100:150] = 255
                await asyncio.to_thread(
                    translate.inpaint_service.inpaint, lama_dummy, lama_mask
                )
                logger.info(f"LaMa inpaint warmup: {(time.time() - lama_start)*1000:.1f}ms")
            except Exception as e:
                logger.warning(f"LaMa warmup failed (non-fatal): {e}")

        # Warmup OCR (PARSeq ONNX) with small crop
        ocr_start = time.time()
        dummy_crop = dummy_image[:100, :100]
        await translate.ocr_service.recognize_text_batch([dummy_crop], batch_size=1)
        logger.info(f"OCR warmup (ONNX): {(time.time() - ocr_start)*1000:.1f}ms")

        # Warmup translation backend (vLLM client or transformers Hy-MT)
        translate_start = time.time()
        warmup = getattr(translate.translation_service, "warmup", None)
        if callable(warmup):
            await warmup()
        else:
            await translate.translation_service.translate_single("テスト", "English")
        logger.info(f"Translation warmup: {(time.time() - translate_start)*1000:.1f}ms")

        logger.info(f"All models warmed up in {(time.time() - warmup_start)*1000:.1f}ms")

        # --- CUDA EP audit: loud ERROR if sessions silently fell back to CPU ---
        try:
            import onnxruntime as _ort
            _available = _ort.get_available_providers()
            logger.info("ORT available providers: %s", _available)
            _cuda_available = "CUDAExecutionProvider" in _available
            if not _cuda_available:
                logger.error(
                    "ORT_CUDA_ABSENT: CUDAExecutionProvider is NOT in onnxruntime's "
                    "available providers %s — all ONNX sessions are running on CPU. "
                    "Ensure onnxruntime-gpu is installed and CUDA libs are preloaded "
                    "via app.services._ort_init.",
                    _available,
                )

            # Check each service session if accessible
            _services = {
                "detector": getattr(translate, "detector_service", None),
                "ocr": getattr(translate, "ocr_service", None),
                "inpaint": getattr(translate, "inpaint_service", None),
            }
            for _svc_name, _svc in _services.items():
                if _svc is None:
                    continue
                # Session may live under .session or ._session or .model
                _sess = (
                    getattr(_svc, "session", None)
                    or getattr(_svc, "_session", None)
                    or getattr(_svc, "model", None)
                )
                if _sess is None or not hasattr(_sess, "get_providers"):
                    continue
                _bound = _sess.get_providers()
                if _bound and _bound[0] == "CPUExecutionProvider" and _cuda_available:
                    logger.error(
                        "ORT_CPU_FALLBACK: %s session is on CPUExecutionProvider "
                        "(providers=%s) despite CUDA being available — expect 5-19x "
                        "slower inference. Check model load order and _ort_init import.",
                        _svc_name,
                        _bound,
                    )
                else:
                    logger.info("ORT provider audit — %s: %s", _svc_name, _bound)
        except Exception as _audit_err:
            logger.warning("ORT provider audit failed (non-fatal): %s", _audit_err)
        # --- end CUDA EP audit ---

    except Exception as e:
        logger.warning(f"Model warmup failed (non-fatal): {e}")

    yield
    logger.info("Shutting down Manga Translation API")


# Create FastAPI app
app = FastAPI(
    title="Manga Translation API",
    description="OCR and translation service for manga images using RapidOCR and Google Gemini",
    version="1.0.0",
    lifespan=lifespan
)

# Pure ASGI middleware for timing (5x faster than BaseHTTPMiddleware)
# BaseHTTPMiddleware has known performance issues - causes 5x RPS reduction
class TimingMiddleware:
    """Pure ASGI middleware to capture request start time before body parsing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Initialize state dict if not present
            if "state" not in scope:
                scope["state"] = {}
            scope["state"]["start_time"] = time.time()
        await self.app(scope, receive, send)

app.add_middleware(TimingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(translate.router, tags=["translation"])
app.include_router(test_page.router)
app.include_router(websocket_upload.router)
app.include_router(inpaint.router)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Manga Translation API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "ocr": "ready",
            "translation": "ready"
        }
    }
