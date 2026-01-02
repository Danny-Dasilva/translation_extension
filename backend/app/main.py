"""FastAPI application entry point"""
import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np

from app.config import settings
from app.routers import translate
from app.routers import test_page

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
        # Create dummy image for warmup (640x640 RGB)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

        # Warmup detector (YOLOv10n)
        detector_start = time.time()
        await translate.detector_service.detect_bubbles(dummy_image)
        logger.info(f"Detector warmup: {(time.time() - detector_start)*1000:.1f}ms")

        # Warmup OCR (manga-ocr) with small crop
        ocr_start = time.time()
        dummy_crop = dummy_image[:100, :100]
        await translate.ocr_service.recognize_text_batch([dummy_crop], batch_size=1)
        logger.info(f"OCR warmup (ONNX): {(time.time() - ocr_start)*1000:.1f}ms")

        # Warmup translation (HY-MT1.5) - warm up ALL instances in pool
        translate_start = time.time()
        if translate.translation_pool:
            warmup_stats = await translate.translation_pool.warmup()
            logger.info(f"Translation warmup ({warmup_stats['num_instances']} instances): {warmup_stats['total_warmup_ms']:.1f}ms")
        else:
            await translate.translation_service.translate_single("テスト", "English")
            logger.info(f"Translation warmup: {(time.time() - translate_start)*1000:.1f}ms")

        logger.info(f"All models warmed up in {(time.time() - warmup_start)*1000:.1f}ms")
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
