"""FastAPI application entry point"""
import logging
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np

from app.config import settings
from app.routers import translate
from app.routers import test_page

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

        # Warmup translation (HY-MT1.5) - use pool if available
        translate_start = time.time()
        if translate.translation_pool:
            await translate.translation_pool.translate_single("テスト", "English")
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


if __name__ == "__main__":
    import os
    import sys

    # Ensure the parent directory is in the path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import uvicorn

    # Reload mode disabled by default for optimal performance with llama-cpp
    # Enable with RELOAD=true environment variable for development
    use_reload = os.getenv("RELOAD", "false").lower() == "true"

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=use_reload,
    )
