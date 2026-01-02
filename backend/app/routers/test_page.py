"""Test page router for debugging the manga translation pipeline."""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import numpy as np

from app.models.request import TranslateRequest
from app.models.response import TranslateResponse, TextBox, TextRegion
from app.services.detector_service import DetectorService
from app.services.manga_ocr_service import MangaOCRService
from app.services.local_translation_service import LocalTranslationService, LocalTranslationPool
from app.config import settings
from app.utils.image_processing import decode_base64_to_numpy

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/test", tags=["test"])

# Setup templates
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Debug output directory
DEBUG_DIR = Path(__file__).parent.parent.parent / "debug_output"
DEBUG_DIR.mkdir(exist_ok=True)

# Reuse services from translate router (they're singletons)
from app.routers.translate import (
    detector_service,
    ocr_service,
    translation_service,
    translation_pool,
)


@router.get("", response_class=HTMLResponse)
async def test_page(request: Request):
    """Serve the test page HTML."""
    return templates.TemplateResponse("test_page.html", {"request": request})


@router.post("/translate")
async def test_translate(request: Request):
    """
    Translate an image and save debug artifacts.

    Returns JSON with:
    - session_id: Unique ID for this request
    - images: Same format as /translate endpoint
    - debug: Timing info and file paths
    """
    start_time = time.time()

    # Parse JSON body
    body = await request.json()
    base64_images = body.get("base64Images", [])
    target_language = body.get("targetLanguage", "English")

    if not base64_images:
        return {"error": "No images provided"}

    # Generate session ID for tracking
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"Test translation session: {session_id}")

    all_results = []
    all_debug = []

    for idx, base64_image in enumerate(base64_images):
        try:
            image_start = time.time()

            # Decode image
            preprocess_start = time.time()
            image_np = decode_base64_to_numpy(base64_image)
            preprocess_time = (time.time() - preprocess_start) * 1000

            # Step 1: Detect bubbles
            detect_start = time.time()
            bubbles = await detector_service.detect_bubbles(
                image_np,
                conf=settings.detection_confidence,
                imgsz=settings.detection_image_size
            )
            detect_time = (time.time() - detect_start) * 1000

            if not bubbles:
                logger.warning(f"No bubbles detected in image {idx}")
                all_results.append([])
                all_debug.append({
                    "image_index": idx,
                    "bubbles": [],
                    "timing": {
                        "preprocess_ms": round(preprocess_time, 2),
                        "detection_ms": round(detect_time, 2)
                    }
                })
                continue

            # Step 2: Crop regions
            crop_start = time.time()
            crops = detector_service.crop_regions(image_np, bubbles)
            crop_time = (time.time() - crop_start) * 1000

            # Step 3: OCR - manga-ocr is fast (~20-100ms per crop)
            ocr_start = time.time()
            ocr_texts, ocr_times = await _test_ocr_batch(crops)
            ocr_time = (time.time() - ocr_start) * 1000

            # Step 4: Translate (uses functions from translate.py with timing wrapper)
            translate_start = time.time()
            translations, translate_times = await _test_translate_with_timing(
                ocr_texts, target_language, parallel=settings.translation_use_parallel
            )
            translate_time = (time.time() - translate_start) * 1000

            # Use bubble bboxes as text regions (entire bubble will be masked)
            all_text_regions = [[{
                'minX': bubble['minX'],
                'minY': bubble['minY'],
                'maxX': bubble['maxX'],
                'maxY': bubble['maxY']
            }] for bubble in bubbles]

            # Build response
            text_boxes = []
            bubble_debug = []

            for i, (bubble, ocr_text, translated_text, text_regions) in enumerate(zip(bubbles, ocr_texts, translations, all_text_regions)):
                text_box = TextBox(
                    ocrText=ocr_text,
                    originalLanguage="ja",
                    minX=bubble["minX"],
                    minY=bubble["minY"],
                    maxX=bubble["maxX"],
                    maxY=bubble["maxY"],
                    background="",
                    fontHeightPx=20,
                    fontColor="#000000",
                    fontStrokeColor="#FFFFFF",
                    zIndex=1,
                    translatedText=translated_text,
                    subtextBoxes=[],
                    textRegions=[TextRegion(**r) for r in text_regions]
                )
                text_boxes.append(text_box)

                bubble_debug.append({
                    "index": i,
                    "bbox": {
                        "minX": bubble["minX"],
                        "minY": bubble["minY"],
                        "maxX": bubble["maxX"],
                        "maxY": bubble["maxY"]
                    },
                    "confidence": bubble.get("confidence", 0),
                    "ocr_text": ocr_text,
                    "translated_text": translated_text,
                    "ocr_time_ms": round(ocr_times[i], 2) if i < len(ocr_times) else 0,
                    "translate_time_ms": round(translate_times[i], 2) if i < len(translate_times) else 0
                })

            image_time = (time.time() - image_start) * 1000

            all_results.append(text_boxes)
            all_debug.append({
                "image_index": idx,
                "bubbles": bubble_debug,
                "timing": {
                    "preprocess_ms": round(preprocess_time, 2),
                    "detection_ms": round(detect_time, 2),
                    "crop_ms": round(crop_time, 2),
                    "ocr_ms": round(ocr_time, 2),
                    "translation_ms": round(translate_time, 2),
                    "total_ms": round(image_time, 2)
                }
            })

            # Log confidence for each bubble
            confidences = [b.get("confidence", 0) for b in bubbles]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            logger.info(
                f"Image {idx}: {len(bubbles)} bubbles (avg conf: {avg_conf:.2f}), "
                f"preprocess={preprocess_time:.1f}ms, detect={detect_time:.1f}ms, "
                f"crop={crop_time:.1f}ms, ocr={ocr_time:.1f}ms, translate={translate_time:.1f}ms"
            )
            for i, (conf, ocr_text) in enumerate(zip(confidences, ocr_texts)):
                logger.debug(f"  Bubble {i+1}: conf={conf:.2f}, text='{ocr_text[:30]}...'" if len(ocr_text) > 30 else f"  Bubble {i+1}: conf={conf:.2f}, text='{ocr_text}'")

        except Exception as e:
            logger.error(f"Error processing image {idx}: {e}", exc_info=True)
            all_results.append([])
            all_debug.append({
                "image_index": idx,
                "error": str(e)
            })

    total_time = (time.time() - start_time) * 1000

    # Build debug data (returned in response, not saved to disk)
    debug_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "target_language": target_language,
        "images": all_debug,
        "total_time_ms": round(total_time, 2)
    }

    # Convert TextBox objects to dicts for JSON response
    # Include confidence from debug data
    images_response = []
    for img_idx, text_boxes in enumerate(all_results):
        bubble_data = all_debug[img_idx].get("bubbles", []) if img_idx < len(all_debug) else []
        images_response.append([
            {
                **tb.model_dump(),
                "confidence": bubble_data[i].get("confidence", 0) if i < len(bubble_data) else 0,
                "ocrTimeMs": bubble_data[i].get("ocr_time_ms", 0) if i < len(bubble_data) else 0,
                "translateTimeMs": bubble_data[i].get("translate_time_ms", 0) if i < len(bubble_data) else 0
            }
            for i, tb in enumerate(text_boxes)
        ])

    # Get timing from first image if available
    first_image_timing = {}
    if debug_data["images"] and "timing" in debug_data["images"][0]:
        first_image_timing = debug_data["images"][0]["timing"]

    return {
        "session_id": session_id,
        "images": images_response,
        "debug": {
            "timing": first_image_timing,
            "total_ms": round(total_time, 2)
        }
    }


@router.get("/debug/{session_id}/{path:path}")
async def get_debug_file(session_id: str, path: str):
    """Serve debug files (images, JSON)."""
    file_path = DEBUG_DIR / session_id / path
    if not file_path.exists():
        return {"error": "File not found"}

    if file_path.suffix == ".json":
        with open(file_path) as f:
            return json.load(f)
    else:
        return FileResponse(file_path)


@router.get("/debug/{session_id}")
async def get_debug_session(session_id: str):
    """Get debug.json for a session."""
    debug_json = DEBUG_DIR / session_id / "debug.json"
    if not debug_json.exists():
        return {"error": "Session not found"}

    with open(debug_json) as f:
        return json.load(f)


@router.post("/benchmark-ocr")
async def benchmark_ocr(request: Request):
    """
    Benchmark OCR approaches: batched VL model vs sequential VL model.

    Returns detailed timing comparison and results from both approaches.
    """
    # Parse JSON body
    body = await request.json()
    base64_images = body.get("base64Images", [])

    if not base64_images:
        return {"error": "No images provided"}

    results = []

    for idx, base64_image in enumerate(base64_images):
        try:
            # Decode image
            image_np = decode_base64_to_numpy(base64_image)

            # Step 1: Detect bubbles
            detect_start = time.time()
            bubbles = await detector_service.detect_bubbles(
                image_np,
                conf=settings.detection_confidence,
                imgsz=settings.detection_image_size
            )
            detect_time = (time.time() - detect_start) * 1000

            if not bubbles:
                results.append({
                    "image_index": idx,
                    "bubbles_detected": 0,
                    "error": "No bubbles detected"
                })
                continue

            # Crop regions
            crops = detector_service.crop_regions(image_np, bubbles)

            # --- manga-ocr benchmark (single approach - already fast) ---
            batched_start = time.time()
            batched_texts = await ocr_service.recognize_text_batch(crops)
            batched_time = (time.time() - batched_start) * 1000

            # --- Second run for comparison (warm cache) ---
            sequential_start = time.time()
            sequential_texts = await ocr_service.recognize_text_batch(crops)
            sequential_time = (time.time() - sequential_start) * 1000

            # Compare results
            comparison = []
            for i, bubble in enumerate(bubbles):
                batched_text = batched_texts[i] if i < len(batched_texts) else ""
                sequential_text = sequential_texts[i] if i < len(sequential_texts) else ""

                comparison.append({
                    "bubble_index": i,
                    "bbox": {
                        "minX": bubble["minX"],
                        "minY": bubble["minY"],
                        "maxX": bubble["maxX"],
                        "maxY": bubble["maxY"]
                    },
                    "results": {
                        "sequential_vl": sequential_text,
                        "batched_vl": batched_text
                    },
                    "match": batched_text == sequential_text
                })

            results.append({
                "image_index": idx,
                "bubbles_detected": len(bubbles),
                "timing": {
                    "detection_ms": round(detect_time, 2),
                    "sequential_vl_ms": round(sequential_time, 2),
                    "batched_vl_ms": round(batched_time, 2)
                },
                "speedup": round(sequential_time / batched_time, 2) if batched_time > 0 else 0,
                "comparison": comparison
            })

            logger.info(
                f"OCR Benchmark - Image {idx}: "
                f"sequential={sequential_time:.1f}ms, batched={batched_time:.1f}ms "
                f"({sequential_time/batched_time:.1f}x speedup)"
            )

        except Exception as e:
            logger.error(f"Error benchmarking image {idx}: {e}", exc_info=True)
            results.append({
                "image_index": idx,
                "error": str(e)
            })

    return {
        "benchmark": "ocr_comparison",
        "approaches": [
            "sequential_vl: Sequential VL model (baseline)",
            "batched_vl: Batched VL model inference"
        ],
        "results": results
    }


# ============================================================================
# Test Helper Functions with Timing
# These wrap the core translation functions from translate.py and add timing.
# ============================================================================


async def _test_ocr_batch(crops: List[np.ndarray]) -> Tuple[List[str], List[float]]:
    """
    OCR using manga-ocr with timing.

    Returns:
        Tuple of (texts, times_ms) where times_ms is estimated per-crop timing.
    """
    if not crops:
        return [], []

    start = time.perf_counter()
    texts = await ocr_service.recognize_text_batch(crops)
    total_time = (time.perf_counter() - start) * 1000

    # Estimate per-crop time (manga-ocr processes sequentially)
    avg_time = total_time / len(crops)
    times = [avg_time] * len(crops)

    logger.info(f"manga-ocr: {len(crops)} crops in {total_time:.1f}ms (avg {avg_time:.1f}ms/crop)")
    return texts, times


async def _test_translate_with_timing(
    texts: List[str],
    target_language: str,
    parallel: bool = False
) -> Tuple[List[str], List[float]]:
    """
    Translate texts with per-text timing, wrapping core functions from translate.py.

    Args:
        texts: List of texts to translate.
        target_language: Target language for translation.
        parallel: If True, use parallel translation; otherwise sequential.

    Returns:
        Tuple of (translations, times_ms) where times_ms is per-text timing.
    """
    if not texts:
        return [], []

    translations = []
    times = []

    if parallel and translation_pool:
        # Use translate_parallel which properly distributes across instances with semaphores
        # This avoids the crash from concurrent access to the same Llama instance
        start = time.perf_counter()
        translations = await translation_pool.translate_parallel(texts, target_language)
        total_time = (time.perf_counter() - start) * 1000

        # Estimate per-text timing (parallel execution = total time / parallelism)
        avg_time = total_time / len(texts)
        times = [avg_time] * len(texts)

        logger.info(f"[DEBUG] Parallel translation: {len(texts)} texts in {total_time:.1f}ms (avg {avg_time:.1f}ms/text)")
    else:
        # Sequential translation with timing
        for text in texts:
            start = time.perf_counter()
            if translation_pool:
                trans = await translation_pool.translate_single(text, target_language)
            else:
                trans = await translation_service.translate_single(text, target_language)
            times.append((time.perf_counter() - start) * 1000)
            translations.append(trans)

    return translations, times
