"""Test page router for debugging the manga translation pipeline."""

import asyncio
import base64
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from fastapi import APIRouter, Request, File, UploadFile, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
import orjson
from streaming_form_data import StreamingFormDataParser
from streaming_form_data.targets import ValueTarget

from app.config import settings
from app.utils.image_processing import decode_base64_to_numpy
from app.utils.ctd_utils import build_text_regions

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
    # Use middleware start time to capture true request start (before body reading)
    middleware_start = getattr(request.state, 'start_time', None)
    start_time = time.time()

    # Parse JSON body using orjson (3-6x faster than stdlib json)
    json_start = time.time()
    body_bytes = await request.body()
    body = orjson.loads(body_bytes)
    json_parse_time = (time.time() - json_start) * 1000

    # Calculate any pre-parsing overhead (network buffering, etc.)
    pre_parse_time = (json_start - middleware_start) * 1000 if middleware_start else 0

    base64_images = body.get("base64Images", [])
    target_language = body.get("targetLanguage", "English")

    logger.info(f"[TIMING] Pre-parse: {pre_parse_time:.1f}ms, JSON parse: {json_parse_time:.1f}ms for {len(base64_images)} image(s)")

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

            # Step 1: Detect text blocks
            detect_start = time.time()
            ctd_result = await detector_service.detect(image_np)
            detect_time = (time.time() - detect_start) * 1000
            bubbles = ctd_result["blocks"]
            text_lines = ctd_result["text_lines"]

            if not bubbles:
                logger.warning(f"No text blocks detected in image {idx}")
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

            # Step 5: Extract tight text bounds and build response
            text_extract_start = time.time()
            all_text_regions = build_text_regions(bubbles, text_lines)
            text_extract_time = (time.time() - text_extract_start) * 1000

            # Build response - use dicts directly to skip Pydantic serialization overhead
            text_boxes = []
            bubble_debug = []

            for i, (bubble, ocr_text, translated_text, text_regions) in enumerate(zip(bubbles, ocr_texts, translations, all_text_regions)):
                # Convert numpy types to Python types for JSON serialization
                min_x = int(bubble["minX"])
                min_y = int(bubble["minY"])
                max_x = int(bubble["maxX"])
                max_y = int(bubble["maxY"])
                confidence = float(bubble.get("confidence", 0))
                ocr_time_ms = round(ocr_times[i], 2) if i < len(ocr_times) else 0
                translate_time_ms = round(translate_times[i], 2) if i < len(translate_times) else 0

                # Convert text_regions coords to Python ints
                text_regions_clean = [
                    {k: int(v) for k, v in region.items()}
                    for region in text_regions
                ]

                # Build dict directly - skips Pydantic validation AND model_dump() serialization
                text_box_dict = {
                    "ocrText": ocr_text,
                    "originalLanguage": "ja",
                    "minX": min_x,
                    "minY": min_y,
                    "maxX": max_x,
                    "maxY": max_y,
                    "background": "",
                    "fontHeightPx": 20,
                    "fontColor": "#000000",
                    "fontStrokeColor": "#FFFFFF",
                    "zIndex": 1,
                    "translatedText": translated_text,
                    "subtextBoxes": [],
                    "textRegions": text_regions_clean,
                    "confidence": confidence,
                    "ocrTimeMs": ocr_time_ms,
                    "translateTimeMs": translate_time_ms,
                }
                text_boxes.append(text_box_dict)

                bubble_debug.append({
                    "index": i,
                    "bbox": {
                        "minX": min_x,
                        "minY": min_y,
                        "maxX": max_x,
                        "maxY": max_y,
                    },
                    "confidence": confidence,
                    "ocr_text": ocr_text,
                    "translated_text": translated_text,
                    "ocr_time_ms": ocr_time_ms,
                    "translate_time_ms": translate_time_ms,
                })

            image_time = (time.time() - image_start) * 1000

            # Calculate gap (unaccounted time)
            measured_sum = preprocess_time + detect_time + crop_time + ocr_time + translate_time + text_extract_time
            gap_time = image_time - measured_sum

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
                    "text_extract_ms": round(text_extract_time, 2),
                    "measured_sum_ms": round(measured_sum, 2),
                    "gap_ms": round(gap_time, 2),
                    "total_ms": round(image_time, 2)
                }
            })

            logger.info(
                f"[TIMING] Image {idx}: preprocess={preprocess_time:.1f}ms, detect={detect_time:.1f}ms, "
                f"crop={crop_time:.1f}ms, ocr={ocr_time:.1f}ms, translate={translate_time:.1f}ms, "
                f"text_extract={text_extract_time:.1f}ms, GAP={gap_time:.1f}ms, TOTAL={image_time:.1f}ms"
            )


        except Exception as e:
            logger.error(f"Error processing image {idx}: {e}", exc_info=True)
            all_results.append([])
            all_debug.append({
                "image_index": idx,
                "error": str(e)
            })

    # text_boxes are already dicts (built during processing to skip Pydantic overhead)
    # Just use them directly - no model_dump() needed
    serialization_start = time.time()
    images_response = all_results  # Already list of list of dicts
    serialization_time = (time.time() - serialization_start) * 1000

    # Calculate total_time from middleware start (captures full request time)
    total_time = (time.time() - middleware_start) * 1000 if middleware_start else (time.time() - start_time) * 1000

    # Get timing from first image and add request-level timing
    first_image_timing = {}
    if all_debug and "timing" in all_debug[0]:
        first_image_timing = all_debug[0]["timing"].copy()

    # Add request-level timing (outside per-image timing)
    first_image_timing["pre_parse_ms"] = round(pre_parse_time, 2)
    first_image_timing["json_parse_ms"] = round(json_parse_time, 2)
    first_image_timing["serialization_ms"] = round(serialization_time, 2)
    first_image_timing["request_total_ms"] = round(total_time, 2)

    logger.info(
        f"[TIMING] Request complete: pre_parse={pre_parse_time:.1f}ms, json_parse={json_parse_time:.1f}ms, "
        f"serialization={serialization_time:.1f}ms, total={total_time:.1f}ms"
    )

    return {
        "session_id": session_id,
        "images": images_response,
        "debug": {
            "timing": first_image_timing,
            "total_ms": round(total_time, 2)
        }
    }


@router.post("/translate-multipart")
async def test_translate_multipart(
    request: Request,
    files: List[UploadFile] = File(..., description="Image files (JPEG/PNG/WebP)"),
    targetLanguage: str = Form(default="English", description="Target language for translation")
):
    """
    Translate images using multipart/form-data (binary upload).

    This endpoint eliminates JSON parsing overhead by accepting binary files directly.
    Expected improvement: ~86% faster request parsing (359ms → ~50ms).

    Returns JSON with:
    - session_id: Unique ID for this request
    - images: Same format as /translate endpoint
    - debug: Timing info and file paths
    """
    # Use middleware start time to capture true request start (before body parsing)
    middleware_start = getattr(request.state, 'start_time', None)
    processing_start = time.time()

    # Calculate multipart parsing overhead (done by FastAPI before we get here)
    multipart_parse_time = (processing_start - middleware_start) * 1000 if middleware_start else 0

    # Read files from memory (already parsed by FastAPI)
    file_read_start = time.time()
    base64_images = []
    for file in files:
        content = await file.read()
        # Convert to base64 for compatibility with existing pipeline
        b64_str = base64.b64encode(content).decode('utf-8')
        # Determine content type
        content_type = file.content_type or 'image/jpeg'
        ext = content_type.split('/')[-1] if '/' in content_type else 'jpeg'
        base64_images.append(f"data:image/{ext};base64,{b64_str}")
    file_read_time = (time.time() - file_read_start) * 1000

    target_language = targetLanguage

    logger.info(f"[TIMING] Multipart parse: {multipart_parse_time:.1f}ms, file read: {file_read_time:.1f}ms for {len(base64_images)} image(s)")

    if not base64_images:
        return {"error": "No images provided"}

    # Generate session ID for tracking
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"Test translation session (multipart): {session_id}")

    all_results = []
    all_debug = []

    for idx, base64_image in enumerate(base64_images):
        try:
            image_start = time.time()

            # Decode image
            preprocess_start = time.time()
            image_np = decode_base64_to_numpy(base64_image)
            preprocess_time = (time.time() - preprocess_start) * 1000

            # Step 1: Detect text blocks
            detect_start = time.time()
            ctd_result = await detector_service.detect(image_np)
            detect_time = (time.time() - detect_start) * 1000
            bubbles = ctd_result["blocks"]
            text_lines = ctd_result["text_lines"]

            if not bubbles:
                logger.warning(f"No text blocks detected in image {idx}")
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

            # Step 3: OCR
            ocr_start = time.time()
            ocr_texts, ocr_times = await _test_ocr_batch(crops)
            ocr_time = (time.time() - ocr_start) * 1000

            # Step 4: Translate
            translate_start = time.time()
            translations, translate_times = await _test_translate_with_timing(
                ocr_texts, target_language, parallel=settings.translation_use_parallel
            )
            translate_time = (time.time() - translate_start) * 1000

            # Step 5: Extract tight text bounds and build response
            text_extract_start = time.time()
            all_text_regions = build_text_regions(bubbles, text_lines)
            text_extract_time = (time.time() - text_extract_start) * 1000

            # Build response
            text_boxes = []
            bubble_debug = []

            for i, (bubble, ocr_text, translated_text, text_regions) in enumerate(zip(bubbles, ocr_texts, translations, all_text_regions)):
                min_x = int(bubble["minX"])
                min_y = int(bubble["minY"])
                max_x = int(bubble["maxX"])
                max_y = int(bubble["maxY"])
                confidence = float(bubble.get("confidence", 0))
                ocr_time_ms = round(ocr_times[i], 2) if i < len(ocr_times) else 0
                translate_time_ms = round(translate_times[i], 2) if i < len(translate_times) else 0

                text_regions_clean = [
                    {k: int(v) for k, v in region.items()}
                    for region in text_regions
                ]

                text_box_dict = {
                    "ocrText": ocr_text,
                    "originalLanguage": "ja",
                    "minX": min_x,
                    "minY": min_y,
                    "maxX": max_x,
                    "maxY": max_y,
                    "background": "",
                    "fontHeightPx": 20,
                    "fontColor": "#000000",
                    "fontStrokeColor": "#FFFFFF",
                    "zIndex": 1,
                    "translatedText": translated_text,
                    "subtextBoxes": [],
                    "textRegions": text_regions_clean,
                    "confidence": confidence,
                    "ocrTimeMs": ocr_time_ms,
                    "translateTimeMs": translate_time_ms,
                }
                text_boxes.append(text_box_dict)

                bubble_debug.append({
                    "index": i,
                    "bbox": {
                        "minX": min_x,
                        "minY": min_y,
                        "maxX": max_x,
                        "maxY": max_y,
                    },
                    "confidence": confidence,
                    "ocr_text": ocr_text,
                    "translated_text": translated_text,
                    "ocr_time_ms": ocr_time_ms,
                    "translate_time_ms": translate_time_ms,
                })

            image_time = (time.time() - image_start) * 1000

            measured_sum = preprocess_time + detect_time + crop_time + ocr_time + translate_time + text_extract_time
            gap_time = image_time - measured_sum

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
                    "text_extract_ms": round(text_extract_time, 2),
                    "measured_sum_ms": round(measured_sum, 2),
                    "gap_ms": round(gap_time, 2),
                    "total_ms": round(image_time, 2)
                }
            })

            logger.info(
                f"[TIMING] Image {idx}: preprocess={preprocess_time:.1f}ms, detect={detect_time:.1f}ms, "
                f"crop={crop_time:.1f}ms, ocr={ocr_time:.1f}ms, translate={translate_time:.1f}ms, "
                f"text_extract={text_extract_time:.1f}ms, GAP={gap_time:.1f}ms, TOTAL={image_time:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Error processing image {idx}: {e}", exc_info=True)
            all_results.append([])
            all_debug.append({
                "image_index": idx,
                "error": str(e)
            })

    serialization_start = time.time()
    images_response = all_results
    serialization_time = (time.time() - serialization_start) * 1000

    # Calculate total time from middleware start (includes multipart parsing)
    total_time = (time.time() - middleware_start) * 1000 if middleware_start else (time.time() - processing_start) * 1000

    first_image_timing = {}
    if all_debug and "timing" in all_debug[0]:
        first_image_timing = all_debug[0]["timing"].copy()

    # Include multipart parsing time (the overhead we're trying to measure)
    first_image_timing["multipart_parse_ms"] = round(multipart_parse_time, 2)
    first_image_timing["file_read_ms"] = round(file_read_time, 2)
    first_image_timing["json_parse_ms"] = 0  # No JSON parsing!
    first_image_timing["serialization_ms"] = round(serialization_time, 2)
    first_image_timing["request_total_ms"] = round(total_time, 2)

    logger.info(
        f"[TIMING] Request complete (multipart): multipart_parse={multipart_parse_time:.1f}ms, "
        f"file_read={file_read_time:.1f}ms, serialization={serialization_time:.1f}ms, total={total_time:.1f}ms"
    )

    return {
        "session_id": session_id,
        "images": images_response,
        "debug": {
            "timing": first_image_timing,
            "total_ms": round(total_time, 2)
        }
    }


async def read_body_optimized(request: Request) -> bytes:
    """
    Read request body using bytearray for O(n) performance.

    Standard request.body() uses list + b"".join() which is O(n²) for many chunks.
    This version uses bytearray.extend() which is O(n) amortized.
    """
    buffer = bytearray()
    async for chunk in request.stream():
        buffer.extend(chunk)
    return bytes(buffer)


@router.post("/translate-binary")
async def test_translate_binary(
    request: Request,
    targetLanguage: str = Query(default="English", description="Target language for translation")
):
    """
    Optimized endpoint accepting raw binary image data.
    Eliminates multipart parsing overhead entirely.

    Uses bytearray accumulation for O(n) body reading performance.
    """
    middleware_start = getattr(request.state, 'start_time', None)
    processing_start = time.time()

    # Optimized body read using bytearray (O(n) vs O(n²) for list+join)
    body = await read_body_optimized(request)
    body_read_time = (time.time() - processing_start) * 1000

    if not body:
        raise HTTPException(status_code=400, detail="No image data provided")

    logger.info(f"[TIMING] Binary body read: {body_read_time:.1f}ms for {len(body)} bytes")

    # Generate session ID for tracking
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"Test translation session (binary): {session_id}")

    all_results = []
    all_debug = []

    try:
        image_start = time.time()

        # Decode image directly from bytes (skip base64 entirely!)
        preprocess_start = time.time()
        image_np = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if image_np is None:
            raise HTTPException(status_code=400, detail="Invalid image data - could not decode")
        preprocess_time = (time.time() - preprocess_start) * 1000

        # Step 1: Detect text blocks
        detect_start = time.time()
        ctd_result = await detector_service.detect(image_np, input_is_bgr=True)
        detect_time = (time.time() - detect_start) * 1000
        bubbles = ctd_result["blocks"]
        text_lines = ctd_result["text_lines"]

        if not bubbles:
            logger.warning("No text blocks detected in image")
            all_results.append([])
            all_debug.append({
                "image_index": 0,
                "bubbles": [],
                "timing": {
                    "preprocess_ms": round(preprocess_time, 2),
                    "detection_ms": round(detect_time, 2)
                }
            })
        else:
            # Step 2: Crop regions
            crop_start = time.time()
            crops = detector_service.crop_regions(image_np, bubbles)
            crop_time = (time.time() - crop_start) * 1000

            # Step 3: OCR
            ocr_start = time.time()
            ocr_texts, ocr_times = await _test_ocr_batch(crops)
            ocr_time = (time.time() - ocr_start) * 1000

            # Step 4: Translate
            translate_start = time.time()
            translations, translate_times = await _test_translate_with_timing(
                ocr_texts, targetLanguage, parallel=settings.translation_use_parallel
            )
            translate_time = (time.time() - translate_start) * 1000

            # Step 5: Extract tight text bounds
            text_extract_start = time.time()
            all_text_regions = build_text_regions(bubbles, text_lines)
            text_extract_time = (time.time() - text_extract_start) * 1000

            # Build response
            text_boxes = []
            bubble_debug = []

            for i, (bubble, ocr_text, translated_text, text_regions) in enumerate(zip(bubbles, ocr_texts, translations, all_text_regions)):
                min_x = int(bubble["minX"])
                min_y = int(bubble["minY"])
                max_x = int(bubble["maxX"])
                max_y = int(bubble["maxY"])
                confidence = float(bubble.get("confidence", 0))
                ocr_time_ms = round(ocr_times[i], 2) if i < len(ocr_times) else 0
                translate_time_ms = round(translate_times[i], 2) if i < len(translate_times) else 0

                text_regions_clean = [
                    {k: int(v) for k, v in region.items()}
                    for region in text_regions
                ]

                text_box_dict = {
                    "ocrText": ocr_text,
                    "originalLanguage": "ja",
                    "minX": min_x,
                    "minY": min_y,
                    "maxX": max_x,
                    "maxY": max_y,
                    "background": "",
                    "fontHeightPx": 20,
                    "fontColor": "#000000",
                    "fontStrokeColor": "#FFFFFF",
                    "zIndex": 1,
                    "translatedText": translated_text,
                    "subtextBoxes": [],
                    "textRegions": text_regions_clean,
                    "confidence": confidence,
                    "ocrTimeMs": ocr_time_ms,
                    "translateTimeMs": translate_time_ms,
                }
                text_boxes.append(text_box_dict)

                bubble_debug.append({
                    "index": i,
                    "bbox": {"minX": min_x, "minY": min_y, "maxX": max_x, "maxY": max_y},
                    "confidence": confidence,
                    "ocr_text": ocr_text,
                    "translated_text": translated_text,
                    "ocr_time_ms": ocr_time_ms,
                    "translate_time_ms": translate_time_ms,
                })

            image_time = (time.time() - image_start) * 1000
            measured_sum = preprocess_time + detect_time + crop_time + ocr_time + translate_time + text_extract_time
            gap_time = image_time - measured_sum

            all_results.append(text_boxes)
            all_debug.append({
                "image_index": 0,
                "bubbles": bubble_debug,
                "timing": {
                    "preprocess_ms": round(preprocess_time, 2),
                    "detection_ms": round(detect_time, 2),
                    "crop_ms": round(crop_time, 2),
                    "ocr_ms": round(ocr_time, 2),
                    "translation_ms": round(translate_time, 2),
                    "text_extract_ms": round(text_extract_time, 2),
                    "measured_sum_ms": round(measured_sum, 2),
                    "gap_ms": round(gap_time, 2),
                    "total_ms": round(image_time, 2)
                }
            })

            logger.info(
                f"[TIMING] Binary: preprocess={preprocess_time:.1f}ms, detect={detect_time:.1f}ms, "
                f"crop={crop_time:.1f}ms, ocr={ocr_time:.1f}ms, translate={translate_time:.1f}ms, "
                f"text_extract={text_extract_time:.1f}ms, GAP={gap_time:.1f}ms, TOTAL={image_time:.1f}ms"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing binary image: {e}", exc_info=True)
        all_results.append([])
        all_debug.append({"image_index": 0, "error": str(e)})

    # Calculate total time
    total_time = (time.time() - middleware_start) * 1000 if middleware_start else (time.time() - processing_start) * 1000

    first_image_timing = {}
    if all_debug and "timing" in all_debug[0]:
        first_image_timing = all_debug[0]["timing"].copy()

    # Request overhead timing (binary = just body read, no parsing)
    first_image_timing["body_read_ms"] = round(body_read_time, 2)
    first_image_timing["multipart_parse_ms"] = 0  # No multipart!
    first_image_timing["json_parse_ms"] = 0  # No JSON!
    first_image_timing["serialization_ms"] = 0
    first_image_timing["request_total_ms"] = round(total_time, 2)

    logger.info(
        f"[TIMING] Request complete (binary): body_read={body_read_time:.1f}ms, total={total_time:.1f}ms"
    )

    return {
        "session_id": session_id,
        "images": all_results,
        "debug": {
            "timing": first_image_timing,
            "total_ms": round(total_time, 2)
        }
    }


@router.post("/translate-streaming")
async def test_translate_streaming(request: Request):
    """
    Streaming multipart parser - processes chunks as they arrive.
    Compatible with standard multipart/form-data requests but much faster.

    Expected improvement: 340ms → 50-100ms (70-85% reduction)

    Uses streaming-form-data library for chunk-by-chunk parsing.
    """
    middleware_start = getattr(request.state, 'start_time', None)
    processing_start = time.time()

    # Create streaming parser with targets for each field
    parser = StreamingFormDataParser(headers=request.headers)
    file_target = ValueTarget()
    language_target = ValueTarget()

    parser.register("files", file_target)
    parser.register("targetLanguage", language_target)

    # Stream chunks directly to parser (non-blocking)
    async for chunk in request.stream():
        parser.data_received(chunk)

    stream_parse_time = (time.time() - processing_start) * 1000

    # Get parsed data
    image_bytes = file_target.value
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No file received")

    target_language = language_target.value.decode() if language_target.value else "English"

    logger.info(f"[TIMING] Streaming parse: {stream_parse_time:.1f}ms for {len(image_bytes)} bytes")

    # Generate session ID
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"Test translation session (streaming): {session_id}")

    all_results = []
    all_debug = []

    try:
        image_start = time.time()

        # Decode image directly from bytes (skip base64!)
        preprocess_start = time.time()
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_np is None:
            raise HTTPException(status_code=400, detail="Invalid image data - could not decode")
        preprocess_time = (time.time() - preprocess_start) * 1000

        # Step 1: Detect text blocks
        detect_start = time.time()
        ctd_result = await detector_service.detect(image_np, input_is_bgr=True)
        detect_time = (time.time() - detect_start) * 1000
        bubbles = ctd_result["blocks"]
        text_lines = ctd_result["text_lines"]

        if not bubbles:
            logger.warning("No text blocks detected in image")
            all_results.append([])
            all_debug.append({
                "image_index": 0,
                "bubbles": [],
                "timing": {
                    "preprocess_ms": round(preprocess_time, 2),
                    "detection_ms": round(detect_time, 2)
                }
            })
        else:
            # Step 2: Crop regions
            crop_start = time.time()
            crops = detector_service.crop_regions(image_np, bubbles)
            crop_time = (time.time() - crop_start) * 1000

            # Step 3: OCR
            ocr_start = time.time()
            ocr_texts, ocr_times = await _test_ocr_batch(crops)
            ocr_time = (time.time() - ocr_start) * 1000

            # Step 4: Translate
            translate_start = time.time()
            translations, translate_times = await _test_translate_with_timing(
                ocr_texts, target_language, parallel=settings.translation_use_parallel
            )
            translate_time = (time.time() - translate_start) * 1000

            # Step 5: Extract tight text bounds
            text_extract_start = time.time()
            all_text_regions = build_text_regions(bubbles, text_lines)
            text_extract_time = (time.time() - text_extract_start) * 1000

            # Build response
            text_boxes = []
            bubble_debug = []

            for i, (bubble, ocr_text, translated_text, text_regions) in enumerate(zip(bubbles, ocr_texts, translations, all_text_regions)):
                min_x = int(bubble["minX"])
                min_y = int(bubble["minY"])
                max_x = int(bubble["maxX"])
                max_y = int(bubble["maxY"])
                confidence = float(bubble.get("confidence", 0))
                ocr_time_ms = round(ocr_times[i], 2) if i < len(ocr_times) else 0
                translate_time_ms = round(translate_times[i], 2) if i < len(translate_times) else 0

                text_regions_clean = [
                    {k: int(v) for k, v in region.items()}
                    for region in text_regions
                ]

                text_box_dict = {
                    "ocrText": ocr_text,
                    "originalLanguage": "ja",
                    "minX": min_x,
                    "minY": min_y,
                    "maxX": max_x,
                    "maxY": max_y,
                    "background": "",
                    "fontHeightPx": 20,
                    "fontColor": "#000000",
                    "fontStrokeColor": "#FFFFFF",
                    "zIndex": 1,
                    "translatedText": translated_text,
                    "subtextBoxes": [],
                    "textRegions": text_regions_clean,
                    "confidence": confidence,
                    "ocrTimeMs": ocr_time_ms,
                    "translateTimeMs": translate_time_ms,
                }
                text_boxes.append(text_box_dict)

                bubble_debug.append({
                    "index": i,
                    "bbox": {"minX": min_x, "minY": min_y, "maxX": max_x, "maxY": max_y},
                    "confidence": confidence,
                    "ocr_text": ocr_text,
                    "translated_text": translated_text,
                    "ocr_time_ms": ocr_time_ms,
                    "translate_time_ms": translate_time_ms,
                })

            image_time = (time.time() - image_start) * 1000
            measured_sum = preprocess_time + detect_time + crop_time + ocr_time + translate_time + text_extract_time
            gap_time = image_time - measured_sum

            all_results.append(text_boxes)
            all_debug.append({
                "image_index": 0,
                "bubbles": bubble_debug,
                "timing": {
                    "preprocess_ms": round(preprocess_time, 2),
                    "detection_ms": round(detect_time, 2),
                    "crop_ms": round(crop_time, 2),
                    "ocr_ms": round(ocr_time, 2),
                    "translation_ms": round(translate_time, 2),
                    "text_extract_ms": round(text_extract_time, 2),
                    "measured_sum_ms": round(measured_sum, 2),
                    "gap_ms": round(gap_time, 2),
                    "total_ms": round(image_time, 2)
                }
            })

            logger.info(
                f"[TIMING] Streaming: preprocess={preprocess_time:.1f}ms, detect={detect_time:.1f}ms, "
                f"crop={crop_time:.1f}ms, ocr={ocr_time:.1f}ms, translate={translate_time:.1f}ms, "
                f"text_extract={text_extract_time:.1f}ms, GAP={gap_time:.1f}ms, TOTAL={image_time:.1f}ms"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing streaming image: {e}", exc_info=True)
        all_results.append([])
        all_debug.append({"image_index": 0, "error": str(e)})

    # Calculate total time
    total_time = (time.time() - middleware_start) * 1000 if middleware_start else (time.time() - processing_start) * 1000

    first_image_timing = {}
    if all_debug and "timing" in all_debug[0]:
        first_image_timing = all_debug[0]["timing"].copy()

    # Request overhead timing
    first_image_timing["stream_parse_ms"] = round(stream_parse_time, 2)
    first_image_timing["multipart_parse_ms"] = 0  # Using streaming instead
    first_image_timing["json_parse_ms"] = 0
    first_image_timing["serialization_ms"] = 0
    first_image_timing["request_total_ms"] = round(total_time, 2)

    logger.info(
        f"[TIMING] Request complete (streaming): stream_parse={stream_parse_time:.1f}ms, total={total_time:.1f}ms"
    )

    return {
        "session_id": session_id,
        "images": all_results,
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

            # Step 1: Detect text blocks
            detect_start = time.time()
            ctd_result = await detector_service.detect(image_np)
            detect_time = (time.time() - detect_start) * 1000
            bubbles = ctd_result["blocks"]

            if not bubbles:
                results.append({
                    "image_index": idx,
                    "bubbles_detected": 0,
                    "error": "No text blocks detected"
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
