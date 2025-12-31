"""Test page router for debugging the manga translation pipeline."""

import base64
import io
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np

from app.models.request import TranslateRequest
from app.models.response import TranslateResponse, TextBox
from app.services.detector_service import DetectorService
from app.services.manga_ocr_service import MangaOCRService
from app.services.local_translation_service import LocalTranslationService
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/test", tags=["test"])

# Setup templates
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Debug output directory
DEBUG_DIR = Path(__file__).parent.parent.parent / "debug_output"
DEBUG_DIR.mkdir(exist_ok=True)

# Reuse services from translate router (they're singletons)
from app.routers.translate import detector_service, ocr_service, translation_service


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

    # Generate session ID
    session_id = str(uuid.uuid4())[:8]
    session_dir = DEBUG_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = session_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    logger.info(f"Test translation session: {session_id}")

    all_results = []
    all_debug = []

    for idx, base64_image in enumerate(base64_images):
        try:
            image_start = time.time()

            # Decode image
            image_data = base64_image
            if ',' in image_data and image_data.startswith('data:image'):
                image_data = image_data.split(',', 1)[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_np = np.array(image)

            # Save original image
            original_path = session_dir / f"original_{idx}.jpg"
            image.save(str(original_path), "JPEG", quality=90)

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
                    "timing": {"detection_ms": detect_time}
                })
                continue

            # Step 2: Crop regions and save
            crops = detector_service.crop_regions(image_np, bubbles)
            crop_paths = []
            for i, crop in enumerate(crops):
                crop_path = crops_dir / f"crop_{idx}_{i}.jpg"
                crop_img = Image.fromarray(crop)
                crop_img.save(str(crop_path), "JPEG", quality=90)
                crop_paths.append(str(crop_path.relative_to(DEBUG_DIR)))

            # Step 3: OCR
            ocr_start = time.time()
            ocr_texts = await ocr_service.recognize_text(crops)
            ocr_time = (time.time() - ocr_start) * 1000

            # Step 4: Translate
            translate_start = time.time()
            if settings.translation_batch_mode == "concatenated":
                translations = await translation_service.translate_batch_concatenated(
                    ocr_texts, target_language
                )
            else:
                translations = await translation_service.translate_batch(
                    ocr_texts, target_language
                )
            translate_time = (time.time() - translate_start) * 1000

            # Build response
            text_boxes = []
            bubble_debug = []

            for i, (bubble, ocr_text, translated_text) in enumerate(zip(bubbles, ocr_texts, translations)):
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
                    subtextBoxes=[]
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
                    "crop_file": crop_paths[i] if i < len(crop_paths) else None
                })

            image_time = (time.time() - image_start) * 1000

            all_results.append(text_boxes)
            all_debug.append({
                "image_index": idx,
                "original_file": str(original_path.relative_to(DEBUG_DIR)),
                "bubbles": bubble_debug,
                "timing": {
                    "detection_ms": round(detect_time, 2),
                    "ocr_ms": round(ocr_time, 2),
                    "translation_ms": round(translate_time, 2),
                    "total_ms": round(image_time, 2)
                }
            })

            logger.info(
                f"Image {idx}: {len(bubbles)} bubbles, "
                f"detect={detect_time:.1f}ms, ocr={ocr_time:.1f}ms, translate={translate_time:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Error processing image {idx}: {e}", exc_info=True)
            all_results.append([])
            all_debug.append({
                "image_index": idx,
                "error": str(e)
            })

    total_time = (time.time() - start_time) * 1000

    # Save debug.json
    debug_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "target_language": target_language,
        "images": all_debug,
        "total_time_ms": round(total_time, 2)
    }
    debug_json_path = session_dir / "debug.json"
    with open(debug_json_path, "w") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)

    # Convert TextBox objects to dicts for JSON response
    images_response = []
    for text_boxes in all_results:
        images_response.append([
            {
                "ocrText": tb.ocrText,
                "originalLanguage": tb.originalLanguage,
                "minX": tb.minX,
                "minY": tb.minY,
                "maxX": tb.maxX,
                "maxY": tb.maxY,
                "background": tb.background,
                "fontHeightPx": tb.fontHeightPx,
                "fontColor": tb.fontColor,
                "fontStrokeColor": tb.fontStrokeColor,
                "zIndex": tb.zIndex,
                "translatedText": tb.translatedText,
                "subtextBoxes": []
            }
            for tb in text_boxes
        ])

    return {
        "session_id": session_id,
        "images": images_response,
        "debug": {
            "session_dir": f"/test/debug/{session_id}",
            "timing": debug_data["images"][0]["timing"] if debug_data["images"] else {},
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
