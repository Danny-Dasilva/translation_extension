"""Translation endpoint router - Local AI Pipeline"""
import logging
import time
from fastapi import APIRouter, HTTPException, status
from typing import List
import base64
import io
from PIL import Image
import numpy as np

from app.models.request import TranslateRequest
from app.models.response import TranslateResponse, TextBox
from app.services.detector_service import DetectorService
from app.services.manga_ocr_service import MangaOCRService
from app.services.local_translation_service import LocalTranslationService
from app.utils.image_processing import (
    calculate_font_size,
    detect_font_colors,
    extract_text_region_background
)
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize local AI services (loaded at startup)
logger.info("Initializing local AI pipeline...")
detector_service = DetectorService()
ocr_service = MangaOCRService()
translation_service = LocalTranslationService()
logger.info("Local AI pipeline ready")


def decode_base64_image(base64_image: str) -> np.ndarray:
    """Decode a base64 image string to numpy array (RGB)."""
    image_data = base64_image
    if ',' in image_data and image_data.startswith('data:image'):
        image_data = image_data.split(',', 1)[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


@router.post("/translate", response_model=TranslateResponse)
async def translate_images(request: TranslateRequest):
    """
    Translate manga images using local AI pipeline:
    1. Decode base64 images
    2. Detect speech bubbles (YOLOv10n)
    3. Crop bubble regions
    4. OCR on crops (PaddleOCR-VL)
    5. Translate texts (HY-MT1.5)
    6. Return structured response with translations and metadata
    """
    start_time = time.time()
    try:
        logger.info(f"Processing {len(request.base64Images)} images")

        all_results = []

        for idx, base64_image in enumerate(request.base64Images):
            try:
                image_start = time.time()
                logger.info(f"Processing image {idx + 1}/{len(request.base64Images)}")

                # Step 1: Decode image
                image_np = decode_base64_image(base64_image)
                logger.debug(f"Image decoded: {image_np.shape}")

                # Step 2: Detect speech bubbles (YOLOv10n - NMS-free, ~2ms)
                detect_start = time.time()
                bubbles = await detector_service.detect_bubbles(
                    image_np,
                    conf=settings.detection_confidence,
                    imgsz=settings.detection_image_size
                )
                detect_time = time.time() - detect_start
                logger.info(f"Detected {len(bubbles)} bubbles in {detect_time*1000:.1f}ms")

                if not bubbles:
                    logger.warning(f"No speech bubbles detected in image {idx + 1}")
                    all_results.append([])
                    continue

                # Step 3: Crop bubble regions
                crops = detector_service.crop_regions(image_np, bubbles)

                # Step 4: OCR on crops (PaddleOCR-VL)
                ocr_start = time.time()
                ocr_texts = await ocr_service.recognize_text(crops)
                ocr_time = time.time() - ocr_start
                logger.info(f"OCR completed in {ocr_time*1000:.1f}ms ({len(crops)} crops)")

                # Step 5: Translate texts (HY-MT1.5)
                translate_start = time.time()
                if settings.translation_batch_mode == "concatenated":
                    translations = await translation_service.translate_batch_concatenated(
                        ocr_texts,
                        request.targetLanguage
                    )
                else:
                    translations = await translation_service.translate_batch(
                        ocr_texts,
                        request.targetLanguage
                    )
                translate_time = time.time() - translate_start
                logger.info(f"Translation completed in {translate_time*1000:.1f}ms ({len(ocr_texts)} texts)")

                # Step 6: Build response
                text_boxes = []
                for bubble, ocr_text, translated_text in zip(bubbles, ocr_texts, translations):
                    # Calculate font size based on bubble dimensions
                    bbox_width = bubble['maxX'] - bubble['minX']
                    bbox_height = bubble['maxY'] - bubble['minY']
                    font_size = calculate_font_size(
                        bbox_width,
                        bbox_height,
                        len(translated_text) if translated_text else 1
                    )

                    # Extract background region
                    background = extract_text_region_background(
                        base64_image,
                        bubble['minX'],
                        bubble['minY'],
                        bubble['maxX'],
                        bubble['maxY']
                    )

                    # Default font colors
                    font_color = "#000000"
                    stroke_color = "#FFFFFF"

                    text_box = TextBox(
                        ocrText=ocr_text,
                        originalLanguage="ja",
                        minX=bubble['minX'],
                        minY=bubble['minY'],
                        maxX=bubble['maxX'],
                        maxY=bubble['maxY'],
                        background=background,
                        fontHeightPx=font_size,
                        fontColor=font_color,
                        fontStrokeColor=stroke_color,
                        zIndex=1,
                        translatedText=translated_text,
                        subtextBoxes=[]
                    )

                    text_boxes.append(text_box)

                image_time = time.time() - image_start
                all_results.append(text_boxes)
                logger.info(
                    f"Image {idx + 1} completed: {len(text_boxes)} boxes in {image_time*1000:.1f}ms "
                    f"(detect: {detect_time*1000:.1f}ms, ocr: {ocr_time*1000:.1f}ms, translate: {translate_time*1000:.1f}ms)"
                )

            except Exception as e:
                logger.error(f"Error processing image {idx + 1}: {e}", exc_info=True)
                all_results.append([])

        elapsed_time = time.time() - start_time
        logger.info(f"Translation request completed in {elapsed_time:.2f} seconds")
        return TranslateResponse(images=all_results)

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Translation request failed after {elapsed_time:.2f} seconds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )
