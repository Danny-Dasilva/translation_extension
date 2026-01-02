"""Translation endpoint router - Local AI Pipeline"""
import asyncio
import logging
import time
from fastapi import APIRouter, HTTPException, status
from typing import List, Tuple
import base64
import io
from PIL import Image
import numpy as np

from app.models.request import TranslateRequest
from app.models.response import TranslateResponse, TextBox, TextRegion
from app.services.detector_service import DetectorService
from app.services.manga_ocr_service import MangaOCRService
from app.services.local_translation_service import LocalTranslationService, LocalTranslationPool
from app.utils.image_processing import (
    calculate_font_size,
    detect_font_colors,
    extract_text_region_background
)
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Semaphore to limit concurrent GPU operations (prevents OOM)
# With 32GB VRAM (RTX 5090), we can safely run more concurrent operations
_gpu_semaphore = asyncio.Semaphore(settings.max_parallel_images)

# Initialize local AI services (loaded at startup)
logger.info("Initializing local AI pipeline...")
detector_service = DetectorService()

# Initialize OCR service for batched inference (always needed)
logger.info("Using OCR service with batched inference")
ocr_service = MangaOCRService()
ocr_pool = None  # Pool deprecated in favor of batched inference

if settings.translation_num_instances > 1:
    logger.info(f"Using Translation Pool with {settings.translation_num_instances} instances")
    translation_pool = LocalTranslationPool()
    translation_service = None  # Not used when pool is available
else:
    logger.info("Using single Translation instance")
    translation_pool = None
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


async def process_single_image(
    idx: int,
    base64_image: str,
    target_language: str,
    semaphore: asyncio.Semaphore
) -> Tuple[int, List[TextBox]]:
    """
    Process a single image through the translation pipeline.

    Pipeline stages:
    1. Detect speech bubbles (YOLOv10n)
    2. Crop bubble regions
    3. Batched OCR (PaddleOCR-VL) - all crops in one model.generate() call
    4. Parallel translation (HY-MT1.5)

    Args:
        idx: Image index for logging
        base64_image: Base64 encoded image
        target_language: Target language for translation
        semaphore: Semaphore for GPU memory management

    Returns:
        Tuple of (index, list of TextBox results)
    """
    try:
        image_start = time.time()
        logger.info(f"Processing image {idx + 1}")

        # GPU-intensive operations inside semaphore (detection + OCR)
        async with semaphore:
            # Step 1: Decode image
            image_np = decode_base64_image(base64_image)
            logger.debug(f"Image {idx + 1} decoded: {image_np.shape}")

            # Step 2: Detect speech bubbles (YOLOv10n - NMS-free, ~2ms)
            detect_start = time.time()
            bubbles = await detector_service.detect_bubbles(
                image_np,
                conf=settings.detection_confidence,
                imgsz=settings.detection_image_size
            )
            detect_time = time.time() - detect_start
            logger.info(f"Image {idx + 1}: Detected {len(bubbles)} bubbles in {detect_time*1000:.1f}ms")

            if not bubbles:
                logger.warning(f"No speech bubbles detected in image {idx + 1}")
                return (idx, [])

            # Step 3: Crop bubble regions
            crops = detector_service.crop_regions(image_np, bubbles)

            # Step 4: Batched OCR (all crops in one model.generate() call)
            ocr_start = time.time()
            ocr_texts = await ocr_service.recognize_text_batch(
                crops,
                batch_size=len(crops)  # Process all crops at once
            )
            ocr_time = time.time() - ocr_start
            logger.info(f"Image {idx + 1}: Batched OCR completed in {ocr_time*1000:.1f}ms ({len(crops)} crops)")

            # Step 5: Translation (parallel or sequential)
            translate_start = time.time()
            if settings.translation_use_parallel:
                translations = await _translate_parallel(ocr_texts, target_language)
            else:
                translations = await _translate_sequential(ocr_texts, target_language)
            translate_time = time.time() - translate_start
            logger.info(f"Image {idx + 1}: Translation completed in {translate_time*1000:.1f}ms ({len(ocr_texts)} texts)")

        # Semaphore released - GPU slot available for other images

        # Use bubble bboxes as text regions (entire bubble will be masked)
        all_text_regions = [[{
            'minX': bubble['minX'],
            'minY': bubble['minY'],
            'maxX': bubble['maxX'],
            'maxY': bubble['maxY']
        }] for bubble in bubbles]

        # Step 6: Build response
        text_boxes = []
        for bubble, ocr_text, translated_text, text_regions in zip(bubbles, ocr_texts, translations, all_text_regions):
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
                subtextBoxes=[],
                textRegions=[TextRegion(**r) for r in text_regions]
            )

            text_boxes.append(text_box)

        image_time = time.time() - image_start
        logger.info(
            f"Image {idx + 1} completed: {len(text_boxes)} boxes in {image_time*1000:.1f}ms "
            f"(detect: {detect_time*1000:.1f}ms, ocr: {ocr_time*1000:.1f}ms, translate: {translate_time*1000:.1f}ms)"
        )

        return (idx, text_boxes)

    except Exception as e:
        logger.error(f"Error processing image {idx + 1}: {e}", exc_info=True)
        return (idx, [])


async def _translate_sequential(texts: List[str], target_language: str) -> List[str]:
    """Translate texts sequentially (original behavior)."""
    translations = []
    for text in texts:
        if translation_pool:
            trans = await translation_pool.translate_single(text, target_language)
        else:
            trans = await translation_service.translate_single(text, target_language)
        translations.append(trans)
    return translations


async def _translate_parallel(texts: List[str], target_language: str) -> List[str]:
    """
    OPTIMIZATION 2: Translate all texts in parallel.

    Uses translation pool if available (true parallelism with multiple instances),
    otherwise falls back to asyncio.gather with single instance.
    """
    if not texts:
        return []

    # Use pool for true parallel translation
    if translation_pool:
        return await translation_pool.translate_parallel(texts, target_language)

    # Fallback to single instance with asyncio.gather
    async def safe_translate(idx: int, text: str) -> Tuple[int, str]:
        """Wrapper that catches exceptions and returns index for ordering."""
        try:
            trans = await translation_service.translate_single(text, target_language)
            return (idx, trans)
        except Exception as e:
            logger.warning(f"Translation failed for text {idx+1}: {e}")
            return (idx, "")

    tasks = [
        asyncio.create_task(safe_translate(i, text))
        for i, text in enumerate(texts)
    ]

    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])
    return [trans for _, trans in results]


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

    Supports parallel processing of multiple images for faster throughput.
    """
    start_time = time.time()
    try:
        num_images = len(request.base64Images)
        logger.info(f"Processing {num_images} images (parallel={settings.parallel_image_processing})")

        # Create semaphore for GPU memory management
        semaphore = asyncio.Semaphore(settings.max_parallel_images)

        if settings.parallel_image_processing and num_images > 1:
            # Parallel processing: process all images concurrently
            tasks = [
                process_single_image(idx, base64_image, request.targetLanguage, semaphore)
                for idx, base64_image in enumerate(request.base64Images)
            ]
            results = await asyncio.gather(*tasks)

            # Sort results by index to maintain order
            results.sort(key=lambda x: x[0])
            all_results = [r[1] for r in results]
        else:
            # Sequential processing (for single image or if disabled)
            all_results = []
            for idx, base64_image in enumerate(request.base64Images):
                _, text_boxes = await process_single_image(
                    idx, base64_image, request.targetLanguage, semaphore
                )
                all_results.append(text_boxes)

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
