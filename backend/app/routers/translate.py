"""Translation endpoint router - Local AI Pipeline"""
import asyncio
import logging
import time
from fastapi import APIRouter, HTTPException, status
from typing import List, Tuple

from app.models.request import TranslateRequest
from app.models.response import TranslateResponse, TextBox, TextRegion
from app.services.detector_factory import create_detector
from app.services.manga_ocr_service import MangaOCRService
from app.services.local_translation_service import LocalTranslationService, LocalTranslationPool
from app.utils.image_processing import (
    calculate_font_size,
    decode_base64_to_numpy,
    extract_text_region_background
)
from app.utils.ctd_utils import build_text_regions
from app.utils.japanese_text_filter import is_japanese_text, filter_japanese_texts
from app.utils.zindex_utils import assign_smart_zindex
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Semaphore to limit concurrent GPU operations (prevents OOM)
# With 32GB VRAM (RTX 5090), we can safely run more concurrent operations
_gpu_semaphore = asyncio.Semaphore(settings.max_parallel_images)

# Initialize local AI services (loaded at startup)
logger.info("Initializing local AI pipeline...")
detector_service = create_detector()

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


async def process_single_image(
    idx: int,
    base64_image: str,
    target_language: str,
    semaphore: asyncio.Semaphore
) -> Tuple[int, List[TextBox]]:
    """
    Process a single image through the translation pipeline.

    Pipeline stages:
    1. Detect text blocks (CTD)
    2. Crop block regions
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
            image_np = decode_base64_to_numpy(base64_image)
            logger.debug(f"Image {idx + 1} decoded: {image_np.shape}")

            # Step 2: Detect text blocks (CTD)
            detect_start = time.time()
            ctd_result = await detector_service.detect(image_np)
            detect_time = time.time() - detect_start

            blocks = ctd_result["blocks"]
            text_lines = ctd_result["text_lines"]

            logger.info(
                f"Image {idx + 1}: Detected {len(blocks)} blocks, "
                f"{len(text_lines)} text lines in {detect_time*1000:.1f}ms"
            )

            if not blocks:
                logger.warning(f"No text blocks detected in image {idx + 1}")
                return (idx, [])

            # Step 3: Crop block regions
            crops = detector_service.crop_regions(image_np, blocks)

            # Build text regions now (before any filtering)
            all_text_regions = build_text_regions(blocks, text_lines)
            original_count = len(crops)

            # Step 4 & 5: OCR and Translation
            ocr_start = time.time()

            if settings.use_pipeline_overlap and len(crops) > 1 and translation_pool:
                # PIPELINE OVERLAP: OCR each crop and start translation immediately
                # This overlaps OCR and translation phases for better throughput
                async def ocr_and_translate_pipelined():
                    results = []
                    for i, crop in enumerate(crops):
                        # OCR single crop
                        text = await ocr_service.recognize_single(crop)

                        # Filter non-Japanese before starting translation
                        if settings.japanese_filter_enabled:
                            if not is_japanese_text(
                                text,
                                settings.japanese_filter_min_ratio,
                                settings.japanese_filter_katakana_max_length
                            ):
                                logger.debug(f"Filtered non-Japanese text at index {i}: '{text[:30]}...'")
                                continue

                        # Start translation immediately (non-blocking)
                        trans_task = asyncio.create_task(
                            translation_pool.translate_single(text, target_language)
                        )
                        results.append((i, text, trans_task))

                    # Await all translation tasks
                    return [(i, text, await task) for i, text, task in results]

                paired = await ocr_and_translate_pipelined()

                if not paired:
                    logger.warning(f"Image {idx + 1}: All text regions filtered as non-Japanese")
                    return (idx, [])

                # Extract results and filter parallel lists to match kept indices
                kept_indices = [i for i, _, _ in paired]
                ocr_texts = [text for _, text, _ in paired]
                translations = [trans for _, _, trans in paired]
                blocks = [blocks[i] for i in kept_indices]
                crops = [crops[i] for i in kept_indices]
                all_text_regions = [all_text_regions[i] for i in kept_indices]

                ocr_time = time.time() - ocr_start
                translate_time = ocr_time  # Combined time for pipelined mode

                filtered_count = original_count - len(kept_indices)
                if filtered_count > 0:
                    logger.info(f"Image {idx + 1}: Pipelined OCR+Translation completed in {ocr_time*1000:.1f}ms ({len(crops)} kept, {filtered_count} filtered)")
                else:
                    logger.info(f"Image {idx + 1}: Pipelined OCR+Translation completed in {ocr_time*1000:.1f}ms ({len(crops)} crops)")
            else:
                # BATCH MODE: All OCR first, then filter, then all translation
                ocr_texts = await ocr_service.recognize_text_batch(
                    crops,
                    batch_size=len(crops)  # Process all crops at once
                )
                ocr_time = time.time() - ocr_start
                logger.info(f"Image {idx + 1}: Batched OCR completed in {ocr_time*1000:.1f}ms ({len(crops)} crops)")

                # Filter non-Japanese OCR results
                if settings.japanese_filter_enabled:
                    valid_indices = filter_japanese_texts(
                        ocr_texts,
                        settings.japanese_filter_min_ratio,
                        settings.japanese_filter_katakana_max_length
                    )

                    filtered_count = len(ocr_texts) - len(valid_indices)
                    if filtered_count > 0:
                        logger.info(f"Image {idx + 1}: Filtered {filtered_count} non-Japanese regions")

                    if not valid_indices:
                        logger.warning(f"Image {idx + 1}: All text regions filtered as non-Japanese")
                        return (idx, [])

                    # Filter all parallel lists to maintain alignment
                    ocr_texts = [ocr_texts[i] for i in valid_indices]
                    blocks = [blocks[i] for i in valid_indices]
                    crops = [crops[i] for i in valid_indices]
                    all_text_regions = [all_text_regions[i] for i in valid_indices]

                # Translation (parallel or sequential)
                translate_start = time.time()
                if settings.translation_use_parallel:
                    translations = await _translate_parallel(ocr_texts, target_language)
                else:
                    translations = await _translate_sequential(ocr_texts, target_language)
                translate_time = time.time() - translate_start
                logger.info(f"Image {idx + 1}: Translation completed in {translate_time*1000:.1f}ms ({len(ocr_texts)} texts)")

            # Calculate per-crop timing (distribute evenly)
            num_items = len(crops) if crops else 1
            ocr_time_per_crop = (ocr_time * 1000) / num_items
            translate_time_per_text = (translate_time * 1000) / num_items

        # Semaphore released - GPU slot available for other images
        # Note: all_text_regions was built and filtered inside the semaphore block

        # Step 6: Build response
        text_boxes = []
        for block, ocr_text, translated_text, text_regions in zip(blocks, ocr_texts, translations, all_text_regions):
            # Calculate font size based on inset text region (where text will be rendered)
            # Use the first text region (the inset box) for font sizing
            region = text_regions[0] if text_regions else block
            bbox_width = region['maxX'] - region['minX']
            bbox_height = region['maxY'] - region['minY']
            font_size = calculate_font_size(
                bbox_width,
                bbox_height,
                len(translated_text) if translated_text else 1
            )

            # Extract background region
            background = extract_text_region_background(
                base64_image,
                block['minX'],
                block['minY'],
                block['maxX'],
                block['maxY']
            )

            # Default font colors
            font_color = "#000000"
            stroke_color = "#FFFFFF"

            text_box = TextBox(
                ocrText=ocr_text,
                originalLanguage="ja",
                minX=block['minX'],
                minY=block['minY'],
                maxX=block['maxX'],
                maxY=block['maxY'],
                background=background,
                fontHeightPx=font_size,
                fontColor=font_color,
                fontStrokeColor=stroke_color,
                zIndex=1,
                translatedText=translated_text,
                subtextBoxes=[],
                textRegions=[TextRegion(**r) for r in text_regions],
                confidence=block.get('confidence', 0.0),
                ocrTimeMs=round(ocr_time_per_crop, 2),
                translateTimeMs=round(translate_time_per_text, 2),
            )

            text_boxes.append(text_box)

        # Assign smart zIndex: smaller boxes get higher zIndex (rendered on top)
        assign_smart_zindex(text_boxes, use_dict=False)

        image_time = time.time() - image_start
        if settings.use_pipeline_overlap and len(crops) > 1 and translation_pool:
            logger.info(
                f"Image {idx + 1} completed: {len(text_boxes)} boxes in {image_time*1000:.1f}ms "
                f"(detect: {detect_time*1000:.1f}ms, pipelined ocr+trans: {ocr_time*1000:.1f}ms)"
            )
        else:
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
    2. Detect text blocks (CTD)
    3. Crop block regions
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
