"""Translation endpoint router - Local AI Pipeline"""
import asyncio
import base64
import io
import logging
import time
import uuid
from fastapi import APIRouter, HTTPException, status
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.models.request import TranslateRequest
from app.models.response import TranslateResponse, TextBox, TextRegion
from app.services.detector_factory import create_detector
from app.services.manga_ocr_service import MangaOCRService
from app.services.parseq_ocr_service import ParseqOCRService
from app.services.local_translation_service import LocalTranslationService, LocalTranslationPool
from app.utils.image_processing import (
    calculate_font_size,
    decode_base64_to_numpy,
    decode_base64_to_pil,
    extract_text_region_background
)
from app.utils.ctd_utils import build_text_regions
from app.utils.japanese_text_filter import is_japanese_text, filter_japanese_texts
from app.utils.zindex_utils import assign_smart_zindex
from app.utils.progress_bus import bus as progress_bus
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
if settings.ocr_backend == "parseq":
    logger.info("Using PARSeq-large OCR (ONNX fp16, CUDA)")
    ocr_service = ParseqOCRService(model_path=settings.parseq_model_path)
else:
    logger.info("Using manga-ocr OCR with batched inference")
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

# Lazy-init inpainting. Heavy model; only load if the feature is on.
inpaint_service = None
if settings.enable_inpainting:
    try:
        from app.services.lama_inpaint_service import LamaInpaintService
        logger.info("Loading LaMa inpainting (koharu-style erase plate)")
        inpaint_service = LamaInpaintService(model_path=settings.lama_model_path)
    except Exception as exc:
        logger.warning(f"LaMa inpainting unavailable ({exc}); continuing without plate")
        inpaint_service = None

logger.info("Local AI pipeline ready")


def _encode_png_base64(image_rgb: np.ndarray) -> str:
    """Encode an HxWx3 uint8 RGB ndarray as a data URL."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _build_inpaint_mask(
    image_shape: Tuple[int, int],
    blocks: List[dict],
    text_lines: List[dict],
    detector_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Combine detector mask + line polygons into a binary 0/255 mask suitable
    for LaMa. Prefers text_lines (tighter) and falls back to block bboxes.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    sources = text_lines if text_lines else blocks
    for region in sources:
        x0 = max(0, int(region.get("minX", 0)))
        y0 = max(0, int(region.get("minY", 0)))
        x1 = min(w, int(region.get("maxX", 0)))
        y1 = min(h, int(region.get("maxY", 0)))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    if detector_mask is not None and detector_mask.size:
        # Detector mask may be smaller res; resize and OR with bbox-derived mask
        dm = detector_mask
        if dm.shape[:2] != (h, w):
            dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_NEAREST)
        _, dm_bin = cv2.threshold(dm, 127, 255, cv2.THRESH_BINARY)
        mask = np.maximum(mask, dm_bin.astype(np.uint8))

    # Dilate a few pixels so stroke edges are covered
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


async def process_single_image(
    idx: int,
    base64_image: str,
    target_language: str,
    semaphore: asyncio.Semaphore,
    job_id: Optional[str] = None,
) -> Tuple[int, List[TextBox], Optional[str]]:
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
    inpainted_b64: Optional[str] = None
    try:
        image_start = time.time()
        logger.info(f"Processing image {idx + 1}")

        async def emit(stage: str, index: int, total: int, note: Optional[str] = None):
            if job_id:
                await progress_bus.emit(job_id, stage, index, total, note=note)

        # GPU-intensive operations inside semaphore (detection + OCR)
        async with semaphore:
            await emit("decode", 0, 5)
            # Step 1: Decode image
            image_np = decode_base64_to_numpy(base64_image)
            logger.debug(f"Image {idx + 1} decoded: {image_np.shape}")

            # Step 2: Detect text blocks (CTD)
            await emit("detect", 1, 5)
            detect_start = time.time()
            ctd_result = await detector_service.detect(image_np)
            detect_time = time.time() - detect_start

            blocks = ctd_result["blocks"]
            text_lines = ctd_result["text_lines"]
            detector_mask = ctd_result.get("mask")

            logger.info(
                f"Image {idx + 1}: Detected {len(blocks)} blocks, "
                f"{len(text_lines)} text lines in {detect_time*1000:.1f}ms"
            )

            if not blocks:
                logger.warning(f"No text blocks detected in image {idx + 1}")
                await emit("done", 5, 5, note="no_blocks")
                return (idx, [], None)

            # Step 3: Crop block regions
            crops = detector_service.crop_regions(image_np, blocks)

            # Build text regions now (before any filtering)
            all_text_regions = build_text_regions(blocks, text_lines)
            original_count = len(crops)

            # Step 4 & 5: OCR and Translation
            await emit("ocr", 2, 5, note=f"{len(crops)} crops")
            ocr_start = time.time()

            # PARSeq is a single-line STR model. If the detector exposes
            # text_lines, precompute per-block OCR by cropping individual lines
            # and stitching — this is how the model was trained. Fall back to
            # the per-batch block-crop path otherwise (matches manga-ocr).
            prefetched_texts: List[str] | None = None
            if isinstance(ocr_service, ParseqOCRService) and text_lines:
                prefetched_texts = await ocr_service.recognize_blocks_with_lines(
                    image_np, blocks, text_lines,
                    batch_size=settings.parseq_batch_size,
                )

            if settings.use_pipeline_overlap and len(crops) > 1 and translation_pool:
                # PIPELINE OVERLAP with mini-batching: OCR crops in batches of 3,
                # then fire translation tasks immediately for each batch.
                # This preserves ONNX batching efficiency while overlapping OCR and translation.
                MINI_BATCH_SIZE = 3

                async def ocr_and_translate_pipelined():
                    results = []
                    for batch_start in range(0, len(crops), MINI_BATCH_SIZE):
                        batch_crops = crops[batch_start:batch_start + MINI_BATCH_SIZE]
                        batch_indices = list(range(batch_start, batch_start + len(batch_crops)))

                        # OCR mini-batch (or slice of prefetched per-block OCR)
                        if prefetched_texts is not None:
                            texts = prefetched_texts[batch_start:batch_start + len(batch_crops)]
                        else:
                            texts = await ocr_service.recognize_text_batch(batch_crops)

                        for i, text in zip(batch_indices, texts):
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
                    await emit("done", 5, 5, note="all_filtered")
                    return (idx, [], None)

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
                if prefetched_texts is not None:
                    ocr_texts = prefetched_texts
                else:
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
                        await emit("done", 5, 5, note="all_filtered")
                        return (idx, [], None)

                    # Filter all parallel lists to maintain alignment
                    ocr_texts = [ocr_texts[i] for i in valid_indices]
                    blocks = [blocks[i] for i in valid_indices]
                    crops = [crops[i] for i in valid_indices]
                    all_text_regions = [all_text_regions[i] for i in valid_indices]

                # Translation (batched page-level [N] protocol, or parallel/sequential fallback)
                await emit("translate", 3, 5, note=f"{len(ocr_texts)} bubbles")
                translate_start = time.time()
                translations = await _run_translation(ocr_texts, target_language)
                translate_time = time.time() - translate_start
                logger.info(f"Image {idx + 1}: Translation completed in {translate_time*1000:.1f}ms ({len(ocr_texts)} texts)")

            # Calculate per-crop timing (distribute evenly)
            num_items = len(crops) if crops else 1
            ocr_time_per_crop = (ocr_time * 1000) / num_items
            translate_time_per_text = (translate_time * 1000) / num_items

            # Step 5b: LaMa inpaint — produce a clean plate the frontend can
            # render translated text onto, replacing the white-rect mask.
            if inpaint_service is not None and settings.enable_inpainting:
                await emit("inpaint", 4, 5, note=f"{len(blocks)} regions")
                inpaint_start = time.time()
                try:
                    inpaint_mask = _build_inpaint_mask(
                        image_np.shape, blocks, text_lines, detector_mask
                    )
                    inpainted_rgb = await asyncio.to_thread(
                        inpaint_service.inpaint, image_np, inpaint_mask
                    )
                    inpainted_b64 = _encode_png_base64(inpainted_rgb)
                    logger.info(
                        f"Image {idx + 1}: LaMa inpaint completed in "
                        f"{(time.time() - inpaint_start)*1000:.1f}ms"
                    )
                except Exception as exc:
                    logger.warning(f"Image {idx + 1}: inpaint failed ({exc}); continuing without plate")
                    inpainted_b64 = None

        # Semaphore released - GPU slot available for other images
        # Note: all_text_regions was built and filtered inside the semaphore block

        # Decode image once for all background extractions (avoids N base64 decodes)
        pil_image = decode_base64_to_pil(base64_image)

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

            # Extract background region (uses pre-decoded PIL image)
            background = extract_text_region_background(
                base64_image,
                block['minX'],
                block['minY'],
                block['maxX'],
                block['maxY'],
                preloaded_image=pil_image,
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

        await emit("done", 5, 5)
        return (idx, text_boxes, inpainted_b64)

    except Exception as e:
        logger.error(f"Error processing image {idx + 1}: {e}", exc_info=True)
        if job_id:
            await progress_bus.finish(job_id, status="error")
        return (idx, [], None)


async def _run_translation(texts: List[str], target_language: str) -> List[str]:
    """Dispatch to batched page-level translation when enabled + worthwhile,
    falling back to the legacy per-bubble parallel/sequential paths.
    """
    if not texts:
        return []
    if settings.use_batched_translation and len(texts) > 1:
        try:
            tgt = translation_pool or translation_service
            assert tgt is not None
            batched = await tgt.translate_batched(texts, target_language)
            if len(batched) == len(texts) and any(b.strip() for b in batched):
                return batched
            logger.warning("Batched translate produced empty/short output; falling back to parallel")
        except Exception as exc:
            logger.warning(f"Batched translate raised {exc!r}; falling back to parallel")
    if settings.translation_use_parallel:
        return await _translate_parallel(texts, target_language)
    return await _translate_sequential(texts, target_language)


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

        # Job id per request — frontend subscribes to /events/{job_id} for progress.
        job_id = getattr(request, "job_id", None) or uuid.uuid4().hex

        # Create semaphore for GPU memory management
        semaphore = asyncio.Semaphore(settings.max_parallel_images)

        if settings.parallel_image_processing and num_images > 1:
            # Parallel processing: process all images concurrently
            tasks = [
                process_single_image(idx, base64_image, request.targetLanguage, semaphore, job_id)
                for idx, base64_image in enumerate(request.base64Images)
            ]
            results = await asyncio.gather(*tasks)

            # Sort results by index to maintain order
            results.sort(key=lambda x: x[0])
            all_results = [r[1] for r in results]
            all_inpainted = [r[2] for r in results]
        else:
            # Sequential processing (for single image or if disabled)
            all_results = []
            all_inpainted = []
            for idx, base64_image in enumerate(request.base64Images):
                _, text_boxes, inpainted = await process_single_image(
                    idx, base64_image, request.targetLanguage, semaphore, job_id
                )
                all_results.append(text_boxes)
                all_inpainted.append(inpainted)

        await progress_bus.finish(job_id, status="ok")

        elapsed_time = time.time() - start_time
        logger.info(f"Translation request completed in {elapsed_time:.2f} seconds")
        return TranslateResponse(images=all_results, inpainted_image_base64=all_inpainted)

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Translation request failed after {elapsed_time:.2f} seconds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )
