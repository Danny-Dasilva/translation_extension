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
from app.utils.image_processing import (
    calculate_font_size,
    decode_base64_to_numpy,
    detect_font_colors,
)
from app.utils.ctd_utils import build_text_regions, build_inpaint_mask, match_blocks_to_bubbles
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

# Translation backend selection. The vLLM (+MTP) path is the production
# default; "transformers" runs the Hy-MT1.5-2bit model in-process. Both
# expose the same async translate_single / translate_batched surface.
_translation_backend = settings.translation_backend.lower()
if _translation_backend == "vllm-openai":
    logger.info("Using vLLM (OpenAI-compatible) translation backend")
    from app.services.vllm_openai_translation_service import VLLMOpenAITranslationService
    translation_service = VLLMOpenAITranslationService(
        base_url=settings.vllm_base_url,
        model_name=settings.vllm_model_name,
    )
elif _translation_backend == "transformers":
    logger.info("Using transformers (Hy-MT1.5) translation backend")
    from app.services.hymt_transformers_service import HyMTTransformersService
    translation_service = HyMTTransformersService()
else:
    raise ValueError(
        f"Unknown translation_backend {settings.translation_backend!r}. "
        "Use 'vllm-openai' or 'transformers'."
    )

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

# Lazy-init speech-bubble detector (YOLOv10n). Separate from the CTD text
# detector above: CTD finds tight text columns, this finds the enclosing
# balloon so the frontend can typeset the translation to the bubble interior.
bubble_detector = None
if settings.enable_bubble_fit:
    try:
        from app.services.detector_service import DetectorService
        logger.info("Loading YOLOv10n speech-bubble detector (bubble-fit typesetting)")
        bubble_detector = DetectorService(model_path=settings.yolo_model_path)
    except Exception as exc:
        logger.warning(f"Bubble detector unavailable ({exc}); typesetting falls back to text blocks")
        bubble_detector = None

logger.info("Local AI pipeline ready")


def _encode_png_base64(image_rgb: np.ndarray) -> str:
    """Encode an HxWx3 uint8 RGB ndarray as a data URL."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _encode_plate_base64(image_rgb: np.ndarray) -> str:
    """Encode the inpainted plate as a browser-renderable data URL.

    Defaults to WebP (lossy, configurable quality) which shrinks the plate
    payload ~91% vs uncompressed PNG (3.38MB -> ~0.28MB/page) with no visible
    loss on manga line-art. WebP decodes natively in the browser canvas, so the
    frontend needs no change. Falls back to PNG if WebP is disabled or the
    encoder is unavailable in this OpenCV build.
    """
    if settings.plate_encode_webp:
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        quality = int(settings.plate_webp_quality)
        ok, buf = cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
        if ok:
            return "data:image/webp;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        logger.warning("WebP plate encode failed; falling back to PNG")
    return _encode_png_base64(image_rgb)


def _build_inpaint_mask(
    image_shape: Tuple[int, int],
    blocks: List[dict],
    text_lines: List[dict],
    detector_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Mask only what will be re-rendered (kept blocks). See
    app.utils.ctd_utils.build_inpaint_mask — `blocks` must be the post-filter
    list so dropped detections keep their original text instead of being
    erased without replacement."""
    return build_inpaint_mask(image_shape, blocks, text_lines, detector_mask)


def _fit_rects_to_bubble_rects(
    fit_rects: List[Optional[dict]],
) -> Optional[List[Optional[Tuple[int, int, int, int]]]]:
    """Convert matched bubble dicts to (minX,minY,maxX,maxY) int tuples for the
    LaMa interior solid-fill tier. None entries (no qualifying bubble) pass
    through as None so those components stay on the neural/classical tiers."""
    if not settings.enable_bubble_solid_fill:
        return None
    rects: List[Optional[Tuple[int, int, int, int]]] = []
    any_rect = False
    for fr in fit_rects:
        if fr is None:
            rects.append(None)
        else:
            rects.append(
                (int(fr["minX"]), int(fr["minY"]), int(fr["maxX"]), int(fr["maxY"]))
            )
            any_rect = True
    return rects if any_rect else None


def _run_inpaint_sync(
    image_np: np.ndarray,
    blocks: List[dict],
    text_lines: List[dict],
    detector_mask: Optional[np.ndarray],
    bubble_rects: Optional[List[Optional[Tuple[int, int, int, int]]]],
) -> Optional[str]:
    """Build the erase mask, run the inpaint router (interior fill → ring fast
    path → classical → LaMa) and return the encoded plate data URL. Runs in a
    worker thread (see overlap_inpaint) so it can overlap OCR+translate."""
    inpaint_mask = _build_inpaint_mask(
        image_np.shape, blocks, text_lines, detector_mask
    )
    inpainted_rgb = inpaint_service.inpaint(
        image_np, inpaint_mask, bubble_rects=bubble_rects
    )
    return _encode_plate_base64(inpainted_rgb)


def _maybe_start_inpaint_task(
    idx: int,
    image_np: np.ndarray,
    blocks: List[dict],
    text_lines: List[dict],
    detector_mask: Optional[np.ndarray],
    fit_rects: List[Optional[dict]],
    emit,
) -> Optional["asyncio.Task"]:
    """Kick off the inpaint in a worker thread so it overlaps OCR+translate.

    Returns the task (await it after translation) or None when inpainting is
    disabled. When settings.overlap_inpaint is False the thread still runs but
    is awaited immediately by the caller, preserving serial behaviour.
    """
    if inpaint_service is None or not settings.enable_inpainting:
        return None
    bubble_rects = _fit_rects_to_bubble_rects(fit_rects)
    return asyncio.create_task(
        asyncio.to_thread(
            _run_inpaint_sync,
            image_np,
            blocks,
            text_lines,
            detector_mask,
            bubble_rects,
        )
    )


async def _await_inpaint_task(
    idx: int, inpaint_task: Optional["asyncio.Task"]
) -> Optional[str]:
    """Await the inpaint worker; failures are non-fatal (no plate returned)."""
    if inpaint_task is None:
        return None
    inpaint_start = time.time()
    try:
        b64 = await inpaint_task
        stats = getattr(inpaint_service, "last_stats", {}) or {}
        logger.info(
            f"Image {idx + 1}: inpaint completed in "
            f"{(time.time() - inpaint_start)*1000:.1f}ms "
            f"(components={stats.get('components')}, "
            f"bubblefill={stats.get('bubblefill_hits')}, "
            f"fastpath={stats.get('fastpath_hits')}, "
            f"classical={stats.get('classical_hits')}, "
            f"forwards={stats.get('forward_calls')})"
        )
        return b64
    except Exception as exc:
        logger.warning(f"Image {idx + 1}: inpaint failed ({exc}); continuing without plate")
        return None


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
    bubbles: List[dict] = []
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

            # Step 2: Detect text blocks (CTD) and speech bubbles (YOLOv10n)
            # CONCURRENTLY. CTD's session.run and YOLO's predict both offload to
            # worker threads, so gathering them overlaps the two GPU forwards
            # instead of running them back-to-back (saved ~YOLO latency/page).
            # Bubble detect failure is non-fatal — we fall back to tight block
            # bboxes downstream.
            await emit("detect", 1, 5)
            detect_start = time.time()

            async def _detect_bubbles_safe() -> List[dict]:
                if bubble_detector is None:
                    return []
                try:
                    return await bubble_detector.detect_bubbles(image_np)
                except Exception as exc:
                    logger.warning(
                        f"Image {idx + 1}: bubble detect failed ({exc}); using text blocks"
                    )
                    return []

            ctd_result, bubbles = await asyncio.gather(
                detector_service.detect(image_np),
                _detect_bubbles_safe(),
            )
            detect_time = time.time() - detect_start

            blocks = ctd_result["blocks"]
            text_lines = ctd_result["text_lines"]
            detector_mask = ctd_result.get("mask")

            logger.info(
                f"Image {idx + 1}: Detected {len(blocks)} blocks, "
                f"{len(text_lines)} text lines, {len(bubbles)} bubbles "
                f"in {detect_time*1000:.1f}ms"
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

            # OCR (GPU) runs INSIDE the semaphore; the inpaint task is launched
            # here too (it only needs detection geometry, not translations) so it
            # can overlap the network-bound translate. Translation + inpaint-await
            # are intentionally deferred to AFTER the semaphore is released so the
            # GPU slot is free during the out-of-process vLLM call.
            inpaint_task: Optional["asyncio.Task"] = None
            if settings.use_pipeline_overlap and len(crops) > 1:
                # PIPELINE OVERLAP with mini-batching: OCR crops in batches of 3,
                # filtering non-Japanese as we go. OCR stays on the GPU inside the
                # semaphore; translation is run after release (see below).
                MINI_BATCH_SIZE = 3

                async def ocr_pipelined():
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
                            # Filter non-Japanese before keeping
                            if settings.japanese_filter_enabled:
                                if not is_japanese_text(
                                    text,
                                    settings.japanese_filter_min_ratio,
                                    settings.japanese_filter_katakana_max_length
                                ):
                                    logger.debug(f"Filtered non-Japanese text at index {i}: '{text[:30]}...'")
                                    continue
                            results.append((i, text))
                    return results

                paired = await ocr_pipelined()

                if not paired:
                    logger.warning(f"Image {idx + 1}: All text regions filtered as non-Japanese")
                    await emit("done", 5, 5, note="all_filtered")
                    return (idx, [], None)

                # Extract OCR results and filter parallel lists to kept indices
                kept_indices = [i for i, _ in paired]
                ocr_texts = [text for _, text in paired]
                blocks = [blocks[i] for i in kept_indices]
                crops = [crops[i] for i in kept_indices]
                all_text_regions = [all_text_regions[i] for i in kept_indices]

                ocr_time = time.time() - ocr_start

                filtered_count = original_count - len(kept_indices)
                if filtered_count > 0:
                    logger.info(f"Image {idx + 1}: Pipelined OCR completed in {ocr_time*1000:.1f}ms ({len(crops)} kept, {filtered_count} filtered)")
                else:
                    logger.info(f"Image {idx + 1}: Pipelined OCR completed in {ocr_time*1000:.1f}ms ({len(crops)} crops)")

                # Match kept blocks to bubbles ONCE (reused for inpaint + response).
                fit_rects = (
                    match_blocks_to_bubbles(blocks, bubbles)
                    if bubbles else [None] * len(blocks)
                )

                # Launch inpaint BEFORE releasing the semaphore so it overlaps the
                # post-release translation. When overlap is disabled, finish it
                # here (serial, inside the semaphore) to keep GPU stages serial.
                await emit("inpaint", 4, 5, note=f"{len(blocks)} regions")
                inpaint_task = _maybe_start_inpaint_task(
                    idx, image_np, blocks, text_lines, detector_mask, fit_rects, emit
                )
                if not settings.overlap_inpaint:
                    inpainted_b64 = await _await_inpaint_task(idx, inpaint_task)
                    inpaint_task = None
            else:
                # BATCH MODE: All OCR first, then filter, then translation (after release)
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

                # Match kept blocks to bubbles ONCE (reused for both the inpaint
                # interior-fill tier and the response build below).
                fit_rects = (
                    match_blocks_to_bubbles(blocks, bubbles)
                    if bubbles else [None] * len(blocks)
                )

                # Step 5b: LaMa inpaint. Inpainting needs only the detection mask
                # (not translated text), so launch it as a background thread task
                # and let it OVERLAP with the post-release translation. The worker
                # thread frees the event loop to drive the concurrent vLLM calls.
                await emit("inpaint", 4, 5, note=f"{len(blocks)} regions")
                inpaint_task = _maybe_start_inpaint_task(
                    idx, image_np, blocks, text_lines, detector_mask, fit_rects, emit
                )

                # When overlap is disabled, finish inpaint before releasing the
                # semaphore so the two GPU stages stay serial (exact prior behaviour).
                if not settings.overlap_inpaint:
                    inpainted_b64 = await _await_inpaint_task(idx, inpaint_task)
                    inpaint_task = None

        # Semaphore released - GPU slot available for other images.
        # Translation is an out-of-process (vLLM/httpx) or in-process-but-non-GPU
        # call, and the inpaint worker runs in its own thread, so neither needs to
        # hold the GPU slot. all_text_regions/blocks/crops/fit_rects were all built
        # and filtered inside the semaphore block above and remain in scope here.

        # Translation (batched page-level [N] protocol, or parallel/sequential fallback)
        await emit("translate", 3, 5, note=f"{len(ocr_texts)} bubbles")
        translate_start = time.time()
        translations = await _run_translation(ocr_texts, target_language)
        translate_time = time.time() - translate_start
        logger.info(f"Image {idx + 1}: Translation completed in {translate_time*1000:.1f}ms ({len(ocr_texts)} texts)")

        # Await the overlapped inpaint (if still running) after translation.
        if inpaint_task is not None:
            inpainted_b64 = await _await_inpaint_task(idx, inpaint_task)

        # Calculate per-crop timing (distribute evenly)
        num_items = len(crops) if crops else 1
        ocr_time_per_crop = (ocr_time * 1000) / num_items
        translate_time_per_text = (translate_time * 1000) / num_items

        # Step 6: Build response
        text_boxes = []
        for block, ocr_text, translated_text, text_regions, fit_rect in zip(
            blocks, ocr_texts, translations, all_text_regions, fit_rects
        ):
            # Font sizing target: the translation is typeset to the bubble
            # INTERIOR when one was matched (wide, horizontal) — not the tight
            # vertical-JP column. Size against the bubbleRect so a roomy balloon
            # yields a roomy font; fall back to the inset text region otherwise.
            if fit_rect is not None:
                bbox_width = int(fit_rect["maxX"]) - int(fit_rect["minX"])
                bbox_height = int(fit_rect["maxY"]) - int(fit_rect["minY"])
            else:
                region = text_regions[0] if text_regions else block
                bbox_width = region['maxX'] - region['minX']
                bbox_height = region['maxY'] - region['minY']
            font_size = calculate_font_size(
                bbox_width,
                bbox_height,
                len(translated_text) if translated_text else 1
            )

            # Content-aware font colors: sample the bubble/block background from
            # the source page and pick dark-on-light vs light-on-dark so text on
            # a black bubble renders white (and vice versa) instead of a fixed
            # black/white default that vanishes on inverted balloons.
            try:
                cy0 = max(0, int(block['minY'])); cy1 = min(image_np.shape[0], int(block['maxY']))
                cx0 = max(0, int(block['minX'])); cx1 = min(image_np.shape[1], int(block['maxX']))
                sample = image_np[cy0:cy1, cx0:cx1]
                if sample.size:
                    font_color, stroke_color = detect_font_colors(sample)
                else:
                    font_color, stroke_color = "#000000", "#FFFFFF"
            except Exception:
                font_color, stroke_color = "#000000", "#FFFFFF"

            text_box = TextBox(
                ocrText=ocr_text,
                originalLanguage="ja",
                minX=block['minX'],
                minY=block['minY'],
                maxX=block['maxX'],
                maxY=block['maxY'],
                background="",
                fontHeightPx=font_size,
                fontColor=font_color,
                fontStrokeColor=stroke_color,
                zIndex=1,
                translatedText=translated_text,
                subtextBoxes=[],
                textRegions=[TextRegion(**r) for r in text_regions],
                bubbleRect=TextRegion(
                    minX=int(fit_rect["minX"]),
                    minY=int(fit_rect["minY"]),
                    maxX=int(fit_rect["maxX"]),
                    maxY=int(fit_rect["maxY"]),
                ) if fit_rect else None,
                confidence=block.get('confidence', 0.0),
                ocrTimeMs=round(ocr_time_per_crop, 2),
                translateTimeMs=round(translate_time_per_text, 2),
            )

            text_boxes.append(text_box)

        # Assign smart zIndex: smaller boxes get higher zIndex (rendered on top)
        assign_smart_zindex(text_boxes, use_dict=False)

        image_time = time.time() - image_start
        if settings.use_pipeline_overlap and len(crops) > 1:
            logger.info(
                f"Image {idx + 1} completed: {len(text_boxes)} boxes in {image_time*1000:.1f}ms "
                f"(detect: {detect_time*1000:.1f}ms, pipelined ocr: {ocr_time*1000:.1f}ms, translate: {translate_time*1000:.1f}ms)"
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
    # TRUE single-call numbered-block path (QUALITY-GATED, default OFF). Packs
    # the whole page into one vLLM generate call. Falls back cleanly to the
    # per-bubble paths below on empty/short/mismatched output.
    if (
        settings.batch_translate
        and len(texts) > 1
        and hasattr(translation_service, "translate_numbered_block")
    ):
        try:
            blocked = await translation_service.translate_numbered_block(texts, target_language)
            if len(blocked) == len(texts) and any(b.strip() for b in blocked):
                return blocked
            logger.warning("Numbered-block translate produced empty/mismatched output; falling back")
        except Exception as exc:
            logger.warning(f"Numbered-block translate raised {exc!r}; falling back")
    if settings.use_batched_translation and len(texts) > 1:
        try:
            batched = await translation_service.translate_batched(texts, target_language)
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
        trans = await translation_service.translate_single(text, target_language)
        translations.append(trans)
    return translations


async def _translate_parallel(texts: List[str], target_language: str) -> List[str]:
    """
    OPTIMIZATION 2: Translate all texts in parallel.

    Fans out per-bubble translate_single calls via asyncio.gather. The vLLM
    backend handles concurrent requests with continuous batching; the
    transformers backend serializes internally but stays correct.
    """
    if not texts:
        return []

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
