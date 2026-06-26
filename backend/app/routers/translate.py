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
from app.utils.english_region import should_skip_as_english
from app.utils.ocr_confidence_gate import (
    is_garbled_low_conf,
    should_erase_dropped,
    is_dialogue_context_candidate,
)
from app.services.translation_postedit import postedit_one
from app.utils.orphan_lines import (
    find_orphan_lines,
    cluster_orphan_lines,
    ocr_orphan_clusters,
    ocr_orphan_clusters_with_conf,
    cluster_bbox,
    merge_orphans_into_blocks,
    reading_order_sort,
)
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
    ocr_service = ParseqOCRService(
        model_path=settings.parseq_model_path,
        hybrid_enabled=settings.hybrid_ocr_enabled,
        ar_model_path=settings.parseq_ar_model_path,
        hybrid_conf_threshold=settings.ocr_confidence_gate_threshold,
    )
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
    erase_only_blocks: Optional[List[dict]] = None,
    fit_rects: Optional[List[Optional[dict]]] = None,
) -> np.ndarray:
    """Mask only what will be re-rendered (kept blocks) PLUS erase-only regions.
    See app.utils.ctd_utils.build_inpaint_mask — `blocks` must be the post-filter
    list so dropped detections keep their original text instead of being
    erased without replacement. `erase_only_blocks` are gate-dropped real-JP SFX
    that are erased (inpaint-only) but never translated/rendered. `fit_rects`
    (per-kept-block bubble match) drives bubble-aware solid fill — only bubble
    dialogue gets solid rects, SFX/over-art uses tight seg-ink."""
    return build_inpaint_mask(
        image_shape, blocks, text_lines, detector_mask,
        erase_blocks=erase_only_blocks or [],
        fit_rects=fit_rects,
    )


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
    erase_only_blocks: Optional[List[dict]] = None,
    fit_rects: Optional[List[Optional[dict]]] = None,
) -> Optional[str]:
    """Build the erase mask, run the inpaint router (interior fill → ring fast
    path → classical → LaMa) and return the encoded plate data URL. Runs in a
    worker thread (see overlap_inpaint) so it can overlap OCR+translate."""
    inpaint_mask = _build_inpaint_mask(
        image_np.shape, blocks, text_lines, detector_mask,
        erase_only_blocks=erase_only_blocks,
        fit_rects=fit_rects,
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
    erase_only_blocks: Optional[List[dict]] = None,
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
            erase_only_blocks,
            fit_rects,
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
            # Per-kept-bubble OCR confidence, aligned 1:1 with the final
            # ocr_texts. Each OCR branch overwrites this; default None means
            # "no confidence info" -> no low-conf name-invention suppression.
            kept_ocr_confs: List[Optional[float]] = []

            # PARSeq is a single-line STR model. If the detector exposes
            # text_lines, precompute per-block OCR by cropping individual lines
            # and stitching — this is how the model was trained. Fall back to
            # the per-batch block-crop path otherwise (matches manga-ocr).
            prefetched_texts: List[str] | None = None
            prefetched_confs: List[float] | None = None
            if isinstance(ocr_service, ParseqOCRService) and text_lines:
                prefetched_texts, prefetched_confs = await ocr_service.recognize_blocks_with_lines(
                    image_np, blocks, text_lines,
                    batch_size=settings.parseq_batch_size,
                    return_confidence=True,
                )

            # Orphan-line recovery: text_lines whose center no detected block
            # claims would otherwise be dropped here (rendering raw Japanese).
            # Paragraph-cluster + OCR them and append as synthetic blocks so they
            # flow through the SAME japanese-filter -> translate -> inpaint mask
            # -> render path as detector blocks. Gated on the prefetched (PARSeq
            # + text_lines) path so the parallel lists stay aligned.
            if (
                settings.orphan_line_recovery
                and prefetched_texts is not None
                and text_lines
            ):
                orphans = find_orphan_lines(blocks, text_lines)
                if orphans:
                    clusters = cluster_orphan_lines(orphans)
                    synth_ocr, synth_conf = await ocr_orphan_clusters_with_conf(
                        ocr_service, image_np, clusters,
                        batch_size=settings.parseq_batch_size,
                    )
                    synth_blocks: List[dict] = []
                    synth_texts: List[str] = []
                    synth_confs: List[float] = []
                    for cluster, text, c in zip(clusters, synth_ocr, synth_conf):
                        if not text.strip():
                            continue
                        synth_blocks.append(cluster_bbox(cluster))
                        synth_texts.append(text)
                        synth_confs.append(c)
                    n_before = len(blocks)
                    # Resolve synthetic-vs-original overlaps: overlapping synth
                    # blocks merge into the original (union bbox + combined text,
                    # synth dropped) so the SMS/balloon renders ONCE; isolated
                    # synth blocks are appended unchanged. Returns merged parallel
                    # blocks/prefetched_texts lists.
                    blocks, prefetched_texts = merge_orphans_into_blocks(
                        blocks, prefetched_texts, synth_blocks, synth_texts
                    )
                    # Keep OCR confidences aligned. Appended orphan blocks carry
                    # their REAL recognition confidence (tail of synth_confs) so
                    # the garble gate can drop low-conf garbled orphan SFX.
                    if prefetched_confs is not None:
                        appended = len(prefetched_texts) - len(prefetched_confs)
                        if appended > 0:
                            tail = (
                                synth_confs[-appended:]
                                if appended <= len(synth_confs)
                                else synth_confs
                            )
                            prefetched_confs = prefetched_confs + list(tail)
                            while len(prefetched_confs) < len(prefetched_texts):
                                prefetched_confs.append(0.0)
                    # build_text_regions/crop_regions depend only on block bbox,
                    # so rebuild the other parallel lists from the merged blocks
                    # (this keeps blocks/crops/text_regions/texts aligned even for
                    # originals whose bbox grew during a merge).
                    if synth_blocks:
                        crops = detector_service.crop_regions(image_np, blocks)
                        all_text_regions = build_text_regions(blocks, text_lines)
                        original_count = len(crops)
                        added = len(blocks) - n_before
                        merged = len(synth_blocks) - added
                        logger.info(
                            f"Image {idx + 1}: orphan-line recovery: "
                            f"{len(orphans)} orphan line(s) -> {len(synth_blocks)} "
                            f"cluster(s) => {added} new block(s), {merged} merged "
                            f"into originals (blocks {n_before} -> {len(blocks)})"
                        )

            # COLUMN-MAJOR RTL reading order over the FULL merged block list.
            # The v11 page-context model was trained with the page's bubbles in
            # column-major right-to-left order (build_v11_dataset.manga_reading_order);
            # the detector emitted a naive (-minX, minY) order that interleaves
            # columns. Re-sort blocks (and EVERY parallel list) into the training
            # order so the served "Page:" context matches training AND orphan
            # blocks land in their true reading position (not appended at the end).
            # reading_order_sort returns the SAME block objects, so we permute the
            # parallel lists by object identity.
            if len(blocks) > 1:
                _ordered = reading_order_sort(blocks)
                _pos = {id(b): i for i, b in enumerate(blocks)}
                _perm = [_pos[id(b)] for b in _ordered]
                blocks = _ordered
                crops = [crops[i] for i in _perm]
                all_text_regions = [all_text_regions[i] for i in _perm]
                if prefetched_texts is not None and len(prefetched_texts) == len(_perm):
                    prefetched_texts = [prefetched_texts[i] for i in _perm]
                if prefetched_confs is not None and len(prefetched_confs) == len(_perm):
                    prefetched_confs = [prefetched_confs[i] for i in _perm]

            # OCR (GPU) runs INSIDE the semaphore; the inpaint task is launched
            # here too (it only needs detection geometry, not translations) so it
            # can overlap the network-bound translate. Translation + inpaint-await
            # are intentionally deferred to AFTER the semaphore is released so the
            # GPU slot is free during the out-of-process vLLM call.
            inpaint_task: Optional["asyncio.Task"] = None
            # Snapshot the pre-filter block list BEFORE the gate reassigns
            # `blocks` to the kept subset. Gate-dropped regions that are real JP
            # ink (e.g. stylized SFX) are collected here for INPAINT-ONLY erase —
            # they are never added to the parallel kept lists used by render.
            orig_blocks = blocks
            erase_only_blocks: List[dict] = []
            # Whole-page v11 context (filled by each OCR branch). Defaults assume
            # every kept line is also context; branches widen the context with
            # dropped DIALOGUE lines so the served page has no holes.
            page_context_lines: List[str] = []
            target_positions: List[int] = []
            if settings.use_pipeline_overlap and len(crops) > 1:
                # PIPELINE OVERLAP with mini-batching: OCR crops in batches of 3,
                # filtering non-Japanese as we go. OCR stays on the GPU inside the
                # semaphore; translation is run after release (see below).
                MINI_BATCH_SIZE = 3

                # Block-ordered indices of gate-dropped DIALOGUE lines (not
                # rendered, but kept as v11 page context for speaker/pronoun
                # continuity). Filled inside ocr_pipelined alongside `results`.
                ctx_dropped: List[Tuple[int, str]] = []

                async def ocr_pipelined():
                    results = []
                    for batch_start in range(0, len(crops), MINI_BATCH_SIZE):
                        batch_crops = crops[batch_start:batch_start + MINI_BATCH_SIZE]
                        batch_indices = list(range(batch_start, batch_start + len(batch_crops)))

                        # OCR mini-batch (or slice of prefetched per-block OCR)
                        if prefetched_texts is not None:
                            texts = prefetched_texts[batch_start:batch_start + len(batch_crops)]
                            confs = (
                                prefetched_confs[batch_start:batch_start + len(batch_crops)]
                                if prefetched_confs is not None
                                else [1.0] * len(texts)
                            )
                        else:
                            tc = await ocr_service.recognize_text_batch_with_conf(batch_crops)
                            texts = [t for t, _c in tc]
                            confs = [c for _t, c in tc]

                        for i, text, conf in zip(batch_indices, texts, confs):
                            # English early-exit: a horizontal/Latin (non-Japanese)
                            # region is left as ORIGINAL pixels — not translated,
                            # not inpainted, no TextBox. Mirror is_leave_intact_label:
                            # just `continue` (do NOT add to kept lists OR to
                            # erase_only_blocks).
                            if settings.english_early_exit_enabled and should_skip_as_english(
                                orig_blocks[i], text_lines, text, is_japanese_text
                            ):
                                logger.info(
                                    "English early-exit skipped idx %d: %r", i, text[:30]
                                )
                                continue
                            # Filter non-Japanese before keeping
                            if settings.japanese_filter_enabled:
                                if not is_japanese_text(
                                    text,
                                    settings.japanese_filter_min_ratio,
                                    settings.japanese_filter_katakana_max_length
                                ):
                                    logger.debug(f"Filtered non-Japanese text at index {i}: '{text[:30]}...'")
                                    continue
                            # OCR-confidence garble gate: drop low-conf garbled
                            # OCR before translation (stops hallucinated captions).
                            if (
                                settings.ocr_confidence_gate_enabled
                                and settings.ocr_confidence_gate_threshold > 0
                                and is_garbled_low_conf(
                                    text, conf,
                                    conf_threshold=settings.ocr_confidence_gate_threshold,
                                )
                            ):
                                logger.info(
                                    "OCR-gate dropped index %d (conf=%.2f): %r — garbled",
                                    i, conf, text[:24],
                                )
                                # Real-JP ink we drop (not translate) must still
                                # be erased so raw Japanese doesn't survive into
                                # the render. Collect for inpaint-only — NOT into
                                # the parallel kept lists.
                                if should_erase_dropped(text):
                                    erase_only_blocks.append(orig_blocks[i])
                                # A dropped DIALOGUE line still belongs in the
                                # v11 page context (no holes for the model); a
                                # pure-SFX / garble drop does not.
                                if (
                                    settings.translation_pagecontext_whole_page
                                    and is_dialogue_context_candidate(text, ocr_confidence=conf)
                                ):
                                    ctx_dropped.append((i, text))
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
                # WHOLE-PAGE v11 context: numbered page = kept dialogue + dropped
                # dialogue lines, in reading (block) order; targets = kept lines.
                ctx_map = {i: t for i, t in paired}
                ctx_map.update({i: t for i, t in ctx_dropped})
                context_order = sorted(ctx_map)
                page_context_lines = [ctx_map[i] for i in context_order]
                ctx_pos = {orig: p for p, orig in enumerate(context_order)}
                target_positions = [ctx_pos[i] for i in kept_indices]
                blocks = [blocks[i] for i in kept_indices]
                crops = [crops[i] for i in kept_indices]
                all_text_regions = [all_text_regions[i] for i in kept_indices]
                # NOTE: the pipelined OCR path consumes recognition confidence
                # inside the garble gate (ocr_pipelined) and does not retain it
                # per kept bubble. To enable low-conf name-invention suppression
                # here too, `paired` would need to carry conf as a 3-tuple
                # (i, text, conf). Until then we pass None (no suppression),
                # which is the prior behaviour. The batch branch threads real
                # confidence via kept_ocr_confs.
                kept_ocr_confs = [None] * len(ocr_texts)

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
                    idx, image_np, blocks, text_lines, detector_mask, fit_rects, emit,
                    erase_only_blocks=erase_only_blocks,
                )
                if not settings.overlap_inpaint:
                    inpainted_b64 = await _await_inpaint_task(idx, inpaint_task)
                    inpaint_task = None
            else:
                # BATCH MODE: All OCR first, then filter, then translation (after release)
                if prefetched_texts is not None:
                    ocr_texts = prefetched_texts
                    ocr_confs = prefetched_confs if prefetched_confs is not None else [1.0] * len(ocr_texts)
                else:
                    tc = await ocr_service.recognize_text_batch_with_conf(
                        crops,
                        batch_size=len(crops)  # Process all crops at once
                    )
                    ocr_texts = [t for t, _c in tc]
                    ocr_confs = [c for _t, c in tc]
                ocr_time = time.time() - ocr_start
                logger.info(f"Image {idx + 1}: Batched OCR completed in {ocr_time*1000:.1f}ms ({len(crops)} crops)")

                # Filter non-Japanese OCR results + OCR-confidence garble gate
                if settings.japanese_filter_enabled:
                    valid_indices = filter_japanese_texts(
                        ocr_texts,
                        settings.japanese_filter_min_ratio,
                        settings.japanese_filter_katakana_max_length
                    )
                    # English early-exit: drop horizontal/Latin (non-Japanese)
                    # regions from valid_indices so they are left as ORIGINAL
                    # pixels — never added to the kept lists, never inpainted
                    # (NOT added to erase_only_blocks), no TextBox. Mirrors
                    # is_leave_intact_label's clean skip.
                    if settings.english_early_exit_enabled:
                        kept_after_english = []
                        for i in valid_indices:
                            if should_skip_as_english(
                                orig_blocks[i], text_lines, ocr_texts[i], is_japanese_text
                            ):
                                logger.info(
                                    "English early-exit skipped idx %d: %r",
                                    i, ocr_texts[i][:30],
                                )
                                continue
                            kept_after_english.append(i)
                        valid_indices = kept_after_english
                    # Block-ordered indices of gate-dropped DIALOGUE lines kept
                    # as v11 page context (not rendered). Pure-SFX drops excluded.
                    ctx_dropped_idx: List[int] = []
                    if settings.ocr_confidence_gate_enabled and settings.ocr_confidence_gate_threshold > 0:
                        gated = []
                        for i in valid_indices:
                            conf = ocr_confs[i] if i < len(ocr_confs) else 1.0
                            if is_garbled_low_conf(
                                ocr_texts[i], conf,
                                conf_threshold=settings.ocr_confidence_gate_threshold,
                            ):
                                logger.info(
                                    "Image %d: OCR-gate dropped index %d (conf=%.2f): %r",
                                    idx + 1, i, conf, ocr_texts[i][:24],
                                )
                                # Real-JP ink we drop (not translate) must still
                                # be erased — collect for inpaint-only, NOT into
                                # the parallel kept lists below.
                                if should_erase_dropped(ocr_texts[i]):
                                    erase_only_blocks.append(orig_blocks[i])
                                if (
                                    settings.translation_pagecontext_whole_page
                                    and is_dialogue_context_candidate(ocr_texts[i], ocr_confidence=conf)
                                ):
                                    ctx_dropped_idx.append(i)
                                continue
                            gated.append(i)
                        valid_indices = gated

                    filtered_count = len(ocr_texts) - len(valid_indices)
                    if filtered_count > 0:
                        logger.info(f"Image {idx + 1}: Filtered {filtered_count} non-Japanese/garbled regions")

                    if not valid_indices:
                        logger.warning(f"Image {idx + 1}: All text regions filtered as non-Japanese")
                        await emit("done", 5, 5, note="all_filtered")
                        return (idx, [], None)

                    # WHOLE-PAGE v11 context: numbered page = kept + dropped
                    # dialogue lines in reading (block) order; targets = kept.
                    context_order = sorted(set(valid_indices) | set(ctx_dropped_idx))
                    page_context_lines = [ocr_texts[i] for i in context_order]
                    ctx_pos = {orig: p for p, orig in enumerate(context_order)}
                    target_positions = [ctx_pos[i] for i in valid_indices]

                    # Filter all parallel lists to maintain alignment
                    ocr_texts = [ocr_texts[i] for i in valid_indices]
                    blocks = [blocks[i] for i in valid_indices]
                    crops = [crops[i] for i in valid_indices]
                    all_text_regions = [all_text_regions[i] for i in valid_indices]
                    # Keep OCR confidence aligned 1:1 so the post-edit can
                    # suppress name invention on low-confidence bubbles.
                    kept_ocr_confs = [
                        ocr_confs[i] if i < len(ocr_confs) else None
                        for i in valid_indices
                    ]
                else:
                    # japanese_filter disabled: no per-index filtering happened,
                    # so confidences (if any) stay aligned with ocr_texts as-is.
                    kept_ocr_confs = list(ocr_confs)
                    # All lines kept -> page context == kept lines, in order.
                    page_context_lines = list(ocr_texts)
                    target_positions = list(range(len(ocr_texts)))

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
                    idx, image_np, blocks, text_lines, detector_mask, fit_rects, emit,
                    erase_only_blocks=erase_only_blocks,
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
        translations = await _run_translation(
            ocr_texts,
            target_language,
            page_context_lines=page_context_lines,
            target_positions=target_positions,
        )
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
        # kept_ocr_confs is set in BOTH OCR branches above, aligned 1:1 with
        # ocr_texts (real confidence in the batch branch, None in the pipelined
        # branch — see the note there). Re-align defensively if the lengths
        # diverged (e.g. an OCR sub-path that didn't populate it).
        if len(kept_ocr_confs) != len(ocr_texts):
            kept_ocr_confs = [None] * len(ocr_texts)
        text_boxes = []
        for block, ocr_text, translated_text, text_regions, fit_rect, ocr_conf in zip(
            blocks, ocr_texts, translations, all_text_regions, fit_rects, kept_ocr_confs
        ):
            # Post-translation glossaries (pure post-edit; v11 prompt untouched).
            # Shared with the batch pipeline via translation_postedit.
            # ocr_conf (when available) suppresses low-confidence name invention.
            translated_text = postedit_one(translated_text, ocr_text, ocr_conf=ocr_conf)

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


async def _run_translation(
    texts: List[str],
    target_language: str,
    page_context_lines: Optional[List[str]] = None,
    target_positions: Optional[List[int]] = None,
) -> List[str]:
    """Dispatch to batched page-level translation when enabled + worthwhile,
    falling back to the legacy per-bubble parallel/sequential paths.

    ``texts`` are the KEPT lines (1:1 with the render). ``page_context_lines`` is
    the WHOLE page's dialogue (kept + dropped-dialogue) in reading order and
    ``target_positions`` indexes the kept lines within it — when both are present
    and the v11 page-context path is on, the model is given the full page as
    context while only the kept lines are translated/returned (no holes where
    dropped dialogue used to be). Output stays aligned 1:1 with ``texts``.
    """
    if not texts:
        return []
    # WHOLE-PAGE v11 context: the full page (incl. dropped dialogue) as context,
    # only the kept lines marked/returned. The strongest train/serve match.
    if (
        settings.batch_translate
        and settings.translation_v11_pagecontext
        and settings.translation_pagecontext_whole_page
        and page_context_lines
        and target_positions is not None
        and len(target_positions) == len(texts)
        and len(page_context_lines) > 1
        and hasattr(translation_service, "translate_page_context_marked")
    ):
        try:
            marked = await translation_service.translate_page_context_marked(
                page_context_lines, target_positions, target_language
            )
            if len(marked) == len(texts) and any(b.strip() for b in marked):
                return marked
            logger.warning("Whole-page context translate produced empty/mismatched output; falling back")
        except Exception as exc:
            logger.warning(f"Whole-page context translate raised {exc!r}; falling back")
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
