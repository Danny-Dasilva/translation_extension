"""PARSeq-large manga OCR service (ONNX runtime, RTX 5090 FP16).

Replaces MangaOCRService with the locally-trained PARSeq-large checkpoint
(5.16% CER on manga109). The model runs in non-autoregressive mode with one
refine iteration at 128x512 resolution, and is ~30x faster than the default
vision2seq manga-ocr model on this machine (~7 ms/crop at fp16 on CUDA).

The tokenizer follows parseq's default layout:
    index 0        -> [E] (EOS)
    indices 1..C   -> charset characters
    C+1, C+2       -> [B] (BOS), [P] (PAD)
The head does not emit BOS/PAD, so logits are (L, C+1) where class 0 is EOS.
"""

# torch is imported first so its bundled CUDA libraries are on the loader
# path before onnxruntime-gpu probes for libcublas/libcudnn.
import torch  # noqa: F401

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort

from app.utils.ocr_postprocess import apply_all as postprocess_ocr

ort.set_default_logger_severity(3)  # ERROR only

logger = logging.getLogger(__name__)

# Repetition guard: run of >=5 identical consecutive chars.
_LONG_RUN_RE = re.compile(r'(.)\1{4,}')


_JP_CHAR = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]'
_JP_SPACE_PATTERN = re.compile(f'({_JP_CHAR})\\s+({_JP_CHAR})')


def normalize_japanese_text(text: str) -> str:
    """Collapse spurious whitespace between Japanese characters."""
    while _JP_SPACE_PATTERN.search(text):
        text = _JP_SPACE_PATTERN.sub(r'\1\2', text)
    return text.strip()


def _has_trigram_loop(text: str, min_repeats: int = 4) -> bool:
    """True if any trigram repeats ``min_repeats`` times consecutively."""
    if len(text) < 3 * min_repeats:
        return False
    for start in range(len(text) - 3 * min_repeats + 1):
        tri = text[start:start + 3]
        ok = True
        for k in range(1, min_repeats):
            if text[start + 3 * k:start + 3 * (k + 1)] != tri:
                ok = False
                break
        if ok:
            return True
    return False


def _repetition_guard(text: str) -> str:
    """Log (do not blank) if repetition artifacts remain after postprocessing."""
    if not text:
        return text
    if _LONG_RUN_RE.search(text):
        logger.warning("PARSeq OCR: long identical-char run detected in %r", text)
    elif _has_trigram_loop(text):
        logger.warning("PARSeq OCR: repeating trigram detected in %r", text)
    return text


def _finalize_ocr(raw: str) -> str:
    """Normalize a single decoded OCR string.

    Order:
      1. ``postprocess_ocr`` (NFC, zero-width strip, fullwidth<->halfwidth,
         punct map, middle-dot collapse, trailing-repeat cap).
      2. ``normalize_japanese_text`` (collapse whitespace between CJK chars).
      3. Repetition guard (logs only; does not mutate).
    """
    cleaned = postprocess_ocr(raw)
    cleaned = normalize_japanese_text(cleaned)
    return _repetition_guard(cleaned)


class ParseqOCRService:
    """ONNX PARSeq-large OCR with batched inference."""

    def __init__(
        self,
        model_path: str = "models/parseq_manga_large_5p16.fp16.onnx",
        fallback_fp32_path: str = "models/parseq_manga_large_5p16.opt.onnx",
        meta_path: str | None = None,
    ):
        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parents[2] / model_file
        if not model_file.exists():
            raise FileNotFoundError(f"PARSeq ONNX model not found: {model_file}")

        if meta_path is None:
            meta_file = model_file.with_suffix(".json")
            if not meta_file.exists():
                # Fallback: metadata is attached to the fp32 export.
                meta_file = model_file.parent / "parseq_manga_large_5p16.json"
        else:
            meta_file = Path(meta_path)
        meta = json.loads(meta_file.read_text())

        self.charset: str = meta["charset"]
        self.img_h, self.img_w = meta["img_size"]
        self.eos_id: int = meta["eos_id"]  # parseq puts EOS at index 0 of head
        self.head_dim: int = meta["head_dim"]
        self.mean = np.array(meta["normalize_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(meta["normalize_std"], dtype=np.float32).reshape(1, 3, 1, 1)
        # itos for the HEAD output. Head drops BOS/PAD, so indices are
        # 0 -> EOS, 1..len(charset) -> charset chars. `_itos` matches `argmax`.
        self._itos: List[str] = ["[E]"] + list(self.charset)

        cuda_first = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        fp32 = Path(fallback_fp32_path)
        if not fp32.is_absolute():
            fp32 = Path(__file__).resolve().parents[2] / fp32

        load_candidates = [
            (model_file, cuda_first),
            (fp32, cuda_first),
            (fp32, ["CPUExecutionProvider"]),
        ]

        start = time.perf_counter()
        last_err: Exception | None = None
        for candidate, providers in load_candidates:
            if not candidate.exists():
                continue
            try:
                self.session = ort.InferenceSession(str(candidate), sess_options=so, providers=providers)
                # Force a tiny inference to surface cuDNN/CUDA memory failures up front.
                _in0 = self.session.get_inputs()[0]
                _dt = np.float16 if "float16" in _in0.type else np.float32
                dummy = np.zeros((1, 3, self.img_h, self.img_w), dtype=_dt)
                self.session.run(None, {_in0.name: dummy})
                model_file = candidate
                break
            except Exception as e:
                last_err = e
                logger.warning("PARSeq load failed for %s with %s: %s", candidate.name, providers, e)
        else:
            raise RuntimeError(f"All PARSeq loaders failed; last error: {last_err}")

        actual_providers = self.session.get_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in actual_providers else "cpu"
        load_time = time.perf_counter() - start
        logger.info(
            "PARSeq loaded: %s (%.1f MB) in %.2fs on %s (providers=%s)",
            model_file.name,
            model_file.stat().st_size / 1e6,
            load_time,
            self.device,
            actual_providers,
        )

        self._input_name = self.session.get_inputs()[0].name
        # Adapt the feed dtype to the model's declared input type: fp16 exports
        # (model.half()) require float16 input, while the fp32/mixed-precision
        # exports require float32. Detect once so _run_sync casts correctly.
        self._input_np_dtype = (
            np.float16 if "float16" in self.session.get_inputs()[0].type else np.float32
        )

    @staticmethod
    def _maybe_rotate_vertical(crop: np.ndarray, thresh_aspect: float = 1.5) -> np.ndarray:
        """Rotate vertical manga bubbles 90° CCW so text reads left→right.

        Training uses img_size=(128, 512) (H×W, 4:1 horizontal). Manga bubbles
        are usually taller than they are wide; rotating CCW maps the rightmost
        vertical column onto the top row (preserving reading order).
        """
        h, w = crop.shape[:2]
        if h > thresh_aspect * w:
            return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return crop

    def _preprocess(self, crops: List[np.ndarray]) -> np.ndarray:
        """Resize to (H, W) bicubic, scale to [0,1], normalize to [-1,1]."""
        batch = np.empty((len(crops), 3, self.img_h, self.img_w), dtype=np.float32)
        for i, crop in enumerate(crops):
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            elif crop.shape[2] == 4:
                crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
            crop = self._maybe_rotate_vertical(crop)
            resized = cv2.resize(crop, (self.img_w, self.img_h), interpolation=cv2.INTER_CUBIC)
            batch[i] = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        batch -= self.mean
        batch /= self.std
        return batch

    def _decode(self, logits: np.ndarray) -> List[str]:
        """Greedy argmax; truncate at first EOS per sample."""
        ids = logits.argmax(-1)  # (B, L)
        texts: List[str] = []
        for row in ids:
            chars: List[str] = []
            for tok in row:
                if tok == self.eos_id:
                    break
                # index 0 is EOS which we already filtered; chars are 1..len(charset)
                if 0 < tok < len(self._itos):
                    chars.append(self._itos[int(tok)])
            texts.append(_finalize_ocr("".join(chars)))
        return texts

    def _run_sync(self, batch: np.ndarray) -> np.ndarray:
        if batch.dtype != self._input_np_dtype:
            batch = batch.astype(self._input_np_dtype, copy=False)
        return self.session.run(None, {self._input_name: batch})[0]

    async def recognize_text_batch(
        self,
        image_crops: List[np.ndarray],
        batch_size: int = 24,
    ) -> List[str]:
        if not image_crops:
            return []

        total_start = time.perf_counter()
        texts: List[str] = []
        current_bs = batch_size
        i = 0
        while i < len(image_crops):
            chunk = image_crops[i : i + current_bs]
            batch = self._preprocess(chunk)
            try:
                logits = await asyncio.to_thread(self._run_sync, batch)
            except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
                if "Failed to allocate memory" in str(e) and current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                    logger.warning("PARSeq OOM; reducing batch to %d", current_bs)
                    continue
                raise
            texts.extend(self._decode(logits))
            i += len(chunk)

        total_ms = (time.perf_counter() - total_start) * 1000
        logger.info(
            "PARSeq OCR batch: %d crops in %.1fms (avg %.1fms/crop)",
            len(image_crops),
            total_ms,
            total_ms / len(image_crops),
        )
        return texts

    async def recognize_single(self, image_crop: np.ndarray) -> str:
        results = await self.recognize_text_batch([image_crop])
        return results[0] if results else ""

    async def recognize_blocks_with_lines(
        self,
        image: np.ndarray,
        blocks: List[dict],
        text_lines: List[dict],
        padding: int = 2,
        batch_size: int = 24,
    ) -> List[str]:
        """OCR per text-line, then concat per block in reading order.

        PARSeq is a single-line STR model. When the detector exposes
        `text_lines` (e.g. the CTD detector), route line-level crops through
        it and stitch results back per block using spatial containment.
        Falls back to block-level OCR when no line overlaps a block.
        """
        if not blocks:
            return []
        if not text_lines:
            crops = []
            h, w = image.shape[:2]
            for b in blocks:
                x0 = max(0, b["minX"] - padding)
                y0 = max(0, b["minY"] - padding)
                x1 = min(w, b["maxX"] + padding)
                y1 = min(h, b["maxY"] + padding)
                crops.append(image[y0:y1, x0:x1])
            return await self.recognize_text_batch(crops, batch_size=batch_size)

        # Assign each text_line to the first block that contains its center.
        block_to_lines: List[List[dict]] = [[] for _ in blocks]
        for ln in text_lines:
            cx = (ln["minX"] + ln["maxX"]) / 2
            cy = (ln["minY"] + ln["maxY"]) / 2
            for bi, b in enumerate(blocks):
                if b["minX"] <= cx <= b["maxX"] and b["minY"] <= cy <= b["maxY"]:
                    block_to_lines[bi].append(ln)
                    break

        # Collect crops in a flat list, remembering the owning block.
        flat_crops: List[np.ndarray] = []
        flat_owner: List[int] = []
        h, w = image.shape[:2]
        for bi, lines in enumerate(block_to_lines):
            if not lines:
                b = blocks[bi]
                x0 = max(0, b["minX"] - padding)
                y0 = max(0, b["minY"] - padding)
                x1 = min(w, b["maxX"] + padding)
                y1 = min(h, b["maxY"] + padding)
                flat_crops.append(image[y0:y1, x0:x1])
                flat_owner.append(bi)
                continue
            # Manga reading order: right-to-left columns, top-to-bottom within.
            lines_sorted = sorted(lines, key=lambda ln: (-ln["minX"], ln["minY"]))
            for ln in lines_sorted:
                x0 = max(0, ln["minX"] - padding)
                y0 = max(0, ln["minY"] - padding)
                x1 = min(w, ln["maxX"] + padding)
                y1 = min(h, ln["maxY"] + padding)
                flat_crops.append(image[y0:y1, x0:x1])
                flat_owner.append(bi)

        texts = await self.recognize_text_batch(flat_crops, batch_size=batch_size)
        per_block: List[List[str]] = [[] for _ in blocks]
        for owner, text in zip(flat_owner, texts):
            if text:
                per_block[owner].append(text)
        return ["".join(parts) for parts in per_block]
