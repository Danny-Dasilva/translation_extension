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

from app.services._ort_init import cuda_provider_options

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
        hybrid_enabled: bool = False,
        ar_model_path: str | None = None,
        hybrid_conf_threshold: float = 0.65,
        vertical_ar_default: bool = True,
        vertical_ar_aspect: float = 1.5,
    ):
        # --- Confidence-gated HYBRID OCR config -----------------------------
        # When enabled, low-confidence crops (the ones the gate treats as
        # garbled) are re-OCR'd by the AR model in one batch and replace the
        # non-AR result. The AR session is loaded lazily on first low-conf hit.
        self.hybrid_enabled = bool(hybrid_enabled)
        self._ar_model_path = ar_model_path
        self.hybrid_conf_threshold = float(hybrid_conf_threshold)
        self._ar_session = None
        self._ar_input_name: str | None = None
        self._ar_input_np_dtype = np.float32
        # Cumulative count of crops re-OCR'd with AR (for per-page logging).
        self.ar_retry_count = 0

        # --- Vertical-AR-by-default routing ---------------------------------
        # The NAR decode duplicates adjacent kana on dense VERTICAL crops at
        # falsely-high confidence (身代わり -> 身身わわ @0.92), so the conf-gated
        # retry above never fires on the worst cases. Route tall/narrow crops
        # (h/w >= aspect) to the AR model UP FRONT, by geometry — independent of
        # confidence AND independent of hybrid_enabled (so it works even when the
        # conf-gated retry is disabled). Horizontal crops stay on the fast NAR
        # path. The same lazily-loaded AR session is reused (no double-load).
        self.vertical_ar_default = bool(vertical_ar_default)
        self.vertical_ar_aspect = float(vertical_ar_aspect)
        # Cumulative count of crops routed to AR by geometry (per-page logging).
        self.vertical_ar_count = 0

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
            ("CUDAExecutionProvider", cuda_provider_options({"cudnn_conv_algo_search": "HEURISTIC"})),
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

    def _is_vertical_crop(self, crop: np.ndarray) -> bool:
        """True if a crop is tall/narrow (the vertical-text signature).

        Uses the SAME ``h > aspect * w`` test as ``_maybe_rotate_vertical`` (with
        the configured ``vertical_ar_aspect``, default 1.5) so the crop set that
        gets ROTATED-for-vertical is exactly the set ROUTED-to-AR — no surprises.
        Strict ``>`` (h exactly == aspect*w is NOT vertical) matches the rotate
        predicate. Vertical crops carry the NAR dup-garble, so they go to AR.
        """
        if crop is None or crop.size == 0:
            return False
        h, w = crop.shape[:2]
        if w <= 0:
            return False
        return h > self.vertical_ar_aspect * w

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

    @staticmethod
    def _softmax_lastdim(logits: np.ndarray) -> np.ndarray:
        """Numerically-stable softmax over the last axis (fp32)."""
        x = logits.astype(np.float32)
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def _decode(self, logits: np.ndarray) -> List[str]:
        """Greedy argmax; truncate at first EOS per sample (text only)."""
        return [t for t, _c in self._decode_with_conf(logits)]

    def _decode_with_conf(self, logits: np.ndarray) -> List[tuple[str, float]]:
        """Greedy decode + per-crop OCR recognition confidence.

        Confidence is the mean softmax max-probability over the DECODED tokens
        (the chars emitted before EOS). High for crisp dialogue (~0.9+), low for
        garbled / stylized SFX the recognizer is unsure about (~0.4-0.6).
        Empty decodes (immediate EOS) get conf 0.0.
        """
        ids = logits.argmax(-1)  # (B, L)
        probs = self._softmax_lastdim(logits)  # (B, L, C)
        maxp = probs.max(-1)  # (B, L) softmax max-prob per step
        out: List[tuple[str, float]] = []
        for row, prow in zip(ids, maxp):
            chars: List[str] = []
            confs: List[float] = []
            for tok, p in zip(row, prow):
                if tok == self.eos_id:
                    break
                # index 0 is EOS which we already filtered; chars are 1..len(charset)
                if 0 < tok < len(self._itos):
                    chars.append(self._itos[int(tok)])
                    confs.append(float(p))
            conf = float(np.mean(confs)) if confs else 0.0
            out.append((_finalize_ocr("".join(chars)), conf))
        return out

    def _run_sync(self, batch: np.ndarray) -> np.ndarray:
        if batch.dtype != self._input_np_dtype:
            batch = batch.astype(self._input_np_dtype, copy=False)
        return self.session.run(None, {self._input_name: batch})[0]

    # ----------------------------- AR session -----------------------------
    def _ar_wanted(self) -> bool:
        """True if any AR path (conf-gated retry OR vertical routing) is on.

        The AR session is shared by BOTH the legacy conf-gated retry
        (``hybrid_enabled``) and the new vertical-AR-by-default routing
        (``vertical_ar_default``). Either one wanting AR is enough to load it —
        in particular vertical routing must work even when the conf-gated retry
        is disabled (the GPU validation runs HYBRID_OCR_ENABLED=false).
        """
        return bool(self.hybrid_enabled) or bool(
            getattr(self, "vertical_ar_default", False)
        )

    def _ensure_ar_session(self) -> bool:
        """Lazily load the AR ONNX session (CUDA-bound, fail-loud on CPU drop).

        Mirrors the non-AR ``_ort_init`` CUDA setup: requests
        CUDAExecutionProvider first and RAISES if ORT silently falls back to
        CPU (the AR model is ~10x heavier; a silent CPU bind would tank
        latency). Returns True once a CUDA-bound session is ready. On any load
        failure it disables ALL AR paths (logs once) and returns False so OCR
        still serves non-AR results. Shared by the conf-gated retry and the
        vertical-AR routing.
        """
        if self._ar_session is not None:
            return True
        if not self._ar_wanted() or not self._ar_model_path:
            return False

        ar_file = Path(self._ar_model_path)
        if not ar_file.is_absolute():
            ar_file = Path(__file__).resolve().parents[2] / ar_file
        if not ar_file.exists():
            logger.error("AR OCR: AR model not found: %s — disabling AR paths", ar_file)
            self.hybrid_enabled = False
            self.vertical_ar_default = False
            return False

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [
            ("CUDAExecutionProvider", cuda_provider_options({"cudnn_conv_algo_search": "HEURISTIC"})),
            "CPUExecutionProvider",
        ]
        try:
            start = time.perf_counter()
            sess = ort.InferenceSession(str(ar_file), sess_options=so, providers=providers)
            actual = sess.get_providers()
            if "CUDAExecutionProvider" not in actual:
                # Fail loud: a silent CPU bind would make the AR retry ~unusable.
                raise RuntimeError(
                    f"HYBRID OCR: AR model bound CPU-only (providers={actual}); "
                    "refusing silent CPU fallback"
                )
            in0 = sess.get_inputs()[0]
            self._ar_input_name = in0.name
            self._ar_input_np_dtype = np.float16 if "float16" in in0.type else np.float32
            # Warm up to surface cuDNN/CUDA failures up front and JIT-compile.
            dummy = np.zeros((1, 3, self.img_h, self.img_w), dtype=self._ar_input_np_dtype)
            sess.run(None, {self._ar_input_name: dummy})
            self._ar_session = sess
            logger.info(
                "HYBRID OCR: AR model loaded %s (%.1f MB) in %.2fs on CUDA (providers=%s)",
                ar_file.name,
                ar_file.stat().st_size / 1e6,
                time.perf_counter() - start,
                actual,
            )
            return True
        except Exception as e:
            logger.error("AR OCR: AR session load failed (%s) — disabling AR paths", e)
            self.hybrid_enabled = False
            self.vertical_ar_default = False
            self._ar_session = None
            return False

    def _run_ar_sync(self, batch: np.ndarray) -> np.ndarray:
        if batch.dtype != self._ar_input_np_dtype:
            batch = batch.astype(self._ar_input_np_dtype, copy=False)
        return self._ar_session.run(None, {self._ar_input_name: batch})[0]

    async def _ar_retry(
        self,
        crops: List[np.ndarray],
        low_idx: List[int],
        results: List[tuple[str, float]],
    ) -> None:
        """Re-OCR ``low_idx`` crops with the AR model; replace in ``results``.

        Runs the AR model as ONE batch over only the low-confidence crops and
        replaces the corresponding non-AR results in place (AR is the
        higher-quality model on hard/stylized crops). Bumps ``ar_retry_count``.
        The garble gate runs downstream on these replaced results, so genuinely
        illegible SFX still drop after the AR pass.
        """
        if not low_idx:
            return
        if not self._ensure_ar_session():
            return
        # The AR model is ~10x heavier than non-AR; on a busy GPU its Softmax
        # over [B,51,4407] can OOM. Run sub-batches and halve on allocation
        # failure (mirrors the non-AR OOM guard); on a bs=1 OOM keep the non-AR
        # result for that crop rather than failing the whole page.
        ar_bs = len(low_idx)
        j = 0
        while j < len(low_idx):
            sub = low_idx[j : j + ar_bs]
            ar_batch = self._preprocess([crops[k] for k in sub])
            try:
                ar_logits = await asyncio.to_thread(self._run_ar_sync, ar_batch)
            except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
                if "Failed to allocate memory" in str(e) and ar_bs > 1:
                    ar_bs = max(1, ar_bs // 2)
                    logger.warning("HYBRID OCR: AR OOM; reducing AR batch to %d", ar_bs)
                    continue
                # bs==1 still OOMs (or other runtime error): skip AR for this
                # crop, keep its non-AR result, and move on.
                logger.warning(
                    "HYBRID OCR: AR retry failed for %d crop(s) (%s); keeping non-AR",
                    len(sub), e,
                )
                j += len(sub)
                continue
            ar_tc = self._decode_with_conf(ar_logits)
            for idx, (text, conf) in zip(sub, ar_tc):
                results[idx] = (text, conf)
            self.ar_retry_count += len(sub)
            j += len(sub)

    async def _ar_decode_indices(
        self,
        crops: List[np.ndarray],
        idxs: List[int],
        results: List[tuple[str, float]],
        batch_size: int,
    ) -> bool:
        """Decode ``idxs`` crops with the AR model, writing into ``results``.

        Used by the vertical-AR-by-default routing: vertical crops are sent to
        the AR model UP FRONT (not as a low-conf retry). Runs as batches over the
        vertical set with the same OOM-halving guard as ``_ar_retry`` (the AR
        model's Softmax over [B,51,4407] is ~10x heavier than NAR). On a bs==1
        OOM (or AR load failure) the caller's NAR result for that crop is kept.
        Returns True if the AR session was available and at least attempted.
        """
        if not idxs:
            return False
        if not self._ensure_ar_session():
            return False
        ar_bs = min(batch_size, len(idxs)) or 1
        j = 0
        while j < len(idxs):
            sub = idxs[j : j + ar_bs]
            ar_batch = self._preprocess([crops[k] for k in sub])
            try:
                ar_logits = await asyncio.to_thread(self._run_ar_sync, ar_batch)
            except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
                if "Failed to allocate memory" in str(e) and ar_bs > 1:
                    ar_bs = max(1, ar_bs // 2)
                    logger.warning(
                        "VERTICAL-AR OCR: AR OOM; reducing AR batch to %d", ar_bs
                    )
                    continue
                logger.warning(
                    "VERTICAL-AR OCR: AR decode failed for %d crop(s) (%s); "
                    "keeping NAR",
                    len(sub), e,
                )
                j += len(sub)
                continue
            ar_tc = self._decode_with_conf(ar_logits)
            for idx, (text, conf) in zip(sub, ar_tc):
                results[idx] = (text, conf)
            self.vertical_ar_count += len(sub)
            j += len(sub)
        return True

    async def _nar_decode_indices(
        self,
        crops: List[np.ndarray],
        idxs: List[int],
        results: List[tuple[str, float]],
        batch_size: int,
    ) -> None:
        """Decode ``idxs`` crops with the fast NAR model, writing into ``results``.

        Same OOM-halving guard as the legacy single-pass loop. ``results`` must be
        pre-sized to the full crop list; this fills only the requested indices.
        """
        if not idxs:
            return
        current_bs = batch_size
        j = 0
        while j < len(idxs):
            sub = idxs[j : j + current_bs]
            batch = self._preprocess([crops[k] for k in sub])
            try:
                logits = await asyncio.to_thread(self._run_sync, batch)
            except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
                if "Failed to allocate memory" in str(e) and current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                    logger.warning("PARSeq OOM; reducing batch to %d", current_bs)
                    continue
                raise
            nar_tc = self._decode_with_conf(logits)
            for idx, (text, conf) in zip(sub, nar_tc):
                results[idx] = (text, conf)
            j += len(sub)

    async def _recognize_batch_with_conf(
        self,
        image_crops: List[np.ndarray],
        batch_size: int = 24,
    ) -> List[tuple[str, float]]:
        """Core batched inference returning (text, ocr_confidence) per crop.

        Routing (when ``vertical_ar_default`` and an AR model are configured):
          * partition crops by geometry — tall/narrow (h/w >= aspect) -> AR,
            the rest -> fast NAR;
          * AR-batch the vertical set UP FRONT (the dup-garble-prone columns),
            NAR-batch the horizontal set;
          * stitch back into original order (results are 1:1 with inputs);
          * the legacy conf-gated AR retry then still runs on any NAR-decoded
            crop below threshold (the vertical crops already went through AR).
        Falls back to the plain NAR-all single pass when routing is disabled.
        """
        if not image_crops:
            return []

        n = len(image_crops)
        total_start = time.perf_counter()
        out: List[tuple[str, float]] = [("", 0.0)] * n

        # --- partition by crop geometry -------------------------------------
        use_vertical_ar = (
            getattr(self, "vertical_ar_default", False)
            and bool(self._ar_model_path)
        )
        if use_vertical_ar:
            vertical_idx = [k for k in range(n) if self._is_vertical_crop(image_crops[k])]
            horizontal_idx = [k for k in range(n) if k not in set(vertical_idx)]
        else:
            vertical_idx = []
            horizontal_idx = list(range(n))

        # AR over the vertical (garble-prone) set, up front, by geometry. If the
        # AR session is unavailable, fall back to NAR for those crops too.
        n_vertical_ar = 0
        vertical_ms = 0.0
        if vertical_idx:
            v_start = time.perf_counter()
            before = self.vertical_ar_count
            ar_ok = await self._ar_decode_indices(
                image_crops, vertical_idx, out, batch_size
            )
            n_vertical_ar = self.vertical_ar_count - before
            vertical_ms = (time.perf_counter() - v_start) * 1000
            if not ar_ok:
                # AR could not load — decode the verticals on NAR instead so we
                # never leave them blank.
                horizontal_idx = horizontal_idx + vertical_idx
                vertical_idx = []

        # NAR over the horizontal (clean) set.
        nar_start = time.perf_counter()
        await self._nar_decode_indices(image_crops, horizontal_idx, out, batch_size)
        nonar_ms = (time.perf_counter() - nar_start) * 1000

        # --- HYBRID: AR-retry on low-confidence NAR crops -------------------
        # Re-OCR ONLY low-conf crops that were decoded by NAR (the verticals
        # already went through AR, so they are excluded). The replaced results
        # flow through the same downstream garble gate.
        ar_ms = 0.0
        n_retry = 0
        if self.hybrid_enabled and self._ar_model_path and horizontal_idx:
            low_idx = [
                k for k in horizontal_idx
                if out[k][1] < self.hybrid_conf_threshold
            ]
            if low_idx:
                ar_start = time.perf_counter()
                before = self.ar_retry_count
                await self._ar_retry(image_crops, low_idx, out)
                n_retry = self.ar_retry_count - before
                ar_ms = (time.perf_counter() - ar_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000
        if n_vertical_ar or n_retry:
            logger.info(
                "PARSeq OCR batch: %d crops in %.1fms (NAR %d/%.1fms + "
                "vertical-AR %d/%.1fms + AR-retry %d/%.1fms; avg %.1fms/crop)",
                n,
                total_ms,
                len(horizontal_idx),
                nonar_ms,
                n_vertical_ar,
                vertical_ms,
                n_retry,
                ar_ms,
                total_ms / n,
            )
        else:
            logger.info(
                "PARSeq OCR batch: %d crops in %.1fms (avg %.1fms/crop)",
                n,
                total_ms,
                total_ms / n,
            )
        return out

    async def recognize_text_batch(
        self,
        image_crops: List[np.ndarray],
        batch_size: int = 24,
    ) -> List[str]:
        return [t for t, _c in await self._recognize_batch_with_conf(image_crops, batch_size)]

    async def recognize_text_batch_with_conf(
        self,
        image_crops: List[np.ndarray],
        batch_size: int = 24,
    ) -> List[tuple[str, float]]:
        """Like ``recognize_text_batch`` but returns (text, ocr_confidence)."""
        return await self._recognize_batch_with_conf(image_crops, batch_size)

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
        return_confidence: bool = False,
    ):
        """OCR per text-line, then concat per block in reading order.

        PARSeq is a single-line STR model. When the detector exposes
        `text_lines` (e.g. the CTD detector), route line-level crops through
        it and stitch results back per block using spatial containment.
        Falls back to block-level OCR when no line overlaps a block.

        When ``return_confidence`` is True, returns ``(texts, confidences)``
        where confidence is the per-block OCR recognition confidence (the MIN
        over the block's lines — a single garbled line should poison the block
        so the gate can drop it). Empty blocks get confidence 0.0.
        """
        if not blocks:
            return ([], []) if return_confidence else []
        if not text_lines:
            crops = []
            h, w = image.shape[:2]
            for b in blocks:
                x0 = max(0, b["minX"] - padding)
                y0 = max(0, b["minY"] - padding)
                x1 = min(w, b["maxX"] + padding)
                y1 = min(h, b["maxY"] + padding)
                crops.append(image[y0:y1, x0:x1])
            tc = await self._recognize_batch_with_conf(crops, batch_size=batch_size)
            texts = [t for t, _c in tc]
            if return_confidence:
                return texts, [c for _t, c in tc]
            return texts

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
            # Use the shared column-grouping ordering (robust to wrapped/
            # overlapping-X columns) rather than a flat (-minX, minY) sort.
            from app.utils.orphan_lines import order_cluster_lines
            lines_sorted = order_cluster_lines(lines)
            for ln in lines_sorted:
                x0 = max(0, ln["minX"] - padding)
                y0 = max(0, ln["minY"] - padding)
                x1 = min(w, ln["maxX"] + padding)
                y1 = min(h, ln["maxY"] + padding)
                flat_crops.append(image[y0:y1, x0:x1])
                flat_owner.append(bi)

        tc = await self._recognize_batch_with_conf(flat_crops, batch_size=batch_size)
        per_block: List[List[str]] = [[] for _ in blocks]
        per_block_conf: List[List[float]] = [[] for _ in blocks]
        for owner, (text, conf) in zip(flat_owner, tc):
            # Always record confidence (even for empty text) so a block whose
            # only line decoded to garbage/empty is flagged low-confidence.
            per_block_conf[owner].append(conf)
            if text:
                per_block[owner].append(text)
        try:
            from app.config import settings
            sep = "\n" if getattr(settings, "ocr_line_join_newline", False) else ""
            gate = float(getattr(settings, "ocr_confidence_gate_threshold", 0.0))
        except Exception:
            sep = ""
            gate = 0.0
        texts = [sep.join(parts) for parts in per_block]
        if return_confidence:
            # Per-line gating instead of a blanket min(): one garbled line must
            # not drop the whole bubble. Block conf = max of lines clearing the
            # gate; if none clear it, keep text but report the (low) max so the
            # downstream garble gate still applies.
            confs: List[float] = []
            for cs in per_block_conf:
                if not cs:
                    confs.append(0.0)
                    continue
                kept = [c for c in cs if c >= gate]
                confs.append(max(kept) if kept else max(cs))
            return texts, confs
        return texts
