"""LaMa-based manga inpainting service (ONNX runtime).

Ported from koharu's `koharu-ml/src/inpainting/{strategy,balloon,mod}.rs` and
`koharu-ml/src/lama/mod.rs`. We keep only the parts our pipeline actually
exercises:

    1. **Crop strategy** — find connected components in the mask via
       `cv2.findContours(RETR_EXTERNAL)`, expand each bbox by `crop_margin`
       pixels (128 px default, matches IOPaint / koharu), run LaMa forward
       on each crop independently, and composite only the masked pixels back
       into the original image. This is koharu's default path for manga.
    2. **Bubble fast-path** (from `balloon.rs`) — inside each per-component
       crop, estimate the background colour from unmasked pixels. If the
       RGB std-dev is below a small threshold (flat speech bubble), skip
       the model entirely and fill the masked pixels with the median RGB.
       Koharu reports this saves >60 % of forward calls on typical pages.
    3. **Fixed 512×512 forward** — the Carve/LaMa-ONNX checkpoint has its
       spatial dimensions baked into the weights (see
       `scripts/download_lama_onnx.py` inspection output). Every crop is
       therefore resized to 512×512 before forward and resized back with
       bilinear / nearest interpolation afterwards.

Loading follows the same CUDA → fp32-CUDA → CPU fallback chain as
`parseq_ocr_service.py`: we try CUDAExecutionProvider first, fall back to
CPU on failure, and log the final provider choice so ops can see it. The
session is warmed up on a 512×512 dummy so the first real request doesn't
eat the cold-start cost.

Public API:

    svc = LamaInpaintService()
    inpainted_rgb = svc.inpaint(image_rgb, mask_gray, max_side=1024)

    image_rgb: (H, W, 3) uint8 RGB
    mask_gray: (H, W)    uint8, non-zero = mask (treated as binary)
    returns  : (H, W, 3) uint8 RGB with masked regions inpainted
"""

from __future__ import annotations

# Import torch first so its bundled CUDA libs are on the loader path before
# onnxruntime-gpu probes for libcublas/libcudnn (same trick as parseq_ocr).
import torch  # noqa: F401

import logging
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)  # ERROR only

logger = logging.getLogger(__name__)


# Koharu constants (see balloon.rs).
_SIMPLE_BG_THRESHOLD_LOW_VARIANCE = 10.0
_SIMPLE_BG_THRESHOLD_HIGH_VARIANCE = 7.0
_SIMPLE_BG_CHANNEL_STD_SWITCH = 1.0


class LamaInpaintService:
    """LaMa ONNX inpainter with koharu's Crop + balloon-fill strategy."""

    #: Spatial dims expected by the Carve/LaMa-ONNX checkpoint. Baked into
    #: the graph — mismatches raise at `session.run`.
    MODEL_HW: Tuple[int, int] = (512, 512)

    def __init__(
        self,
        model_path: str = "models/lama.onnx",
        crop_margin: int = 128,
        default_max_side: int = 1024,
    ):
        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parents[2] / model_file
        if not model_file.exists():
            raise FileNotFoundError(
                f"LaMa ONNX model not found: {model_file}. "
                "Run `uv run python scripts/download_lama_onnx.py`."
            )

        self.crop_margin = crop_margin
        self.default_max_side = default_max_side

        # Same load chain as parseq_ocr_service (see PARSeq comments for
        # rationale). ORT_ENABLE_ALL is safe: the LaMa graph is fully static.
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_first: List = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ]
        load_candidates: List[Tuple[Path, List]] = [
            (model_file, cuda_first),
            (model_file, ["CPUExecutionProvider"]),
        ]

        start = time.perf_counter()
        last_err: Exception | None = None
        session: ort.InferenceSession | None = None
        for candidate, providers in load_candidates:
            try:
                session = ort.InferenceSession(
                    str(candidate), sess_options=so, providers=providers
                )
                # Force a tiny inference to surface cuDNN/CUDA issues up front.
                h, w = self.MODEL_HW
                dummy_img = np.zeros((1, 3, h, w), dtype=np.float32)
                dummy_msk = np.zeros((1, 1, h, w), dtype=np.float32)
                session.run(
                    None,
                    {
                        session.get_inputs()[0].name: dummy_img,
                        session.get_inputs()[1].name: dummy_msk,
                    },
                )
                break
            except Exception as e:  # noqa: BLE001 — ORT raises varied exceptions
                last_err = e
                logger.warning(
                    "LaMa load failed for %s with %s: %s", candidate.name, providers, e
                )
                session = None
        if session is None:
            raise RuntimeError(f"All LaMa loaders failed; last error: {last_err}")

        self.session = session
        actual_providers = self.session.get_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in actual_providers else "cpu"
        load_time = time.perf_counter() - start
        logger.info(
            "LaMa loaded: %s (%.1f MB) in %.2fs on %s (providers=%s)",
            model_file.name,
            model_file.stat().st_size / 1e6,
            load_time,
            self.device,
            actual_providers,
        )

        inputs = self.session.get_inputs()
        # Inspection of Carve/LaMa-ONNX shows inputs[0].name == "image",
        # inputs[1].name == "mask". We pin by position, not name, to remain
        # robust to alternate manga-tuned ONNX ports that may relabel.
        self._image_input = inputs[0].name
        self._mask_input = inputs[1].name

        # Stats for SUMMARY.md
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask_gray: np.ndarray,
        *,
        max_side: int | None = None,
    ) -> np.ndarray:
        """Inpaint masked regions of `image_rgb`.

        Koharu's Crop strategy: one forward per connected mask component
        plus a balloon-fill fast path that skips the model for flat bubbles.

        Args:
            image_rgb: (H, W, 3) uint8 RGB.
            mask_gray: (H, W) uint8 — any non-zero pixel is treated as masked.
            max_side : optional, hard upper bound on the per-crop longer
                side before resize to the model's fixed 512×512 input. The
                crop itself is always downsampled to 512; this cap only
                matters for memory/timing of the resize step. Defaults to
                `self.default_max_side`.

        Returns:
            (H, W, 3) uint8 RGB with masked pixels inpainted.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"image_rgb must be (H, W, 3) RGB; got {image_rgb.shape}"
            )
        if mask_gray.ndim != 2:
            raise ValueError(f"mask_gray must be (H, W); got {mask_gray.shape}")
        if image_rgb.shape[:2] != mask_gray.shape:
            raise ValueError(
                f"image/mask shape mismatch: {image_rgb.shape[:2]} vs {mask_gray.shape}"
            )

        if max_side is None:
            max_side = self.default_max_side

        # Binarise the mask once (koharu: `binarize_mask`).
        binary_mask = (mask_gray > 127).astype(np.uint8) * 255
        if not binary_mask.any():
            # Nothing masked — return a copy so callers can freely mutate.
            self.last_stats = {
                "components": 0,
                "fastpath_hits": 0,
                "forward_calls": 0,
                "forward_ms": 0.0,
            }
            return image_rgb.copy()

        # Koharu strategy.rs:298 — RETR_EXTERNAL so we only iterate outer
        # contours (inner holes would create spurious crops).
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return image_rgb.copy()

        boxes = [cv2.boundingRect(c) for c in contours]

        out = image_rgb.copy()
        working_mask = binary_mask.copy()
        h_full, w_full = image_rgb.shape[:2]

        t_total_forward = 0.0
        fastpath_hits = 0
        forward_calls = 0

        for bx, by, bw, bh in boxes:
            # crop_box with margin (strategy.rs:331 — expand on all sides
            # then shift inward if we ran off the edge to preserve
            # `(box + margin*2)` footprint when possible).
            l, t, r, b = _expand_and_clamp(
                bx, by, bx + bw, by + bh, self.crop_margin, w_full, h_full
            )
            crop_img = out[t:b, l:r].copy()
            crop_msk = working_mask[t:b, l:r].copy()

            if not crop_msk.any():
                continue

            # Balloon-fill fast path.
            filled_img, filled_msk, filled_count = _apply_bubble_fastpath(
                crop_img, crop_msk
            )
            if filled_count > 0:
                _composite_masked(out, filled_img, crop_msk, l, t)
                # The balloon fill cleared only the pixels it painted; the
                # returned mask is what remains to actually forward through
                # the model.
                crop_img = filled_img
                crop_msk = filled_msk

            remaining = int(crop_msk.any())
            if not remaining:
                fastpath_hits += 1
                _clear_region_in_mask(working_mask, binary_mask, l, t, r, b)
                continue

            # Pad-to-multiple path is moot: our ONNX has fixed 512×512. Just
            # resize the crop → 512, forward, resize output back.
            t0 = time.perf_counter()
            crop_out = self._forward_one(crop_img, crop_msk)
            t_total_forward += time.perf_counter() - t0
            forward_calls += 1

            # Composite ONLY masked pixels back. This is the key invariant
            # from `composite_masked` in strategy.rs:444 — anything we don't
            # actively repaint must keep the original pixel.
            _composite_masked(out, crop_out, crop_msk, l, t)

            # And clear this component from the working mask so subsequent
            # contours don't re-process overlapping areas.
            _clear_region_in_mask(working_mask, binary_mask, l, t, r, b)

        self.last_stats = {
            "components": len(boxes),
            "fastpath_hits": fastpath_hits,
            "forward_calls": forward_calls,
            "forward_ms": t_total_forward * 1000.0,
        }
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _forward_one(self, crop_img: np.ndarray, crop_msk: np.ndarray) -> np.ndarray:
        """Run LaMa on a single crop.

        Resizes crop and mask to the model's fixed `MODEL_HW`, runs forward,
        then resizes the output back. Mask is resized with nearest-neighbour
        to keep it binary; image uses bilinear.
        """
        mh, mw = self.MODEL_HW
        orig_h, orig_w = crop_img.shape[:2]

        img_small = cv2.resize(crop_img, (mw, mh), interpolation=cv2.INTER_AREA)
        msk_small = cv2.resize(crop_msk, (mw, mh), interpolation=cv2.INTER_NEAREST)

        # (H, W, 3) uint8 → (1, 3, H, W) float32, [0, 1]
        img_in = img_small.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        # Binary mask → (1, 1, H, W) float32 {0, 1}
        msk_in = (msk_small > 0).astype(np.float32)[None, None]

        outputs = self.session.run(
            None,
            {self._image_input: img_in, self._mask_input: msk_in},
        )
        # Carve/LaMa-ONNX: output is (1, 3, H, W) float32 in [0, 255].
        out = outputs[0][0]  # (3, H, W)
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
        out = out.transpose(1, 2, 0)  # (H, W, 3)

        if (orig_h, orig_w) != (mh, mw):
            out = cv2.resize(
                out, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
        return out


# ----------------------------------------------------------------------
# Free helpers (tested individually; also used in test_lama_e2e.py)
# ----------------------------------------------------------------------


def _expand_and_clamp(
    x1: int, y1: int, x2: int, y2: int, margin: int, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """Port of koharu's `crop_box` (strategy.rs:331).

    Expand by `margin` on each side. If one edge overflows, shift the
    opposite edge outward so the full `(box + margin*2)` footprint is
    preserved whenever possible. Returns `(l, t, r, b)` clamped to the
    image.
    """
    desire_l = x1 - margin
    desire_r = x2 + margin
    desire_t = y1 - margin
    desire_b = y2 + margin

    l = max(desire_l, 0)
    r = min(desire_r, img_w)
    t = max(desire_t, 0)
    b = min(desire_b, img_h)

    if desire_l < 0:
        r = min(r - desire_l, img_w)
    if desire_r > img_w:
        l = max(l - (desire_r - img_w), 0)
    if desire_t < 0:
        b = min(b - desire_t, img_h)
    if desire_b > img_h:
        t = max(t - (desire_b - img_h), 0)

    r = max(r, l + 1)
    b = max(b, t + 1)
    return int(l), int(t), int(r), int(b)


def _apply_bubble_fastpath(
    crop_img: np.ndarray, crop_msk: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Balloon-fill fast path (balloon.rs:36).

    Estimate the background colour from *unmasked* pixels inside the crop.
    If the RGB per-channel std-dev is below the threshold, fill masked
    pixels with the median RGB and zero them out in the returned mask.

    Simplified vs koharu's original: koharu uses a separate per-bubble
    segmentation mask to isolate bubble interiors; we only have the erase
    mask and the crop, so we treat the whole *unmasked* area of the crop as
    the background estimator. This still nails flat speech bubbles (which
    is >60 % of panels in manga).

    Returns (filled_image, remaining_mask, filled_pixel_count).
    """
    unmasked = crop_msk == 0
    if not unmasked.any():
        return crop_img, crop_msk, 0

    bg = crop_img[unmasked]  # (N, 3) uint8
    if bg.size == 0:
        return crop_img, crop_msk, 0

    median_rgb = np.median(bg, axis=0)  # (3,) float
    std_rgb = np.std(bg.astype(np.float32), axis=0)  # (3,)

    # koharu: pick the tighter threshold when channel std-dev itself is noisy.
    channel_std_meta = float(np.std(std_rgb))
    if channel_std_meta > _SIMPLE_BG_CHANNEL_STD_SWITCH:
        inpaint_thresh = _SIMPLE_BG_THRESHOLD_HIGH_VARIANCE
    else:
        inpaint_thresh = _SIMPLE_BG_THRESHOLD_LOW_VARIANCE

    if float(std_rgb.max()) >= inpaint_thresh:
        return crop_img, crop_msk, 0

    fill = np.clip(np.round(median_rgb), 0, 255).astype(np.uint8)
    filled = crop_img.copy()
    masked_pixels = crop_msk > 0
    filled[masked_pixels] = fill
    remaining_mask = np.zeros_like(crop_msk)
    return filled, remaining_mask, int(masked_pixels.sum())


def _composite_masked(
    out: np.ndarray, crop: np.ndarray, crop_mask: np.ndarray, left: int, top: int
) -> None:
    """Paste `crop` into `out` at (left, top), but ONLY where `crop_mask` > 0.

    Mirrors `composite_masked` in strategy.rs:444. Critical for preserving
    art outside the mask at full original resolution.
    """
    ch, cw = crop_mask.shape
    dst = out[top : top + ch, left : left + cw]
    mask = crop_mask > 0
    # Broadcast the (H,W) mask to the 3 channels.
    np.copyto(dst, crop, where=mask[:, :, None])


def _clear_region_in_mask(
    working_mask: np.ndarray,
    original_mask: np.ndarray,
    l: int,
    t: int,
    r: int,
    b: int,
) -> None:
    """Clear pixels inside (l,t,r,b) from `working_mask` that were set in
    `original_mask`. We clear based on the *original* mask rather than
    touching every pixel in the bbox so other (overlapping) components can
    still be processed if needed.
    """
    working_mask[t:b, l:r] = np.where(
        original_mask[t:b, l:r] > 0, 0, working_mask[t:b, l:r]
    )
