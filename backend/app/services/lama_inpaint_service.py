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

# Ring-based fast path: a masked component is flat-filled when ≥ this fraction
# of the pixels in a ~12 px band around the mask sit within ±tolerance of the
# band's median colour. Catches text in white boxes / bubbles that the legacy
# whole-crop std test misses (crop margin always contains borders/art).
_RING_UNIFORM_FRACTION = 0.90
_RING_TOLERANCE = 12.0

# 3-way router tier-2 cutoff: residual regions whose ring std (max channel) is
# below this go to classical cv2.inpaint (smooth/gradient — diffusion handles
# them); at or above, they're textured/screentone and go to the LaMa model.
_CLASSICAL_STD_MAX = 20.0

# bubbleRect interior solid-fill tier (R1 hybrid). For a masked component whose
# center falls inside a matched speech-bubble rect, sample the bubble interior
# pixels that are NOT under the erase mask, drop dark glyph remnants (they are a
# minority of a flat balloon), take the robust median, and fill iff ≥ this
# fraction of the trimmed pixels sit within ±tolerance of the median. Tuned so
# only genuinely flat balloon interiors fire; tinted/screentoned bubbles fail
# the uniformity test and fall through to classical/LaMa.
_BUBBLEFILL_UNIFORM_FRACTION = 0.88
_BUBBLEFILL_TOLERANCE = 22.0
# Pixels darker than this luminance are treated as un-erased glyph remnants and
# dropped from the background estimator — but only while they remain a minority
# (≤ this fraction) of the sample, so a genuinely dark balloon is not mistaken
# for "white-with-text".
_DARK_LUMA_THRESHOLD = 128.0
_DARK_REMNANT_MAX_FRACTION = 0.70

# Flat-fill safety guards (FIX #2 — blob/rectangle artifacts). With neural LaMa
# off in prod, fills come from the solid-color tiers below. Two failure modes:
#   (a) a dark/red background median gets flat-filled over the masked region,
#       painting opaque dark blobs over faces. Reject any flat-fill whose median
#       luminance is below this BT.601 threshold — legit speech bubbles are
#       near-white (luma ≫ 110), so they are unaffected — and route to inpaint.
_MIN_FLATFILL_LUMA = 110.0
#   (b) flat-filling a LARGE connected component produces a big opaque patch.
#       Components whose masked-pixel area exceeds this cap skip the flat-fill
#       tiers entirely and fall through to classical/neural inpaint. ~6000 px²
#       comfortably covers normal dialogue glyphs/lines while excluding the
#       oversized regions that read as patches.
_MAX_FLATFILL_AREA = 6000


def _luma(px: np.ndarray) -> np.ndarray:
    """Rec.601 luminance of an (N,3) RGB float array."""
    return px[:, 0] * 0.299 + px[:, 1] * 0.587 + px[:, 2] * 0.114


def _luma_scalar(rgb: np.ndarray) -> float:
    """Rec.601 (BT.601) luminance of a single RGB triple. Matches the RGB
    channel order used throughout this module (R=ch0, G=ch1, B=ch2)."""
    return float(rgb[0]) * 0.299 + float(rgb[1]) * 0.587 + float(rgb[2]) * 0.114


def _trim_dark_remnants(px: np.ndarray) -> np.ndarray:
    """Drop dark glyph-remnant pixels from a background sample when they are a
    minority. Returns the trimmed sample (or the original if dark pixels are the
    majority, i.e. the background itself is dark)."""
    if px.size == 0:
        return px
    dark = _luma(px) < _DARK_LUMA_THRESHOLD
    frac_dark = float(dark.mean())
    if 0.0 < frac_dark <= _DARK_REMNANT_MAX_FRACTION:
        kept = px[~dark]
        if kept.size:
            return kept
    return px


def _ring_std_max(crop_img: np.ndarray, crop_msk: np.ndarray) -> float | None:
    """Max per-channel std of the ~12px ring around the mask. None if no ring.
    Low = smooth/gradient (classical-inpaint-safe); high = textured."""
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    ring = (cv2.dilate(crop_msk, ring_kernel, iterations=1) > 0) & (crop_msk == 0)
    if not ring.any():
        return None
    px = crop_img[ring].astype(np.float32)
    return float(px.std(axis=0).max())


class LamaInpaintService:
    """LaMa ONNX inpainter with koharu's Crop + balloon-fill strategy."""

    #: Spatial dims expected by the Carve/LaMa-ONNX checkpoint. Baked into
    #: the graph — mismatches raise at `session.run`.
    MODEL_HW: Tuple[int, int] = (512, 512)

    def __init__(
        self,
        model_path: str = "models/lama.onnx",
        crop_margin: int = 128,
        # 2048 keeps typical manga pages (≤2048 on the long side) at native
        # resolution: the global downscale pass added blur on the pasted-back
        # patches while per-component crops already bound model cost (512²).
        default_max_side: int = 2048,
        enable_classical_inpaint: bool = True,
        use_neural: bool | None = None,
    ):
        # 3-way router tier 2: route smooth/gradient residual regions to
        # classical cv2.inpaint instead of the LaMa model. Set False to force
        # the old flat-fill-or-LaMa behavior.
        self.enable_classical_inpaint = enable_classical_inpaint

        # Tier-3 backend. When neural is disabled, the textured/screentone
        # residual is reconstructed with cv2.inpaint (Navier-Stokes) instead of
        # the LaMa ONNX model — and the 208MB model is NOT loaded at all (no GPU
        # working set, no cold-start). See settings.use_neural_inpaint and the
        # 2026-06-13 no-AI inpaint audit for the validation behind this default.
        if use_neural is None:
            try:
                from app.config import settings as _settings
                use_neural = bool(getattr(_settings, "use_neural_inpaint", False))
            except Exception:  # noqa: BLE001 — config import is best-effort
                use_neural = False
        self.use_neural = use_neural

        self.crop_margin = crop_margin
        self.default_max_side = default_max_side
        self.last_stats: dict = {}

        if not self.use_neural:
            # No neural model: tier-3 is classical NS. Nothing to load.
            self.session = None
            self.device = "cpu"
            self._image_input = None
            self._mask_input = None
            logger.info(
                "LaMa inpaint service in NON-NEURAL mode "
                "(tier-3 = cv2.inpaint NS; ONNX model not loaded)"
            )
            return

        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parents[2] / model_file
        if not model_file.exists():
            raise FileNotFoundError(
                f"LaMa ONNX model not found: {model_file}. "
                "Run `uv run python scripts/download_lama_onnx.py`."
            )

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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask_gray: np.ndarray,
        *,
        max_side: int | None = None,
        bubble_rects: List[Tuple[int, int, int, int]] | List[None] | None = None,
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
            bubble_rects: optional list of (minX,minY,maxX,maxY) speech-bubble
                interiors in FULL-image coords (None entries allowed). When a
                masked component's center falls inside one, the interior
                solid-fill tier (R1 hybrid) tries to skip the neural forward.
                None disables the tier.

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

        # Optional full-image downsample: if the page's longer side exceeds
        # `max_side`, shrink image+mask by a uniform factor before the
        # per-component pipeline and upsample the result back. This cuts
        # per-crop resize cost (the 512×512 model still runs the same count
        # of forwards, but crops are smaller and may even hit the fast path
        # more often at low-res). Mask resampled with NEAREST to stay binary.
        h_in, w_in = image_rgb.shape[:2]
        longer = max(h_in, w_in)
        downscale = 1.0
        if max_side and longer > max_side:
            downscale = max_side / float(longer)
            new_w = max(1, int(round(w_in * downscale)))
            new_h = max(1, int(round(h_in * downscale)))
            image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_gray = cv2.resize(mask_gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Binarise the mask once (koharu: `binarize_mask`).
        binary_mask = (mask_gray > 127).astype(np.uint8) * 255
        if not binary_mask.any():
            # Nothing masked — return a copy so callers can freely mutate.
            self.last_stats = {
                "components": 0,
                "fastpath_hits": 0,
                "bubblefill_hits": 0,
                "classical_hits": 0,
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
        bubblefill_hits = 0
        classical_hits = 0
        forward_calls = 0

        # Scale provided bubble rects into the (possibly downscaled) working
        # coordinate space, dropping None entries.
        scaled_rects: List[Tuple[int, int, int, int]] = []
        if bubble_rects:
            for rect in bubble_rects:
                if rect is None:
                    continue
                rx0, ry0, rx1, ry1 = rect
                if downscale != 1.0:
                    rx0 = int(round(rx0 * downscale)); ry0 = int(round(ry0 * downscale))
                    rx1 = int(round(rx1 * downscale)); ry1 = int(round(ry1 * downscale))
                if rx1 > rx0 and ry1 > ry0:
                    scaled_rects.append((rx0, ry0, rx1, ry1))

        def _rect_for_component(cx: float, cy: float) -> Tuple[int, int, int, int] | None:
            """Smallest scaled bubble rect containing (cx,cy), or None."""
            best = None
            best_area = None
            for rx0, ry0, rx1, ry1 in scaled_rects:
                if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
                    area = (rx1 - rx0) * (ry1 - ry0)
                    if best_area is None or area < best_area:
                        best, best_area = (rx0, ry0, rx1, ry1), area
            return best

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

            # FIX #2b — oversized-component guard. Flat-filling a large masked
            # component produces a big opaque patch, so skip both flat-fill tiers
            # (interior fill + fastpath) for components above the area cap and let
            # them go to classical/neural inpaint. Small components are unchanged.
            comp_area = int((crop_msk > 0).sum())
            allow_flatfill = comp_area <= _MAX_FLATFILL_AREA

            # Tier 0 — bubbleRect interior solid-fill (R1 hybrid). If this
            # component's center sits inside a matched bubble, try to fill its
            # flat interior background and skip everything downstream.
            if allow_flatfill and scaled_rects:
                ccx = bx + bw / 2.0
                ccy = by + bh / 2.0
                rect = _rect_for_component(ccx, ccy)
                if rect is not None:
                    # Convert the full-image rect to crop-local coords.
                    local_rect = (rect[0] - l, rect[1] - t, rect[2] - l, rect[3] - t)
                    bf_img, bf_msk, bf_count = _apply_bubble_interior_fill(
                        crop_img, crop_msk, local_rect
                    )
                    if bf_count > 0:
                        _composite_masked(out, bf_img, crop_msk, l, t)
                        bubblefill_hits += 1
                        _clear_region_in_mask(working_mask, binary_mask, l, t, r, b)
                        continue

            # Balloon-fill fast path (skipped for oversized components — FIX #2b).
            filled_count = 0
            if allow_flatfill:
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

            # Tier 2 — classical inpaint for smooth / gradient backgrounds.
            # The flat-fill fast path already handled uniform regions; whatever
            # remains has a non-flat ring. If that ring is merely smooth (low
            # std — gradients, soft shading), cv2.inpaint (Navier-Stokes,
            # radius 3) reconstructs the thin text strokes in ~tens of ms with
            # quality indistinguishable from LaMa, and avoids LaMa entirely.
            # Only genuinely textured/screentone rings (high std) fall through
            # to the neural model, which is the one case classical smears.
            if self.enable_classical_inpaint:
                ring_std = _ring_std_max(crop_img, crop_msk)
                if ring_std is not None and ring_std < _CLASSICAL_STD_MAX:
                    m = (crop_msk > 0).astype(np.uint8)
                    classical = cv2.inpaint(crop_img, m, 3, cv2.INPAINT_NS)
                    _composite_masked(out, classical, crop_msk, l, t)
                    classical_hits += 1
                    _clear_region_in_mask(working_mask, binary_mask, l, t, r, b)
                    continue

            # Tier 3 — textured/screentone residual. Backend depends on
            # `use_neural`: classical cv2.inpaint NS (default) or the LaMa ONNX
            # model. `forward_calls`/`forward_ms` count this tier either way.
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
            "bubblefill_hits": bubblefill_hits,
            "classical_hits": classical_hits,
            "forward_calls": forward_calls,
            "forward_ms": t_total_forward * 1000.0,
            "downscale": downscale,
        }

        # Upsample back to original size if we downscaled at entry.
        if downscale < 1.0:
            out = cv2.resize(out, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _forward_one(self, crop_img: np.ndarray, crop_msk: np.ndarray) -> np.ndarray:
        """Reconstruct the textured/screentone residual of a single crop.

        Tier-3 backend. When `use_neural` is False (default), this is a purely
        classical cv2.inpaint (Navier-Stokes, r=3) — no model, no GPU — which the
        no-AI audit found visually indistinguishable from LaMa once the
        translation is rendered on top (85% of inpainted px are hidden; the
        residual is imperceptible on dialogue). When `use_neural` is True it runs
        the LaMa ONNX model: resize crop+mask to the fixed `MODEL_HW`, forward,
        resize back (mask nearest, image bilinear).
        """
        if not self.use_neural:
            m = (crop_msk > 0).astype(np.uint8)
            return cv2.inpaint(crop_img, m, 3, cv2.INPAINT_NS)

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


def _apply_bubble_interior_fill(
    crop_img: np.ndarray,
    crop_msk: np.ndarray,
    interior_rect: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """bubbleRect-gated interior solid-fill (R1 hybrid tier 0).

    `interior_rect` is the matched bubble rect in CROP-LOCAL coords (l,t,r,b).
    Estimate the background from the bubble-interior pixels that are NOT under
    the erase mask, drop dark glyph remnants (minority), take the robust median,
    and if ≥ _BUBBLEFILL_UNIFORM_FRACTION of the trimmed interior sits within
    ±_BUBBLEFILL_TOLERANCE of it, fill the masked pixels with that median.

    Returns (filled_image, remaining_mask, filled_pixel_count). filled_count==0
    means the gate rejected this component (it stays on the downstream tiers).
    """
    l, t, r, b = interior_rect
    h, w = crop_msk.shape
    l = max(0, min(l, w)); r = max(0, min(r, w))
    t = max(0, min(t, h)); b = max(0, min(b, h))
    if r <= l or b <= t:
        return crop_img, crop_msk, 0

    interior_msk = crop_msk[t:b, l:r]
    interior_img = crop_img[t:b, l:r]
    unmasked = interior_msk == 0
    if not unmasked.any():
        return crop_img, crop_msk, 0

    sample = interior_img[unmasked].astype(np.float32)  # (N,3)
    trimmed = _trim_dark_remnants(sample)
    if trimmed.size == 0:
        return crop_img, crop_msk, 0

    med = np.median(trimmed, axis=0)
    within = (np.abs(trimmed - med) <= _BUBBLEFILL_TOLERANCE).all(axis=1)
    if float(within.mean()) < _BUBBLEFILL_UNIFORM_FRACTION:
        return crop_img, crop_msk, 0

    # Reject dark medians (FIX #2a): never flat-fill a dark/red background over
    # the mask — that paints opaque blobs over faces. Fall through to inpaint.
    if _luma_scalar(med) < _MIN_FLATFILL_LUMA:
        return crop_img, crop_msk, 0

    fill = np.clip(np.round(med), 0, 255).astype(np.uint8)
    filled = crop_img.copy()
    masked_pixels = crop_msk > 0
    filled[masked_pixels] = fill
    return filled, np.zeros_like(crop_msk), int(masked_pixels.sum())


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

    # --- Primary estimator: ring of pixels immediately AROUND the mask. ---
    # The crop bbox carries `crop_margin` (≈128 px) of context, which almost
    # always contains box borders / panel frames / art — so a whole-crop
    # std-dev test never passes (observed fastpath_hits: 0 across full
    # galleries). The pixels that actually matter are the ones the masked
    # strokes sit ON: a thin band adjacent to the mask. For text inside a
    # white box or speech bubble that band is flat white even when the box
    # outline is 30 px away.
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    ring = (cv2.dilate(crop_msk, ring_kernel, iterations=1) > 0) & (crop_msk == 0)
    if ring.any():
        ring_px = crop_img[ring].astype(np.float32)  # (N, 3)
        # Drop residual dark stroke remnants (anti-aliased glyph edges the tight
        # erase mask did not fully cover) before estimating the background, but
        # only while they are a minority — otherwise the ring is genuinely dark.
        # This widens the path to catch white-box / bubble text it previously
        # missed (measured: fired on only 6/75 components without the trim).
        ring_px = _trim_dark_remnants(ring_px)
        med = np.median(ring_px, axis=0)
        # Trimmed uniformity: tolerate a small fraction of outliers (stray
        # screentone dots, neighbouring stroke tips) instead of a global std.
        within = (np.abs(ring_px - med) <= _RING_TOLERANCE).all(axis=1)
        # Reject dark medians (FIX #2a): a dark/red ring median would paint an
        # opaque blob over the mask (e.g. over a face) — route to inpaint.
        if (
            float(within.mean()) >= _RING_UNIFORM_FRACTION
            and _luma_scalar(med) >= _MIN_FLATFILL_LUMA
        ):
            fill = np.clip(np.round(med), 0, 255).astype(np.uint8)
            filled = crop_img.copy()
            masked_pixels = crop_msk > 0
            filled[masked_pixels] = fill
            return filled, np.zeros_like(crop_msk), int(masked_pixels.sum())

    # --- Legacy whole-crop estimator (koharu balloon.rs) as fallback. ---
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

    # Reject dark medians (FIX #2a): never flat-fill a dark background over the
    # mask — route to inpaint so we don't paint opaque blobs over art.
    if _luma_scalar(median_rgb) < _MIN_FLATFILL_LUMA:
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
