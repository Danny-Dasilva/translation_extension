"""Comic Text Detector (CTD) service for manga text detection."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.config import settings
from app.utils.ctd_utils import ERASE_SEG_THRESHOLD
# Pure geometry helpers reused for DETECTION-TIME balloon-column fusion. These are
# the SAME guarded predicates the pre-translation `bubble_grouping` grouper uses
# (column adjacency, glyph-width similarity, Y-overlap, RTL direction, panel-area
# guard, YOLO-bubble membership) — reused here so fusion happens before OCR.
from app.utils.bubble_grouping import (
    _adjacent_columns,
    _bubble_implausibly_large,
    bubble_id_of,
)
from app.utils.orphan_lines import reading_order_sort

logger = logging.getLogger(__name__)

# Reduce ONNX runtime verbosity
ort.set_default_logger_severity(3)  # 3 = ERROR only


class ComicTextDetectorService:
    """
    Comic Text Detector - detects text blocks and tight text line regions.

    Replaces:
    - detector_service.py (YOLOv10n bubble detection)
    - text_region_extractor.py (CV-based text bounds)
    """

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = settings.ctd_model_path

        # Prefer FP16 variant when available (koharu parity - faster inference)
        model_file = Path(model_path)
        fp16_candidate = model_file.with_name(model_file.stem + ".fp16" + model_file.suffix)
        if fp16_candidate.exists():
            logger.info(f"CTD: preferring FP16 model at {fp16_candidate}")
            model_file = fp16_candidate
            model_path = str(fp16_candidate)

        if not model_file.exists():
            raise FileNotFoundError(
                f"Comic Text Detector model not found at {model_path}. "
                "Run scripts/download_models.py --ctd to download it."
            )

        self.input_size = settings.ctd_input_size
        self.text_threshold = settings.ctd_text_threshold
        self.block_confidence = settings.ctd_block_confidence
        self.min_area = settings.ctd_min_text_area
        self.nms_free = settings.ctd_nms_free  # YOLOv10 uses one-to-one assignment

        providers = self._select_providers()
        logger.info(f"Loading Comic Text Detector from {model_path}")
        self.session = self._create_session(model_path, providers)

        provider = self.session.get_providers()[0] if self.session.get_providers() else "unknown"
        logger.info(f"CTD using execution provider: {provider}")

    def _select_providers(self) -> List[str]:
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available]
        return providers or available

    def _create_session(self, model_path: str, providers: List[str]) -> ort.InferenceSession:
        from app.services._ort_init import cuda_provider_options

        # Inject env-gated CUDA arena options (avoid greedy VRAM fill beside vLLM).
        cuda_opts = cuda_provider_options()
        if cuda_opts:
            providers = [
                ("CUDAExecutionProvider", cuda_opts) if p == "CUDAExecutionProvider" else p
                for p in providers
            ]
        try:
            return ort.InferenceSession(model_path, providers=providers)
        except Exception as exc:
            names = [p[0] if isinstance(p, tuple) else p for p in providers]
            if "CUDAExecutionProvider" in names:
                logger.warning("CTD CUDA init failed (%s). Falling back to CPU.", exc)
                return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            raise

    def _letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize with padding while maintaining aspect ratio."""
        h, w = img.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        return padded, scale, (new_w, new_h)

    def _preprocess(
        self,
        img: np.ndarray,
        input_is_bgr: bool
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for CTD model."""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            if input_is_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.shape[2] == 3:
            if input_is_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img.copy()
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        padded, scale, (pw, ph) = self._letterbox(img)

        img_in = padded.astype(np.float32) / 255.0
        img_in = img_in.transpose(2, 0, 1)[None]  # [1, 3, H, W]

        return img_in, scale, (pw, ph)

    async def detect(self, img: np.ndarray, input_is_bgr: bool = False) -> Dict:
        """
        Detect text blocks and tight text regions.

        Args:
            img: Input image (RGB or BGR numpy array)
            input_is_bgr: Set True if the input is BGR (e.g., cv2.imdecode)

        Returns:
            {
                "blocks": List of text block bboxes,
                "text_lines": List of tight text line bboxes,
                "mask": Binary text mask (H, W) or None
            }
        """
        h, w = img.shape[:2]
        img_in, scale, padded_size = self._preprocess(img, input_is_bgr)

        input_name = self.session.get_inputs()[0].name
        outputs = await asyncio.to_thread(self.session.run, None, {input_name: img_in})

        blks, mask, lines_map = self._assign_outputs(outputs)

        blocks = self._parse_blocks(blks, scale, (w, h))
        text_lines = self._extract_text_lines(lines_map, scale, padded_size, (w, h))

        # When block detection is unavailable, derive blocks from text_lines
        # This supports models that only output segmentation + text line maps
        if not blocks and text_lines:
            logger.info("No block detections; deriving blocks from text lines")
            blocks = self._derive_blocks_from_text_lines(text_lines)

        # Expand line bboxes using koharu's font-aware padding (item #6)
        text_lines = self._expand_text_lines(text_lines, (w, h))

        # Erase mask uses a LOWER seg threshold than detection (ERASE_SEG_THRESHOLD
        # ~0.45 vs ctd_text_threshold ~0.8) so faint glyph tails are erased too,
        # otherwise LaMa reseeds text-shaped ghosts. Detection geometry is
        # unaffected (blocks/lines were parsed at the detection threshold above).
        text_mask = (
            self._process_mask(
                mask, padded_size, (w, h), blocks,
                erase_threshold=ERASE_SEG_THRESHOLD,
            )
            if mask is not None
            else None
        )

        logger.debug(f"CTD detected {len(blocks)} blocks, {len(text_lines)} text lines")

        return {
            "blocks": blocks,
            "text_lines": text_lines,
            "mask": text_mask,
        }

    def _assign_outputs(self, outputs: List[np.ndarray]) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Assign CTD outputs using names and shape heuristics."""
        output_names = [out.name.lower() for out in self.session.get_outputs()]
        name_map = {name: output for name, output in zip(output_names, outputs)}

        # --- v26 contract: det (axis-aligned blocks) + mask (2ch seg) + obb
        # (oriented text lines). The legacy name heuristic below misroutes the
        # v26 "det" output (which is BLOCKS) into lines_map and ignores "obb",
        # so detect it explicitly. obb is a [1,N,>=6] box tensor, NOT a DBNet
        # probability map, so it is parsed by _parse_obb_lines (not the prob-map
        # path); we return it as lines_map and branch on ndim downstream.
        if "det" in name_map and "mask" in name_map and "obb" in name_map:
            # v26's axis-aligned "det" block head is unreliable in this export
            # (blk_det is an external path in the .pt, not fused into the ONNX;
            # confidences peak ~0.1, below block_confidence). The "obb" oriented
            # line head IS trained, so we drive text-lines from obb and let
            # detect() derive blocks from those lines. blks=None forces that path.
            return None, name_map["mask"], name_map["obb"]

        blks = None
        lines_map = None
        mask = None

        for name, output in name_map.items():
            if "blk" in name or "box" in name:
                blks = output
                continue
            if "mask" in name or "seg" in name:
                mask = output
                continue
            if "det" in name or "line" in name or "db" in name or "prob" in name:
                lines_map = output

        if blks is None:
            for output in outputs:
                if output.ndim == 3:
                    blks = output
                    break

        candidates = [output for output in outputs if output is not blks and output.ndim in (3, 4)]
        if len(candidates) >= 2:
            if lines_map is None and mask is None:
                mask, lines_map = candidates[0], candidates[1]
            else:
                if lines_map is None:
                    lines_map = candidates[0] if candidates[0] is not mask else candidates[1]
                if mask is None:
                    mask = candidates[0] if candidates[0] is not lines_map else candidates[1]
        elif len(candidates) == 1:
            if lines_map is None:
                lines_map = candidates[0]
            elif mask is None:
                mask = candidates[0]

        if blks is None:
            logger.warning("CTD outputs missing blocks; check model output order.")
        if lines_map is None:
            logger.warning("CTD outputs missing text lines map; tight text regions unavailable.")
        if mask is None:
            logger.debug("CTD outputs missing mask; text segmentation disabled.")

        return blks, mask, lines_map

    @staticmethod
    def _calculate_iou(box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1["minX"], box2["minX"])
        y1 = max(box1["minY"], box2["minY"])
        x2 = min(box1["maxX"], box2["maxX"])
        y2 = min(box1["maxY"], box2["maxY"])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1["maxX"] - box1["minX"]) * (box1["maxY"] - box1["minY"])
        area2 = (box2["maxX"] - box2["minX"]) * (box2["maxY"] - box2["minY"])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _apply_nms(self, blocks: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping boxes."""
        if not blocks:
            return blocks

        # Sort by confidence (highest first)
        sorted_blocks = sorted(blocks, key=lambda b: b["confidence"], reverse=True)
        keep: List[Dict] = []

        while sorted_blocks:
            best = sorted_blocks.pop(0)
            keep.append(best)

            # Remove boxes with high IoU overlap with the best box
            sorted_blocks = [
                box for box in sorted_blocks
                if self._calculate_iou(best, box) < iou_threshold
            ]

        return keep

    @staticmethod
    def _filter_contained_boxes(blocks: List[Dict]) -> List[Dict]:
        """Remove boxes that fully contain other boxes (keep innermost only)."""
        if len(blocks) <= 1:
            return blocks

        # Find boxes that contain other boxes
        to_remove = set()
        for i, box_a in enumerate(blocks):
            for j, box_b in enumerate(blocks):
                if i == j:
                    continue
                # Check if box_a fully contains box_b
                if (box_a["minX"] <= box_b["minX"] and
                    box_a["minY"] <= box_b["minY"] and
                    box_a["maxX"] >= box_b["maxX"] and
                    box_a["maxY"] >= box_b["maxY"]):
                    to_remove.add(i)  # Remove the container (outer box)
                    break  # A contains at least one box, mark for removal

        return [b for i, b in enumerate(blocks) if i not in to_remove]

    def _parse_blocks(
        self,
        blks: np.ndarray | None,
        scale: float,
        orig_size: Tuple[int, int]
    ) -> List[Dict]:
        """Parse YOLO-style block detections."""
        if blks is None:
            return []

        blocks: List[Dict] = []
        w, h = orig_size

        if blks.ndim == 3:
            blks = blks[0]

        # Vectorized confidence prefilter: the per-row loop below recomputes the
        # identical conf (det[4] * det[5:].max()) and `continue`s when it falls
        # below block_confidence. Dropping those rows up-front with the same
        # formula is provably output-identical, but avoids ~64k Python-level
        # iterations to keep a handful of survivors (~177ms -> ~1.6ms).
        if blks.shape[0] and blks.shape[1] >= 5:
            obj = blks[:, 4]
            cls = blks[:, 5:].max(axis=1) if blks.shape[1] > 5 else np.ones_like(obj)
            conf_all = obj * cls
            blks = blks[conf_all >= self.block_confidence]

        for det in blks:
            if len(det) < 5:
                continue

            coords = np.asarray(det[:4], dtype=np.float32)
            if coords.max() <= 1.5:
                coords = coords * self.input_size

            x1, y1, x2, y2 = coords
            if x2 < x1 or y2 < y1:
                cx, cy, bw, bh = coords
                x1 = cx - (bw / 2)
                y1 = cy - (bh / 2)
                x2 = cx + (bw / 2)
                y2 = cy + (bh / 2)

            if len(det) > 5:
                class_scores = np.asarray(det[5:], dtype=np.float32)
                if class_scores.size > 0:
                    conf = float(det[4]) * float(class_scores.max())
                else:
                    conf = float(det[4])
            else:
                conf = float(det[4])
            if conf < self.block_confidence:
                continue

            min_x = int(max(0, min(x1 / scale, w)))
            min_y = int(max(0, min(y1 / scale, h)))
            max_x = int(max(0, min(x2 / scale, w)))
            max_y = int(max(0, min(y2 / scale, h)))

            if max_x <= min_x or max_y <= min_y:
                continue

            blocks.append({
                "minX": min_x,
                "minY": min_y,
                "maxX": max_x,
                "maxY": max_y,
                "confidence": float(conf),
            })

        # Apply NMS to remove duplicate/overlapping boxes
        # Skip NMS for YOLOv10 models that use one-to-one assignment (NMS-free architecture)
        if not self.nms_free:
            blocks = self._apply_nms(blocks, iou_threshold=0.5)
        else:
            logger.debug("Skipping NMS (model uses one-to-one assignment)")

        # Filter out boxes that fully contain other boxes (keep innermost)
        blocks = self._filter_contained_boxes(blocks)

        blocks.sort(key=lambda b: (-b["minX"], b["minY"]))
        return blocks

    def _extract_text_lines(
        self,
        lines_map: np.ndarray | None,
        scale: float,
        padded_size: Tuple[int, int],
        orig_size: Tuple[int, int]
    ) -> List[Dict]:
        """Extract tight text boxes from the DBNet probability map (V5) or from
        the v26 OBB oriented-line box tensor."""
        if lines_map is None:
            return []

        # v26 OBB path: a [1, N, C] box tensor (C ~ 7), NOT a spatial prob map.
        # Columns: [cx, cy, w, h, cls0_conf, cls1_conf, angle] in input-letterbox
        # pixel space. We emit axis-aligned bboxes (the small angle is folded into
        # the AABB) so downstream gating/crops work; blocks are then derived from
        # these lines by detect().
        arr = lines_map[0] if lines_map.ndim == 3 else lines_map
        if arr.ndim == 2 and arr.shape[1] <= 16:
            w, h = orig_size
            line_conf = min(self.block_confidence, 0.3)  # OBB op-point (~0.25-0.4)
            out: List[Dict] = []
            for row in arr:
                cx, cy, bw, bh = (float(v) for v in row[:4])
                conf = float(np.max(row[4:6])) if arr.shape[0] and row.shape[0] >= 6 else float(row[4])
                if conf < line_conf or bw <= 1 or bh <= 1:
                    continue
                x1, y1 = cx - bw / 2.0, cy - bh / 2.0
                x2, y2 = cx + bw / 2.0, cy + bh / 2.0
                min_x = int(max(0, min(x1 / scale, w)))
                min_y = int(max(0, min(y1 / scale, h)))
                max_x = int(max(0, min(x2 / scale, w)))
                max_y = int(max(0, min(y2 / scale, h)))
                if max_x <= min_x or max_y <= min_y:
                    continue
                area_orig = (max_x - min_x) * (max_y - min_y)
                if area_orig < self.min_area:
                    continue
                out.append({
                    "minX": min_x, "minY": min_y, "maxX": max_x, "maxY": max_y,
                    "area": int(area_orig), "confidence": conf,
                })
            return out

        if lines_map.ndim == 4:
            prob_map = lines_map[0, 0]
        elif lines_map.ndim == 3:
            prob_map = lines_map[0]
        else:
            prob_map = lines_map

        binary = (prob_map > self.text_threshold).astype(np.uint8) * 255

        pw, ph = padded_size
        binary = binary[:ph, :pw]

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_lines: List[Dict] = []
        w, h = orig_size

        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_orig = area / (scale * scale)
            if area_orig < self.min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            min_x = int(max(0, min(x / scale, w)))
            min_y = int(max(0, min(y / scale, h)))
            max_x = int(max(0, min((x + bw) / scale, w)))
            max_y = int(max(0, min((y + bh) / scale, h)))

            if max_x <= min_x or max_y <= min_y:
                continue

            text_lines.append({
                "minX": min_x,
                "minY": min_y,
                "maxX": max_x,
                "maxY": max_y,
                "area": int(area_orig),
                "polygon": (cnt / scale).astype(np.int32).tolist(),
            })

        return text_lines

    def _process_mask(
        self,
        mask: np.ndarray,
        padded_size: Tuple[int, int],
        orig_size: Tuple[int, int],
        blocks: List[Dict] | None = None,
        legacy: bool = False,
        erase_threshold: float | None = None,
    ) -> np.ndarray:
        """Process segmentation mask to original image size.

        Koharu-style refinement (default):
          1. Threshold + clip to union of expanded per-block bboxes.
          2. Morphological close (radius ~10) to fill text-stroke gaps.
          3. L1 dilation radius=2 via two passes of the 3x3 cross kernel.
          4. Final dilate radius=3 with an ellipse kernel.
          5. Clip again to expanded block bounds so dilation never escapes.

        If ``legacy=True``, falls back to the pre-koharu behavior (plain
        threshold without block-aware refinement) for A/B comparisons.
        """
        # v26 emits a 2-channel mask: ch0=text, ch1=onomatopoeia (SFX). Combine
        # both (pixelwise max) so SFX ink is in the erase mask, not just dialogue
        # text. The block-bounds refinement below still clips it to detected
        # blocks/SFX regions, so this cannot leak onto un-detected art. (When the
        # model's ono channel is dead — e.g. round1seg — this is a near no-op; the
        # SFX capability lives in the model's ono head, trained on COO/MS92 ono.)
        if mask.ndim == 4:
            mask = mask[0].max(axis=0) if mask.shape[1] >= 2 else mask[0, 0]
        elif mask.ndim == 3:
            mask = mask.max(axis=0) if mask.shape[0] >= 2 else mask[0]

        pw, ph = padded_size
        mask = mask[:ph, :pw]

        w, h = orig_size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # ERASE mask threshold is decoupled from (and never higher than) the
        # detection threshold, so faint stroke tails are captured for inpainting
        # while detection stays at ``self.text_threshold``.
        seg_thr = (
            self.text_threshold
            if erase_threshold is None
            else min(erase_threshold, self.text_threshold)
        )
        binary = (mask > seg_thr).astype(np.uint8) * 255

        if legacy or not blocks:
            # Legacy path (or no blocks available to constrain): plain threshold.
            return binary

        # --- Koharu refine_segmentation_mask (postprocess.rs:25-77) ---

        # Build union of expanded per-block bboxes (item #5).
        in_bounds = self._build_block_bounds_mask(blocks, (w, h))

        if in_bounds is None:
            return binary

        # Step 1: clip threshold mask to block bounds.
        base = cv2.bitwise_and(binary, in_bounds)

        # Step 2: morph close (radius ~10) to connect gaps within a block.
        # RECT (not ELLIPSE): cv2 decomposes a rectangular SE into separable
        # passes (~20ms -> ~2ms). Measured ~0.2% pixel delta vs ellipse, harmless
        # here since the result is re-clipped to in_bounds and then dilated.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        closed = cv2.morphologyEx(base, cv2.MORPH_CLOSE, close_kernel)
        closed = cv2.bitwise_and(closed, in_bounds)

        # Step 3: L1 dilation by radius=2 (two passes of a 3x3 cross kernel
        # is equivalent to an L1 distance dilation of radius 2).
        cross3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilated = cv2.dilate(closed, cross3, iterations=2)

        # Step 4: final dilate radius=3 with an ellipse kernel.
        ellipse7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(dilated, ellipse7, iterations=1)

        # Step 5: final clip to block bounds so dilation never escapes.
        refined = cv2.bitwise_and(dilated, in_bounds)

        return refined

    def _build_block_bounds_mask(
        self,
        blocks: List[Dict],
        image_size: Tuple[int, int],
    ) -> np.ndarray | None:
        """Rasterize the union of expanded per-block bboxes.

        Ports expanded_text_block_crop_bounds from
        /tmp/koharu/koharu-ml/src/comic_text_detector/postprocess.rs:107-168.
        Each block is padded by font_px*0.1 horizontally/vertically (plus a
        small floor) and the union of those rectangles is rasterized.
        """
        if not blocks:
            return None

        w, h = image_size
        in_bounds = np.zeros((h, w), dtype=np.uint8)

        for block in blocks:
            bw = block["maxX"] - block["minX"]
            bh = block["maxY"] - block["minY"]
            if bw <= 0 or bh <= 0:
                continue

            # font size heuristic: use the shorter bbox dimension if the
            # caller did not pre-compute it.
            font = block.get("font_size_px") or min(bw, bh)
            font = max(float(font), 1.0)
            pad = max(font * 0.1, 2.0)

            x1 = int(max(0, block["minX"] - pad))
            y1 = int(max(0, block["minY"] - pad))
            x2 = int(min(w, block["maxX"] + pad))
            y2 = int(min(h, block["maxY"] + pad))
            if x2 <= x1 or y2 <= y1:
                continue
            in_bounds[y1:y2, x1:x2] = 255

        if in_bounds.max() == 0:
            return None
        return in_bounds

    def _expand_text_lines(
        self,
        text_lines: List[Dict],
        image_size: Tuple[int, int],
    ) -> List[Dict]:
        """Font-aware line-crop expansion (item #6).

        Ports maybe_expand_ctd_line + expanded_text_block_crop_bounds from
        /tmp/koharu/koharu-ml/src/comic_text_detector/postprocess.rs:107-262.
        For each line bbox, the font size is estimated from the shorter of
        (width, height). We pad horizontally/vertically with the
        direction-aware ratios from koharu:
          horizontal text: pad_x=font*0.12, pad_y=font*0.18
          vertical   text: pad_x=font*0.18, pad_y=font*0.12
        Direction is inferred from aspect ratio (height > width => vertical).
        """
        if not text_lines:
            return text_lines

        w, h = image_size
        out: List[Dict] = []
        for line in text_lines:
            bw = line["maxX"] - line["minX"]
            bh = line["maxY"] - line["minY"]
            if bw <= 0 or bh <= 0:
                out.append(line)
                continue

            vertical = bh > bw * 1.2
            font = min(bw, bh)
            font = max(float(font), 1.0)
            base_pad = max(font * 0.08, 2.0)
            if vertical:
                pad_x = max(font * 0.18, base_pad)
                pad_y = max(font * 0.12, base_pad)
            else:
                pad_x = max(font * 0.12, base_pad)
                pad_y = max(font * 0.18, base_pad)

            new_line = dict(line)
            new_line["minX"] = int(max(0, line["minX"] - pad_x))
            new_line["minY"] = int(max(0, line["minY"] - pad_y))
            new_line["maxX"] = int(min(w, line["maxX"] + pad_x))
            new_line["maxY"] = int(min(h, line["maxY"] + pad_y))
            new_line["font_size_px"] = float(font)
            new_line["direction"] = "vertical" if vertical else "horizontal"
            out.append(new_line)

        return out


    @staticmethod
    def _box_area(box: List[float]) -> float:
        w = max(0.0, box[2] - box[0])
        h = max(0.0, box[3] - box[1])
        return w * h

    @staticmethod
    def _calc_iou_raw(a: List[float], b: List[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _contained_ratio(cls, smaller: List[float], larger: List[float]) -> float:
        """Fraction of ``smaller``'s area that is contained within ``larger``."""
        x1 = max(smaller[0], larger[0])
        y1 = max(smaller[1], larger[1])
        x2 = min(smaller[2], larger[2])
        y2 = min(smaller[3], larger[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        sm_area = cls._box_area(smaller)
        return inter / sm_area if sm_area > 0 else 0.0

    @classmethod
    def _should_merge_koharu(cls, a: List[float], b: List[float]) -> bool:
        """Koharu's multi-check merge predicate.

        Ports merge_slice_regions from
        /tmp/koharu/koharu-ml/src/comic_text_bubble_detector/mod.rs:539-608.
        Returns True iff the two bboxes should be merged into one block.
        The merged bbox area must also be <= 3x the larger input area
        (caller-enforced after the merge).
        """
        area_a = cls._box_area(a)
        area_b = cls._box_area(b)
        if area_a <= 0 or area_b <= 0:
            return False

        # Rule 1: IoU >= 0.5 => same region.
        if cls._calc_iou_raw(a, b) >= 0.5:
            return True

        # Rule 2: One is >=85% contained inside the other.
        if area_a < area_b:
            if cls._contained_ratio(a, b) >= 0.85:
                return True
        else:
            if cls._contained_ratio(b, a) >= 0.85:
                return True

        # Rule 3: vertical proximity + horizontal alignment + size compat.
        width_a = max(a[2] - a[0], 1.0)
        height_a = max(a[3] - a[1], 1.0)
        width_b = max(b[2] - b[0], 1.0)
        height_b = max(b[3] - b[1], 1.0)
        min_height = min(height_a, height_b)

        # Smallest vertical gap between top/bottom edges.
        y_dist = min(abs(a[1] - b[3]), abs(a[3] - b[1]))
        x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        x_overlap_ratio = x_overlap / min(width_a, width_b)
        size_ratio = min(area_a, area_b) / max(area_a, area_b)

        edge_tolerance = 0.5 * max(width_a, width_b)
        horizontal_edge_aligned = (
            abs(a[0] - b[0]) < edge_tolerance
            and abs(a[2] - b[2]) < edge_tolerance
        )

        if (
            y_dist < min_height * 0.1
            and x_overlap_ratio > 0.2
            and size_ratio > 0.3
            and horizontal_edge_aligned
        ):
            return True

        return False

    def _derive_blocks_from_text_lines(
        self,
        text_lines: List[Dict],
        min_block_area: int = 500,
    ) -> List[Dict]:
        """Derive blocks from text lines using koharu's multi-check merge.

        Ports merge_slice_regions from
        /tmp/koharu/koharu-ml/src/comic_text_bubble_detector/mod.rs:518-608.
        Lines merge iff _should_merge_koharu returns True AND the resulting
        merged area is <= 3x the larger input area.
        """
        if not text_lines:
            return []

        boxes: List[List[float]] = [
            [float(t["minX"]), float(t["minY"]), float(t["maxX"]), float(t["maxY"])]
            for t in text_lines
        ]

        merged_any = True
        while merged_any:
            merged_any = False
            used: set = set()
            new_boxes: List[List[float]] = []

            for i, box_i in enumerate(boxes):
                if i in used:
                    continue
                current = list(box_i)
                for j in range(i + 1, len(boxes)):
                    if j in used:
                        continue
                    box_j = boxes[j]
                    if not self._should_merge_koharu(current, box_j):
                        continue
                    # Candidate merge; enforce area cap (<= 3x larger).
                    merged_box = [
                        min(current[0], box_j[0]),
                        min(current[1], box_j[1]),
                        max(current[2], box_j[2]),
                        max(current[3], box_j[3]),
                    ]
                    larger_area = max(self._box_area(current), self._box_area(box_j))
                    if self._box_area(merged_box) > 3.0 * larger_area:
                        continue
                    current = merged_box
                    used.add(j)
                    merged_any = True

                new_boxes.append(current)
                used.add(i)
            boxes = new_boxes

        blocks: List[Dict] = []
        for box in boxes:
            if self._box_area(box) < min_block_area:
                continue
            blocks.append({
                "minX": int(box[0]),
                "minY": int(box[1]),
                "maxX": int(box[2]),
                "maxY": int(box[3]),
                "confidence": 0.9,
            })

        # Manga reading order: right-to-left, then top-to-bottom.
        blocks.sort(key=lambda b: (-b["minX"], b["minY"]))
        return blocks

    @staticmethod
    def _union_block_from_members(members: Sequence[Dict]) -> Dict:
        """Union bbox of the fused column blocks (int coords, max confidence)."""
        return {
            "minX": int(min(float(m["minX"]) for m in members)),
            "minY": int(min(float(m["minY"]) for m in members)),
            "maxX": int(max(float(m["maxX"]) for m in members)),
            "maxY": int(max(float(m["maxY"]) for m in members)),
            "confidence": max(float(m.get("confidence", 0.9)) for m in members),
        }

    @staticmethod
    def _dedup_near_identical_blocks(
        blocks: List[Dict], *, iou_thresh: float = 0.9
    ) -> List[Dict]:
        """Drop near-identical duplicate CTD blocks BEFORE fusion (BUG D2).

        CTD occasionally emits two near-identical raw boxes for the SAME
        glyph region (verified on p082: two raw column detections at the
        exact same bbox). Without this pass one instance gets absorbed into
        a fused multi-column union while its exact duplicate survives as its
        own standalone block downstream — a blank/duplicate phantom box on
        the page (the duplicate has no distinguishing text of its own, since
        it IS the same glyphs, so OCR/translation produces an empty or
        redundant render).

        Two blocks are treated as duplicates ONLY when their IoU exceeds
        ``iou_thresh`` (0.9). This is conservative by construction: genuine
        side-by-side balloon COLUMNS are adjacent, not overlapping — their
        bounding boxes share little or no area (IoU near 0) even when they
        touch edge-to-edge, so this can never eat two distinct adjacent
        columns, only near-exact overlapping repeats. Keeps the
        larger-area box of each duplicate pair (ties broken by confidence);
        drops the rest. Preserves input order for survivors; non-mutating.
        """
        n = len(blocks)
        if n < 2:
            return list(blocks)
        dropped = [False] * n
        for i in range(n):
            if dropped[i]:
                continue
            box_i = [blocks[i]["minX"], blocks[i]["minY"], blocks[i]["maxX"], blocks[i]["maxY"]]
            for j in range(i + 1, n):
                if dropped[j]:
                    continue
                box_j = [blocks[j]["minX"], blocks[j]["minY"], blocks[j]["maxX"], blocks[j]["maxY"]]
                if ComicTextDetectorService._calc_iou_raw(box_i, box_j) <= iou_thresh:
                    continue
                area_i = ComicTextDetectorService._box_area(box_i)
                area_j = ComicTextDetectorService._box_area(box_j)
                conf_i = float(blocks[i].get("confidence", 0.9))
                conf_j = float(blocks[j].get("confidence", 0.9))
                if area_j > area_i or (area_j == area_i and conf_j > conf_i):
                    dropped[i] = True
                    break  # i is gone; stop comparing it against later boxes
                dropped[j] = True
        return [b for i, b in enumerate(blocks) if not dropped[i]]

    @staticmethod
    def _is_furigana_pair(
        a: Dict,
        b: Dict,
        *,
        height_ratio_max: float = 0.42,
        glyph_mult: float = 1.8,
        y_overlap_min: float = 0.50,
    ) -> bool:
        """True when one of ``a``/``b`` is a furigana/ruby gloss beside the other (BUG D3).

        A ruby annotation is a TINY side-column (a couple of phonetic
        characters) running alongside a portion of the kanji column it
        glosses — geometrically it looks like a normal adjacent column
        (passes ``_adjacent_columns``: close X, overlapping Y) but it is
        markedly SHORTER than its neighbour (verified on p082: ruby height
        ~60px beside a kanji column ~159px tall, ratio ~0.38). Fusing it as
        an independent reading-order element interleaves the ruby reading
        into the middle of the sentence, producing incoherent JP
        (``ruby + kanji + rest`` instead of ``kanji + rest``).

        Conservative guard: ``height_ratio_max`` sits BELOW the ratios of
        genuine short trailing dialogue columns. A 25-page GPU re-audit
        (2026-07-04) measured true furigana at ratio ~0.377 but real short
        columns that must still fuse (p110 ``締めの``/…, p114 ``たかったんだよな``/…,
        p093 ``理性``/…) at 0.46-0.49 — so the earlier 0.5 default wrongly
        excluded them and reintroduced blank/omitted translations. 0.42
        cleanly separates the two (below 0.46, above 0.377) and errs LOW:
        a false exclusion of a real column (-> blank output) is worse than
        an occasional mis-fused ruby. Also requires the SAME tight
        column-adjacency geometry (``_adjacent_columns``) the fuser demands,
        so boxes that merely differ in height but sit far apart or do not
        Y-overlap are never treated as a ruby pair.
        """
        ha = float(a["maxY"]) - float(a["minY"])
        hb = float(b["maxY"]) - float(b["minY"])
        if ha <= 0 or hb <= 0:
            return False
        short, tall = (a, b) if ha <= hb else (b, a)
        h_short, h_tall = min(ha, hb), max(ha, hb)
        if h_short / h_tall > height_ratio_max:
            return False
        return _adjacent_columns(
            short, tall, glyph_mult=glyph_mult, y_overlap_min=y_overlap_min
        )

    @staticmethod
    def fuse_balloon_columns(
        blocks: List[Dict],
        bubbles: Optional[Sequence[Dict]],
        *,
        max_span: int = 3,   # Re-audit (2026-07-04): cap at 3. Keeps the FULL -66%
                             # omission win (corrected_omission 17, byte-identical to
                             # span=6) AND eliminates the >=4-col blank over-merge
                             # (p070 recovered, 1->0), zero new omissions. (p082/p110/
                             # p114 blanks are a separate <=3-col/garbled-OCR issue.)
        glyph_mult: float = 1.8,
        y_overlap_min: float = 0.50,
        width_ratio: float = 2.2,
        bubble_area_mult: float = 8.0,
    ) -> List[Dict]:
        """DETECTION-TIME balloon-column fusion (BEFORE OCR).

        CTD emits one block per text COLUMN, so a multi-column vertical speech
        balloon arrives downstream as N independent OCR/translation units — the
        page-context model then reconstructs the sentence on EACH column
        (duplication) or folds it onto one and BLANKS the siblings (omission),
        and every fragment gets its own render box. This fuses the side-by-side
        columns of ONE balloon into a SINGLE block *here*, so OCR sees one crop
        and translation sees one JP string per balloon.

        Unlike the pre-translation ``bubble_grouping`` re-segmentation (disabled
        after the 2026-06-29 regression) there is NO merge->translate->resplit
        roundtrip to lose text on — the fused balloon is one unit end-to-end, so
        the failure that killed that attempt (resplit blanking continuations of a
        correctly-grouped long balloon) cannot occur here.

        Membership-gated and conservative (fusing two DISTINCT balloons is worse
        than leaving a split):
          * two blocks fuse only when they share a YOLO parent bubble — different
            or no parent NEVER fuse. Requires ``bubbles`` (the input is returned
            unchanged when the bubble detector did not run),
          * they must pass the SAME tight same-balloon column-adjacency geometry
            the pre-translation grouper uses (``_adjacent_columns``: X within
            ``glyph_mult`` glyph-widths, near-equal glyph width so a fat SFX is
            not absorbed as a column, ``y_overlap_min`` Y-overlap, RTL
            left-stepping direction),
          * the shared bubble must not be panel-sized relative to the fragments
            (``_bubble_implausibly_large``) — a panel does not authorise fusing
            everything inside it,
          * only STRICTLY reading-order-adjacent columns fuse, runs capped at
            ``max_span``.

        Two hardening passes/guards (2026-07-04 audit, p082):
          * ``_dedup_near_identical_blocks`` drops true near-duplicate raw
            CTD boxes (IoU > 0.9) BEFORE the greedy loop even runs, so an
            exact-duplicate detection never survives fusion as a second,
            blank phantom box (BUG D2),
          * ``_is_furigana_pair`` EXCLUDES a tiny ruby/furigana gloss from
            fusing with the taller kanji column it sits beside, inside
            ``_can_fuse`` (same bubble + column-adjacency geometry it would
            otherwise pass + markedly shorter). Note this must be an
            EXCLUSION, not a "merge them together first" pre-pass: since
            bbox union is associative, pre-merging ruby+kanji produces the
            IDENTICAL final union bbox/crop the naive loop would have
            produced anyway (verified: a no-op), which still hands OCR a
            crop containing both the ruby glyphs and the kanji column and
            reproduces the exact garbling this guard exists to prevent.
            Excluding the pairing instead leaves the ruby as its own
            isolated single-element block (its own small, low-value crop)
            while the kanji fuses normally with its OTHER genuine
            neighbours, WITHOUT the ruby's pixels ever entering that crop
            (BUG D3).
          Both are conservative geometry gates tuned to never touch two
          distinct adjacent columns — see their docstrings for the guards.

        Blocks are compared in column-major RTL reading order
        (``reading_order_sort``) so the columns of one balloon are consecutive,
        and each pairwise test is between two ORIGINAL single columns (the
        glyph-width guard is only meaningful column-vs-column). Returns a NEW
        block list: a fused run becomes one union block (max member confidence);
        untouched blocks are the same dict objects. Non-mutating.
        """
        if not bubbles or len(blocks) < 2:
            return list(blocks)

        blocks = ComicTextDetectorService._dedup_near_identical_blocks(blocks)
        if len(blocks) < 2:
            return list(blocks)

        ordered = reading_order_sort(blocks)
        n = len(ordered)
        bids = [bubble_id_of(b, bubbles) for b in ordered]

        def _can_fuse(j: int, k: int) -> bool:
            ba, bb = bids[j], bids[k]
            # Different or absent parent bubble => never fuse.
            if ba is None or bb is None or ba != bb:
                return False
            a, b = ordered[j], ordered[k]
            # A panel-sized container is not a balloon; it must not authorise
            # fusing its contents.
            if _bubble_implausibly_large(
                bubbles[ba], a, b, area_mult=bubble_area_mult
            ):
                return False
            # A furigana/ruby gloss beside a kanji column is not an
            # independent utterance -- excluding the pairing here (rather
            # than merging their boxes) keeps the ruby's pixels out of the
            # fused crop entirely (BUG D3).
            if ComicTextDetectorService._is_furigana_pair(
                a, b, glyph_mult=glyph_mult, y_overlap_min=y_overlap_min
            ):
                return False
            # Same tight same-balloon column geometry as the grouper.
            return _adjacent_columns(
                a, b,
                glyph_mult=glyph_mult,
                y_overlap_min=y_overlap_min,
                directional=True,
                max_width_ratio=width_ratio,
            )

        used = [False] * n
        fused: List[Dict] = []
        for i in range(n):
            if used[i]:
                continue
            members = [i]
            j = i
            # Greedy contiguous run; each step compares the last-added ORIGINAL
            # single column with the next single column.
            while len(members) < max_span and j + 1 < n and not used[j + 1]:
                if not _can_fuse(j, j + 1):
                    break
                members.append(j + 1)
                j += 1
            for m in members:
                used[m] = True
            if len(members) == 1:
                fused.append(ordered[i])
            else:
                fused.append(
                    ComicTextDetectorService._union_block_from_members(
                        [ordered[m] for m in members]
                    )
                )
        return fused

    def crop_regions(
        self,
        image: np.ndarray,
        blocks: List[Dict],
        padding: int = 5
    ) -> List[np.ndarray]:
        """Crop detected text blocks for OCR."""
        h, w = image.shape[:2]
        crops: List[np.ndarray] = []

        for block in blocks:
            x1 = max(0, block["minX"] - padding)
            y1 = max(0, block["minY"] - padding)
            x2 = min(w, block["maxX"] + padding)
            y2 = min(h, block["maxY"] + padding)
            crops.append(image[y1:y2, x1:x2])

        return crops
