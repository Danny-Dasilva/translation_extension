"""AnimeText YOLO12s detector service - optimized for speed."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.config import settings

logger = logging.getLogger(__name__)


class AnimeTextDetectorService:
    """
    AnimeText YOLO12s FP16 text detector - 3.1x faster than CTD.

    Performance (RTX 5090):
    - Mean: 2.41ms | P95: 2.45ms | 414 FPS
    - vs CTD: 7.52ms | 133 FPS

    Features:
    - 640x640 input size (vs CTD's 1024x1024)
    - 18MB model (vs CTD's 91MB)
    - Blocks-only output (no text_lines or mask)
    """

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = settings.animetext_model_path

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"AnimeText model not found at {model_path}. "
                "Run scripts/download_models.py --animetext to download it."
            )

        self.input_size = settings.animetext_input_size
        self.confidence_threshold = settings.animetext_confidence_threshold

        providers = self._select_providers()
        logger.info(f"Loading AnimeText YOLO12s from {model_path}")
        self.session = self._create_session(model_path, providers)

        provider = self.session.get_providers()[0] if self.session.get_providers() else "unknown"
        logger.info(f"AnimeText using execution provider: {provider}")

    def _select_providers(self) -> List[str]:
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available]
        return providers or available

    def _create_session(self, model_path: str, providers: List[str]) -> ort.InferenceSession:
        try:
            return ort.InferenceSession(model_path, providers=providers)
        except Exception as exc:
            if "CUDAExecutionProvider" in providers:
                logger.warning("AnimeText CUDA init failed (%s). Falling back to CPU.", exc)
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
        """Preprocess image for AnimeText model."""
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
        Detect text blocks (blocks only, no text_lines or mask).

        Args:
            img: Input image (RGB or BGR numpy array)
            input_is_bgr: Set True if the input is BGR (e.g., cv2.imdecode)

        Returns:
            {
                "blocks": List of text block bboxes with minX, minY, maxX, maxY, confidence,
                "text_lines": [] (empty - not supported by this detector),
                "mask": None (not supported by this detector)
            }
        """
        h, w = img.shape[:2]
        img_in, scale, padded_size = self._preprocess(img, input_is_bgr)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img_in})

        blocks = self._parse_yolo_output(outputs[0], scale, (w, h))

        logger.debug(f"AnimeText detected {len(blocks)} blocks")

        return {
            "blocks": blocks,
            "text_lines": [],  # AnimeText doesn't provide text lines
            "mask": None,      # AnimeText doesn't provide mask
        }

    def _parse_yolo_output(
        self,
        output: np.ndarray,
        scale: float,
        orig_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Parse YOLO12 output format (vectorized for speed).

        Input shape: [1, 5, 8400] where 5 = [cx, cy, w, h, conf]
        """
        w, h = orig_size

        # Reshape: [1, 5, 8400] -> [8400, 5]
        if output.ndim == 3:
            output = output[0]  # -> [5, 8400]
        if output.shape[0] == 5:
            output = output.T  # -> [8400, 5]

        # Extract columns
        cx, cy, bw, bh, conf = output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4]

        # Filter by confidence (vectorized)
        mask = conf >= self.confidence_threshold
        if not np.any(mask):
            return []

        cx, cy, bw, bh, conf = cx[mask], cy[mask], bw[mask], bh[mask], conf[mask]

        # Convert center to corner format (vectorized)
        x1 = (cx - bw / 2) / scale
        y1 = (cy - bh / 2) / scale
        x2 = (cx + bw / 2) / scale
        y2 = (cy + bh / 2) / scale

        # Clip to image bounds (vectorized)
        x1 = np.clip(x1, 0, w).astype(np.int32)
        y1 = np.clip(y1, 0, h).astype(np.int32)
        x2 = np.clip(x2, 0, w).astype(np.int32)
        y2 = np.clip(y2, 0, h).astype(np.int32)

        # Filter invalid boxes (vectorized)
        valid = (x2 > x1) & (y2 > y1)
        x1, y1, x2, y2, conf = x1[valid], y1[valid], x2[valid], y2[valid], conf[valid]

        # Build blocks list
        blocks = [
            {"minX": int(x1[i]), "minY": int(y1[i]), "maxX": int(x2[i]), "maxY": int(y2[i]), "confidence": float(conf[i])}
            for i in range(len(conf))
        ]

        # Apply NMS (YOLO12 is not NMS-free like YOLOv10)
        blocks = self._apply_nms(blocks, iou_threshold=0.5)

        # Filter out boxes that fully contain other boxes (keep innermost)
        blocks = self._filter_contained_boxes(blocks)

        # Sort by reading order (right-to-left for manga, then top-to-bottom)
        blocks.sort(key=lambda b: (-b["minX"], b["minY"]))
        return blocks

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
