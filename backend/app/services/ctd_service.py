"""Comic Text Detector (CTD) service for manga text detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.config import settings

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

        model_file = Path(model_path)
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
        try:
            return ort.InferenceSession(model_path, providers=providers)
        except Exception as exc:
            if "CUDAExecutionProvider" in providers:
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
        outputs = self.session.run(None, {input_name: img_in})

        blks, mask, lines_map = self._assign_outputs(outputs)

        blocks = self._parse_blocks(blks, scale, (w, h))
        text_lines = self._extract_text_lines(lines_map, scale, padded_size, (w, h))
        text_mask = self._process_mask(mask, padded_size, (w, h)) if mask is not None else None

        # When block detection is unavailable, derive blocks from text_lines
        # This supports models that only output segmentation + text line maps
        if not blocks and text_lines:
            logger.info("No block detections; deriving blocks from text lines")
            blocks = self._derive_blocks_from_text_lines(text_lines)

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

        blocks.sort(key=lambda b: (-b["minX"], b["minY"]))
        return blocks

    def _extract_text_lines(
        self,
        lines_map: np.ndarray | None,
        scale: float,
        padded_size: Tuple[int, int],
        orig_size: Tuple[int, int]
    ) -> List[Dict]:
        """Extract tight text boxes from the DBNet probability map."""
        if lines_map is None:
            return []

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
        orig_size: Tuple[int, int]
    ) -> np.ndarray:
        """Process segmentation mask to original image size."""
        if mask.ndim == 4:
            mask = mask[0, 0]
        elif mask.ndim == 3:
            mask = mask[0]

        pw, ph = padded_size
        mask = mask[:ph, :pw]

        w, h = orig_size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        return (mask > self.text_threshold).astype(np.uint8) * 255


    def _derive_blocks_from_text_lines(
        self,
        text_lines: List[Dict],
        merge_distance: int = 30,
        min_block_area: int = 500
    ) -> List[Dict]:
        """
        Derive text blocks from text_lines by merging nearby lines.
        
        Uses a simple spatial clustering approach:
        1. Start with each text_line as a potential block
        2. Merge lines that are within merge_distance of each other
        3. Filter by minimum area
        
        Args:
            text_lines: List of text line dicts with minX, minY, maxX, maxY
            merge_distance: Max distance to merge nearby lines into same block
            min_block_area: Minimum block area to keep
            
        Returns:
            List of block dicts with minX, minY, maxX, maxY, confidence
        """
        if not text_lines:
            return []
        
        # Convert to list of [minX, minY, maxX, maxY]
        boxes = [[t["minX"], t["minY"], t["maxX"], t["maxY"]] for t in text_lines]
        
        # Greedy merge: iteratively merge overlapping/nearby boxes
        merged = True
        while merged:
            merged = False
            new_boxes = []
            used = set()
            
            for i, box1 in enumerate(boxes):
                if i in used:
                    continue
                    
                current = list(box1)
                for j, box2 in enumerate(boxes):
                    if j <= i or j in used:
                        continue
                    
                    # Check if boxes are close enough to merge
                    # Expand box1 by merge_distance and check overlap
                    expanded = [
                        current[0] - merge_distance,
                        current[1] - merge_distance,
                        current[2] + merge_distance,
                        current[3] + merge_distance
                    ]
                    
                    # Check if box2 overlaps with expanded box1
                    if (expanded[0] <= box2[2] and expanded[2] >= box2[0] and
                        expanded[1] <= box2[3] and expanded[3] >= box2[1]):
                        # Merge: take union of bounding boxes
                        current = [
                            min(current[0], box2[0]),
                            min(current[1], box2[1]),
                            max(current[2], box2[2]),
                            max(current[3], box2[3])
                        ]
                        used.add(j)
                        merged = True
                
                new_boxes.append(current)
                used.add(i)
            
            boxes = new_boxes
        
        # Convert to block format and filter by area
        blocks = []
        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            
            if area >= min_block_area:
                blocks.append({
                    "minX": box[0],
                    "minY": box[1],
                    "maxX": box[2],
                    "maxY": box[3],
                    "confidence": 0.9,  # Derived blocks have fixed confidence
                })
        
        # Sort by reading order (right-to-left for manga, then top-to-bottom)
        blocks.sort(key=lambda b: (-b["minX"], b["minY"]))
        
        return blocks

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
