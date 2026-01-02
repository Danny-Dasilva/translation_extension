"""YOLOv10n bubble detection service for manga pages."""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from app.config import settings

logger = logging.getLogger(__name__)


class DetectorService:
    """
    Speech bubble detection using YOLOv10n fine-tuned on manga.

    YOLOv10 uses "Consistent Dual Assignments" which eliminates NMS,
    providing deterministic latency regardless of detection count.
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize the detector with a YOLOv10n model.

        Args:
            model_path: Path to the fine-tuned weights. Defaults to settings.
        """
        if model_path is None:
            model_path = settings.yolo_model_path

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"YOLOv10n model not found at {model_path}. "
                "Run training or download pre-trained weights."
            )

        logger.info(f"Loading YOLOv10n model from {model_path}")
        self.model = YOLO(model_path)
        logger.info("YOLOv10n model loaded successfully")

    async def detect_bubbles(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        imgsz: int = 640
    ) -> List[Dict]:
        """
        Detect speech bubbles in a manga image.

        Args:
            image: Input image as numpy array (RGB or BGR)
            conf: Confidence threshold for detections
            imgsz: Input image size for inference

        Returns:
            List of detected bubbles with bounding boxes:
            [{"minX": int, "minY": int, "maxX": int, "maxY": int, "confidence": float}, ...]

            Sorted in manga reading order (right-to-left, top-to-bottom).
        """
        # Run inference (YOLOv10 is NMS-free)
        results = self.model.predict(
            image,
            imgsz=imgsz,
            conf=conf,
            verbose=False
        )

        boxes = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append({
                    "minX": int(coords[0]),
                    "minY": int(coords[1]),
                    "maxX": int(coords[2]),
                    "maxY": int(coords[3]),
                    "confidence": float(box.conf[0])
                })

        # Sort by manga reading order: right-to-left (descending X), top-to-bottom (ascending Y)
        boxes.sort(key=lambda b: (-b["minX"], b["minY"]))

        logger.debug(f"Detected {len(boxes)} speech bubbles")
        return boxes

    def crop_regions(
        self,
        image: np.ndarray,
        boxes: List[Dict],
        padding: int = 5
    ) -> List[np.ndarray]:
        """
        Crop detected bubble regions from the image.

        Args:
            image: Source image as numpy array
            boxes: List of bounding boxes from detect_bubbles()
            padding: Pixels to add around each box (default: 5)

        Returns:
            List of cropped image regions as numpy arrays
        """
        h, w = image.shape[:2]
        crops = []

        for box in boxes:
            # Apply padding with bounds checking
            x1 = max(0, box["minX"] - padding)
            y1 = max(0, box["minY"] - padding)
            x2 = min(w, box["maxX"] + padding)
            y2 = min(h, box["maxY"] + padding)

            crop = image[y1:y2, x1:x2]
            crops.append(crop)

        return crops
