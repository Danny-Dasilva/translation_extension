# Comic-Text-Detector (CTD) Implementation Plan

## Overview

Replace YOLOv10n bubble detector + CV-based text extraction with Comic-Text-Detector ONNX model for accurate tight text bounding boxes.

**Model Source**: https://huggingface.co/mayocream/comic-text-detector-onnx
- License: Apache 2.0
- Training Data: Manga109-s dataset
- Size: ~95MB

## Problem Statement

**Current state**: Green bounding boxes cover entire speech bubbles (lots of whitespace)
**Desired state**: Boxes tightly wrap only the actual text characters

The CV-based approach (`text_region_extractor.py`) fundamentally cannot solve this because:
- Morphological operations connect text to bubble edges
- Otsu threshold includes bubble boundary pixels
- Classical CV can't distinguish text from bubble interior

## Architecture Comparison

| Current Pipeline | New CTD Pipeline |
|-----------------|------------------|
| Image → YOLOv10n (bubbles) → Crop → CV text bounds → OCR → Translate | Image → CTD (blocks + text lines + mask) → OCR → Translate |
| Text detection constrained by bubble crop | Text detection on full image |
| Two models + CV heuristics | Single model, three outputs |

## CTD Model Outputs

The CTD ONNX model outputs 3 tensors:

| Output | Description | Use For |
|--------|-------------|---------|
| `blks` | YOLO-format text block boxes | Replaces YOLOv10n bubble detection |
| `lines_map` | DBNet probability map | **Tight text polygons** (the main goal) |
| `mask` | Binary text segmentation | Text removal/inpainting |

## Files to Modify/Create

| File | Action |
|------|--------|
| `scripts/download_models.py` | Add CTD ONNX download from HuggingFace |
| `app/services/ctd_service.py` | **NEW** - Comic Text Detector service |
| `app/services/detector_service.py` | Deprecate or remove |
| `app/routers/translate.py` | Replace detector_service with ctd_service |
| `app/routers/test_page.py` | Update to use CTD |
| `app/utils/text_region_extractor.py` | Deprecate (CTD provides this) |
| `app/config.py` | Add CTD config options |

## Implementation Steps

### Step 1: Model Download Script

```python
# scripts/download_models.py - add:

def download_ctd_model():
    """Download Comic Text Detector ONNX model from HuggingFace."""
    from huggingface_hub import hf_hub_download
    import os

    model_path = "models/comictextdetector.onnx"
    if not os.path.exists(model_path):
        print("Downloading Comic Text Detector ONNX model...")
        hf_hub_download(
            repo_id="mayocream/comic-text-detector-onnx",
            filename="model.onnx",
            local_dir="models",
            local_dir_use_symlinks=False
        )
        # Rename if needed
        if os.path.exists("models/model.onnx"):
            os.rename("models/model.onnx", model_path)
    return model_path
```

### Step 2: CTD Service Class

```python
# app/services/ctd_service.py
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ComicTextDetectorService:
    """
    Comic Text Detector - detects text blocks and tight text regions.

    Replaces:
    - detector_service.py (YOLOv10n bubble detection)
    - text_region_extractor.py (CV-based text bounds)
    """

    def __init__(self, model_path: str = "models/comictextdetector.onnx"):
        logger.info(f"Loading Comic Text Detector from {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_size = 1024
        self.text_threshold = 0.3
        self.min_area = 100

        # Log which provider is being used
        provider = self.session.get_providers()[0]
        logger.info(f"CTD using execution provider: {provider}")

    def _letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize with padding maintaining aspect ratio."""
        h, w = img.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        return padded, scale, (new_w, new_h)

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for CTD model."""
        # Handle different input formats
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.shape[2] == 3:
            # Assume BGR from cv2, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        padded, scale, (pw, ph) = self._letterbox(img)

        # Normalize and transpose to NCHW
        img_in = padded.astype(np.float32) / 255.0
        img_in = img_in.transpose(2, 0, 1)[None]  # [1, 3, H, W]

        return img_in, scale, (pw, ph)

    async def detect(self, img: np.ndarray) -> Dict:
        """
        Detect text blocks and tight text regions.

        Args:
            img: Input image (BGR or RGB numpy array)

        Returns:
            {
                'blocks': List of text block bboxes (bubble-level),
                'text_lines': List of tight text polygons,
                'mask': Binary text mask (H, W)
            }
        """
        h, w = img.shape[:2]
        img_in, scale, (pw, ph) = self._preprocess(img)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img_in})

        # CTD outputs: blks, mask, lines_map
        # Note: exact order may vary - verify with model inspection
        blks, mask, lines_map = outputs[0], outputs[1], outputs[2]

        # Extract text blocks (bubble-level boxes from YOLO head)
        blocks = self._parse_blocks(blks, scale, (w, h))

        # Extract tight text lines from DBNet output
        text_lines = self._extract_text_lines(lines_map, scale, (pw, ph), (w, h))

        # Process segmentation mask
        text_mask = self._process_mask(mask, (pw, ph), (w, h))

        logger.debug(f"CTD detected {len(blocks)} blocks, {len(text_lines)} text lines")

        return {
            'blocks': blocks,
            'text_lines': text_lines,
            'mask': text_mask
        }

    def _parse_blocks(self, blks: np.ndarray, scale: float,
                      orig_size: Tuple[int, int]) -> List[Dict]:
        """
        Parse YOLO-format block detections.

        blks format: [batch, num_boxes, 6] where 6 = [x1, y1, x2, y2, conf, cls]
        or similar depending on model export.
        """
        blocks = []
        w, h = orig_size

        # Handle different possible output shapes
        if blks.ndim == 3:
            blks = blks[0]  # Remove batch dimension

        for det in blks:
            if len(det) < 5:
                continue

            x1, y1, x2, y2, conf = det[:5]

            # Filter by confidence
            if conf < 0.4:
                continue

            # Scale back to original coordinates
            blocks.append({
                'minX': int(x1 / scale),
                'minY': int(y1 / scale),
                'maxX': int(x2 / scale),
                'maxY': int(y2 / scale),
                'confidence': float(conf)
            })

        return blocks

    def _extract_text_lines(self, lines_map: np.ndarray, scale: float,
                            padded_size: Tuple[int, int],
                            orig_size: Tuple[int, int]) -> List[Dict]:
        """
        Extract tight text polygons from DBNet probability map.

        This is the key function that produces tight text bounds.
        """
        # Get probability map from output
        if lines_map.ndim == 4:
            prob_map = lines_map[0, 0]  # [batch, channel, H, W] -> [H, W]
        elif lines_map.ndim == 3:
            prob_map = lines_map[0]
        else:
            prob_map = lines_map

        # Threshold to binary
        binary = (prob_map > self.text_threshold).astype(np.uint8) * 255

        # Remove padding
        pw, ph = padded_size
        binary = binary[:ph, :pw]

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        text_lines = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            # Get axis-aligned bounding box
            x, y, bw, bh = cv2.boundingRect(cnt)

            # Scale back to original image coordinates
            text_lines.append({
                'minX': int(x / scale),
                'minY': int(y / scale),
                'maxX': int((x + bw) / scale),
                'maxY': int((y + bh) / scale),
                'area': int(area / (scale * scale)),
                'polygon': (cnt / scale).astype(np.int32).tolist()
            })

        return text_lines

    def _process_mask(self, mask: np.ndarray, padded_size: Tuple[int, int],
                      orig_size: Tuple[int, int]) -> np.ndarray:
        """Process segmentation mask to original image size."""
        # Handle different output shapes
        if mask.ndim == 4:
            mask = mask[0, 0]
        elif mask.ndim == 3:
            mask = mask[0]

        # Remove padding
        pw, ph = padded_size
        mask = mask[:ph, :pw]

        # Resize to original dimensions
        w, h = orig_size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Binarize
        return (mask > self.text_threshold).astype(np.uint8) * 255

    def crop_regions(self, img: np.ndarray, blocks: List[Dict]) -> List[np.ndarray]:
        """Crop image regions for each block (for OCR)."""
        crops = []
        for block in blocks:
            x1 = max(0, block['minX'])
            y1 = max(0, block['minY'])
            x2 = min(img.shape[1], block['maxX'])
            y2 = min(img.shape[0], block['maxY'])
            crops.append(img[y1:y2, x1:x2])
        return crops
```

### Step 3: Update translate.py

Key changes to `app/routers/translate.py`:

```python
# OLD imports to remove:
# from app.services.detector_service import DetectorService
# from app.utils.text_region_extractor import extract_text_bounds, calculate_inset_region

# NEW imports:
from app.services.ctd_service import ComicTextDetectorService

# OLD initialization to remove:
# detector_service = DetectorService()

# NEW initialization:
logger.info("Initializing Comic Text Detector...")
ctd_service = ComicTextDetectorService()

# In process_single_image(), replace:
#   bubbles = await detector_service.detect_bubbles(image_np, ...)
#   crops = detector_service.crop_regions(image_np, bubbles)
#   ... extract_text_bounds(crop, method='morphological') ...

# With:
async def process_single_image(idx, base64_image, target_language, semaphore):
    async with semaphore:
        image_np = decode_base64_to_numpy(base64_image)

        # Single CTD call gets blocks + tight text lines + mask
        result = await ctd_service.detect(image_np)

        blocks = result['blocks']          # Bubble-level boxes
        text_lines = result['text_lines']  # Tight text bounds
        mask = result['mask']              # Text segmentation

        if not blocks:
            return (idx, [])

        # Crop regions for OCR
        crops = ctd_service.crop_regions(image_np, blocks)

        # OCR on crops
        ocr_texts = await ocr_service.recognize_text_batch(crops)

        # Match text_lines to blocks
        # Each block may contain multiple text_lines
        block_text_regions = match_text_lines_to_blocks(blocks, text_lines)

        # ... rest of pipeline (translation, response building)


def match_text_lines_to_blocks(blocks: List[Dict],
                                text_lines: List[Dict]) -> List[List[Dict]]:
    """
    Match tight text line boxes to their parent blocks.

    Returns list of text_line lists, one per block.
    """
    result = [[] for _ in blocks]

    for line in text_lines:
        # Find which block contains this text line
        line_cx = (line['minX'] + line['maxX']) / 2
        line_cy = (line['minY'] + line['maxY']) / 2

        for i, block in enumerate(blocks):
            if (block['minX'] <= line_cx <= block['maxX'] and
                block['minY'] <= line_cy <= block['maxY']):
                result[i].append(line)
                break

    return result
```

### Step 4: Update Config

```python
# app/config.py - add:
class Settings(BaseSettings):
    # ... existing settings ...

    # Comic Text Detector settings
    ctd_model_path: str = "models/comictextdetector.onnx"
    ctd_input_size: int = 1024
    ctd_text_threshold: float = 0.3
    ctd_block_confidence: float = 0.4
    ctd_min_text_area: int = 100
```

## Testing Checklist

- [ ] Model downloads successfully from HuggingFace
- [ ] CTD service initializes with CUDA provider
- [ ] Test page shows tight text boxes (green) matching actual text
- [ ] `blocks` output replaces YOLO bubble detection
- [ ] `text_lines` output provides tight text bounds
- [ ] OCR still works on detected regions
- [ ] Translation pipeline end-to-end works
- [ ] Performance is acceptable (~50-100ms per image)

## Research Sources

- [comic-text-detector GitHub](https://github.com/dmMaze/comic-text-detector)
- [manga-image-translator GitHub](https://github.com/zyddnys/manga-image-translator)
- [HuggingFace model](https://huggingface.co/mayocream/comic-text-detector-onnx)
- Training data: [Manga109-s dataset](https://huggingface.co/datasets/hal-utokyo/Manga109-s)

## Notes

1. **CTD detects TEXT directly**, not bubbles first. This is fundamentally different from the current YOLO approach.

2. The `lines_map` DBNet output is what gives tight text polygons - this is the key improvement over CV-based extraction.

3. The `mask` output can be used for text inpainting/removal if needed later.

4. **CRAFT is NOT recommended for manga** - per manga-image-translator docs.

5. The model was trained on manga specifically (Manga109 + Digital Comic Museum datasets).
