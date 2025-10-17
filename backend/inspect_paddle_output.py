#!/usr/bin/env python3
"""Inspect PaddleOCR output format"""
from paddleocr import PaddleOCR
from PIL import Image
import json

# Initialize detector
detector = PaddleOCR(
    use_textline_orientation=True,
    lang='japan',
    text_det_thresh=0.3,
    text_det_box_thresh=0.5,
    text_det_unclip_ratio=1.8,
)

# Test image
image_path = "/home/danny/Documents/personal/extension/backend/.venv/lib/python3.11/site-packages/manga_ocr/assets/example.jpg"

# Run detection with file path
result = detector.predict(image_path)

print("Result type:", type(result))
print("Result length:", len(result))
print("\nFull result structure:")
print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
