#!/usr/bin/env python3
"""Debug script to test PaddleOCR coordinate output"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# Test with the latest input image
image_path = Path("debug_output/input_image_1760799624632.jpg")
if not image_path.exists():
    print(f"Image not found: {image_path}")
    sys.exit(1)

# Load image
image = Image.open(image_path)
image_np = np.array(image)
print(f"Original image size: {image_np.shape} (H x W x C)")
print(f"Original dimensions: {image.width}x{image.height}")

# Initialize PaddleOCR with our settings
print("\n" + "="*80)
print("Testing PaddleOCR with text_det_limit_side_len=9999")
print("="*80)

ocr = PaddleOCR(
    lang='japan',
    use_textline_orientation=True,
    text_det_limit_side_len=9999,
    text_det_limit_type='max',
    text_det_thresh=0.3,
    text_det_box_thresh=0.5,
    text_det_unclip_ratio=1.8,
)

# Run OCR
result = ocr.ocr(image_np)

print(f"\nResult type: {type(result)}")
print(f"Result length: {len(result)}")

if result and len(result) > 0:
    page = result[0]
    print(f"First page type: {type(page)}")

    if isinstance(page, list):
        print(f"\n✓ Using LEGACY LIST FORMAT")
        print(f"Number of text boxes: {len(page)}")

        for i, line in enumerate(page[:3]):  # Show first 3
            if len(line) >= 2:
                bbox = line[0]
                text_info = line[1]
                text = text_info[0] if len(text_info) > 0 else ""

                print(f"\nBox {i+1}:")
                print(f"  Text: {text}")
                print(f"  Raw bbox type: {type(bbox)}")
                print(f"  Raw bbox: {bbox}")

                # Calculate min/max
                bbox_arr = np.array(bbox)
                if bbox_arr.ndim == 2:
                    minX = int(bbox_arr[:, 0].min())
                    minY = int(bbox_arr[:, 1].min())
                    maxX = int(bbox_arr[:, 0].max())
                    maxY = int(bbox_arr[:, 1].max())
                    print(f"  Computed bounds: ({minX}, {minY}) -> ({maxX}, {maxY})")
                    print(f"  Box size: {maxX-minX}x{maxY-minY}")

                    # Check if coordinates are in original image space
                    if maxX > image.width or maxY > image.height:
                        print(f"  ⚠️  WARNING: Coordinates EXCEED image dimensions!")
                    elif maxX < image.width * 0.5 and maxY < image.height * 0.5:
                        print(f"  ⚠️  WARNING: Coordinates seem SCALED DOWN (much smaller than image)")
                    else:
                        print(f"  ✓ Coordinates seem to be in original image space")

    elif hasattr(page, 'json'):
        print(f"\n✓ Using OCRResult FORMAT")
        page_data = page.json.get('res', {})
        texts = page_data.get('rec_texts', [])
        rec_boxes = page_data.get('rec_boxes')
        polys = page_data.get('rec_polys') or page_data.get('dt_polys')

        print(f"Number of texts: {len(texts)}")
        print(f"rec_boxes type: {type(rec_boxes)}")
        print(f"polys type: {type(polys)}")

        if rec_boxes is not None:
            print(f"\nFirst 3 rec_boxes:")
            for i in range(min(3, len(rec_boxes))):
                print(f"\nBox {i+1}:")
                print(f"  Text: {texts[i] if i < len(texts) else 'N/A'}")
                print(f"  rec_boxes[{i}]: {rec_boxes[i]}")

        if polys is not None:
            print(f"\nFirst 3 polys:")
            for i in range(min(3, len(polys) if isinstance(polys, list) else 0)):
                print(f"\nBox {i+1}:")
                print(f"  Text: {texts[i] if i < len(texts) else 'N/A'}")
                print(f"  polys[{i}]: {polys[i]}")

print("\n" + "="*80)
print("DONE")
print("="*80)
