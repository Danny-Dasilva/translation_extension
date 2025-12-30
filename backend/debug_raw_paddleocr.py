#!/usr/bin/env python3
"""Check what PaddleOCR ACTUALLY returns - raw format inspection"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import glob

# Get latest input image
files = sorted(glob.glob('debug_output/input_image_*.jpg'))
if not files:
    print("No input images found")
    sys.exit(1)

img_path = files[-1]
image = Image.open(img_path)
image_np = np.array(image)

print(f"Image: {img_path}")
print(f"Size: {image.size} (W x H)")
print(f"Shape: {image_np.shape} (H x W x C)")
print()

# Initialize with current settings
ocr = PaddleOCR(
    lang='japan',
    use_textline_orientation=True,
    text_det_limit_side_len=9999,
    text_det_limit_type='max',
    text_det_thresh=0.3,
    text_det_box_thresh=0.5,
    text_det_unclip_ratio=1.8,
)

result = ocr.ocr(image_np)

print("="*80)
print("RAW PADDLEOCR OUTPUT INSPECTION")
print("="*80)
print()

if result and len(result) > 0:
    page = result[0]
    print(f"Page type: {type(page)}")
    print(f"Has .json attr: {hasattr(page, 'json')}")

    if hasattr(page, 'json'):
        page_data = page.json.get('res', {})
        print(f"\nKeys in page_data: {list(page_data.keys())}")

        rec_boxes = page_data.get('rec_boxes')
        polys = page_data.get('rec_polys') or page_data.get('dt_polys')
        texts = page_data.get('rec_texts', [])

        print(f"\nrec_boxes type: {type(rec_boxes)}")
        print(f"rec_boxes length: {len(rec_boxes) if rec_boxes else 0}")

        if rec_boxes and len(rec_boxes) > 0:
            print(f"\n1st rec_box RAW: {rec_boxes[0]}")
            print(f"  Type: {type(rec_boxes[0])}")
            print(f"  Text: {texts[0] if texts else 'N/A'}")

            # Try to understand format
            box = rec_boxes[0]
            if isinstance(box, list) and len(box) == 4:
                print(f"\n  Interpreting as [a, b, c, d] = {box}")
                print(f"  If [minX, minY, maxX, maxY]: box at ({box[0]}, {box[1]}) size {box[2]-box[0]}x{box[3]-box[1]}")
                print(f"  If [minY, minX, maxY, maxX]: box at ({box[1]}, {box[0]}) size {box[3]-box[1]}x{box[2]-box[0]}")

        if polys and len(polys) > 0:
            print(f"\n1st poly RAW: {polys[0]}")
            print(f"  Type: {type(polys[0])}")

print("\nDone")
