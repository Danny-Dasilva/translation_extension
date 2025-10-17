#!/usr/bin/env python3
"""Test script to verify translation visualization"""
import asyncio
import base64
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.ocr_service import OCRService
from app.services.translation_service import TranslationService
import numpy as np
from PIL import Image


async def test_translation_viz(image_path: str):
    """Test full OCR + Translation + Visualization pipeline"""
    print(f"Testing translation visualization on: {image_path}")

    # Read image and convert to base64
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

    # Load image as numpy array for visualization
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_np = np.array(image)

    # Initialize services
    print("Initializing services...")
    ocr_service = OCRService()
    translation_service = TranslationService()

    # Step 1: Detect text
    print("Detecting text...")
    ocr_results = await ocr_service.detect_text(base64_image)
    print(f"Detected {len(ocr_results)} text regions")

    if not ocr_results:
        print("No text detected!")
        return

    # Step 2: Group nearby text regions
    print("Grouping text regions...")
    grouped_results = ocr_service.group_text_regions(ocr_results)
    print(f"Grouped into {len(grouped_results)} text regions")

    # Step 3: Translate texts
    print("Translating text...")
    texts_to_translate = [box['text'] for box in grouped_results]
    translations = await translation_service.translate_batch(texts_to_translate, "English")

    # Step 4: Combine results
    translated_boxes = []
    for ocr_box, translation in zip(grouped_results, translations):
        translated_boxes.append({
            **ocr_box,
            'translation': translation
        })

    # Display results
    print(f"\n{'='*80}")
    print("TRANSLATION RESULTS")
    print(f"{'='*80}\n")

    for idx, box in enumerate(translated_boxes, 1):
        print(f"Region {idx}:")
        print(f"  Japanese: {box['text']}")
        print(f"  English:  {box['translation']}")
        print(f"  BBox: ({box['minX']}, {box['minY']}) -> ({box['maxX']}, {box['maxY']})")
        print()

    # Step 4: Visualize translated text
    print("Creating translation visualization...")
    ocr_service.visualize_translated_text(image_np, translated_boxes)

    print(f"{'='*80}")
    print("Check backend/debug_output/ for:")
    print("  - ocr_debug_*.jpg (detected Japanese text with boxes)")
    print("  - translation_debug_*.jpg (English translations overlaid)")
    print(f"{'='*80}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use the manga-ocr example
        image_path = str(Path(__file__).parent / ".venv/lib/python3.11/site-packages/manga_ocr/assets/example.jpg")

    asyncio.run(test_translation_viz(image_path))
