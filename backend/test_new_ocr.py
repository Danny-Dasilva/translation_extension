#!/usr/bin/env python3
"""Test script to verify new OCR setup with a manga image"""
import asyncio
import base64
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.ocr_service import OCRService


async def test_ocr(image_path: str):
    """Test OCR on a manga image"""
    print(f"Testing OCR on: {image_path}")

    # Read image and convert to base64
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

    # Initialize OCR service
    print("Initializing OCR service...")
    ocr_service = OCRService()

    # Detect text
    print("Detecting text...")
    results = await ocr_service.detect_text(base64_image)

    print(f"\n{'='*80}")
    print(f"RESULTS: Detected {len(results)} text regions")
    print(f"{'='*80}\n")

    for idx, result in enumerate(results, 1):
        print(f"Region {idx}:")
        print(f"  Text: {result['text']}")
        print(f"  BBox: ({result['minX']}, {result['minY']}) -> ({result['maxX']}, {result['maxY']})")
        print(f"  Confidence: {result['confidence']:.2f}")
        print()

    print(f"{'='*80}")
    print("Check backend/debug_output/ for visualization")
    print(f"{'='*80}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use the latest debug image if no argument provided
        debug_dir = Path(__file__).parent / "debug_output"
        images = list(debug_dir.glob("*.jpg"))
        if images:
            # Get the most recent non-debug image (original manga page)
            image_path = str(sorted(images)[-1])
        else:
            print("Please provide an image path as argument")
            sys.exit(1)

    asyncio.run(test_ocr(image_path))
