#!/usr/bin/env python3
"""Test what OCR actually returns with current server code"""
import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import base64
from app.services.ocr_service import OCRService

async def test():
    # Load the most recent input image
    import glob
    images = sorted(glob.glob("debug_output/input_image_*.jpg"))
    if not images:
        print("No input images found!")
        return
    img_path = Path(images[-1])

    with open(img_path, 'rb') as f:
        img_bytes = f.read()
        base64_img = base64.b64encode(img_bytes).decode()

    print(f"Testing with image: {img_path}")
    print(f"Base64 length: {len(base64_img)}")

    # Use the actual OCRService from the server
    service = OCRService()

    print("\n" + "="*80)
    print("Running OCR with server's OCRService...")
    print("="*80)

    results = await service.detect_text(base64_img)

    print(f"\nTotal boxes detected: {len(results)}")
    print("\nFirst 3 boxes:")
    for i, box in enumerate(results[:3]):
        print(f"\nBox {i+1}:")
        print(f"  Text: {box['text']}")
        print(f"  Coords: ({box['minX']}, {box['minY']}) -> ({box['maxX']}, {box['maxY']})")
        print(f"  Size: {box['maxX']-box['minX']}x{box['maxY']-box['minY']}")
        print(f"  Confidence: {box['confidence']:.2f}")

if __name__ == "__main__":
    asyncio.run(test())
