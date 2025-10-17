"""Translation endpoint router"""
import logging
import time
from fastapi import APIRouter, HTTPException, status
from typing import List
import base64
import io
from PIL import Image
import numpy as np

from app.models.request import TranslateRequest
from app.models.response import TranslateResponse, TextBox
from app.services.ocr_service import OCRService
from app.services.translation_service import TranslationService
from app.utils.image_processing import (
    calculate_font_size,
    detect_font_colors,
    extract_text_region_background
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
ocr_service = OCRService()
translation_service = TranslationService()


@router.post("/translate", response_model=TranslateResponse)
async def translate_images(request: TranslateRequest):
    """
    Translate manga images:
    1. Decode base64 images
    2. Run OCR to detect Japanese text + bounding boxes
    3. Translate detected text using Gemini
    4. Return structured response with translations and metadata
    """
    start_time = time.time()
    try:
        logger.info(f"Processing {len(request.base64Images)} images")
        
        all_results = []
        
        for idx, base64_image in enumerate(request.base64Images):
            try:
                logger.info(f"Processing image {idx + 1}/{len(request.base64Images)}")
                
                # Step 1: Run OCR to detect text
                ocr_results = await ocr_service.detect_text(base64_image)

                if not ocr_results:
                    logger.warning(f"No text detected in image {idx + 1}")
                    all_results.append([])
                    continue

                # Step 2: Group nearby text regions into complete sentences
                grouped_results = ocr_service.group_text_regions(ocr_results)
                logger.info(f"Grouped {len(ocr_results)} boxes into {len(grouped_results)} text regions")

                # Step 3: Extract texts for translation
                texts_to_translate = [box['text'] for box in grouped_results]

                # Step 4: Translate all texts in batch
                translations = await translation_service.translate_batch(
                    texts_to_translate,
                    request.targetLanguage
                )
                
                # Step 5: Combine grouped results with translations
                text_boxes = []
                translated_boxes = []  # For debug visualization

                for ocr_box, translated_text in zip(grouped_results, translations):
                    # Calculate font size
                    bbox_width = ocr_box['maxX'] - ocr_box['minX']
                    bbox_height = ocr_box['maxY'] - ocr_box['minY']
                    font_size = calculate_font_size(
                        bbox_width,
                        bbox_height,
                        len(translated_text)
                    )

                    # Extract background region
                    background = extract_text_region_background(
                        base64_image,
                        ocr_box['minX'],
                        ocr_box['minY'],
                        ocr_box['maxX'],
                        ocr_box['maxY']
                    )

                    # Detect font colors (using a simple default for now)
                    font_color = "#000000"
                    stroke_color = "#FFFFFF"

                    # Create TextBox
                    text_box = TextBox(
                        ocrText=ocr_box['text'],
                        originalLanguage="",
                        minX=ocr_box['minX'],
                        minY=ocr_box['minY'],
                        maxX=ocr_box['maxX'],
                        maxY=ocr_box['maxY'],
                        background=background,
                        fontHeightPx=font_size,
                        fontColor=font_color,
                        fontStrokeColor=stroke_color,
                        zIndex=1,
                        translatedText=translated_text,
                        subtextBoxes=[]
                    )

                    text_boxes.append(text_box)

                    # For debug visualization
                    translated_boxes.append({
                        **ocr_box,
                        'translation': translated_text
                    })

                # DEBUG: Visualize translated text on image
                try:
                    # Decode image for visualization
                    image_data = base64_image
                    if ',' in image_data and image_data.startswith('data:image'):
                        image_data = image_data.split(',', 1)[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image_np = np.array(image)

                    ocr_service.visualize_translated_text(image_np, translated_boxes)
                except Exception as viz_error:
                    logger.warning(f"Failed to create translation visualization: {viz_error}")

                all_results.append(text_boxes)
                logger.info(f"Successfully processed image {idx + 1} with {len(text_boxes)} text boxes")

            except Exception as e:
                logger.error(f"Error processing image {idx + 1}: {e}")
                # Return empty result for this image
                all_results.append([])

        elapsed_time = time.time() - start_time
        logger.info(f"Translation request completed in {elapsed_time:.2f} seconds")
        return TranslateResponse(images=all_results)
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Translation request failed after {elapsed_time:.2f} seconds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )
