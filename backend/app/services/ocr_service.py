"""OCR service using PaddleOCR for unified detection and recognition"""
import base64
import io
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

# Create debug directory
DEBUG_DIR = Path(__file__).parent.parent.parent / "debug_output"
DEBUG_DIR.mkdir(exist_ok=True)


class OCRService:
    """Service for detecting and recognizing text using PaddleOCR unified pipeline"""

    def __init__(self):
        """Initialize PaddleOCR pipeline with recognition enabled"""
        try:
            # Initialize PaddleOCR - matching run.py configuration for consistent coordinates
            self.pipeline = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
            logger.info("PaddleOCR pipeline initialized successfully with detection + recognition")

        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR pipeline: {e}")
            raise

    async def detect_text(self, base64_image: str) -> List[Dict[str, Any]]:
        """
        Detect text and bounding boxes in a base64-encoded image using PaddleOCR.

        This replaces the previous two-stage pipeline:
        - Old: PaddleOCR detection only → crop regions → manga-ocr recognition
        - New: Single PaddleOCR call with built-in detection + recognition

        Args:
            base64_image: Base64-encoded image string (with or without data URI prefix)

        Returns:
            List of dictionaries containing:
            - text: detected Japanese text
            - minX, minY, maxX, maxY: bounding box coordinates
            - confidence: recognition confidence (0-1)
        """
        try:
            # Decode base64 image
            image_np = self._decode_base64_image(base64_image)

            orig_height, orig_width = image_np.shape[:2]
            logger.info(f"Original image dimensions: {orig_width}x{orig_height}")

            # Save input image to temp file (predict() requires file path, not numpy array)
            timestamp = int(time.time() * 1000)
            input_image_path = DEBUG_DIR / f"input_image_{timestamp}.jpg"
            Image.fromarray(image_np).save(input_image_path, 'JPEG', quality=95)
            logger.info(f"Saved input image to: {input_image_path}")

            # Run PaddleOCR using predict() method (matches run.py approach)
            # predict() returns correct dt_polys coordinates, unlike ocr()
            logger.info("Running PaddleOCR predict() for detection and recognition...")
            result = self.pipeline.predict(str(input_image_path))
            logger.info(f"PaddleOCR processing complete for image: {orig_width}x{orig_height}")
            # DEBUG: Log raw result structure
            logger.info(f"PaddleOCR raw result type: {type(result)}")
            logger.info(f"PaddleOCR raw result length: {len(result) if result else 0}")
            if result and len(result) > 0:
                logger.info(f"First result element type: {type(result[0])}")

            ocr_boxes = []

            # Parse PaddleOCR predict() output - simplified approach matching run.py
            if result and len(result) > 0:
                # Get first result (single image)
                page = result[0]

                # Access the 'res' dictionary within json (matching run.py lines 38-42)
                res_dict = page.json.get('res', {})

                # Extract dt_polys (detection polygons), rec_texts, and rec_scores
                dt_polys = res_dict.get('dt_polys', [])
                rec_texts = res_dict.get('rec_texts', [])
                rec_scores = res_dict.get('rec_scores', [])

                logger.info(f"PaddleOCR detected {len(dt_polys)} text boxes with {len(rec_texts)} recognized texts")

                # Process each detected text box
                for idx, (poly, text, score) in enumerate(zip(dt_polys, rec_texts, rec_scores)):
                    # Skip empty text
                    if not text or not text.strip():
                        logger.debug(f"Skipping box {idx + 1} due to empty text")
                        continue

                    # Convert polygon to bounding box (minX, minY, maxX, maxY)
                    poly_np = np.array(poly)
                    minX = int(np.min(poly_np[:, 0]))
                    minY = int(np.min(poly_np[:, 1]))
                    maxX = int(np.max(poly_np[:, 0]))
                    maxY = int(np.max(poly_np[:, 1]))

                    # Clip to image bounds
                    minX = max(0, min(minX, orig_width - 1))
                    minY = max(0, min(minY, orig_height - 1))
                    maxX = max(0, min(maxX, orig_width - 1))
                    maxY = max(0, min(maxY, orig_height - 1))

                    # Filter sound effects: huge boxes with small text (area/char > 8000)
                    box_area = (maxX - minX) * (maxY - minY)
                    text_len = len(text.strip())
                    area_per_char = box_area / text_len if text_len > 0 else 0

                    if area_per_char > 6000:
                        logger.debug(f"Skipping sound effect box {idx + 1}: '{text}' (area/char: {area_per_char:.0f})")
                        continue

                    entry = {
                        'text': text.strip(),
                        'confidence': float(score),
                        'minX': minX,
                        'minY': minY,
                        'maxX': maxX,
                        'maxY': maxY,
                    }
                    ocr_boxes.append(entry)
                    logger.info(
                        f"Added text box {len(ocr_boxes)}: '{text[:30]}...' "
                        f"(confidence: {score:.2f}, bounds: {minX},{minY} to {maxX},{maxY})"
                    )

            else:
                logger.warning(f"PaddleOCR returned empty or invalid result: {result}")

            logger.info(f"PaddleOCR detected {len(ocr_boxes)} text regions total")

            # DEBUG: Visualize OCR boxes
            if ocr_boxes:
                self._visualize_ocr_boxes(image_np, ocr_boxes)
            else:
                logger.warning("No text detected in image")

            return ocr_boxes

        except Exception as e:
            logger.error(f"OCR detection failed: {e}", exc_info=True)
            raise ValueError(f"Failed to detect text: {str(e)}")

    def _decode_base64_image(self, base64_image: str) -> np.ndarray:
        """
        Decode base64 image string to numpy array

        Args:
            base64_image: Base64 string (with or without data URI prefix)

        Returns:
            Numpy array representing the image
        """
        try:
            # Remove data URI prefix if present
            if ',' in base64_image and base64_image.startswith('data:image'):
                base64_image = base64_image.split(',', 1)[1]

            # Decode base64
            image_bytes = base64.b64decode(base64_image)

            # Load as PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to numpy array
            return np.array(image)

        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise ValueError(f"Invalid base64 image data: {str(e)}")

    def extract_text_region(self, image_np: np.ndarray, bbox: Dict[str, int]) -> Optional[str]:
        """
        Extract a text region from the image as base64

        Args:
            image_np: Image as numpy array
            bbox: Bounding box dict with minX, minY, maxX, maxY

        Returns:
            Base64-encoded cropped region
        """
        try:
            # Crop region
            cropped = image_np[
                bbox['minY']:bbox['maxY'],
                bbox['minX']:bbox['maxX']
            ]

            # Convert to PIL Image
            cropped_img = Image.fromarray(cropped)

            # Encode as JPEG
            buffer = io.BytesIO()
            cropped_img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)

            # Encode as base64
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_str}"

        except Exception as e:
            logger.error(f"Failed to extract text region: {e}")
            return None

    def _visualize_ocr_boxes(self, image_np: np.ndarray, ocr_boxes: List[Dict[str, Any]]) -> None:
        """
        Visualize OCR bounding boxes on the image for debugging

        Args:
            image_np: Original image as numpy array
            ocr_boxes: List of detected text boxes with coordinates
        """
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_np)
            draw = ImageDraw.Draw(image)

            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()

            # Draw each bounding box
            for idx, box in enumerate(ocr_boxes):
                minX = box['minX']
                minY = box['minY']
                maxX = box['maxX']
                maxY = box['maxY']
                text = box['text']
                confidence = box.get('confidence', 0)

                # Draw rectangle (red color)
                draw.rectangle(
                    [(minX, minY), (maxX, maxY)],
                    outline='red',
                    width=3
                )

                # Draw text label with background
                label = f"{idx+1}: {text[:30]}..."

                # Get text bounding box for background
                bbox = draw.textbbox((minX, minY - 22), label, font=font)

                # Draw background for text
                draw.rectangle(bbox, fill='red')

                # Draw text
                draw.text(
                    (minX, minY - 22),
                    label,
                    fill='white',
                    font=font
                )

                # Draw corner markers
                corner_size = 10
                # Top-left
                draw.line([(minX, minY), (minX + corner_size, minY)], fill='green', width=2)
                draw.line([(minX, minY), (minX, minY + corner_size)], fill='green', width=2)
                # Top-right
                draw.line([(maxX, minY), (maxX - corner_size, minY)], fill='green', width=2)
                draw.line([(maxX, minY), (maxX, minY + corner_size)], fill='green', width=2)
                # Bottom-left
                draw.line([(minX, maxY), (minX + corner_size, maxY)], fill='blue', width=2)
                draw.line([(minX, maxY), (minX, maxY - corner_size)], fill='blue', width=2)
                # Bottom-right
                draw.line([(maxX, maxY), (maxX - corner_size, maxY)], fill='blue', width=2)
                draw.line([(maxX, maxY), (maxX, maxY - corner_size)], fill='blue', width=2)

            # Save debug image
            timestamp = int(time.time() * 1000)
            output_path = DEBUG_DIR / f"ocr_debug_{timestamp}.jpg"
            image.save(output_path, 'JPEG', quality=95)

            logger.info(f"OCR visualization saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to visualize OCR boxes: {e}")

    def visualize_translated_text(self, image_np: np.ndarray, translated_boxes: List[Dict[str, Any]]) -> None:
        """
        Visualize translated English text overlaid on the manga image

        Args:
            image_np: Original image as numpy array
            translated_boxes: List of boxes with 'translation' field added
        """
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_np)
            draw = ImageDraw.Draw(image)

            # Try to load fonts
            try:
                # Larger font for translations
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
                font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()

            # Draw translated text in each region
            for idx, box in enumerate(translated_boxes):
                minX = box['minX']
                minY = box['minY']
                maxX = box['maxX']
                maxY = box['maxY']
                translation = box.get('translation', '')

                if not translation:
                    continue

                # Fill the text region with white background
                draw.rectangle(
                    [(minX, minY), (maxX, maxY)],
                    fill='white',
                    outline='lightgray',
                    width=2
                )

                # Calculate text positioning
                box_width = maxX - minX
                box_height = maxY - minY

                # Word wrap the translation to fit the box
                words = translation.split()
                lines = []
                current_line = []

                for word in words:
                    test_line = ' '.join(current_line + [word])
                    bbox = draw.textbbox((0, 0), test_line, font=font_medium)
                    text_width = bbox[2] - bbox[0]

                    if text_width <= box_width - 10:  # 10px padding
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            # Word too long, add it anyway
                            lines.append(word)
                            current_line = []

                if current_line:
                    lines.append(' '.join(current_line))

                # Calculate starting Y position to center text vertically
                line_height = 18
                total_text_height = len(lines) * line_height
                start_y = minY + (box_height - total_text_height) // 2

                # Ensure text doesn't go above the box
                if start_y < minY:
                    start_y = minY + 5

                # Draw each line of text
                for i, line in enumerate(lines):
                    # Center text horizontally
                    bbox = draw.textbbox((0, 0), line, font=font_medium)
                    text_width = bbox[2] - bbox[0]
                    text_x = minX + (box_width - text_width) // 2
                    text_y = start_y + (i * line_height)

                    # Ensure text stays within box bounds
                    if text_x < minX:
                        text_x = minX + 5
                    if text_y + line_height > maxY:
                        break  # Stop if we run out of vertical space

                    # Draw text with shadow for better readability
                    # Shadow
                    draw.text(
                        (text_x + 1, text_y + 1),
                        line,
                        fill='lightgray',
                        font=font_medium
                    )
                    # Main text
                    draw.text(
                        (text_x, text_y),
                        line,
                        fill='black',
                        font=font_medium
                    )

            # Save debug image
            timestamp = int(time.time() * 1000)
            output_path = DEBUG_DIR / f"translation_debug_{timestamp}.jpg"
            image.save(output_path, 'JPEG', quality=95)

            logger.info(f"Translation visualization saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to visualize translated text: {e}")
