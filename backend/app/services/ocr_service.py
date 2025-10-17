"""OCR service using PaddleOCR for detection and manga-ocr for recognition"""
import base64
import io
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from paddleocr import PaddleOCR
from manga_ocr import MangaOcr

logger = logging.getLogger(__name__)

# Create debug directory
DEBUG_DIR = Path(__file__).parent.parent.parent / "debug_output"
DEBUG_DIR.mkdir(exist_ok=True)


class OCRService:
    """Service for detecting text in images using PaddleOCR + manga-ocr"""

    def __init__(self):
        """Initialize PaddleOCR for detection and manga-ocr for recognition"""
        try:
            # Initialize PaddleOCR for text detection only (Japanese optimized)
            self.detector = PaddleOCR(
                use_doc_orientation_classify=False,  # Disable document orientation (manga already oriented)
                use_doc_unwarping=False,  # Disable unwarping (not needed for flat manga images)
                use_textline_orientation=True,  # Enable text angle classification for rotated text
                lang='japan',  # Japanese language
                text_det_thresh=0.3,  # Lower threshold for better detection
                text_det_box_thresh=0.5,  # Box threshold
                text_det_unclip_ratio=1.8,  # Unclip ratio for text regions
            )
            logger.info("PaddleOCR detector initialized successfully")

            # Initialize manga-ocr for Japanese text recognition
            self.recognizer = MangaOcr()
            logger.info("manga-ocr recognizer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OCR engines: {e}")
            raise

    async def detect_text(self, base64_image: str) -> List[Dict[str, Any]]:
        """
        Detect text and bounding boxes in a base64-encoded image using two-stage pipeline:
        1. PaddleOCR detects text regions (bounding boxes)
        2. manga-ocr recognizes Japanese text in each region

        Args:
            base64_image: Base64-encoded image string (with or without data URI prefix)

        Returns:
            List of dictionaries containing:
            - text: detected Japanese text
            - minX, minY, maxX, maxY: bounding box coordinates
            - confidence: detection confidence (0-1)
        """
        try:
            # Decode base64 image
            image_np = self._decode_base64_image(base64_image)
            image_pil = Image.fromarray(image_np)

            # Stage 1: Detect text regions using PaddleOCR
            logger.info("Running PaddleOCR text detection...")
            detection_result = self.detector.predict(image_np)

            ocr_boxes = []
            if detection_result and len(detection_result) > 0:
                result_dict = detection_result[0]
                dt_polys = result_dict.get('dt_polys', [])

                logger.info(f"PaddleOCR detected {len(dt_polys)} text regions")

                # Stage 2: Recognize text in each detected region using manga-ocr
                for idx, bbox_points in enumerate(dt_polys):
                    # Convert bbox to minX, minY, maxX, maxY format
                    x_coords = [point[0] for point in bbox_points]
                    y_coords = [point[1] for point in bbox_points]
                    minX = int(min(x_coords))
                    minY = int(min(y_coords))
                    maxX = int(max(x_coords))
                    maxY = int(max(y_coords))

                    # Crop the text region
                    try:
                        cropped_region = image_pil.crop((minX, minY, maxX, maxY))

                        # Recognize text using manga-ocr
                        recognized_text = self.recognizer(cropped_region)

                        # Skip empty or very short detections (likely noise)
                        if recognized_text and len(recognized_text.strip()) > 0:
                            ocr_boxes.append({
                                'text': recognized_text,
                                'minX': minX,
                                'minY': minY,
                                'maxX': maxX,
                                'maxY': maxY,
                                'confidence': 0.9  # manga-ocr doesn't provide confidence scores
                            })
                            logger.debug(f"Region {idx+1}: '{recognized_text}'")

                    except Exception as e:
                        logger.warning(f"Failed to recognize text in region {idx+1}: {e}")
                        continue

                logger.info(f"Successfully recognized {len(ocr_boxes)} text regions")

                # DEBUG: Visualize OCR boxes
                self._visualize_ocr_boxes(image_np, ocr_boxes)
            else:
                logger.warning("No text detected in image")

            return ocr_boxes

        except Exception as e:
            logger.error(f"OCR detection failed: {e}", exc_info=True)
            raise ValueError(f"Failed to detect text: {str(e)}")

    def group_text_regions(
        self,
        ocr_boxes: List[Dict[str, Any]],
        vertical_threshold: float = 15.0,
        horizontal_threshold: float = 15.0
    ) -> List[Dict[str, Any]]:
        """
        Group nearby text regions into complete sentences/paragraphs.

        Japanese manga typically has vertical text in speech bubbles where multiple
        small text regions should be combined into one sentence before translation.

        Args:
            ocr_boxes: List of individual OCR detection boxes
            vertical_threshold: Max vertical distance (pixels) to group boxes
            horizontal_threshold: Max horizontal distance (pixels) to group boxes

        Returns:
            List of grouped text regions with:
            - text: combined text from all boxes in group
            - minX, minY, maxX, maxY: merged bounding box
            - original_boxes: list of original boxes in this group
            - confidence: average confidence
        """
        if not ocr_boxes:
            return []

        try:
            logger.info(f"Grouping {len(ocr_boxes)} text regions...")

            # Create list of boxes with indices
            boxes = [(i, box) for i, box in enumerate(ocr_boxes)]
            groups = []
            visited = set()

            def get_box_center(box):
                """Get center point of box"""
                cx = (box['minX'] + box['maxX']) / 2
                cy = (box['minY'] + box['maxY']) / 2
                return cx, cy

            def boxes_are_close(box1, box2):
                """Check if two boxes are close enough to be in same group"""
                # Get box dimensions
                b1_width = box1['maxX'] - box1['minX']
                b1_height = box1['maxY'] - box1['minY']
                b2_width = box2['maxX'] - box2['minX']
                b2_height = box2['maxY'] - box2['minY']

                # Calculate gaps between boxes
                # Horizontal gap: distance between boxes in X direction
                if box1['maxX'] < box2['minX']:
                    h_gap = box2['minX'] - box1['maxX']
                elif box2['maxX'] < box1['minX']:
                    h_gap = box1['minX'] - box2['maxX']
                else:
                    h_gap = 0  # Boxes overlap horizontally

                # Vertical gap: distance between boxes in Y direction
                if box1['maxY'] < box2['minY']:
                    v_gap = box2['minY'] - box1['maxY']
                elif box2['maxY'] < box1['minY']:
                    v_gap = box1['minY'] - box2['maxY']
                else:
                    v_gap = 0  # Boxes overlap vertically

                # Detect text orientation (vertical vs horizontal)
                avg_height = (b1_height + b2_height) / 2
                avg_width = (b1_width + b2_width) / 2
                is_vertical = avg_height > avg_width * 1.5  # Text is vertical if height >> width

                # For vertical text (typical in manga):
                # - Allow small horizontal gap (text columns are close)
                # - Allow moderate vertical gap (text flows top to bottom)
                if is_vertical:
                    return h_gap <= horizontal_threshold and v_gap <= vertical_threshold
                else:
                    # For horizontal text:
                    # - Allow moderate horizontal gap (text flows left to right)
                    # - Allow small vertical gap (same line)
                    return h_gap <= vertical_threshold and v_gap <= horizontal_threshold / 2

            def get_reading_order(group_boxes):
                """
                Sort boxes in reading order.
                For vertical Japanese: top-to-bottom, right-to-left columns
                For horizontal text: left-to-right, top-to-bottom
                """
                if not group_boxes:
                    return []

                # Detect if text is vertical or horizontal
                heights = [b['maxY'] - b['minY'] for _, b in group_boxes]
                widths = [b['maxX'] - b['minX'] for _, b in group_boxes]
                avg_height = sum(heights) / len(heights)
                avg_width = sum(widths) / len(widths)
                is_vertical = avg_height > avg_width * 1.5

                if is_vertical:
                    # Vertical text: sort by X (right to left), then Y (top to bottom)
                    return sorted(group_boxes, key=lambda x: (-x[1]['minX'], x[1]['minY']))
                else:
                    # Horizontal text: sort by Y (top to bottom), then X (left to right)
                    return sorted(group_boxes, key=lambda x: (x[1]['minY'], x[1]['minX']))

            # Group boxes using connected components approach
            for idx, box in boxes:
                if idx in visited:
                    continue

                # Start new group with this box
                group = [(idx, box)]
                visited.add(idx)

                # Find all connected boxes
                queue = [(idx, box)]
                while queue:
                    current_idx, current_box = queue.pop(0)

                    # Check all unvisited boxes
                    for other_idx, other_box in boxes:
                        if other_idx in visited:
                            continue

                        if boxes_are_close(current_box, other_box):
                            group.append((other_idx, other_box))
                            visited.add(other_idx)
                            queue.append((other_idx, other_box))

                groups.append(group)

            # Create merged boxes for each group
            merged_boxes = []
            for group in groups:
                # Sort boxes in reading order
                sorted_group = get_reading_order(group)

                # Merge text
                merged_text = ''.join([box['text'] for _, box in sorted_group])

                # Merge bounding boxes
                all_boxes = [box for _, box in sorted_group]
                merged_minX = min(b['minX'] for b in all_boxes)
                merged_minY = min(b['minY'] for b in all_boxes)
                merged_maxX = max(b['maxX'] for b in all_boxes)
                merged_maxY = max(b['maxY'] for b in all_boxes)

                # Average confidence
                avg_confidence = sum(b['confidence'] for b in all_boxes) / len(all_boxes)

                merged_boxes.append({
                    'text': merged_text,
                    'minX': merged_minX,
                    'minY': merged_minY,
                    'maxX': merged_maxX,
                    'maxY': merged_maxY,
                    'confidence': avg_confidence,
                    'original_boxes': all_boxes,
                    'group_size': len(all_boxes)
                })

            logger.info(f"Grouped into {len(merged_boxes)} text regions (from {len(ocr_boxes)} original boxes)")

            return merged_boxes

        except Exception as e:
            logger.error(f"Text grouping failed: {e}", exc_info=True)
            # Return original boxes if grouping fails
            return ocr_boxes

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
            # timestamp = int(time.time() * 1000)
            # output_path = DEBUG_DIR / f"ocr_debug_{timestamp}.jpg"
            # image.save(output_path, 'JPEG', quality=95)

            # logger.info(f"OCR visualization saved to: {output_path}")

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
            # timestamp = int(time.time() * 1000)
            # output_path = DEBUG_DIR / f"translation_debug_{timestamp}.jpg"
            # image.save(output_path, 'JPEG', quality=95)

            # logger.info(f"Translation visualization saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to visualize translated text: {e}")
