"""Image processing utilities"""
import base64
import io
import logging
from typing import Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

logger = logging.getLogger(__name__)


def calculate_font_size(bbox_width: int, bbox_height: int, text_length: int) -> int:
    """
    Calculate appropriate font size based on bounding box dimensions and text length
    
    Args:
        bbox_width: Width of bounding box in pixels
        bbox_height: Height of bounding box in pixels
        text_length: Length of translated text
    
    Returns:
        Font size in pixels
    """
    if text_length == 0:
        return 20  # Default
    
    # Calculate area and estimate characters per line
    area = bbox_width * bbox_height
    chars_per_line = max(1, bbox_width // 12)  # Rough estimate: 12px per char
    num_lines = max(1, (text_length + chars_per_line - 1) // chars_per_line)
    
    # Calculate font size based on height and number of lines
    font_height = bbox_height // (num_lines * 1.2)  # 1.2 = line height factor
    
    # Clamp to reasonable range
    font_size = max(12, min(int(font_height), 50))
    
    logger.debug(f"Calculated font size {font_size}px for bbox({bbox_width}x{bbox_height}), text_len={text_length}")
    return font_size


def detect_font_colors(image_region: np.ndarray) -> Tuple[str, str]:
    """
    Detect appropriate font and stroke colors based on background
    
    Args:
        image_region: Cropped image region as numpy array
    
    Returns:
        Tuple of (font_color_hex, stroke_color_hex)
    """
    try:
        # Calculate average brightness
        avg_brightness = np.mean(image_region)
        
        # Dark background -> white text, black stroke
        # Light background -> black text, white stroke
        if avg_brightness < 128:
            return "#FFFFFF", "#000000"
        else:
            return "#000000", "#FFFFFF"
    
    except Exception as e:
        logger.warning(f"Font color detection failed, using defaults: {e}")
        return "#000000", "#FFFFFF"


def extract_text_region_background(
    base64_image: str, 
    minX: int, 
    minY: int, 
    maxX: int, 
    maxY: int
) -> str:
    """
    Extract the background image for a text region
    
    Args:
        base64_image: Full image as base64 string
        minX, minY, maxX, maxY: Bounding box coordinates
    
    Returns:
        Base64-encoded cropped region
    """
    try:
        # Decode base64
        if ',' in base64_image and base64_image.startswith('data:image'):
            base64_image = base64_image.split(',', 1)[1]
        
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Crop to bounding box
        cropped = image.crop((minX, minY, maxX, maxY))
        
        # Encode as JPEG
        buffer = io.BytesIO()
        cropped.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        # Return as base64
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"
    
    except Exception as e:
        logger.error(f"Failed to extract background region: {e}")
        return ""


def compress_image(base64_image: str, max_size_mb: float = 2.0) -> str:
    """
    Compress image if it exceeds max size
    
    Args:
        base64_image: Base64-encoded image
        max_size_mb: Maximum size in megabytes
    
    Returns:
        Compressed base64 image
    """
    try:
        # Decode
        if ',' in base64_image and base64_image.startswith('data:image'):
            prefix, data = base64_image.split(',', 1)
        else:
            prefix = "data:image/jpeg;base64"
            data = base64_image
        
        image_bytes = base64.b64decode(data)
        current_size_mb = len(image_bytes) / (1024 * 1024)
        
        # Return original if under limit
        if current_size_mb <= max_size_mb:
            return base64_image
        
        # Load and compress
        image = Image.open(io.BytesIO(image_bytes))
        
        # Calculate scale factor
        scale = (max_size_mb / current_size_mb) ** 0.5
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        # Resize
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save with compression
        buffer = io.BytesIO()
        resized.save(buffer, format='JPEG', quality=80, optimize=True)
        buffer.seek(0)
        
        # Encode
        compressed_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        logger.info(f"Compressed image from {current_size_mb:.2f}MB to ~{max_size_mb}MB")
        return f"{prefix},{compressed_data}"
    
    except Exception as e:
        logger.error(f"Image compression failed: {e}")
        return base64_image
