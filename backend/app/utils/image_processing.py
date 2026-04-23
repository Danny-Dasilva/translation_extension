"""Image processing utilities"""
import base64
import io
import logging
from typing import Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def _strip_data_uri(base64_image: str) -> str:
    """Strip the `data:image/...;base64,` prefix if present."""
    if ',' in base64_image and base64_image.startswith('data:image'):
        return base64_image.split(',', 1)[1]
    return base64_image


def decode_base64_to_numpy_fast(b64: str) -> np.ndarray:
    """
    Fast base64 -> BGR numpy array decode using cv2.imdecode.

    Handles PNG/JPEG/WebP transparently (cv2 auto-detects the format).
    Alpha channels (RGBA) are flattened onto a white background.
    Grayscale images are promoted to 3 channels.

    Returns:
        BGR ndarray of shape (H, W, 3), dtype=uint8.
    """
    data = _strip_data_uri(b64)
    raw = base64.b64decode(data)
    buf = np.frombuffer(raw, dtype=np.uint8)

    # IMREAD_UNCHANGED preserves alpha so we can flatten it ourselves.
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("cv2.imdecode failed - invalid or unsupported image data")

    # Normalize channel layout
    if image.ndim == 2:
        # Grayscale (L) -> BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif channels == 2:
            # LA (grayscale + alpha) -> flatten then promote
            gray = image[:, :, 0:1].astype(np.float32)
            alpha = image[:, :, 1:2].astype(np.float32) / 255.0
            flattened = gray * alpha + 255.0 * (1.0 - alpha)
            flattened = np.clip(flattened, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(flattened, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            # BGRA -> flatten onto white background.
            # Fast path: if alpha is fully opaque, skip the blend entirely.
            alpha_ch = image[:, :, 3]
            if alpha_ch.min() == 255:
                image = image[:, :, :3].copy()
            else:
                bgr = image[:, :, :3].astype(np.float32)
                alpha = alpha_ch.astype(np.float32)[:, :, None] / 255.0
                flattened = bgr * alpha + 255.0 * (1.0 - alpha)
                image = np.clip(flattened, 0, 255).astype(np.uint8)
        # else channels == 3 -> already BGR, no-op

    return image


def decode_base64_to_pil(base64_image: str) -> Image.Image:
    """
    Decode base64 image string (with optional data URI prefix) to PIL Image (RGB).

    Wraps the fast cv2 path and converts BGR -> RGB at the boundary so downstream
    PIL callers see the same RGB image they always have.
    """
    bgr = decode_base64_to_numpy_fast(base64_image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def decode_base64_to_numpy(base64_image: str) -> np.ndarray:
    """Decode base64 image string (with optional data URI prefix) to RGB numpy array."""
    bgr = decode_base64_to_numpy_fast(base64_image)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def snap_font_color(
    rgb: Tuple[int, int, int], threshold: int = 20
) -> Tuple[int, int, int]:
    """
    Snap a font color to clean (0,0,0) or (255,255,255) when it's close.

    Matches koharu's behavior where detected font colors snap to cleaner values.

    Args:
        rgb: 3-tuple of ints in [0, 255].
        threshold: How close (per-channel) to 0 or 255 counts as "snap".

    Returns:
        Snapped RGB tuple.
    """
    r, g, b = rgb
    if r <= threshold and g <= threshold and b <= threshold:
        return (0, 0, 0)
    if r >= 255 - threshold and g >= 255 - threshold and b >= 255 - threshold:
        return (255, 255, 255)
    return (int(r), int(g), int(b))


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
            font_rgb: Tuple[int, int, int] = (255, 255, 255)
            stroke_rgb: Tuple[int, int, int] = (0, 0, 0)
        else:
            font_rgb = (0, 0, 0)
            stroke_rgb = (255, 255, 255)

        # Snap to cleaner values (matches koharu's behavior). Safe because
        # pure (0,0,0)/(255,255,255) stay the same; future sampled colors
        # within threshold of those extremes will also snap cleanly.
        font_rgb = snap_font_color(font_rgb)
        stroke_rgb = snap_font_color(stroke_rgb)

        return (
            "#{:02X}{:02X}{:02X}".format(*font_rgb),
            "#{:02X}{:02X}{:02X}".format(*stroke_rgb),
        )

    except Exception as e:
        logger.warning(f"Font color detection failed, using defaults: {e}")
        return "#000000", "#FFFFFF"


def extract_text_region_background(
    base64_image: str,
    minX: int,
    minY: int,
    maxX: int,
    maxY: int,
    preloaded_image: Image.Image | None = None,
) -> str:
    """
    Extract the background image for a text region

    Args:
        base64_image: Full image as base64 string (ignored if preloaded_image given)
        minX, minY, maxX, maxY: Bounding box coordinates
        preloaded_image: Pre-decoded PIL Image to avoid repeated base64 decoding

    Returns:
        Base64-encoded cropped region
    """
    try:
        if preloaded_image is not None:
            image = preloaded_image
        else:
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
