# Translation Visualization Feature

## Overview

Added automatic debug visualization that shows translated English text overlaid on manga images. This helps you preview what the final translation will look like before rendering it in the browser extension.

## What It Does

When you send a manga image to the `/translate` endpoint, the system now automatically generates two debug images:

### 1. OCR Detection Visualization (`ocr_debug_{timestamp}.jpg`)
- **Red bounding boxes** around detected Japanese text
- **Text labels** showing recognized Japanese characters
- **Corner markers** (green = top corners, blue = bottom corners)
- **Numbered regions** for easy reference

**Purpose**: Verify that all text regions are detected correctly and Japanese text is recognized accurately.

### 2. Translation Visualization (`translation_debug_{timestamp}.jpg`)
- **White-filled boxes** replacing Japanese text regions
- **English translations** overlaid in each box
- **Word-wrapped text** that fits within the original text bounds
- **Centered alignment** for readability
- **Text shadow** for better contrast

**Purpose**: Preview what the translated manga will look like with English text.

## How It Works

### Pipeline Flow

```
1. User sends manga image (base64) to /translate endpoint
   ↓
2. PaddleOCR detects text regions (bounding boxes)
   ↓
3. manga-ocr recognizes Japanese text in each region
   ↓
4. [DEBUG] Save ocr_debug_*.jpg with detected text
   ↓
5. Gemini translates Japanese text to English
   ↓
6. [DEBUG] Save translation_debug_*.jpg with English overlay
   ↓
7. Return JSON response with translations to extension
```

### Code Integration

**OCR Service** (`app/services/ocr_service.py`):
- `_visualize_ocr_boxes()` - Creates OCR detection visualization
- `visualize_translated_text()` - Creates translation visualization

**Translation Router** (`app/routers/translate.py`):
- Calls `visualize_translated_text()` after translation completes
- Handles image decoding for visualization
- Logs warnings if visualization fails (non-blocking)

## Output Location

All debug images are saved to:
```
backend/debug_output/
├── ocr_debug_1760616182338.jpg
├── translation_debug_1760616396608.jpg
└── ... (timestamped files)
```

## Text Rendering Features

### Word Wrapping
- Automatically breaks long translations into multiple lines
- Respects bounding box width
- Adds 10px padding for readability

### Text Centering
- Horizontally centered within each box
- Vertically centered within available space
- Falls back to top-left if text too long

### Text Shadow
- Light gray shadow (1px offset) for better contrast
- Black text on white background
- Ensures readability over manga artwork

### Font Fallback
```python
Primary: DejaVuSans.ttf (14px)
Fallback: System default font
```

## Testing

Use the test script to verify the visualization:

```bash
cd backend
uv run python test_translation_viz.py path/to/manga.jpg
```

This will:
1. Detect Japanese text
2. Translate to English
3. Generate both debug visualizations
4. Show translation results in terminal

## Example Output

**Terminal Output:**
```
Region 1:
  Japanese: ついやあ!!
  English:  No way--
  BBox: (126, 0) -> (247, 235)

Region 2:
  Japanese: 顔はなんなんだよ...
  English:  What's with your face...
  BBox: (230, 0) -> (347, 231)
```

**Visual Output:**
- `ocr_debug_*.jpg` - Shows "ついやあ!!" in red box with label
- `translation_debug_*.jpg` - Shows "No way--" in white box, centered

## Customization

### Adjust Text Appearance

Edit `ocr_service.py` line 280-285:

```python
# Change font size
font_medium = ImageFont.truetype("...", 16)  # Increase from 14

# Change colors
fill='white'        # Background color
outline='lightgray' # Border color
fill='black'        # Text color
fill='lightgray'    # Shadow color
```

### Adjust Word Wrapping

Edit `ocr_service.py` line 321:

```python
if text_width <= box_width - 10:  # Change padding from 10px
```

### Adjust Line Spacing

Edit `ocr_service.py` line 336:

```python
line_height = 20  # Increase from 18 for more spacing
```

## Troubleshooting

**Problem**: No translation_debug_*.jpg files generated

**Solution**: Check logs for visualization errors:
```bash
grep "Failed to create translation visualization" backend.log
```

**Problem**: Text appears cut off or overlaps box

**Solution**: Decrease font size or adjust word wrapping width in `ocr_service.py`

**Problem**: Text is unreadable

**Solution**:
- Increase font size
- Adjust shadow offset/color
- Try white text with black background instead

## Performance Impact

- **Minimal**: Visualization runs in background after response sent
- **Non-blocking**: Errors in visualization don't affect API response
- **Disk usage**: ~500KB per visualization image (JPEG quality 95)

## Future Enhancements

Potential improvements:
- [ ] Configurable font colors based on original manga colors
- [ ] Preserve vertical text orientation from original
- [ ] Adjustable font size based on box dimensions
- [ ] Optional: Save visualization to user-specified directory
- [ ] Optional: Return visualization images in API response

## References

- PIL/Pillow ImageDraw: https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
- Word wrapping algorithm: Custom implementation based on text width measurement
