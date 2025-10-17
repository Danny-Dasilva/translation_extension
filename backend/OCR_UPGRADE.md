# OCR System Upgrade

## Summary

Successfully upgraded from RapidOCR to a hybrid PaddleOCR + manga-ocr pipeline for significantly improved Japanese manga text detection and recognition.

## Changes Made

### 1. Dependencies Updated (`pyproject.toml`)

**Removed:**
- `rapidocr-onnxruntime` - Generic OCR, poor manga support

**Added:**
- `paddleocr>=2.9.2` - State-of-the-art text detection
- `paddlepaddle>=3.0.0` - PaddleOCR framework
- `manga-ocr>=0.1.11` - Specialized Japanese manga text recognition
- `torch>=2.0.0` - PyTorch for manga-ocr
- `transformers>=4.30.0` - Transformers for manga-ocr

### 2. OCR Service Rewritten (`app/services/ocr_service.py`)

**New Two-Stage Pipeline:**

1. **Stage 1: Text Detection** (PaddleOCR)
   - Detects text region bounding boxes
   - Optimized parameters for Japanese text:
     - `lang='japan'` - Japanese language model
     - `text_det_thresh=0.3` - Lower threshold for better detection
     - `text_det_box_thresh=0.5` - Box threshold
     - `text_det_unclip_ratio=1.8` - Expanded text regions
     - `use_textline_orientation=True` - Handles rotated text

2. **Stage 2: Text Recognition** (manga-ocr)
   - Specialized Japanese manga character recognition
   - Trained specifically on manga fonts and styles
   - Handles furigana, vertical text, and manga-specific layouts

### 3. Key Improvements

**Before (RapidOCR):**
- ❌ Over-segmented text into tiny fragments
- ❌ Missed many text regions completely
- ❌ Poor Japanese character recognition
- ❌ Individual characters instead of complete phrases

**After (PaddleOCR + manga-ocr):**
- ✅ Detects complete text regions (speech bubbles)
- ✅ Much better coverage - finds more text
- ✅ Accurate Japanese character recognition
- ✅ Preserves full context for translation

## Testing

Test scripts created:

1. **OCR Detection Test** (`test_new_ocr.py`):
```bash
uv run python test_new_ocr.py path/to/manga/image.jpg
```
Tests OCR detection and recognition, outputs detected Japanese text.

2. **Translation Visualization Test** (`test_translation_viz.py`):
```bash
uv run python test_translation_viz.py path/to/manga/image.jpg
```
Tests full pipeline: OCR → Translation → Visualization.

Results show proper detection of complete text regions with accurate Japanese recognition.

## Performance Notes

- First run downloads models automatically (~200MB total)
- Models are cached in `~/.paddlex/official_models/`
- manga-ocr uses CUDA if available for faster inference
- Expected initialization time: ~5-10 seconds (first run only)

## Debug Visualizations

The system now automatically generates two debug images for each translation request:

1. **`ocr_debug_{timestamp}.jpg`**:
   - Shows detected Japanese text regions with red bounding boxes
   - Displays recognized Japanese text above each box
   - Useful for verifying OCR detection accuracy

2. **`translation_debug_{timestamp}.jpg`**:
   - Shows English translations overlaid on the manga
   - Japanese text regions are filled with white background
   - English text is word-wrapped and centered in each region
   - Useful for previewing final translation appearance

Both images are saved to `backend/debug_output/` directory.

## Next Steps

To use with your manga translation extension:

1. Install dependencies: `uv sync` (already done)
2. Restart your backend server
3. Test with real manga pages
4. Check `debug_output/` folder for visualization previews
5. Adjust detection thresholds if needed (in `ocr_service.py`)

## Bug Fix: Coordinate Mismatch (Resolved)

### Issue
Initial implementation had bounding boxes appearing in wrong locations on manga pages because PaddleOCR's document preprocessing (orientation detection, unwarping) was transforming the image internally, causing coordinate space mismatch.

### Fix Applied
Disabled document preprocessing parameters:
```python
use_doc_orientation_classify=False  # Manga pages already properly oriented
use_doc_unwarping=False            # Not needed for flat digital images
```

This ensures coordinates from PaddleOCR match the original image space, enabling correct text cropping and visualization.

## Troubleshooting

If GPU memory issues occur, models are configured to use CPU by default. To enable GPU:
- PaddleOCR: Set environment variable `CUDA_VISIBLE_DEVICES=0`
- manga-ocr: Automatically uses CUDA if available

**If bounding boxes appear in wrong locations:**
- Ensure `use_doc_orientation_classify=False` and `use_doc_unwarping=False` in PaddleOCR init
- These preprocessing steps transform the image, causing coordinate mismatches

## References

- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- manga-ocr: https://github.com/kha-white/manga-ocr
