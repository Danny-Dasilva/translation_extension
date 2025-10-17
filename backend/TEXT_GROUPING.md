# Text Region Grouping

## Problem

OCR detection engines like PaddleOCR often break single sentences into multiple small text regions. When translating manga, this causes loss of context and poor translation quality.

**Example:**
```
Individual boxes:              Should be grouped as:
"お"          → "Oh"          "お人気ある" → "Popular"
"人"          → "Person"
"気"          → "Spirit"
"ある"        → "There is"
```

Without grouping, the translator sees individual characters and produces nonsensical translations. With grouping, the translator sees complete phrases and produces accurate translations.

## Solution

Implemented a **proximity-based text region grouping algorithm** that:
1. Finds text boxes that are spatially close to each other
2. Groups them using connected components approach
3. Sorts grouped boxes in correct reading order (Japanese vertical/horizontal)
4. Merges the text and bounding boxes
5. Sends merged text for translation

## Algorithm Details

### Spatial Proximity Detection

Two boxes are considered "close" if:
- **For vertical text** (height > 1.5 × width):
  - Horizontal gap ≤ 15 pixels (columns are close together)
  - Vertical gap ≤ 15 pixels (text flows top-to-bottom)

- **For horizontal text**:
  - Horizontal gap ≤ 15 pixels (text flows left-to-right)
  - Vertical gap ≤ 7.5 pixels (same line)

### Reading Order Detection

**Vertical Japanese text:**
- Primary sort: Right to left (X coordinate descending)
- Secondary sort: Top to bottom (Y coordinate ascending)

**Horizontal text:**
- Primary sort: Top to bottom (Y coordinate ascending)
- Secondary sort: Left to right (X coordinate ascending)

### Connected Components Grouping

Uses BFS (Breadth-First Search) to find all boxes connected through proximity:
1. Start with an unvisited box
2. Find all boxes close to it
3. Recursively find boxes close to those
4. Group all connected boxes together
5. Repeat for remaining unvisited boxes

## Configuration

Adjust thresholds in `ocr_service.py`:

```python
ocr_service.group_text_regions(
    ocr_boxes,
    vertical_threshold=15.0,    # Max vertical gap in pixels
    horizontal_threshold=15.0   # Max horizontal gap in pixels
)
```

**Increase thresholds** to group more aggressively (merge more regions)
**Decrease thresholds** to group more conservatively (keep regions separate)

## Usage

Grouping is automatically applied in the translation pipeline:

```python
# 1. Detect individual text regions
ocr_results = await ocr_service.detect_text(base64_image)
# Result: [{"text": "お", ...}, {"text": "人", ...}, {"text": "気", ...}]

# 2. Group nearby regions
grouped = ocr_service.group_text_regions(ocr_results)
# Result: [{"text": "お人気", "original_boxes": [...], ...}]

# 3. Translate grouped text
translations = await translation_service.translate_batch([g['text'] for g in grouped])
# Result: ["Popular"]
```

## Integration

The grouping is integrated into the `/translate` endpoint:

```
/translate endpoint flow:
├─ OCR Detection (PaddleOCR + manga-ocr)
├─ **Text Grouping** ← NEW
├─ Translation (Gemini)
└─ Response formatting
```

## Examples

### Example 1: Vertical Speech Bubble

**Before grouping:**
```
Box 1: "お" (0, 100) -> (20, 120)
Box 2: "人" (0, 125) -> (20, 145)  ← 5px gap
Box 3: "気" (0, 150) -> (20, 170)  ← 5px gap
```

**After grouping:**
```
Merged: "お人気" (0, 100) -> (20, 170)
```

### Example 2: Separate Speech Bubbles

**Before grouping:**
```
Box 1: "こんにちは" (50, 100) -> (150, 120)
Box 2: "さよなら"   (300, 100) -> (400, 120)  ← 150px gap (too far!)
```

**After grouping:**
```
Group 1: "こんにちは" (50, 100) -> (150, 120)
Group 2: "さよなら"   (300, 100) -> (400, 120)
```

## Grouped Box Format

```python
{
    'text': str,              # Merged text from all boxes
    'minX': int,              # Min X of all boxes
    'minY': int,              # Min Y of all boxes
    'maxX': int,              # Max X of all boxes
    'maxY': int,              # Max Y of all boxes
    'confidence': float,      # Average confidence
    'original_boxes': [...],  # List of original OCR boxes
    'group_size': int         # Number of boxes merged
}
```

## Debug Information

Grouping logs show the consolidation:
```
[INFO] Successfully recognized 10 text regions
[INFO] Grouping 10 text regions...
[INFO] Grouped into 3 text regions (from 10 original boxes)
```

## Performance

- **Algorithmic complexity**: O(n²) where n = number of text boxes
- **Typical case**: <50 boxes per manga page → <2500 comparisons
- **Overhead**: Negligible (~1-2ms for typical manga page)

## Troubleshooting

**Problem**: Too much grouping (entire page becomes one region)

**Solution**: Decrease thresholds (try 10px or 8px)

**Problem**: Not enough grouping (sentences still fragmented)

**Solution**: Increase thresholds (try 20px or 25px)

**Problem**: Wrong reading order

**Solution**: Check if text orientation detection is working correctly. Vertical text should have height > 1.5 × width.

## Future Enhancements

Potential improvements:
- [ ] Machine learning-based grouping (train on labeled manga data)
- [ ] Speech bubble detection to group only within same bubble
- [ ] Adaptive thresholds based on text density
- [ ] Direction vector analysis for better reading order
- [ ] Column detection for multi-column vertical text

## References

- Research paper: "Unconstrained Text Detection in Manga" (arXiv:2009.04042)
- Stack Overflow: "Merging regions in MSER for identifying text lines in OCR"
- manga-image-translator repository (text grouping inspiration)
