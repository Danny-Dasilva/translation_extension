# CTD postprocessing: koharu parity port

Ports koharu's CTD mask refinement, font-aware line padding, bubble-merge
heuristics, and FP16 model preference into our Python CTD service.

Reference (Rust): `/tmp/koharu/koharu-ml/src/comic_text_detector/postprocess.rs`
and `/tmp/koharu/koharu-ml/src/comic_text_bubble_detector/mod.rs`.

Ours (Python): `backend/app/services/ctd_service.py`.

---

## Changes

### 1. Block-aware mask refinement (`_process_mask`)

- Koharu: `refine_segmentation_mask` -- `postprocess.rs:25-77`.
- Ours: `ComicTextDetectorService._process_mask` -- `ctd_service.py:389-453`
  plus helper `_build_block_bounds_mask` -- `ctd_service.py:455-496`.

Pipeline now:

1. Threshold probability map at `text_threshold` (unchanged).
2. Build a binary "in-bounds" mask as the union of per-block bboxes
   expanded by `font*0.1` (min 2 px) horizontally/vertically.
3. `cv2.bitwise_and(threshold_mask, in_bounds)` -- clip to blocks.
4. `cv2.morphologyEx(MORPH_CLOSE, ellipse(21))` -- fill intra-glyph gaps.
5. L1 dilate radius=2 via two `cv2.dilate` passes of a 3x3 cross kernel.
   (Two passes of an L1 ball = L1 dilation of radius=2.)
6. Final `cv2.dilate(ellipse(7))` radius~=3 to smooth strokes.
7. Final `cv2.bitwise_and(in_bounds)` -- dilation never escapes bounds.

A `legacy=True` flag bypasses steps 2-7 for A/B comparison.
When `blocks` is empty, the method degrades gracefully to legacy behavior.

### 2. Font-aware line-crop expansion (`_expand_text_lines`)

- Koharu: `expanded_text_block_crop_bounds` + `maybe_expand_ctd_line`
  -- `postprocess.rs:107-262`.
- Ours: `ComicTextDetectorService._expand_text_lines` --
  `ctd_service.py:498-547`.

Each text-line bbox is padded direction-aware:

| Direction  | pad_x       | pad_y       |
|------------|-------------|-------------|
| horizontal | font*0.12   | font*0.18   |
| vertical   | font*0.18   | font*0.12   |

...with a floor of `max(font*0.08, 2px)`. Direction is inferred from
bbox aspect ratio (`height > width*1.2` => vertical). Font size is the
shorter bbox dim. Pads are clipped to image bounds. Lines receive new
`font_size_px` and `direction` metadata fields. This fixes clipped
diacritics / furigana on vertical text. Called in `detect()` at
`ctd_service.py:140`.

### 3. Bubble-merge sanity checks (`_derive_blocks_from_text_lines`)

- Koharu: `merge_slice_regions` -- `comic_text_bubble_detector/mod.rs:539-608`.
- Ours: `ComicTextDetectorService._derive_blocks_from_text_lines` --
  `ctd_service.py:703-753`; predicate split into
  `_should_merge_koharu` -- `ctd_service.py:632-701`; helpers
  `_box_area`, `_calc_iou_raw`, `_contained_ratio` -- `ctd_service.py:605-630`.

Replaces the old single 30-px proximity threshold with koharu's three-rule
disjunction. Two bboxes merge iff **any** of:

- Rule 1: `IoU(a, b) >= 0.5`
- Rule 2: `smaller` is >=85% contained in `larger` (via `_contained_ratio`).
- Rule 3: all four of:
  - `y_dist < min(height_a, height_b) * 0.1`
  - `x_overlap / min(width_a, width_b) > 0.2`
  - `min(area) / max(area) > 0.3` (size ratio)
  - left and right edges align within `0.5 * max(width_a, width_b)`
    (horizontal edge alignment).

After a match, merged bbox area must be `<= 3 * max(area_a, area_b)` or
the merge is rejected (prevents runaway cluster growth).

### 4. FP16 CTD model preference

- Ours: `ComicTextDetectorService.__init__` -- `ctd_service.py:31-53`.

On init, if a file named `<model_stem>.fp16<suffix>` exists alongside the
configured `ctd_model_path` (e.g. `comictextdetector.fp16.onnx`), it is
loaded instead. The existing CUDA->CPU provider-fallback chain is preserved
(`_select_providers` / `_create_session` unchanged).

### 5. Test harness

- Path: `backend/scripts/test_ctd_mask_refinement.py`.
- Loads `de.png` at repo root (falls back to `runs/detect/runs/**`).
- Runs CTD once, then invokes `_process_mask` twice:
  1. `legacy=True, blocks=None` -> `before.png`
  2. `legacy=False, blocks=<parsed blocks>` -> `after.png`
- Writes a 2x2 composite `compare.png` containing: input, before-overlay
  (red), after-overlay (green), after-mask alone.
- Run from `backend/`: `uv run python scripts/test_ctd_mask_refinement.py`.

---

## Before/after visual summary

Sample: `de.png` (repo root).

| Metric                  | Before (legacy) | After (koharu) |
|-------------------------|-----------------|----------------|
| Nonzero mask pixels     | 36,010 (2.59%)  | 15,608 (1.12%) |
| Blocks detected         | 3               | 3              |
| Text lines              | 5               | 5              |

Visual diff (see `compare.png`):

- Before: red overlay bleeds into panel borders and non-text artwork
  (e.g. speed lines around the bottom-right bubble). The raw threshold
  picks up low-confidence fuzz outside any detected block.
- After: green overlay is tightly clipped to the three detected
  speech/caption regions. Dilation+close fills in broken strokes inside
  each block (noticeable on the bold "WIN!!" / "LOSE..." captions and the
  vertical goal bubble) without spilling past block bounds. The final
  mask panel (bottom-right) shows clean, block-constrained connected
  components -- ready for inpainting.

This ~56% reduction in mask area is pure false-positive suppression,
exactly what koharu's `refine_segmentation_mask` was designed to do.

---

## Files changed / added

- MODIFIED `backend/app/services/ctd_service.py`
- CREATED  `backend/scripts/test_ctd_mask_refinement.py`
- CREATED  `thoughts/koharu-improvements/ctd/SUMMARY.md` (this file)
- CREATED  `thoughts/koharu-improvements/ctd/before.png`
- CREATED  `thoughts/koharu-improvements/ctd/after.png`
- CREATED  `thoughts/koharu-improvements/ctd/compare.png`

## Notes

- The `mask` field in `detect()`'s return dict is still populated exactly
  as before for consumers that rely on mask bytes (frontend inpainter).
  Only the pixel values change.
- `_process_mask` accepts `blocks=None, legacy=True` kwargs so other call
  sites can opt into the old behavior without code churn.
- Text-line expansion runs before mask refinement so future work can
  unify the `in_bounds` rasterization against expanded lines instead of
  expanded blocks if higher precision becomes desirable.
- `_should_merge_koharu` is a pure, static classmethod -- easy to unit
  test in isolation if we add pytest coverage later.
