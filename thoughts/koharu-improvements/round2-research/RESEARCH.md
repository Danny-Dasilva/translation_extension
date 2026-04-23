# Koharu Round-2 Research — Unported Improvements
Generated: 2026-04-22

Scope: Deep read of /tmp/koharu HEAD (v0.47.7). Items below are **not yet** in our backend. Ranked by (impact × effort⁻¹).

---

### 1. Text-region-guided LaMa crop windows with overlap merging — L/M
Koharu: `koharu-ml/src/lama/mod.rs:244-345` (`crop_windows_from_text_blocks` + `enlarge_window` + `merge_overlapping_windows`). For each detected text block: build a bbox, enlarge via a quadratic that targets `area × 1.7` with aspect 1.0, clamp to image, then greedily union any overlapping/touching rects. Pass as `crop_windows` to `run_inpaint_with_windows`.
Us: `backend/app/services/lama_inpaint_service.py:212` — we iterate raw `cv2.findContours` boxes from the mask with a fixed 128-px margin. A dense page with 10 bubbles ⇒ 10+ forward passes even when clusters could share a window.
Port cost: ~60 LOC. Reuse our existing `_expand_and_clamp`; add `_plan_crop_windows(text_regions, W, H, ratio=1.7)` + iterative `_merge_overlapping` (O(n²) is fine for n<30), then gate the contour path behind `if not text_regions`. Expect 30–50 % fewer LaMa forwards on busy pages.

### 2. Perspective-warped line crops for OCR — L/M
Koharu: `koharu-ml/src/comic_text_detector/postprocess.rs:170-241` (`warp_line_region`). Each `line_polygon` quad is expanded (direction-aware), clipped to the image, then `Projection::from_control_points` warps the (possibly tilted) quad into an axis-aligned rect sized to the quad's true text-axis length; vertical text gets rotated 270°.
Us: `parseq_ocr_service.py:177` only does a 90° CCW rotate on tall crops; we pass axis-aligned bboxes from `_extract_text_lines` straight to PARSeq. Slanted speech-bubble tails and angled SFX text lose accuracy.
Port cost: ~40 LOC helper `_warp_line_polygon(img, quad, direction)` using `cv2.getPerspectiveTransform` + `cv2.warpPerspective` with `INTER_LINEAR`. Wire into `parseq_ocr_service` just before `_maybe_rotate_vertical`. Biggest quality lift for tilted SFX/hand-lettering.

### 3. PaddleOCR-VL-style **online** repetition trimming — M/S
Koharu: `koharu-llm/src/paddleocr_vl.rs:319-329` + `repeated_ocr_suffix_start:651-689`. Inside the token loop, after every decoded token they detect if the tail has `N ≥ 4` repetitions of a ≤12-char unit; if so they `break` and trim. They ALSO set `repetition_penalty=1.2` on the sampler (`DEFAULT_REPETITION_PENALTY` constant + `LlamaSampler::penalties` chain). See CHANGELOG v0.47.6/7 — they specifically landed this last week.
Us: `backend/app/utils/ocr_postprocess.py` has a post-hoc regex guard, but we keep generating all 128+ tokens before trimming. We don't apply a `repeat_penalty` on llama-cpp-python.
Port cost: (a) add `repeat_penalty=1.2` to our `Llama(…).create_completion` / sampler config — one line. (b) Add a streaming callback that inspects the accumulated text every N tokens and early-aborts. Saves ~30–40 % of wasted OCR tokens on degenerate crops. If we can't use streaming cleanly, just `repeat_penalty` alone is a 10-min win.

### 4. Recursive XY-Cut manga reading order — M/M
Koharu: `koharu-app/src/pipeline/engines/support.rs:311-434` (`sort_manga_reading_order`). Computes median block w/h, uses `min_gap_x = max(median_w*0.15, 10)` and y counterpart; recursively finds the widest inter-block gap on each axis, prefers Y cuts when `width_y > 12 || width_y > width_x*0.4`, then right-half-first on X and top-half-first on Y. Much better than `(-minX, minY)` for pages with stacked panels of different column counts.
Us: `ctd_service.py:342`, `animetext_service.py:199`, `detector_service.py:85`, `parseq_ocr_service.py:314`, `routers/translate.py:476` all sort by `(-minX, minY)` — a panel with two side-by-side sub-columns in the same Y band gets interleaved.
Port cost: ~80 LOC in a new `utils/reading_order.py::sort_manga_reading_order(blocks)`; call from the 5 sites listed. Pure CPU, no model touch. Fixes translation **coherence** which ripples through LLM context because we feed sources in order.

### 5. Emphasis-punctuation ligature normalizer for rendering — M/S
Koharu: `koharu-renderer/src/layout.rs:618-700` (`emphasis_mark_kind` / `emphasis_pair_symbol` / `normalize_vertical_emphasis_punctuation`). Collapses `"!!"`→`‼ (U+203C)`, `"??"`→`⁇ (U+2047)`, `"!?"`→`⁉ (U+2049)`, `"?!"`→`⁈ (U+2048)` before shaping, especially important for vertical text where two stacked bangs look awful.
Us: `backend/app/utils/image_processing.py` draws `!!` as two glyphs. No ligature pass.
Port cost: ~20 LOC pre-layout normalizer in our Python compositor + `overlay-renderer.ts` rendering path. 10 minutes, instant visual upgrade in shonen panels.

### 6. BubbleIndex: grow layout box to segmented bubble bbox — L/L
Koharu: `koharu-renderer/src/text/latin.rs:74-185` (`BubbleIndex::lookup_match`). Scans bubble seg mask once → per-ID bbox; for each seed text bbox, counts pixels of each bubble ID under the seed, picks majority, returns bubble bbox inset by 12 % (horizontal) / 20 % (vertical). The layout engine then uses that (not the tight CTD bbox) as the wrap width, so translated text **fills the bubble** instead of cramming into the original Japanese bbox.
Us: Our Python compositor + frontend overlay both use the raw detection bbox, which for Japanese vertical text can be 60 % narrower than the English translation needs. We re-shrink with binary-fit to compensate, which hurts legibility.
Port cost: ~100 LOC. We already expose `bubble_mask` from CTD (it's on disk for LaMa). Add `backend/app/utils/bubble_index.py::BubbleIndex` + integrate into `image_processing.py:compose_text` auto-sizing. Biggest readability win — stop rendering English into a tiny vertical strip.

### 7. Median (not mean) background color for balloon fastpath — S/S
Koharu: `koharu-ml/src/inpainting/balloon.rs:130-161`. Uses `median_rgb` per channel for the fill color; std-dev threshold switches between 10.0 (single-channel noise) and 7.0 (colored noise). Switch rule: if `stddev(per-channel stddevs) > 1.0`, use tighter threshold.
Us: `lama_inpaint_service.py:360` — `median_rgb = np.median(bg, axis=0)` already matches (good), but we use a single `_SIMPLE_BG_THRESHOLD` rather than the dual threshold switch. Thin black-framed bubbles on busy art get rejected from fastpath when they shouldn't.
Port cost: 6 LOC — add `channel_std_meta = np.std(std_rgb)` and pick 7 vs 10. Also, we don't restrict the background sampler to pixels *inside the bubble mask* (koharu uses `bubble_mask == id`). Adding that makes the median robust against adjacent panels bleeding in.

### 8. `expanded_text_block_crop_bounds` direction-aware padding WITH line-polygon union — M/S
Koharu: `koharu-ml/src/comic_text_detector/postprocess.rs:107-168`. For CTD blocks, they expand the *bounding rectangle* to include each line polygon's `maybe_expand_ctd_line` expansion (not just the block bbox), then pad by `(font*0.12, font*0.18)` for horizontal or `(0.18, 0.12)` for vertical.
Us: We ported the dilate + clip half (bullet #1 of the already-landed list) but the OCR crop path (`parseq_ocr_service`) passes each line_polygon's own bbox, not the block-level expanded union. When a single line's polygon is imprecise, we clip glyph ascenders/descenders.
Port cost: ~15 LOC — when cropping per-line for PARSeq, also consult the block-level expanded bounds as a floor. Tiny code, clear OCR accuracy gain on slanted / thin-stroke text.

### 9. Post-prediction font-color clamping — S/S
Koharu: `koharu-app/src/pipeline/engines/yuzumarker_font.rs:118-151`. After the font model returns RGB: snap near-black to `[0,0,0]` (threshold 60 for grayscale, 12 for saturated); snap near-white to `[255,255,255]`. Drop stroke if text and stroke are within 16 per channel (prevents invisible halo rendering).
Us: `utils/image_processing.py:snap_font_color` snaps to pure B/W but only one threshold; no "drop stroke if colors collapse" rule. That produces "white text with off-white stroke" artifacts on scanned pages.
Port cost: ~20 LOC in `snap_font_color` — add grayscale vs color detection (range-of-channels ≤ 10) and a `colors_similar(a,b,16)` guard that zeros the stroke. Instant cleanup of faux-chromatic aberration on translated text.

### 10. `repetition_penalty` + decoder-start handling on our local LLM — S/S
Koharu: `koharu-llm/src/paddleocr_vl.rs:531-545` — sampler chain `penalties(last_n=-1, penalty=1.2) → greedy`. Note `last_n = -1` means "consider the entire context". For the translation LLM (`koharu-llm/src/prompt.rs`) they disable thinking via template variable `enable_thinking => false` to suppress chain-of-thought in Qwen-family templates.
Us: `local_translation_service.py` uses llama-cpp-python defaults (no `repeat_penalty`, no `repeat_last_n`). We already have `strip_thinking_block` post-hoc, but we're paying tokens we then throw away.
Port cost: Two lines — `repeat_penalty=1.15, repeat_last_n=-1` on `create_chat_completion`. For Qwen-3 models also send `"chat_template_kwargs": {"enable_thinking": False}` so the runtime skips the `<think>` block entirely. Saves tokens and trims tail repetition loops mid-generation.

---

## Honorable mentions (skipped — higher effort or lower payoff)

- **candle tensor-based morph_close/dilate** (`comic_text_detector/mod.rs:375-392`) — we already use cv2 which is faster on CPU than reimplementing with torch. Skip.
- **`MAX_UBATCH=512` for llama.cpp** — only matters if we switch to llama.cpp native multimodal; we use ORT for PARSeq. Skip.
- **Sentry session tracking** (v0.46.0) — ops-only, no quality impact.
- **Brush-layer compositing** (renderer.rs:37-41) — UI feature, not per-page quality.
- **Keybind/undo configuration** (v0.47.0) — UI-only.

## Sources
1. `/tmp/koharu/koharu-ml/src/lama/mod.rs` — crop window planning
2. `/tmp/koharu/koharu-ml/src/comic_text_detector/postprocess.rs` — warp_line_region
3. `/tmp/koharu/koharu-llm/src/paddleocr_vl.rs` — online repeat trim + rep penalty
4. `/tmp/koharu/koharu-app/src/pipeline/engines/support.rs` — XY-Cut
5. `/tmp/koharu/koharu-renderer/src/layout.rs` — emphasis ligatures
6. `/tmp/koharu/koharu-renderer/src/text/latin.rs` — BubbleIndex
7. `/tmp/koharu/koharu-ml/src/inpainting/balloon.rs` — median + dual stddev switch
8. `/tmp/koharu/koharu-app/src/pipeline/engines/yuzumarker_font.rs` — color clamps
9. `/tmp/koharu/CHANGELOG.md` v0.47.6–7 — repeat guard + rep_penalty
