# Round 4 — Koharu Findings

Scanned: koharu-rpc (SSE), pipeline engines (bubble_seg/lama/aot/ctd_full), koharu-ml inpainting (balloon.rs, mask.rs), koharu-core events.rs, scripts, workflows, CHANGELOG 0.40–0.47.

### 1. Per-bubble ID mask (not binary) — high impact / medium effort
Koharu: `koharu-app/src/pipeline/engines/bubble_segmentation.rs:49-65` sorts bubbles by descending area then paints each into a Luma8 mask with a unique ID `(i+1) as u8` (up to 255). Smaller bubbles overwrite larger ones, so text seeds inside nested bubbles resolve correctly.
Us: `backend/app/services/lama_inpaint_service.py:369` explicitly notes "koharu uses a separate per-bubble segmentation mask to isolate bubble interiors; we only have the erase mask" — we treat all bubble pixels as one bin.
Port: ~40 LOC in `detector_service.py` or wherever bubble mask is produced. Sort `regions.sort(key=lambda r: -r.area)`, iterate with `id = i+1`, fill bbox with that value. Persist as uint8 PNG; downstream services accept it as-is since non-zero is still truthy for binary use.
Expected: unblocks findings 2+3 below — bubble-bounded dilation, per-bubble background color, correct fill on multi-bubble panels.

### 2. Bubble-bounded mask dilation via `dominant_bubble_id` — high impact / medium effort
Koharu: `koharu-ml/src/inpainting/mask.rs:15-128`. For each text block: crop the binary mask, dilate by `radius = clamp(font_size * 0.16, 2, 8)`, then when merging back reject pixels whose bubble_id differs from the block's dominant_bubble_id (`merge_expanded_region:119`). Prevents the erase mask from bleeding through bubble outlines into adjacent panels.
Us: `lama_inpaint_service.py` dilates uniformly and clips to the whole-bubble mask only as a binary gate. No per-block dilation radius, no per-bubble containment.
Port: ~80 LOC helper `expand_mask_for_inpainting(mask, bubble_id_mask, text_blocks)`. For each block compute dilate radius from `detected_font_size_px`, run `cv2.dilate` on the local ROI, accept pixels only where `bubble_id_mask[y,x] == dominant_id`. Chain a residual pass over any original mask pixels not covered (`mask.rs:63-73`).
Expected: cleaner inpaint seams, no "ghost erase" through bubble borders, tighter painted region for small fonts.

### 3. Per-bubble median-RGB bubble-fill (not crop-level) — high impact / low effort
Koharu: `koharu-ml/src/inpainting/balloon.rs:36-101`. For each bubble_id that overlaps the erase mask, compute median RGB from *unmasked pixels inside that bubble only*, check channel stddev vs dual thresholds (low-variance=10, high-variance=7 switched by `stddev3(std_rgb) > 1.0`), and fill only pixels inside *that bubble*. Our version treats the whole crop's unmasked area as one background estimator.
Us: `lama_inpaint_service.py:360-403` has TODO note acknowledging this simplification.
Port: ~50 LOC — replace `_apply_bubble_fastpath` with a loop `for bubble_id in overlapping_bubble_ids(mask, bubble_mask):`. Requires finding 1 first. Compute `median_rgb` on mask `(bubble_mask == bubble_id) & (mask == 0)` only.
Expected: fixes textured-background panels wrongly triggering fast-path when one bubble is flat and another is not; eliminates color bleed between bubbles.

### 4. Per-step non-fatal warning recovery — medium impact / low effort
Koharu: `koharu-app/src/pipeline/mod.rs:155-234` `'pages:` loop — when any engine step fails (load or run), emits `JobWarningEvent { page_index, step_id, message }`, credits remaining steps on that page to progress, and `continue 'pages`. Job finishes with status `CompletedWithErrors` (`koharu-core/src/events.rs:114`).
Us: `backend/app/routers/translate.py:406` on error calls `progress_bus.finish(job_id, status="error")` — whole job fails hard on one bad image.
Port: ~30 LOC. Add a `warnings: list[dict]` field threaded through the multi-image loop; on `except Exception` per image, append `{index, stage, message}`, emit a `stage="warning"` progress event, and continue to next image. Final status becomes `"completed_with_errors"` if warnings non-empty.
Expected: batch uploads of 20 pages don't die because page 7 had one bad OCR crop.

### 5. SSE `Last-Event-ID` replay with ring buffer — medium impact / medium effort
Koharu: `koharu-rpc/src/events.rs:55-101`. On reconnect reads `Last-Event-ID` header, replays buffered events `seq > last_id` from a broadcast ring; on buffer gap emits a fresh `Snapshot` to reseed; on `BroadcastStreamRecvError::Lagged` also reseeds. Each frame carries `Event::default().id(sev.seq.to_string())` so browsers retry correctly.
Us: `backend/app/utils/progress_bus.py:27` docstring: "our version is simpler: per-job asyncio.Queue, no ring buffer — reconnects only see future events." Means if the extension background worker disconnects mid-translation (common — service-worker sleep), the user sees progress jump instead of the real state.
Port: ~60 LOC. Add `collections.deque(maxlen=256)` per job in `ProgressBus`. In the SSE route read `request.headers.get("last-event-id")`, replay matching seqs, then switch to live. Send each event with `f"id: {seq}\n"` in the SSE frame.
Expected: Firefox/Chrome service-worker reconnects resume cleanly; users see deterministic progress bars.

### 6. `MAX_NEW_TOKENS = 256` + repetition_penalty=1.2 for OCR VL — low impact / trivial effort
Koharu: CHANGELOG 0.47.6/0.47.7 — `paddleocr_vl.rs:30-31` `DEFAULT_MAX_NEW_TOKENS=256`, `DEFAULT_REPETITION_PENALTY=1.2`, plus `repeated_ocr_suffix_start` (line 651) that strips trailing repeats before returning the decoded text. They tuned these in *the last 24 hours* specifically to kill PaddleOCR-VL's "stuck repeating" failure mode.
Us: `backend/app/services/manga_ocr_service.py` and `local_translation_service.py` — we already have normalizer + repetition guard per rounds 1-2, but haven't aligned the exact knobs for Qwen/Paddle VL variants.
Port: ~10 LOC. In any VL-OCR backend invocation: set `max_new_tokens=256` (not the often-default 512+), `repetition_penalty=1.2`, and post-process with the suffix-detection from `paddleocr_vl.rs:651-689` (detect k-char unit repeating ≥4 times at end, truncate).
Expected: fewer runaway OCR outputs on ambiguous tiny text; faster per-crop latency.

### 7. `detector="ctd"` conditional crop-bounds expansion — low impact / low effort
Koharu: `koharu-ml/src/comic_text_detector/postprocess.rs:107-128` — `expanded_text_block_crop_bounds` only expands when `detector == "ctd"` OR `line_polygons` present; PP-DocLayout / comic-text-bubble-detector get a plain bbox because their outputs already have the tight polygon geometry. This is reused by *both* OCR crop and inpaint mask expansion (a single source of truth).
Us: We padd uniformly regardless of detector in `ctd_utils.py` / OCR crop paths.
Port: ~20 LOC. Thread `detector` into `TextRegion`/dict, branch in crop-bounds helper. Only CTD-produced boxes get the vertical-aware padding; animetext/yolo boxes stay tight.
Expected: tighter OCR crops for non-CTD detectors (less surrounding noise), better OCR accuracy on animetext pipeline.

### 8. Ascent+descent+leading line-height (not magic 1.1) — low impact / trivial effort
Koharu: `koharu-renderer/src/layout.rs:186` `line_height = (ascent + descent + metrics.leading).max(font_size)`. Leading comes from the font file itself.
Us: `src/services/overlay-renderer.ts:48` `const LINE_GAP_FACTOR = 1.1;` multiplied onto `(ascent+descent)`. 1.1 works for Latin fonts but produces cramped lines for fonts whose designer set nonzero leading (comic fonts like Bangers have tight design metrics; our 1.1x looks worse than the font's own 1.25x native).
Port: ~5 LOC. Replace `LINE_GAP_FACTOR` with reading the Canvas TextMetrics' `fontBoundingBoxAscent + fontBoundingBoxDescent` (if available) and using that directly; fallback to actualBoundingBox with a 1.05 floor only when leading is unavailable.
Expected: more faithful rendering across Bangers/Fredoka/CC Wild Words; no "too-tight" lines.

### 9. CI integration-test harness + cached llama runtime — low impact / medium effort (process)
Koharu: `.github/workflows/test.yml:43-55` + `tests/integration-tests/` — a dedicated crate that spins up the full axum server (`src/harness.rs`) and runs real HTTP calls (`tests/binary.rs`, `tests/pipelines.rs`, `tests/scene.rs`). The runtime (llama.cpp shared lib) is cached across CI runs keyed on `hashFiles('koharu-runtime/**/*.rs', 'koharu-llm/**/*.rs')`.
Us: `backend/scripts/test_parseq_e2e.py` exists but no full-pipeline HTTP harness + no CI config guarding `/translate`.
Port: ~200 LOC — `backend/tests/integration/test_translate_e2e.py` that boots FastAPI (`TestClient`/`httpx.AsyncClient`), POSTs a fixture image, asserts response has `boxes`, `text_blocks`, correct count. Add a `.github/workflows/backend.yml` caching `backend/models/` by hash of model URLs.
Expected: regressions in `/translate` wiring caught before merge.

