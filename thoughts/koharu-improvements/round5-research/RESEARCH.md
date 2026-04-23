# Koharu Round-5 Research — Consolidated Master List + Deep-Dive
Generated: 2026-04-22. Koharu HEAD v0.47.7.

---

## Part A — Consolidated Ranked Action List (rounds 2+3+4 dedup)

### 1. Predicted stroke_width_px + stroke_color + line_height + angle_deg from font_detector — HIGH/LOW
Origins: round-3 #1, round-3 #2, round-3 #10
Koharu refs: `koharu-ml/src/font_detector/mod.rs:136-148`
Us: `overlay-renderer.ts:673` hardcodes `fontSize*0.12`; ignores regression[4-9] that the model already emits for free.
Port: ~40 LOC. Thread `strokeWidthPx, strokeColor, lineHeight, angleDeg` through `animetext_service.py` → TextBox JSON → `overlay-renderer.ts` drawStep (apply `ctx.rotate` around centroid, pass line-height into binary fit, use predicted stroke). Zero extra inference cost.

### 2. LaMa text-region-guided crop windows w/ overlap merge — HIGH/MED
Origins: round-2 #1
Koharu refs: `koharu-ml/src/lama/mod.rs:244-345`
Us: `lama_inpaint_service.py:212` iterates raw contour boxes w/ fixed 128-px margin → 10+ passes on a busy page.
Port: ~60 LOC `_plan_crop_windows(blocks, W, H, area_ratio=1.7)` + O(n²) merge. 30-50% fewer LaMa forwards.

### 3. Per-bubble ID mask + bubble-bounded dilation + per-bubble median fill — HIGH/MED
Origins: round-4 #1, round-4 #2, round-4 #3 (and round-2 #7 dual-stddev feeds into it)
Koharu refs: `koharu-app/src/pipeline/engines/bubble_segmentation.rs:49-65`, `koharu-ml/src/inpainting/mask.rs:15-128`, `koharu-ml/src/inpainting/balloon.rs:36-161`
Us: `lama_inpaint_service.py:369` notes the simplification; `:360` uses crop-level median only.
Port: ~170 LOC total. (a) Emit area-sorted Luma8 ID mask in `detector_service.py`. (b) `expand_mask_for_inpainting` using dominant_bubble_id + font-scaled dilate (`clamp(font*0.16, 2, 8)`). (c) Loop per overlapping bubble_id with dual-threshold stddev switch (7/10). Prerequisite for #9.

### 4. Perspective-warped line crops for OCR — HIGH/MED
Origins: round-2 #2
Koharu refs: `koharu-ml/src/comic_text_detector/postprocess.rs:170-241`
Us: `parseq_ocr_service.py:177` only does axis-aligned + 90° rot.
Port: ~40 LOC `_warp_line_polygon` via `cv2.getPerspectiveTransform` + `warpPerspective`. Biggest OCR-accuracy lift on slanted SFX/hand-lettering.

### 5. BubbleIndex: grow layout box to segmented bubble bbox — HIGH/HIGH
Origins: round-2 #6
Koharu refs: `koharu-renderer/src/text/latin.rs:74-185`
Us: Compositor + overlay use the raw tight CTD bbox → English crammed into narrow JP vertical strip.
Port: ~100 LOC `BubbleIndex` (majority-vote pixel count → inset 12%/20%) in `image_processing.py:compose_text`. Biggest readability win. Prereq: #3's ID mask.

### 6. Recursive XY-Cut manga reading order — HIGH/MED
Origins: round-2 #4
Koharu refs: `koharu-app/src/pipeline/engines/support.rs:311-434`
Us: 5 sites sort `(-minX, minY)` → interleaved side-by-side columns.
Port: ~80 LOC `utils/reading_order.py::sort_manga_reading_order`; wire into `ctd_service.py:342`, `animetext_service.py:199`, `detector_service.py:85`, `parseq_ocr_service.py:314`, `routers/translate.py:476`. Improves LLM context coherence.

### 7. LLM + OCR repetition_penalty=1.2 + online trim + enable_thinking=False — HIGH/LOW
Origins: round-2 #3, round-2 #10, round-4 #6
Koharu refs: `koharu-llm/src/paddleocr_vl.rs:319-329,531-545,651-689`, `koharu-llm/src/prompt.rs`
Us: `local_translation_service.py` uses llama-cpp defaults; `ocr_postprocess.py` only post-hoc trims.
Port: ~15 LOC total. (a) `repeat_penalty=1.2, repeat_last_n=-1, max_tokens=256`. (b) `chat_template_kwargs={"enable_thinking": False}`. (c) Streaming callback: every N tokens detect ≥4× repeat of ≤12-char suffix, early-abort. 30-40% fewer wasted OCR tokens.

### 8. CTD threshold alignment (0.235) + NMS 0.35 + min-5px filter + contain-merge 0.3 — MED/LOW
Origins: round-3 #4, round-3 #5
Koharu refs: `koharu-ml/src/comic_text_detector/mod.rs:24,26`, `comic_text_bubble_detector/mod.rs:478,518-537`
Us: `config.py:51` text=0.3, NMS=0.5 (ctd_service.py:246); no small-box filter, no containment merge.
Port: ~30 LOC. Defaults `ctd_text_threshold=0.235, ctd_nms_iou=0.35`; post-NMS: drop w≤5 or h≤5, then merge pairs w/ containment≥0.3.

### 9. Post-prediction font-color clamping + stroke-collapse drop — MED/LOW
Origins: round-2 #9
Koharu refs: `koharu-app/src/pipeline/engines/yuzumarker_font.rs:118-151`
Us: `snap_font_color` single threshold, no "drop stroke when text≈stroke within 16/ch".
Port: ~20 LOC — grayscale-vs-color detection (channel-range ≤10) + `colors_similar(a,b,16)` guard. Kills faux-chromatic halos.

### 10. Emphasis-punctuation ligatures (!!→‼ etc.) — MED/LOW
Origins: round-2 #5
Koharu refs: `koharu-renderer/src/layout.rs:618-700`
Us: `!!` draws as two glyphs in both Python compositor and overlay-renderer.
Port: ~20 LOC normalizer (U+203C, U+2047, U+2049, U+2048). 10-min visual upgrade on shonen panels.

### 11. Per-image non-fatal warning recovery — MED/LOW
Origins: round-4 #4
Koharu refs: `koharu-app/src/pipeline/mod.rs:155-234`, `koharu-core/src/events.rs:114`
Us: `routers/translate.py:406` hard-fails the whole batch on any exception.
Port: ~30 LOC. Per-image try/except, append `{index, stage, message}` to warnings list, emit `stage="warning"` SSE, final `completed_with_errors`.

### 12. SSE Last-Event-ID replay w/ ring buffer — MED/MED
Origins: round-4 #5
Koharu refs: `koharu-rpc/src/events.rs:55-101`
Us: `utils/progress_bus.py:27` explicitly says no ring buffer.
Port: ~60 LOC `deque(maxlen=256)` per job, read `last-event-id` header, replay future seqs. Fixes service-worker reconnect jumps.

### 13. Ranged parallel model download w/ retry — MED/MED
Origins: round-3 #6
Koharu refs: `koharu-runtime/src/downloads.rs:27,60,249`
Us: `scripts/download_lama_onnx.py` single-stream, no retry, no resume.
Port: ~80 LOC `httpx.AsyncClient` + 10 MiB ranges + `asyncio.gather(n=cpu_count)` + `.part` file pre-alloc. 3-6× faster first-run.

### 14. block-level expanded bounds as floor for per-line OCR crop — MED/LOW
Origins: round-2 #8
Koharu refs: `koharu-ml/src/comic_text_detector/postprocess.rs:107-168`
Us: Per-line crop in PARSeq uses line_polygon bbox only — clips ascenders on slanted text.
Port: ~15 LOC — consult block-expanded bounds as min-floor when cropping.

### 15. `detector="ctd"` conditional crop expansion — MED/LOW
Origins: round-4 #7
Koharu refs: `koharu-ml/src/comic_text_detector/postprocess.rs:107-128`
Us: `ctd_utils.py` uniform padding regardless of detector.
Port: ~20 LOC. Thread `detector` into TextRegion; only CTD boxes get vertical-aware padding; animetext/yolo stay tight.

### 16. Probability-map skip gate (`max < 0.05` ⇒ skip inpaint) — LOW/TRIVIAL
Origins: round-3 #7
Koharu refs: `koharu-ml/tests/manga_text_segmentation_2025.rs:21-24`
Us: always runs LaMa even on empty pages.
Port: ~10 LOC. Saves 30-200 ms on blank covers / split pages.

### 17. Per-stage tracing (preprocess_ms/forward_ms/postprocess_ms/total_ms) — LOW/LOW
Origins: round-3 #9
Koharu refs: `koharu-ml/src/speech_bubble_segmentation/mod.rs:208-218` (see also deep-dive #D1)
Us: aggregate SSE only.
Port: ~15 LOC/service via `time.perf_counter()` at DEBUG. Zero prod cost, faster diagnosis.

### 18. DeepSeek temperature=1.3 override in provider adapter — LOW/LOW
Origins: round-3 #3
Koharu refs: `koharu-llm/src/providers/deepseek.rs:36-37`
Us: uniform 0.1-0.3 in `local_translation_service.py:297,504`.
Port: ~5 LOC `model_temperature_map`.

### 19. Ascent+descent+leading line-height (not LINE_GAP_FACTOR=1.1) — LOW/TRIVIAL
Origins: round-4 #8
Koharu refs: `koharu-renderer/src/layout.rs:186`
Us: `overlay-renderer.ts:48` hardcoded 1.1.
Port: ~5 LOC — use `fontBoundingBoxAscent+fontBoundingBoxDescent` from TextMetrics; 1.05 floor fallback.

### 20. Backend /translate E2E integration test harness + CI — LOW/MED (process)
Origins: round-4 #9
Koharu refs: `.github/workflows/test.yml:43-55`, `tests/integration-tests/`
Us: only `test_parseq_e2e.py`, no FastAPI HTTP harness.
Port: ~200 LOC `backend/tests/integration/test_translate_e2e.py` + `.github/workflows/backend.yml` w/ model-cache key.

---

## Part B — Deep-Dive: Unexplored Corners

### D1. Koharu's own benchmark signal: PaddleOCR-VL `generation_ms` is THE hot path — HIGH/MED
Koharu refs: `koharu-ml/src/paddleocr_vl/mod.rs:420-432` emits `preprocess_ms, generation_ms, decode_ms, total_ms` separately. Every other `tracing::info!` call site (font_detector, speech_bubble_seg, manga_text_seg_2025, pp_doclayout_v3, aot_inpainting, comic_text_bubble_detector) uses the 3-field template. Only PaddleOCR-VL carves out `generation_ms` vs `decode_ms` → that's where koharu spends effort. `koharu-renderer/benches/rendering.rs` is the ONLY criterion bench in the repo; no ML benches, because ML is IO-bound on CUDA/Metal and they rely on tracing.
Us: `manga_ocr_service.py` emits one aggregate time. We can't tell if slowness is ONNX forward vs Python pre/post.
Finding: VL-OCR autoregressive generation dominates. Any batching/bucketing gain on the generation step is a force multiplier. Implications for our roadmap: prioritize #7 (repeat_penalty + early-abort + max_tokens=256) over #2/#4/#8 if VL-OCR is in our pipeline.
Port: adopt the 4-field timing template; immediately visible bottleneck tiers.

### D2. PaddleOCR-VL **bucket-batching** by identical resized (W,H) — HIGH/MED
Koharu refs: `koharu-ml/src/paddleocr_vl/mod.rs:180-184, 357-395, 728-751` (`build_batch_groups`). After `smart_resize` with `patch_size=14, merge_size=2 ⇒ factor=28`, images are grouped by `(processed_width, processed_height)` via `BTreeMap`, each bucket forwards as one batch with shared `grid_thw`. `max_bucket_image_tokens` logged per run. This avoids padding the batch to the biggest image (which an ordinary "batch everything" approach pays for).
Us: `parseq_ocr_service._process_batch` (line ~240) uses a fixed target size. If we adopt any VL-OCR, we should bucket by post-resize dims.
Port: ~50 LOC `collections.defaultdict(list)` keyed by `(w,h)` after `smart_resize`, one `sess.run` per bucket. Directly targets the D1 bottleneck on mixed crop sizes.

### D3. Cargo feature gating exposes a 3-way CPU/CUDA/Metal build matrix with ZERO Vulkan flag — MED/LOW
Koharu refs: `/tmp/koharu/Cargo.toml:95-110` declares `cudarc` + `objc2-metal*` workspace-deps; `koharu-ml/Cargo.toml [features]` defines ONLY `cuda` and `metal`. Vulkan (per docs/acceleration-and-runtime.md:26-30) is only for "OCR and LLM" via llama.cpp's own vulkan backend — NOT candle's. Detection and inpainting are CUDA-or-Metal-or-CPU.
Us: backend uses `onnxruntime` with providers list in `config.py`. We support `CUDAExecutionProvider` + `CoreMLExecutionProvider` + CPU. No Vulkan. No Metal tracking for DirectML either.
Finding: koharu's HARD CHOICE is "don't bother with Vulkan for detection/inpaint". We should NOT implement DirectML/Vulkan for detector/LaMa — koharu tried and rejected. Keep ONNX Runtime CUDA+CoreMLEP only; route LLM through llama.cpp where Vulkan is free.
Port: document this as an ADR; rename a backend option to mirror koharu's split (detector_provider vs ocr_provider).

### D4. `config.toml` operator-tunables are surprisingly minimal — LOW/LOW
Koharu refs: `docs/en-US/reference/settings.md:66-82` + `koharu-runtime/src/runtime.rs:20-30`. Koharu exposes exactly FOUR runtime knobs: `data_path, connect_timeout_secs=20, read_timeout_secs=300, max_retries=3`. Everything else (thresholds, max_pixels, repetition_penalty) is compile-time. The "Engines" tab picks backends; no per-engine numeric tuning surface.
Us: `config.py` has ~40 tunable fields. We're over-configured relative to koharu's philosophy.
Finding: koharu treats ML thresholds as product decisions, not ops knobs. This is a dev-velocity win (fewer combinations to regress-test) and a UX win (users don't mis-tune). 
Port: reduce `config.py` to ~8 ops knobs; move `ctd_text_threshold`, `nms_iou`, `max_pixels` behind a `--profile {quality,fast}` switch. No user-facing churn.

### D5. PaddleOCR-VL config defaults reveal the right OCR resolution budget — MED/LOW
Koharu refs: `koharu-ml/src/paddleocr_vl/mod.rs:147` `default_max_pixels = 1536*1536 = 2_359_296`; `:143` `default_min_pixels = 384*384 = 147_456`; factor=28 (patch 14 × merge 2). `manga_text_segmentation_2025/mod.rs:20-21` `GPU_MAX_PIXELS=1_536²`, `CPU_MAX_PIXELS=1_280²`.
Us: `parseq_ocr_service` uses PARSeq's native 32×128. We don't have a global page-level max_pixels gate before ML.
Finding: 1536² is the "good enough" ceiling for text-related ML; 1280² on CPU. Our code currently lets the CTD preprocess see arbitrary-resolution inputs.
Port: ~15 LOC in `ctd_service.py` preprocess: if `w*h > 1536*1536` (or 1280² on CPU) resize so longest side keeps aspect, record `scale_factor`, unscale detections. Caps worst-case latency on 4K scans.

---

## Sources
1. `/tmp/koharu/koharu-ml/src/paddleocr_vl/mod.rs` — timings, bucketing, max_pixels
2. `/tmp/koharu/koharu-ml/src/font_detector/mod.rs` — regression outputs + timings
3. `/tmp/koharu/Cargo.toml` + `koharu-ml/Cargo.toml` — feature flags
4. `/tmp/koharu/docs/en-US/reference/settings.md` — operator tunables
5. `/tmp/koharu/docs/en-US/explanation/acceleration-and-runtime.md` — backend matrix
6. `/tmp/koharu/koharu-runtime/src/runtime.rs` + `downloads.rs` — HTTP config
7. `/tmp/koharu/koharu-renderer/benches/rendering.rs` — only bench in repo
