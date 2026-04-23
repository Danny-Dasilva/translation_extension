# Round 3 Research — Koharu Port Candidates

Ranked by impact × effort⁻¹. Every item cites a concrete `file:line` in `/tmp/koharu` and proposes a concrete port plan. Items already ported (rounds 1–2) are excluded.

---

### 1. Use font_detector's regressed `stroke_width_px` / `stroke_color` instead of fixed `0.12 × fontSize` — HIGH / LOW
Koharu: `koharu-ml/src/font_detector/mod.rs:136-141` exposes `stroke_width_px = sigmoid(regression[4]) * width`, `stroke_color = regression[5..8]`. The model outputs 10 regression values; we currently use only `text_color` and clamp only to color. `stroke_width` and `stroke_color` ride on the same tensor row — zero extra inference cost.
Us: `src/services/overlay-renderer.ts:673` hardcodes `strokeWidth = Math.max(2, Math.round(fontSize * 0.12))`; `resolveColors` at `:514` never reads a predicted stroke color.
Port: ~30 LOC. Thread `strokeWidthPx` + `strokeColor` through the FontPrediction JSON in `backend/app/services/animetext_service.py` (or wherever font_detector output lands) and in `overlay-renderer.ts:657` `drawStrokedText`, swap the coefficient for `region.strokeWidthPx ?? Math.round(fontSize * 0.12)` and the default `#000000` for the API value (only when contrast is sufficient — reuse existing contrast gate at `:527`).
Expected: correct stroke thickness on thin/bold fonts (manga titles and onomatopoeia rendered with visibly wrong weight today); fewer misreads on high-contrast double-stroke balloons.

---

### 2. Use regressed `line_height` directly instead of koharu's `font_size/size_px` formula — HIGH / LOW
Koharu: `koharu-ml/src/font_detector/mod.rs:142-147`: `line_height = 1.0 + line_spacing_px / font_size_px` (floored to 1.2). Derived from a separate regression head (`regression[8]`), not from font metrics.
Us: `overlay-renderer.ts` uses a hardcoded coefficient per block. Round 2 #? discussed line_height but from the PSD export path (`koharu-psd/src/export.rs:676-699`), not the regressed value.
Port: ~10 LOC. Propagate `line_height` through the same JSON plumbing as #1; pass directly to the Canvas layout loop in `overlay-renderer.ts:386`'s binary-fit section.
Expected: better multi-line balloon fit, fewer iterations in the binary fit (it already narrows faster when line_height matches the actual typeset spacing).

---

### 3. DeepSeek-specific `temperature=1.3` override in provider adapter — MED / LOW
Koharu: `koharu-llm/src/providers/deepseek.rs:36-37` overrides `temperature: Some(1.3), max_tokens: None` while OpenAI (`openai.rs:36-37`), Gemini (no temperature passed), Claude (no temperature), and DeepSeek all share the same `send_chat_completion` path. DeepSeek explicitly wants a much higher temperature because their model's scoring calibration differs.
Us: `backend/app/services/local_translation_service.py:297,504` uses `temperature=0.1`–`0.3` uniformly.
Port: ~5 LOC. Add a `model_temperature_map` dict (keys: `deepseek-chat`, `qwen3*`, default) consulted when we build the translate request. Round 2 #10 covered Qwen's `enable_thinking=False`; this is an orthogonal per-provider knob.
Expected: less literal / more fluent DeepSeek output when remote-mode is used as a translation backend.

---

### 4. Port `detections_to_text_blocks` drop-small filter (min 5×5 px) + `merge_text_regions` overlap-IoU≥0.5 + contained≥0.3 — MED / LOW
Koharu: `koharu-ml/src/comic_text_bubble_detector/mod.rs:478` drops regions with `width ≤ 5.0 || height ≤ 5.0`. `merge_text_regions` at `:518-537` merges any pair overlapping with IoU≥0.5 or 30%-contained. This is stricter than our `iou_threshold=0.5` NMS (`ctd_service.py:335,246`) and runs a second pass specifically to eliminate duplicates after the NMS pass.
Us: `ctd_service.py:_apply_nms` only does IoU-based NMS at 0.5 and a hardcoded block_confidence threshold; no small-box filter, no post-NMS containment merge.
Port: ~25 LOC. After the existing NMS in `_apply_nms`, filter `blocks` where `width <= 5 or height <= 5`, then run a contained-pass with the 0.3 threshold (bbox_A ∩ bbox_B / min(area_A, area_B) ≥ 0.3 ⇒ merge).
Expected: fewer phantom 1–2 word fragments from CTD on noisy manga pages; ~5–15% fewer OCR calls on pages with heavy artifacts.

---

### 5. Align CTD binary threshold with koharu's `60/255 ≈ 0.235` (we use `0.3`) — MED / TRIVIAL
Koharu: `koharu-ml/src/comic_text_detector/mod.rs:26` `BINARY_THRESHOLD: u8 = 60` (i.e. probability ≥ 0.235). Also NMS at `0.35` (`:24`) vs. our `0.5`.
Us: `config.py:51` `ctd_text_threshold: 0.3`; `ctd_service.py:246` uses `iou_threshold=0.5` for NMS.
Port: ~3 LOC. Change defaults: `ctd_text_threshold=0.235`, `ctd_nms_iou=0.35`. Keep them as tunables (our block-aware refine pass in round 1 already clips to block bounds, so lowering the threshold is safe — the block gate filters spurious pixels).
Expected: more complete text masks (fewer holes in bold strokes / small kana); better inpainting because edges aren't under-masked. Lower NMS recovers side-by-side narrow vertical columns CTD currently fuses.

---

### 6. Ranged-download for model weights with 10 MiB chunks + parallelism = num_cpus — MED / MED
Koharu: `koharu-runtime/src/downloads.rs:27` `CHUNK_SIZE: 10 * 1024 * 1024`, `:249` `.buffer_unordered(num_cpus::get())` for parallel range requests. Combined with `ExponentialBackoff` retry middleware (`:60`) and a 30s metadata timeout (`:31`), this is why first-run model pulls don't hang on slow mirrors.
Us: we pull weights in one shot in `scripts/download_lama_onnx.py` and friends; single-stream, no retry, no resume, no progress events.
Port: ~80 LOC Python equivalent using `httpx.AsyncClient` + `asyncio.gather` + `Range: bytes=start-stop`. Pre-allocate `.part` file with `os.truncate(size)`, seek-write per range, rename on success.
Expected: 3–6× faster first-run on residential links; resume on connection drops; eliminates the existing "20-min stall silently" failure mode when users install cold.

---

### 7. Use `probability_map.max_value() > 0.05` sanity gate after segmentation — LOW / TRIVIAL
Koharu: `koharu-ml/tests/manga_text_segmentation_2025.rs:21-24` pins a regression: if segmentation max probability is ≤ 0.05, treat the page as text-free (skip inpaint entirely).
Us: we always run LaMa even on pages with no detections. The fast-path in `lama_inpaint_service.py` handles this somewhat, but we still do mask construction + dilate + gray-close.
Port: ~10 LOC. After CTD segmentation in `/translate`, short-circuit if `mask.max() / 255.0 < 0.05`: skip inpaint, return original as `inpainted`. Emit progress event "skipped inpaint (no text)".
Expected: 30–200 ms saved per empty page (blank panels, opening covers, split-layout title pages).

---

### 8. `looks_like_runtime_library` + shallowest-path archive extract policy — LOW / LOW
Koharu: `koharu-runtime/src/archive.rs:153-168` `remember_shallower_path` resolves name collisions in multi-variant archives (e.g. ZLUDA ships `nvcuda.dll` at three nesting depths for trace/debug/release — pick release). `:177-181` `looks_like_runtime_library` filters by file extension (.dll/.so/.dylib) during extract.
Us: N/A today. We don't ship CUDA/ZLUDA runtimes, but we will bundle an ONNX Runtime GPU wheel if we support remote GPU installs, and the pattern applies to any multi-variant archive (e.g. PARSeq wheels).
Port: defer until we actually ship optional GPU archives. Note the pattern for future use.
Expected: future-proofing; no immediate win.

---

### 9. Track per-stage timings with `tracing::info!(preprocess_ms=..., forward_ms=..., postprocess_ms=..., total_ms=...)` — LOW / LOW
Koharu: `koharu-ml/src/speech_bubble_segmentation/mod.rs:208-218` emits a structured tracing event per inference with the three stage times plus detection count. Same pattern in `manga_text_segmentation_2025/mod.rs:97-107` and `font_detector/mod.rs:163-171`. Used to identify bottlenecks — the benchmark in `koharu-renderer/benches/rendering.rs:36,45` splits "layout" from "render" for the exact same reason.
Us: we emit aggregate per-stage timings in SSE progress events but don't split preprocess/forward/postprocess. When PARSeq is slow we can't tell if it's the batch-encode or the ONNX forward.
Port: ~15 LOC per service. Wrap each sub-step in `time.perf_counter()` deltas and log at DEBUG; emit as a `stage.detail` SSE event for the UI dev panel.
Expected: zero production cost; significantly faster diagnosis when a customer reports slow pages ("is it OCR forward or OCR preprocess?").

---

### 10. PSD text-layer metadata pattern → live canvas text-box metadata — LOW / MED
Koharu: `koharu-psd/src/export.rs:57-71` `TextLayerMetadata` captures `rotation_rad`-derived `transform[6]` (affine matrix), `orientation`, `justification`, `font_name`, `faux_bold`, `faux_italic`, `box_width`, `box_height`. The justification inference at `:634-652` uses `TextAlign` from style with fallback to "center for Latin horizontal, left otherwise" — a direct rule we could port for overlay positioning when `PsdTextAlign` is absent.
Us: `overlay-renderer.ts` has ad-hoc alignment and no rotation support for rotated bubbles. We lose the per-block angle the font_detector already predicts (`regression[9]`, `angle_deg`).
Port: ~40 LOC. Add `rotationDeg` to the TextBox JSON (`animetext_service.py` already has access via the font_prediction block). In `overlay-renderer.ts` drawStep, apply `ctx.rotate(rotationDeg * π/180)` around the block centroid before drawing. Reuse the Latin-vs-CJK justification rule.
Expected: tilted speech bubbles (~5–10% of manga pages have at least one rotated balloon) render correctly instead of axis-aligned.

---

## Skipped / not worth porting

- **PSD packbits RLE encoding** (`koharu-psd/src/packbits.rs`) — only relevant if we export PSDs. We don't.
- **`koharu-llm/src/providers/caiyun.rs`** 17-language whitelist pattern — useful only if we add Caiyun. No current user demand.
- **`koharu-renderer` ICU4X line-breaker** — our Canvas renderer already uses the browser's line-break tables which are equivalent on Chromium.
- **`koharu-runtime/src/zluda.rs`** — AMD GPU acceleration via ZLUDA translation shim; out of scope for Python/ONNX backend.
- **`manga_text_segmentation_2025` GPU_MAX_PIXELS 1536×1536 vs CPU_MAX_PIXELS 1280×1280** — we already downsample CTD input; adding per-device budget for a segmentation model we don't use is premature.
- **Recursive XY-cut already proposed in round 2** — reading-order sort in `comic_text_bubble_detector` at `:458-494` doesn't sort, koharu sorts later in `koharu-app` which we haven't copied (out of scope for backend port).
