# Round 8 Research — Strict novelty hunt

Generated: 2026-04-22. Corners: runtime/install.rs, runtime/loader.rs, pipeline/engines/support.rs, pipeline/mod.rs, comic_text_bubble_detector ImageSlicer, UI canvas (Workspace, TextBlockLayer), threshold grep across koharu-ml, font_detector/mod.rs, engines/mit48px_ocr.rs, engines/paddle_ocr.rs. Everything else is already on R2-R7 backlog (font regression, XY-Cut reading order, color clamps, per-engine tracing, merge_slice_regions) or non-applicable (Windows DLL search, CUDA loader, Tauri project save).

## Novel Items

### 1. ImageSlicer for tall pages (webtoon/vertical strip) — HIGH / LOW
Koharu: `koharu-ml/src/comic_text_bubble_detector/mod.rs:92-95, 700-775`. Before RT-DETR bubble detection, checks `height/width > 3.5`. If true, slices into overlapping horizontal strips of `width × (3.0·width)` with 20% overlap, detects per-slice, y-offsets each bbox, then runs `merge_slice_regions` to dedupe. Trailing slices shorter than `0.7·slice_height` fold into the previous slice.

Us: `ctd_service.py`, `animetext_service.py`, `detector_service.py` always pass the full image at whatever `det_size` the backbone expects (~1024). A 1800×8000 webtoon gets squashed to 1024-square, destroying small-text resolution. We ported the *merger* (ctd_service.py:597,656) but not the *slicer* — the merger has no slices to merge. Directly relevant to `animetext_1104718` (zero-detection on tall display-text pages) because character pixel area collapses below the detector's receptive field after whole-page downscale.

Port: ~60 LOC new `backend/app/utils/image_slicer.py` (constants 3.5 / 3.0 / 0.2 / 0.7). Gate behind `settings.slicer_enabled=True`. Call in both detector services before the model; existing `merge_slice_regions` handles dedup. Detect time scales linearly with slices (~3-5 for a tall webtoon).

Refs: `/tmp/koharu/koharu-ml/src/comic_text_bubble_detector/mod.rs:700-775` + `:92-95`.

### 2. Per-step failure isolation with WarningTick — MED / LOW
Koharu: `koharu-app/src/pipeline/mod.rs:112-272`. Each engine step (detect/OCR/translate/inpaint) runs inside its own `match step_result`. On failure, emits one `WarningTick`, increments `warning_count`, and `continue 'pages` — **prior stage ops stay committed**. Final return is `RunOutcome { warning_count }`, letting callers mark the run `CompletedWithErrors` rather than empty.

Us: `routers/translate.py:395-399` catches at `process_single_image` level; on exception returns `(idx, [], None)` — zero boxes, no partial output. Inpaint has graceful fallback (line 328 `continuing without plate`), but a translate-pool exception nukes OCR output too. We already tolerate `japanese_filter_enabled` producing empty; the symmetric handling of a translate exception returning `TextBox(ocrText=JP, translatedText="")` would let the frontend overlay source text rather than hiding the bubble entirely.

Port: ~40 LOC inside `process_single_image`. Separate try/except around translate (and a second one around OCR) that preserve prior-stage output. Add `warning_count` to the response for observability. Fixes the "everything-or-nothing" class for multi-image requests where one translate hiccup currently yields an empty page.

Refs: `/tmp/koharu/koharu-app/src/pipeline/mod.rs:153, 198-272, 284-295`.

## Saturation signal
Seven corners deep-dived; 20+ files skimmed. Two novel items, down from R7's three. ImageSlicer is the stronger find — directly maps to a known failure. Per-step isolation is robustness polish. R9 should expect 0-1; recommend declaring saturation after 1 more pass unless a new signal (new failure captures, different koharu subsystem) enters scope.
