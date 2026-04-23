# Round 7 Koharu Novelty Hunt

## Coverage checks performed
- git log: shallow clone (1 commit), no behavioural history available.
- speech_bubble_segmentation/mod.rs: fully read; geometry helpers reusable but seg-model itself is covered by R5-#3 (per-bubble ID mask) and R5-#5 (BubbleIndex growth).
- integration-tests/*.rs: pipelines.rs, scene.rs, events.rs, binary.rs, meta.rs inspected.
- koharu-app/src/pipeline/: `engines/` enumerated (aot, bubble_segmentation, comic_text_bubble, ctd_full, ctd_segment, lama, llm_translate, manga_ocr, mit48px_ocr, paddle_ocr, pp_doclayout, renderer, support, yuzumarker_font). `support.rs` is the only one not already cited in R2-R6.
- Error-handling grep (unwrap_or_else/expect/bail): graceful-degradation path we hadn't cited before — mit48px emits `bail!("missing local mit48px assets…")` instead of auto-downloading, consistent with our offline-only posture; not a port candidate.
- Magic-number grep: CTB vertical-merge thresholds and seg mask_crop_window biases examined against ours.

## Genuinely novel items

### R7-1. CTD vertical-merge y-threshold uses wrong axis — LOW/LOW
Koharu: `koharu-ml/src/comic_text_bubble_detector/mod.rs:543,583` computes
`y_distance_threshold = image_height * 0.1`, then
`local_y_threshold = y_distance_threshold.min(max(h1,h2) * 0.1)` — i.e. "10% of the image, capped by 10% of the **taller** box".
Us: `backend/app/services/ctd_service.py:625,640` uses `min_height = min(h1,h2)` with test `y_dist < min_height * 0.1`. Two divergences: (a) uses **min** instead of **max** of the two heights — stricter for mismatched pairs, so we under-merge a small+large neighbour pair even when vertically adjacent; (b) no image-height ceiling, so very tall boxes on a short image get an unbounded threshold.
Port: ~4 LOC. In `_should_merge_koharu`, add `image_height` parameter (thread through from `_derive_blocks_from_text_lines` which has image dims in `postprocess`), replace line 640 with `local_thr = min(image_height * 0.1, max(height_a, height_b) * 0.1); if y_dist < local_thr`. Expected effect: a few extra vertical-stack merges per page (1-2 blocks/page), closer parity with koharu's empirical tuning.

### R7-2. SSE `Last-Event-ID` replay is stubbed, not implemented — MED/MED
Koharu: `tests/integration-tests/tests/events.rs:1-14` explicitly asserts "`Last-Event-ID` reconnect replays missed events from the ring buffer" and "Lag fallback: subscriber that falls off the broadcast buffer is re-seeded with a fresh `Snapshot` instead of terminating". Their pipeline test at `tests/integration-tests/tests/pipelines.rs` also exercises the `/events` contract.
Us: `backend/app/routers/events.py:73-77` reads the `Last-Event-ID` header, logs it, then comments "replay unsupported; continuing forward-only". The endpoint accepts the header for spec compliance but silently drops events that occurred during the disconnect window — extension reconnects after a network blip lose the `finished` frame and hang in "processing" state until timeout.
Port: ~40 LOC in `progress_bus.py`. Add a `collections.deque(maxlen=256)` per `job_id` that retains `ProgressEvent` after emit. In `_event_stream`, before subscribing, if `last_event_id` parses as int, yield every buffered event with `seq > last_event_id`, then attach to the live subscription. On lag overflow emit a single synthetic `event: snapshot` frame carrying the job's current `JobStatus` (mirrors koharu's "re-seed with Snapshot" fallback). Eliminates the hung-spinner reconnect bug we see in flaky-network testing.

### R7-3. Speech-bubble mask_crop_window letterbox-unpad helper is reusable — LOW/LOW
Koharu: `koharu-ml/src/speech_bubble_segmentation/mod.rs:483-504` — given `(original_w, original_h, proto_w, proto_h)`, computes the letterbox padding in proto-space and returns the inner crop `(top, left, bottom, right)` with a `-0.1 / +0.1` round-bias that keeps the crop 1-pixel inclusive on the padded edge. Unit-tested at `:569-573` (`mask_crop_window(1000,500,160,160) == (40,0,120,160)`).
Us: R5-#3 (per-bubble ID mask) and the eventual R5-#5 (BubbleIndex) both need to map a YOLO-seg proto-mask back to original image coordinates after letterboxing. Today `backend/app/utils/image_processing.py` has ad-hoc letterbox code in the inpainting path but no shared "crop out the padding after upsample" helper; the naive `imageops.resize` path we use also silently stretches the padded mask, leaking a thin probability fringe along two edges of every bubble.
Port: ~15 LOC as `backend/app/utils/image_processing.py::letterbox_unpad_crop(orig_w, orig_h, mask_w, mask_h) -> (top, left, bottom, right)` with the same `-0.1/+0.1` bias, plus the same golden unit test value. Prerequisite when we port R5-#3; without it the per-bubble ID mask will inherit a 1-2px bleed at the letterbox seam.

## Dependencies
R7-1 is isolated. R7-2 is isolated and the only non-ML item. R7-3 is a strict prerequisite for R5-#3 and R5-#5 — pull it in as part of that work, not standalone.
