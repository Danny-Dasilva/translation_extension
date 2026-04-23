# Round 9 Koharu Novelty Check

**SATURATED** — 7 corners checked, 0 novel items. Recommend stopping research rounds; switch to implementation of existing 28-item backlog.

## Corners audited

1. **LaMa FFT boundary handling** (`koharu-ml/src/lama/fft/{mod,cpu,cuda,metal}.rs`, 813 LOC) — Pure rfft2/irfft2 custom ops for FFC block. **No** windowing, partial-conv, or boundary-specific logic. Boundary handling lives upstream in `symmetric_pad_rgb` / `pad_mod=8` reflection-pad — already captured as **R6-1**. The cpu.rs uses conjugate-symmetry reconstruction (line 108-118) which is numerically standard; no novelty.

2. **Integration-test numerical fixtures** — `tests/integration-tests/` contains only Rust RPC harness tests (`binary.rs`, `scene.rs`, `pipelines.rs`, `projects.rs`, `meta.rs`, `llm.rs`, `events.rs`) plus openapi client. **Zero `fixtures/*.json`** with pinned numerical values. Koharu does not publish golden-output tensors.

3. **Additional `#[bench]` / criterion blocks** — Confirmed: `Cargo.toml:153` declares criterion, but grep across repo shows ONLY `koharu-renderer/benches/rendering.rs` uses it. **Zero `#[bench]`** blocks anywhere. Already captured in R5 findings #13/#14 (koharu chose tracing spans over ML microbenches).

4. **CHANGELOG 0.40-0.47 bug-fix scan** — Reviewed `[0.47.7]` down to `[0.44.0]`. All substantive items map to existing backlog: `repetition_penalty`/`MAX_NEW_TOKENS=256` (master R2 #3, R4 #6), `Inpainting OOM` (R6-1 padding), `Download stuck at 100%` / `ExponentialBackoff` (R5 #13), font-fallback/RTL (R2 #5, R5 BubbleIndex). Remaining items are Tauri-only (keyring on Linux, panic handler, updater, sentry, ZLUDA/AMD) — not portable to our Python backend.

5. **Release performance claims** — 0.47.2 "Improve inpainting" and 0.46.0 miscellaneous are **unquantified**. No benchmark numbers in release notes. No architectural surprise.

6. **Provider retry/backoff** — `koharu-llm/src/providers/mod.rs:77-103` `ensure_provider_success` detects 429 + quota strings (`insufficient_quota`, `resource_exhausted`, `rate limit exceeded`, `credit balance is too low`) and maps to `provider_quota_exceeded:{provider}`. Our backend is local-model-only; we don't ship provider adapters, so this is **not applicable**. Download retry via `reqwest_retry::ExponentialBackoff` already captured as R5 #13.

7. **Blob cache eviction policy** — `koharu-app/src/blobs.rs:20` `IMAGE_CACHE_CAPACITY: usize = 64` bounded `lru::LruCache<BlobRef, DynamicImage>` over a content-addressed blake3 blob store sharded by first-2-hex (`blob_path` at `:121`). **Architecturally absent from our backend** — we have no persistent content-addressed blob layer; decoded images are per-request. Not a backlog fit — porting would require inventing the blob store itself (which is koharu-specific project-file design, not a translation-pipeline concern).

## Conclusion

Trend confirms saturation: R6=3, R7=3, R8=2, **R9=0**. All seven residual corners either (a) already in the 28-item backlog, (b) architecturally N/A to our Python FastAPI backend, or (c) Tauri/desktop-only concerns. Stop researching. Begin implementing top-priority items (master #1 font_detector regression, #2 LaMa window merging, #3 per-bubble ID mask, R6-1 pad_mod=8 + restore-unmasked-pixels, R6-2 emphasis ligatures, R4 #6 online repetition trim).

