# Koharu Round-6 Research — Strict Novelty Check
Generated: 2026-04-22. Koharu HEAD v0.47.7.

**Answer: YES — 3 genuinely novel items found** after verifying none of them appear on the round-5 consolidated master list or its Part-B deep-dive.

---

## Corners checked (exhaustive, to justify "novelty-only" claim)

- `/tmp/koharu/koharu-runtime/src/{zluda,archive,cuda,downloads,runtime,install,loader,packages,llama}.rs` — nothing net-new beyond master items #3/#13/#18 and deep-dive D3
- `/tmp/koharu/koharu-llm/src/providers/{claude,gemini,openai,deepseek,caiyun,deepl,google_translate,openai_compatible,chat_completions,credentials}.rs` — thin adapters, no novel prompts/knobs beyond D3/D4 and master #18
- `/tmp/koharu/koharu-llm/src/prompt.rs` — `BLOCK_TAG_INSTRUCTIONS` already ported to `local_translation_service.py:42,89-92`
- `/tmp/koharu/koharu-core/src/op.rs` — editor state management only, not pipeline-relevant
- `/tmp/koharu/koharu-psd/src/*.rs` — PSD export format, not applicable to our JSON/overlay output
- `/tmp/koharu/.github/workflows/{test,lint,build,release,pr,publish,labeler,summary,docs}.yml` — cargo + bun + Swatinem rust-cache + integration-runtime cache — already covered as master #20
- `/tmp/koharu/scripts/*.{py,ts}` — mostly dataset-prep & model-conversion. The `connected_text_candidates` directional-morphology line-splitter in `refine_manga109.py:335` is interesting but trainer-side, not runtime
- TODO/FIXME scan across koharu-ml/koharu-app/koharu-llm/koharu-core/koharu-renderer — only 6 non-trivial, all in `koharu-llm/src/safe/` (llama.cpp safe-wrapper internals) and `koharu-renderer/src/{shape,segment}.rs` caching TODOs — not actionable for us
- `/tmp/koharu/koharu-ml/src/{pp_doclayout_v3,manga_text_segmentation_2025,probability_map}` — alt detector stack (out of scope) or already referenced by master #16

All three findings below originate in files not deep-dived in earlier rounds: `koharu-renderer/src/text/script.rs` and `koharu-ml/src/inpainting/strategy.rs`.

---

## R6-1. LaMa reflection-pad to pad_mod=8 + restore-unmasked-pixels — HIGH/MED

**Koharu refs:** `koharu-ml/src/inpainting/strategy.rs:272-293` (`pad_forward` using `symmetric_pad_rgb`), `:208-251` (`run_resize` with original-pixel restore at :243-248), `:131-142` (3-way dispatch: Original / Resize / Crop based on `max_side` vs `crop_trigger=800` / `resize_limit=1280`).

**Us:** `backend/app/services/lama_inpaint_service.py:287-318` (`_forward_one`) hard-codes 512×512 resize for every crop regardless of size. No `pad_mod=8` reflection padding — we resize-and-forward even when the crop is already smaller than 512. No original-pixel restore after forward — LaMa's tiny reconstruction error bleeds into *unmasked* pixels on every crop.

**Port (~110 LOC):**
1. New helper `_pad_forward(crop_img, crop_msk, pad_mod=8)`: compute `pad_w = ceil_multiple(w, 8)`, `pad_h = ceil_multiple(h, 8)`, `cv2.copyMakeBorder(..., cv2.BORDER_REFLECT)`, forward at native resolution, then slice back to `(w, h)`. Skip resize entirely when `max(w, h) ≤ 512`.
2. Fallback: if `max(w, h) > 1280`, take the current resize-to-512 path BUT add post-forward restore: `out[mask<127] = orig[mask<127]` before compositing.
3. Gate via `config.py:lama_use_native_pad=True`.

**Impact:** (a) Small bubbles (~64-256px wide — the majority) get inpainted at native resolution — sharper result, no upscale blur. (b) Quality loss on the Resize fallback drops from "whole crop" to "just the masked pixels". Addresses the artifact where inpainting leaves a faint halo of softened texture around every bubble.

**Dependencies:** independent. Stacks cleanly with master #2 (LaMa crop-window merging) and #3 (per-bubble ID mask).

---

## R6-2. Latin-only uppercase normalization for translated text — MED/TRIVIAL

**Koharu refs:** `koharu-renderer/src/text/script.rs:44-50` — `normalize_translation_for_layout` uppercases the entire string iff every code-point is Latin/Common/Inherited. Called before layout in `layout.rs`. Matches manga convention (English manga lettering is overwhelmingly ALL-CAPS).

**Us:** `src/services/overlay-renderer.ts` draws the raw translation string; `backend/app/utils/image_processing.py` same. No uppercase normalization. A Japanese → English translation renders as `"hello there"` instead of the expected `"HELLO THERE"`.

**Port (~10 LOC):**
- `src/utils/text-utils.ts::normalizeForLayout(s)`: regex test `/^[\x00-\x7F\p{Common}\p{Inherited}]*$/u` or simpler `/^[A-Za-z0-9\s\p{P}]*$/u` — if true, return `s.toUpperCase()`; else return `s`.
- Call at `overlay-renderer.ts` just before text measurement/drawStep.
- Mirror in `image_processing.py::compose_text` for the compositor path.
- Opt-out via `config.py:uppercase_latin_translations=True`.

**Impact:** Instantly raises perceived translation quality on every Japanese-source → English-target panel. Zero inference cost. Trivial regression risk (affects only glyph case, not layout math — but note it may slightly change fitted font size since uppercase glyphs are wider; re-run auto-size AFTER normalization).

**Dependencies:** None. Interacts positively with master #5 (BubbleIndex gives more room) and #10 (emphasis ligatures).

---

## R6-3. Prefer detector `source_direction` over bbox aspect-ratio for writing-mode decisions — MED/LOW

**Koharu refs:** `koharu-renderer/src/text/script.rs:8-32` — `writing_mode_for_block`:
- Rule A: if text has no CJK → always Horizontal (regardless of bubble shape).
- Rule B: else prefer `block.source_direction` (from OCR/detector metadata) over bbox aspect ratio. Aspect ratio is only a fallback for user-added blocks.
- Test at `:274-283`: "Latin text in tall box stays horizontal" — this is the exact bug where our pipeline would stack "HELLO" vertically because the Japanese bubble was tall.

**Us:** `src/services/overlay-renderer.ts` and `backend/app/utils/image_processing.py:537` both use `vertical = bh > bw * 1.2` — a pure bbox-aspect heuristic. We never consult whether the *original* source was vertically-written, and we never special-case "the translation is Latin, so render horizontally regardless".

**Port (~40 LOC):**
1. Thread `source_direction: "horizontal" | "vertical" | null` through `TextBox` / `TextRegion` JSON. In `ctd_service.py` it's already computable from the CTD block's `vertical` flag — just propagate it.
2. In `image_processing.py:537` replace the single `vertical = bh > bw * 1.2` line with:
   ```python
   if is_latin_only(block.translation): vertical = False
   elif source_direction is not None: vertical = (source_direction == "vertical")
   else: vertical = bh > bw * 1.2
   ```
3. Mirror in `overlay-renderer.ts`.

**Impact:** Fixes the common failure mode where an English translation gets stacked one-letter-per-line inside a tall manga speech bubble. Doesn't help CJK→CJK translations, but those are rare for our use case.

**Dependencies:** Complements R6-2 (uppercase). Overlaps with master #5 (BubbleIndex box growth) but solves a different axis of the same readability problem — #5 is "make the box bigger", R6-3 is "stop flipping it to vertical when text is Latin".

---

## Summary — adding to the master list

Updated running total: **23 unported items** (20 from round 5 + R6-1, R6-2, R6-3).

All three R6 items are independent of each other and independent of the top-3 master chain (`font_detector regression`, `LaMa window merging`, `per-bubble ID mask`). R6-2 is the cheapest win on the entire list (10 LOC, trivial effort, high perceived-quality delta on every rendered page).
