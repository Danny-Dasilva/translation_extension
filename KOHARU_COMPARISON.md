# Koharu vs Our Extension — Quality & Performance Audit

Cross-cutting analysis across 10 parallel investigations. Each agent's full report is in the conversation transcript.

## Executive Summary

Koharu's biggest advantages over us: **(1) actual inpainting** (we have none), **(2) a Scene+Op graph that makes per-bubble edit & re-run trivial** (we are one-shot), **(3) batched page-level LLM translation** (we do per-bubble, losing coherence), **(4) mask refinement & crop padding around CTD output** (we threshold only, resulting in bleed & clipped OCR crops).

Our advantages: FP16 PARSeq-large at 7 ms/crop (koharu is FP32 end-to-end), 3-level CUDA/CPU fallback chain, GPU-OOM batch-halving, and fresh 5p16 manga charset training.

## Ranked Action Plan

### Tier 1 — Fix architectural gaps (days of work, big wins)

| # | Change | Impact | Effort |
|---|---|---|---|
| 1 | **Add LaMa inpainting service** — koharu's #1 missing piece on our side. Port `mask.rs` dilation + `balloon.rs` flat-fill fast-path + `strategy.rs` Crop-with-margin (ref `/tmp/koharu/koharu-ml/src/inpainting/`). Use manga-tuned LaMa ONNX (IOPaint `big-lama-manga`), ~100MB fp16. ~150–500 ms/page on 5090. Return inpainted page in API; frontend renders text on clean plate. | **Visual quality transformation** — kills the "translated text over visible Japanese" look. | L |
| 2 | **Page-level batched translation** via koharu's `[1]…[2]…` tagged-block protocol (`/tmp/koharu/koharu-app/src/llm.rs:204-524`). One LLM call per page instead of per-bubble. Port `format_sources` / `parse_tagged_blocks` / `split_legacy_lines` verbatim. Rewrite our system prompt to koharu's manga-translator prompt (`/tmp/koharu/koharu-llm/src/prompt.rs:50-57`). | **Coherence** — intra-page tone/pronoun/name consistency. Also usually faster (one prompt-processing pass). | M |
| 3 | **Scene+Op server-side sessions** (`/tmp/koharu/koharu-core/src/{scene,op}.rs`). Keep per-page state; accept `POST /pipelines {steps, pages, node_id?}` to re-run just translate or just OCR on one bubble. Enables per-bubble retry UX and slashes cost of edits. | Enables every UX improvement below. | L |
| 4 | **Lazy per-engine loading registry** (`/tmp/koharu/koharu-app/src/pipeline/engine.rs:94-131`). Today we eagerly load 6× Llama + PARSeq + CTD at import (~10 GB VRAM). Load on first use behind an `asyncio.Lock`. Add `/unload` endpoint. | Faster startup, lower baseline VRAM, makes detector-only use-cases viable. | M |

### Tier 2 — Quality quick wins (hours of work, measurable wins)

| # | Change | Impact | Effort |
|---|---|---|---|
| 5 | **Block-aware mask refinement** — port `refine_segmentation_mask` from `/tmp/koharu/koharu-ml/src/comic_text_detector/postprocess.rs:12-77`. Threshold → L1 dilate(radius=2) → clip back inside per-block expanded bbox. Currently `ctd_service._process_mask` threshold-only; mask bleeds onto art. | Cleaner inpainting, fewer missed stroke edges. | S |
| 6 | **CTD line-crop expansion** — port `expanded_text_block_crop_bounds`/`maybe_expand_ctd_line` (`postprocess.rs:107-168`). Pad crops by `font*0.10-0.18` direction-aware; our raw `boundingRect` clips kana/diacritics. | ~5–15 % OCR WER improvement, especially on vertical text. | S |
| 7 | **OCR repetition guard** — detect `n`-gram repetition (≥5× same char or trigram) in PARSeq output and emit blank. Koharu added this specifically for VLM degeneration (CHANGELOG 0.47.6-7). | Kills "ててててて…" failure mode. | S |
| 8 | **Morphological close + dilate** on CTD mask (`morph_close(radius=10)` + `dilate(radius=3)` — `comic_text_detector/mod.rs:308-309`). | Residual ink-specks disappear; better inpainting input. | S |
| 9 | **Unicode OCR postproc** — halfwidth→fullwidth (ASCII + 0xFEE0), collapse `…`/`・`/`...` (`/tmp/koharu/koharu-ml/src/manga_ocr/mod.rs:184-226`). | Cleaner JP passed to translator → better outputs. | S |
| 10 | **Binary-search font fit** in overlay-renderer.ts — replace single-pass `(height-20)/totalHeight` (`overlay-renderer.ts:206-214`) with `[8..72]px` binary search that measures actual text bounds per candidate (`/tmp/koharu/koharu-renderer/src/layout.rs:133-167`). | Text always fits; no tiny/overflow cases. | M |
| 11 | **Bubble-merge sanity checks** — port koharu's IoU + containment + size-ratio + edge-alignment tests in `merge_slice_regions` (`comic_text_bubble_detector/mod.rs:518-608`). Our `_derive_blocks_from_text_lines` uses a single 30-px threshold; over-merges across panels. | Fewer "two speakers merged" bubbles. | M |

### Tier 3 — Performance plumbing

| # | Change | Est. saving | Effort |
|---|---|---|---|
| 12 | **Raise PARSeq batch from 4 → 24** with warm-up at N=24. 5090 fp16 fits easily; OOM fallback already exists. | 40–80 ms/page | S |
| 13 | **FP16 CTD export** to ONNX, load via CUDA EP. CTD is currently FP32 (~80 ms → ~45 ms). | 30–40 ms/page | S |
| 14 | **`cv2.imdecode` over PIL base64 decode** in `image_processing.py:12-26`. | 10–25 ms/page | Trivial |
| 15 | **ORT IOBinding + pinned pre-alloc buffer** for PARSeq. Eliminates numpy-↔-CUDA roundtrips per batch. | 15–25 ms/page | M |
| 16 | **TensorRT EP with fp16 engine cache** for CTD (`trt_engine_cache_enable=1`). 30–60 s build, cached thereafter. | 20–35 ms/page on top of #13 | M |
| 17 | **Wire up or delete `app/utils/gpu_semaphore.py`** — defined, imported nowhere. Real GPU gating is `_gpu_semaphore` in `translate.py:30`. Dead code today, but it flags a real bug: llama-cpp and onnxruntime sharing CUDA without gating can hit error 900. | Bug fix | S |

### Tier 4 — UX polish (content-script side)

| # | Change | Impact | Effort |
|---|---|---|---|
| 18 | **Per-bubble retry button** on overlay → resend single-box payload. Biggest UX win. | High | S (depends on #3) |
| 19 | **Inline text editor** — dblclick overlay → contenteditable (`/tmp/koharu/ui/components/panels/TextBlocksPanel.tsx:228`). | High | S |
| 20 | **Floating progress badge** with cancel (port `ActivityBubble.tsx`). Requires SSE stream from `websocket_upload.py`. | M | M |
| 21 | **"Show original" hotkey** (hold Alt → hide overlay). Trivial on top of existing canvas overlay. | M | Trivial |
| 22 | **Draggable/resizable bubble boxes** (`@use-gesture/react` pattern from `TextBlockLayer.tsx:137`). | M | M |

## Non-issues — we already match or exceed koharu

- **FP16 precision.** Koharu is FP32 everywhere (`/tmp/koharu/koharu-ml/src/loading.rs:39`). Our FP16 PARSeq already outperforms their manga-ocr on throughput.
- **Fallback chain robustness.** Our `fp16-CUDA → fp32-CUDA → fp32-CPU` (`parseq_ocr_service.py:96-111`) is more robust than koharu's single CUDA-or-nothing path.
- **Dynamic OOM batch-halving.** We have it (`parseq_ocr_service.py:186-200`), koharu does not.
- **Vertical text handling.** Our explicit 90°-CCW rotation for `h > 1.5*w` crops is simpler and more reliable than relying on koharu's per-line polygon warp.
- **Line-level OCR.** PARSeq-large + CTD text_lines is already koharu's mit48px pattern.

## Reference model repos (for download)

| Purpose | Repo | Size |
|---|---|---|
| LaMa manga inpaint | `mayocream/lama-manga` (safetensors) or IOPaint `big-lama` ONNX fp16 | ~100 MB |
| AOT-GAN alt inpaint | `mayocream/aot-inpainting` | ~40 MB |
| manga-text-seg-2025 | `mayocream/manga-text-segmentation-2025` | ~160 MB |
| speech-bubble-seg | `mayocream/speech-bubble-segmentation` (YOLOv8-seg) | ~25 MB |
| comic-text-bubble-detector | `ogkalu/comic-text-and-bubble-detector` (RT-DETR v2) | ~100 MB |

## Key koharu file map (for future reference)

```
/tmp/koharu/
├── koharu-ml/src/
│   ├── inpainting/{mod,mask,balloon,strategy}.rs   ← mask dilation, bubble fast-path, crop strategy
│   ├── lama/mod.rs                                  ← LaMa model wrapper (safetensors loader)
│   ├── aot_inpainting/mod.rs                        ← AOT-GAN wrapper
│   ├── comic_text_detector/{mod,postprocess}.rs     ← CTD + block-aware mask refine + crop expand
│   ├── comic_text_bubble_detector/mod.rs            ← RT-DETR bubble/text separator + merge heuristic
│   ├── manga_text_segmentation_2025/mod.rs          ← newer text seg model
│   ├── speech_bubble_segmentation/mod.rs            ← YOLOv8-seg instance bubble masks
│   ├── pp_doclayout_v3/mod.rs                       ← panel layout + reading order
│   ├── paddleocr_vl/mod.rs                          ← VLM OCR (opt-in heavy fallback)
│   ├── manga_ocr/mod.rs                             ← classic ViT+BERT OCR
│   ├── mit48px_ocr/mod.rs                           ← line-level CNN+transformer OCR
│   └── probability_map.rs                           ← shared prob-map utility
├── koharu-renderer/src/{layout,renderer,shape,segment}.rs  ← binary-search fit, ICU wrap, harfrust
├── koharu-llm/src/{prompt,model,providers}.rs       ← prompt templates, GGUF + providers
├── koharu-app/src/
│   ├── pipeline/{mod,engine}.rs                     ← DAG toposort, lazy engine registry, SSE ticks
│   ├── llm.rs                                       ← translate_texts batched tagged-block protocol
│   └── blobs.rs                                     ← blake3 content-addressed image cache
├── koharu-core/src/{scene,op,protocol,blob,font,google_fonts}.rs  ← scene graph + op/inverse
├── koharu-rpc/src/{events,routes/{pipelines,operations}}.rs      ← SSE stream, cancellation
└── ui/components/
    ├── panels/TextBlocksPanel.tsx                   ← per-block edit + retry UI
    ├── canvas/TextBlockLayer.tsx                    ← drag/resize boxes
    └── ActivityBubble.tsx                           ← progress/cancel pill
```
