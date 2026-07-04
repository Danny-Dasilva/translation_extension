"""Configuration management using Pydantic Settings"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Keys (optional - only needed for cloud fallback)
    gemini_api_key: Optional[str] = None

    # Server
    # Canonical port layout: vLLM serves on 8000 (vllm_base_url below), this
    # FastAPI backend serves on 8001, and the browser extension talks to 8001.
    # This avoids the old collision where both defaulted to 8000.
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = True

    # CORS
    allowed_origins: str = "*"

    # Rate Limiting
    max_requests_per_minute: int = 60
    max_images_per_request: int = 5

    # Translation
    default_target_language: str = "English"

    # Local AI Model Paths
    yolo_model_path: str = "app/models/yolov10n_manga.pt"
    # Note: manga-ocr auto-downloads its model, no path config needed

    # OCR backend selection: "parseq" (local trained model) or "manga-ocr"
    ocr_backend: str = "parseq"
    parseq_model_path: str = "models/parseq_manga_ep60_nonAR_dynbatch.fp16.onnx"
    # Batched non-autoregressive export with a dynamic batch axis: one forward
    # pass OCRs all lines on a page (~10x faster than the old AR_single model,
    # which had a hardcoded batch=1 Reshape and required N sequential forwards).
    #
    # NOTE: this non-AR export was evaluated 2026-06-14 against the AR_single model
    # (parseq_manga_best_ep60_AR_single.onnx) on a labeled per-line GT set. RAW, it
    # REGRESSED accuracy (+1.04pp mean CER over the +0.5pp bar) and emitted non-AR
    # repeat artifacts on single-line crops ('体体体体体', '...。..', 'うっ!!!ー!!').
    # We now run it with the repeat-collapse postprocess (collapse_cjk_runs +
    # collapse_trailing_loop in app/utils/ocr_postprocess.py, wired into apply_all)
    # which neutralizes those artifacts; the speed win is taken pending the
    # postprocess-on per-line A/B re-run. To fall back, set this to the AR_single
    # path and parseq_batch_size=1. See
    # thoughts/shared/research/translation-perf-display/2026-06-13_parseq-dynamic-batch-proposal.md
    parseq_batch_size: int = 8

    # Detector Selection: "animetext" (fast) or "ctd" (full-featured).
    # CTD is recommended when ocr_backend="parseq" because PARSeq is a
    # line-level STR model and needs per-line crops (text_lines) for best
    # quality; AnimeText only produces block-level bboxes.
    detector_type: str = "ctd"

    # AnimeText YOLO12s FP16 (3.1x faster than CTD: 414 FPS vs 133 FPS)
    animetext_model_path: str = "models/animetext_yolo12s_fp16.onnx"
    animetext_input_size: int = 640
    animetext_confidence_threshold: float = 0.272  # From model's threshold.json

    # Comic Text Detector (CTD) - includes text_lines and mask
    #
    # v26 retrain (round1seg / "v4", 2026-06-12). Contract differs from V5:
    #   outputs = det[1,300,6] (axis-aligned blocks, NMS-free) +
    #             mask[1,2,H,W] (ch0=text, ch1=onomatopoeia) +
    #             obb[1,300,7] (oriented text lines).
    # The export has a FIXED 1280x1280 input, so ctd_input_size MUST be 1280
    # (feeding 1024 raises a shape error). The seg head is tuned for a 0.8 text
    # threshold; the legacy 0.3 over-flags ~2x on screentone/dark art.
    ctd_model_path: str = "models/comictextdetector_v26_round9_onofix_20260622.onnx"
    ctd_input_size: int = 1280
    ctd_text_threshold: float = 0.8
    ctd_block_confidence: float = 0.4
    ctd_min_text_area: int = 100
    ctd_nms_free: bool = True  # v26 det head is NMS-free (one-to-one, 300 slots)

    # Orphan-line recovery: text_lines whose center sits inside NO detected
    # block are otherwise silently dropped before OCR (SMS balloons, vertical
    # narration columns, dense paragraphs the block detector misses). When on,
    # those orphans are paragraph-clustered, OCR'd, and appended as synthetic
    # blocks so they flow through the SAME filter -> translate -> render path.
    # ALWAYS ON: missing-bubble loss renders raw Japanese to the reader.
    orphan_line_recovery: bool = True

    # Translation backend: "vllm-openai" (vLLM serving an OpenAI-compatible
    # chat endpoint — the v10-it Gemma 4 E4B merged model + Google's MTP
    # drafter) or "transformers" (HF transformers, used for Hy-MT1.5-2bit).
    #
    # PROMOTED 2026-07-03: v1 = Qwen3-VL-8B-abliterated + text-SFT LoRA (merged,
    # bf16), +10.104 chrF++ over the prior Gemma-4 E4B v11fix8 (CI95 [+7.7,+12.9]
    # p=0, fair same-code A/B). Served on the box via box_serve_v1_merged.sh
    # (merged weights + CUDA graphs, VLLM_USE_FLASHINFER_SAMPLER=0, multimodal
    # image=1 so translation_serve_image_context works). No MTP drafter (Qwen3-VL
    # has none). Contract: translation_v11_pagecontext=True + short_utterance_
    # normalize OFF (v1's byte-exact train format; validated by the +10.104 cert).
    # Prior default: v10it (Gemma-4 v11fix8 + MTP, serve_v10it_vllm.sh, local :8000).
    translation_backend: str = "vllm-openai"
    vllm_base_url: str = "http://100.64.235.63:8001/v1"
    vllm_model_name: str = "v1"
    # Max concurrent in-flight requests to the vLLM chat endpoint. This gates
    # every VLLMOpenAITranslationService._chat() call via an asyncio.Semaphore.
    # The service is a module-level singleton shared across ALL concurrently
    # processed pages (up to max_parallel_images pages x ~N bubbles/page each
    # fan out into individual _chat calls), so a low value here bottlenecks
    # translate_ms independent of vLLM's own batching capacity. Raise with the
    # server's max concurrent sequences in mind.
    translation_client_concurrency: int = 32

    # Translation model (transformers backend)
    hymt_transformers_model_dir: str = "app/weights/hymt15-2bit"

    # Weights directory (for downloaded models)
    weights_dir: str = "app/weights"

    # Flag-for-finetune storage. Users flag poor translations from the
    # extension; POST /flag persists the ORIGINAL source image + metadata here
    # as a fine-tune dataset seed (image PNGs + flagged.jsonl record per flag).
    # Relative to the backend working directory; gitignored (data/).
    flagged_dir: str = "data/flagged"

    # Async/non-blocking logging. Structured translation logs are written to a
    # rotating JSONL file under this directory via a QueueListener background
    # thread (logging never blocks the request path). See app/logging_config.py.
    log_dir: str = "logs"

    # Performance Tuning
    detection_confidence: float = 0.25
    detection_image_size: int = 640
    parallel_image_processing: bool = True  # Process multiple images in parallel
    # Max concurrent image pipelines. RTX 5090 (32GB) comfortably holds CTD +
    # YOLO + PARSeq + LaMa working sets for 4 in-flight pages; vLLM runs
    # out-of-process so it is not bounded by this. Raise further only with a VRAM
    # headroom check.
    max_parallel_images: int = 4

    # Translation parallelization
    translation_use_parallel: bool = True  # Use parallel translation with asyncio.gather

    # Pipeline optimization
    use_pipeline_overlap: bool = True  # Start translation as each OCR completes (overlap OCR+translation)

    # Koharu-inspired stages
    # When enabled, run LaMa inpainting after OCR/translate and return inpainted PNG
    enable_inpainting: bool = True
    lama_model_path: str = "models/lama.onnx"
    # Encode the inpainted "plate" as WebP (lossy, q=82) instead of uncompressed
    # PNG base64. Cuts the per-page plate payload ~91% (PNG 3.38MB -> WebP ~0.28MB)
    # with no visible quality loss on manga line-art. WebP decodes natively in the
    # browser canvas, so no frontend change is needed. Set False to restore PNG.
    plate_encode_webp: bool = True
    plate_webp_quality: int = 82
    # SFX ono-mask erasure (v26 detector only, DEFAULT OFF). The active detector
    # (comictextdetector_v26_round9_onofix_20260622.onnx) emits a 2-channel seg
    # mask (ch0=text, ch1=onomatopoeia/SFX). round9's ch1 fires accurately on
    # stylized SFX glyphs drawn directly over artwork (no text-line/block box),
    # which the existing erase-mask pipeline silently drops: both
    # ``ComicTextDetectorService._process_mask`` (clips to
    # ``_build_block_bounds_mask``) and ``ctd_utils.build_inpaint_mask`` (clips
    # detector-seg ink to ``detected_area``) only erase pixels inside a detected
    # text region, so on-art SFX with no box ships raw Japanese. When True, the
    # unclipped ch1 mask (``ctd_result["ono_mask"]``) is OR-ed into the final
    # erase mask WITHOUT the block/line clip, so free-floating SFX ink actually
    # gets erased. When False (default), ono_mask is never consumed downstream
    # and behaviour is byte-identical to before this flag existed. Gated pending
    # a full-page GPU render audit (Step-0 experiment covered detection-only,
    # not the composited output) before flipping the default on.
    inpaint_ono_mask: bool = True

    # --- Final-composite font readability + page consistency ---------------
    # READABILITY FLOOR (resolution-aware). The minimum rendered font size is
    # ``max(render_font_abs_floor, image_height * render_font_floor_frac)`` so the
    # floor scales with page resolution (manga pages here are ~1000-2000px tall).
    # When text doesn't fit at this floor we wrap to more lines / allow modest
    # bubble overflow rather than shrinking below it (see compose_final). The
    # absolute floor is the hard minimum for tiny pages; the fraction dominates
    # at full manga resolution (e.g. 1791px * 0.012 ≈ 21px).
    render_font_abs_floor: int = 18
    render_font_floor_frac: float = 0.012
    # Hard floor for clamped (no-bubble) SFX/caption blocks that genuinely cannot
    # fit at the resolution-aware floor without burying neighbouring art. These
    # are legitimately variable, so they get a lower hard minimum than dialogue.
    render_font_clamped_hard_floor: int = 12
    # Largest font the binary-search fit may select (prevents one-word bubbles
    # from rendering at absurd sizes).
    render_font_max_cap: int = 96
    # PAGE-LEVEL CONSISTENCY. When True, dialogue (bubble-matched) blocks on a
    # page are driven toward a SHARED target size — a low percentile of the
    # per-bubble max-fit sizes — so most bubbles render at one readable size and
    # only genuinely tiny bubbles deviate. SFX/caption (clamped, no-bubble)
    # blocks stay on their own independent track (they are legitimately
    # variable). Set False to A/B against the prior per-bubble-independent fit.
    render_consistent_font: bool = True
    # Percentile (0-100) of per-bubble max-fit sizes used as the shared dialogue
    # target. A MODERATE percentile (not the min) keeps most bubbles readable and
    # biases the page LARGER; the existing overflow_frac slack absorbs the few
    # bubbles that then spill. Was 35 (biased the whole page small); 60 reads
    # closer to human scanlation sizing.
    render_consistent_font_percentile: int = 60
    # bubbleRect-gated interior solid-fill inpaint tier (R1 hybrid). When on, the
    # LaMa service fills flat speech-bubble interiors with their robust median
    # background and skips the neural forward for those components. Purely
    # additive + gated; False instantly restores the prior 3-tier behaviour.
    enable_bubble_solid_fill: bool = True
    # Final inpaint tier: when True, the textured/screentone residual that used to
    # go through the LaMa neural forward is instead handled by cv2.inpaint
    # (Navier-Stokes, r=3) — a purely classical (no-AI) reconstruction. The
    # bubble solid-fill / ring fast-path / classical-NS tiers are unchanged; this
    # only swaps tier-3. Audit (2026-06-13_noai-inpaint-audit.md) over 11
    # benchmark pages found 85% of inpainted pixels are hidden by the re-rendered
    # translation and the visible residual is imperceptible on dialogue; only
    # large SFX-over-detailed-art (which is largely left un-erased in production
    # anyway) is mildly softer than LaMa. Removing the neural tier drops the
    # 208MB ONNX model load + GPU working set and the ~28ms/forward on the ~40%
    # of components that previously hit the model. Set False to restore LaMa.
    use_neural_inpaint: bool = False
    # Overlap LaMa inpaint with OCR+translate. Inpainting only needs the detection
    # mask (not translated text), so it can run concurrently with the OCR/translate
    # stage instead of serially after it. Runs in a worker thread so the event loop
    # stays free to drive the vLLM translate calls.
    overlap_inpaint: bool = True
    # When enabled, detect speech bubbles (YOLOv10n) and expose the matched
    # bubble interior per text box (bubbleRect) so the frontend can typeset the
    # translation to the bubble rather than the tight (vertical-JP) text column.
    enable_bubble_fit: bool = True
    # When enabled, use page-level [N]-tagged batched translation (coherence win)
    # instead of per-bubble parallel calls. Fallback to parallel on failure.
    # NOTE: the current vLLM translate_batched fans out concurrent single-bubble
    # calls — it does NOT pack bubbles into one prompt. See batch_translate below
    # for the true single-call numbered-block path.
    use_batched_translation: bool = True

    # TRUE single-call numbered-block translation: pack all of a page's bubbles
    # into ONE vLLM generate call (1.,2.,3.… prompt, numbered output parsed back).
    # ON: page-level translation gives the model intra-page context (speaker /
    # possessive consistency) + a system prompt that locks output to the target
    # language. Falls back to per-bubble parallel on any count/parse mismatch.
    batch_translate: bool = True

    # A/B FLAG: prepend a SHORT genre/self-reference system message to the
    # page-level numbered-block call (translate_numbered_block). Default False
    # because v10it is prompt-sensitive (it collapses on the heavy few-shot
    # BATCHED_SYSTEM_PROMPT). When True, sends [system, user]; the light prompt
    # (LIGHT_SYSTEM_PROMPT) targets the お母さん/母さん -> "my mom" self-reference
    # error. The Part13 A/B decides the default. Override via env
    # TRANSLATION_SYSTEM_PROMPT_ENABLED=true|false.
    translation_system_prompt_enabled: bool = False

    # v11 PAGE-CONTEXT translation format (default ON). When True, page
    # translation uses the EXACT context-augmented single-line format the v11
    # `gemma4_e4b_v11_pagecontext` LoRA was trained on (see
    # backend/scripts/data/v11/build_v11_dataset.py): for a page of N bubbles we
    # issue N calls, each carrying the FULL numbered page as context and ONE
    # "Translate line k: …" marked line; the assistant returns just that one
    # line. The shared "Page:\n1. …\nN. …" prefix is byte-identical across the N
    # calls so vLLM prefix-caching amortizes it. `translate_single` (no page
    # context) uses the matching PLAIN v11 format
    # ("Translate the following Japanese to English. …\n\nJapanese: {jp}").
    # When False, the prior numbered-block ([N]/tagged) path is used unchanged.
    #
    # NOTE: only meaningful when serving the v11 merged model
    # (backend/training/runs/manga-bubbles/gemma4_e4b_v11_pagecontext/merged).
    # Serving v10it with this ON would feed v10it an out-of-distribution prompt.
    translation_v11_pagecontext: bool = True

    # WHOLE-PAGE context for v11 page-context translation. The v11 LoRA was
    # trained to translate one marked line while seeing the WHOLE page's dialogue
    # (all lines, in reading order) as the numbered "Page:" context. When True
    # (default), DIALOGUE lines dropped before translation (OCR-gate / garble,
    # see is_dialogue_context_candidate) are still passed as CONTEXT-ONLY lines so
    # the served page has no holes — but they are NOT translated/rendered. Pure
    # SFX boxes are never included as dialogue context. When False, only the KEPT
    # lines form the context (the prior behaviour).
    #
    # NOTE: garbled-OCR dropped lines carry noisy text; including them is a
    # train/serve-faithfulness win (no gaps) but adds OCR noise to the context.
    # This flag exists so the trade-off is empirically tunable; validated ON.
    translation_pagecontext_whole_page: bool = True

    # CROSS-BUBBLE SENTENCE MERGE (pre-translation re-segmentation). One JP
    # sentence typeset across 2-3 stacked bubbles in the SAME column is otherwise
    # translated as independent fragments whose halves contradict (p8
    # "今朝はあの子達が" negating the paired "didn't come"; "からな" -> "It's from
    # you"). When True, strictly-adjacent same-column lines are fused into ONE
    # translation unit when the LEADING line dangles on a connective (て/で/が/の/
    # けど, no terminal 。!?…) or the TRAILING line is a bare sentence-final
    # particle (からな/のに/なさい/だろう). The merged JP is translated as one
    # marked line (NO prompt-template change -> NO train/serve risk) and the
    # English is re-split back to member bubbles (full EN in the first bubble,
    # blank continuation bubbles). Pure re-segmentation; validated main-side on
    # GPU. See app.utils.sentence_merge.
    translation_sentence_merge: bool = True

    # COLUMN -> PARENT-BUBBLE GROUPING (pre-translation re-segmentation; P1). A
    # multi-column vertical speech balloon is detected/OCR'd as ONE box PER
    # COLUMN, so it arrives downstream as N independent translation units. The
    # v11 page-context model then either folds the whole sentence onto ONE
    # fragment and BLANKS the rest (silent omissions) or reconstructs it on EACH
    # fragment (identical EN duplicated across adjacent bubbles); every fragment
    # also gets its own render box (clutter). When True, the column-fragments of
    # ONE balloon are grouped into a SINGLE translation unit BEFORE the marked
    # translate call (one balloon = one marked JP line + one render box, EN
    # rendered once). Grouping prefers the CTD parent-bubble membership (the YOLO
    # bubble detector, present on both pipelines) and falls back to geometric
    # column adjacency; conservative guards (same parent only, Y-overlap
    # required, capped X gap + span) keep genuinely-separate balloons apart. Pure
    # re-segmentation reusing the sentence-merge plan -> NO prompt/template change
    # -> NO train/serve risk. See app.utils.bubble_grouping.
    translation_bubble_grouping: bool = False  # DISABLED. Rework #2 (membership column-adjacency+RTL+glyph-width+panel guards) FAILED validation 2026-06-29: Ikenie4 regen corrected-omissions 14(off)->50(on), naive 86->277. Root cause is now DEEPER than membership over-merge: even a CORRECTLY-grouped long multi-column balloon loses text in the merge->translate->resplit roundtrip (the model consolidates the fused JP onto the lead, resplit blanks the continuations without redistributing — e.g. p113 6-column plea truncated). Needs a resplit/redistribution fix or a span cap before re-enable. P2 backfill+dedup stay on.

    # DETECTION-TIME BALLOON-COLUMN FUSION (pre-OCR re-segmentation). The
    # systemic-defect fix all four pipeline-audit lenses converged on. CTD emits
    # one block per text COLUMN, so a multi-column vertical balloon arrives as N
    # independent OCR/translation units (the model duplicates or blanks siblings).
    # When True, the side-by-side columns of ONE balloon are fused into a SINGLE
    # block *before* crop/OCR (see ComicTextDetectorService.fuse_balloon_columns),
    # so OCR sees ONE crop and translation sees ONE JP string per balloon. This is
    # DISTINCT from (and cleaner than) the disabled `translation_bubble_grouping`
    # above: there is NO merge->translate->resplit roundtrip to lose text on (the
    # exact failure that killed the 2026-06-29 attempt, which retrofitted grouping
    # onto already-split OCR) — the fused balloon is one unit end-to-end. Fusion
    # is membership-gated + guarded (same YOLO parent bubble ONLY, tight
    # column-adjacency geometry reused from app.utils.bubble_grouping, panel-area
    # guard, span cap), so different balloons / wide SFX / panel containers never
    # fuse. Requires the YOLO bubble detector (no bubbles => no fusion). Default
    # OFF: safe opt-in pending GPU regen + 3-way omission audit.
    detection_time_balloon_grouping: bool = True

    # SAFETY NET 1 (P2.1): EMPTY-BUBBLE BACKFILL. After the marked page-context
    # output is parsed, any KEPT high-OCR-confidence non-empty JP bubble that
    # ended up with an EMPTY translation (the model folded its sentence onto a
    # neighbour) is re-translated via the deterministic single-line PLAIN path.
    # Intentionally-blanked merge continuations are skipped (their EN is on the
    # lead bubble). Recovers ~45-55 omitted bubbles/chapter. See
    # app.utils.bubble_grouping.select_backfill_targets.
    translation_empty_bubble_backfill: bool = True

    # SAFETY NET 2 (P2.2): ADJACENT IDENTICAL-EN DE-DUP. After finalization,
    # adjacent same-balloon bubbles whose normalized EN is identical / one a
    # substring of the other (the model independently reconstructed the whole
    # sentence on each column P1 missed) are collapsed — full EN kept on the lead
    # bubble, continuation blanked (mirrors the merge contract). Conservative:
    # only column-adjacent bubbles with >=8-char EN. See
    # app.utils.bubble_grouping.dedup_adjacent_identical.
    translation_adjacent_dedup: bool = True

    # P0 coverage (2026-06-30): bubble-keyed final dedup ("1 balloon = 1 string").
    # When a detector ran, collapse same-balloon bubbles to ONE EN (winner =
    # largest-area block) — supersedes the narrow adjacent dedup for the in-balloon
    # case. See app.utils.bubble_grouping.dedup_by_bubble.
    translation_bubble_dedup: bool = True

    # P0 content-drop fix (2026-07-01): FUSED-BALLOON RETRANSLATE. dedup_by_bubble
    # enforces "1 balloon = 1 string" — within a detected balloon it keeps ONE
    # non-empty EN (largest-area winner) and BLANKS the rest, independent of
    # string equality. That is correct for the DUP case (the model reconstructed
    # the same utterance on each column-fragment), but when a multi-column balloon
    # holds DISTINCT lines (e.g. 平然と家族で朝ごはんを + 違う…!!) each fragment was
    # translated separately, so the winner's EN covers ONLY its own line and the
    # blanked siblings' content is silently DROPPED from the page — the same
    # merge->translate->resplit content-loss class documented on
    # translation_bubble_grouping above. When True, a balloon whose blanked
    # siblings MEANINGFULLY DIVERGE from the winner (not near-duplicates) triggers
    # ONE extra marked-line call on the balloon's FUSED JP (member OCR joined in
    # reading order, page context otherwise unchanged, SAME build_v11_context_prompt
    # path); the fused EN lands on the winner and the siblings stay blank (the
    # sentence_merge contract). Near-duplicate siblings keep the plain blank
    # (no extra call). When False, behavior == current (blank divergent siblings).
    # Cost is bounded: one call per multi-block balloon with divergent ENs.
    # See app.utils.bubble_grouping.plan_bubble_dedup / apply_fused_balloon_retranslate.
    translation_balloon_fused_retranslate: bool = True
    # Backfill safeguard: a merge-continuation is recovered standalone when its
    # lead's EN is shorter than this fraction of the fused-group JP length (i.e.
    # the lead was truncated and did NOT carry the whole sentence).
    translation_backfill_lead_truncation_ratio: float = 0.5

    # A/B FLAG (item 4): CAST / ROLE ANCHOR for the v11 page-context prompt.
    # When True, build_v11_context_prompt inserts ONE in-body "Cast: Name (role,
    # pronoun); ..." context line BETWEEN the instruction and the "Page:" block,
    # to anchor pronoun/gender + named-entity resolution (the dominant remaining
    # model bucket). Default FALSE => the served prompt is BYTE-IDENTICAL to the
    # trained v11 template (proven by tests/unit/test_cast_anchor_prompt.py).
    #
    # CRITICAL: the anchor is an IN-BODY context line, NEVER a `system` message —
    # a system message on this format-sensitive page-context path is the
    # ~95% chrF++-collapse risk class (see MEMORY.md chat-template-mismatch).
    # To be A/B'd on backend/scripts/data/v11/eval_pagecontext_heldout.jsonl.
    # Override via env TRANSLATION_CAST_ANCHOR=true|false.
    translation_cast_anchor: bool = False

    # A/B FLAG (item 5): NARRATION-CAPTION 3rd-person conditioning for the v11
    # page-context prompt. Manga carries two box kinds: spoken dialogue bubbles
    # and NARRATION captions (rectangular boxes — a narrator's aside, not a
    # character speaking). The caption-vs-dialogue box kind is detected upstream
    # but discarded before the prompt is built, so narration is translated with
    # the same speaker/pronoun pressure as dialogue and often comes out in an
    # inappropriate first/second person. When True AND the marked line is a
    # narration caption, build_v11_context_prompt inserts ONE in-body directive
    # line (BETWEEN the "Page:" block and the "Translate line" directive) asking
    # for a third-person render. Default FALSE => the served prompt is
    # BYTE-IDENTICAL to the trained v11 template (proven by
    # tests/unit/test_narration_prompt.py); the directive is opt-in AND only
    # fires for lines the caller marks as narration.
    #
    # CRITICAL: like the cast anchor, this is an IN-BODY context line, NEVER a
    # `system` message — a system message on this format-sensitive page-context
    # path is the ~95% chrF++-collapse risk class (see MEMORY.md
    # chat-template-mismatch). Override via env
    # TRANSLATION_RENDER_NARRATION_3RD_PERSON=true|false.
    translation_render_narration_3rd_person: bool = False

    # IMAGE-CONTEXT SERVE PATH (v1 Qwen3-VL-8B text-SFT). When True, each v11
    # page-context marked call is sent as a MULTIMODAL message — the page image
    # block FIRST, then the BYTE-IDENTICAL build_v11_context_prompt text — and a
    # one-shot warm call pre-warms the shared image+instruction prefix so the
    # image KV is prefilled once per page (multimodal prefix caching, verified by
    # backend/scripts/eval/bench_image_prefix.py). v1 is text-trained but
    # measurably exploits a page image supplied at inference (best POV arm).
    #
    # ON by default — validated E2E and rolled out. MUST only be enabled for an
    # IMAGE-CAPABLE serve (a VL model behind /v1); enabling it against a
    # text-only served model would send image blocks it cannot consume. The TEXT
    # portion stays byte-identical to the trained template (no train/serve
    # drift) whether on or off — proven by tests/unit/test_image_context_serve.py.
    translation_serve_image_context: bool = True

    # Normalize short Japanese utterances (interpunct/dot/space-separated kana,
    # runaway repeated kana) before translation so the model isn't destabilized.
    #
    # v1 (Qwen3-VL-8B text-SFT) was trained on RAW builder output — normalize-on
    # diverges on 571/29,467 training rows; the eval/build harnesses already force
    # this False, so defaulting it False makes prod match the certified serve
    # contract (the byte-exact trained template). Do not flip back without
    # re-certifying builder parity (build_textsft_refusalstripped.verify_builder_parity).
    short_utterance_normalize_enabled: bool = False

    # Per-bubble translation generation budget. Manga lines are short, but
    # longer context-aware lines (page-level numbered-block translation, multi-
    # clause narration) need headroom so they aren't truncated mid-sentence.
    # Lower = fewer decode steps on overlong generations. Raise if truncation seen.
    translate_max_tokens: int = 64

    # English early-exit (post-OCR). When a detected region reads HORIZONTALLY
    # (Latin layout, per the CTD per-line `direction`) AND its OCR text is NOT
    # recognized as Japanese, the region is left as original pixels: not
    # OCR-translated, not inpainted, no TextBox emitted. Japanese is near-always
    # vertical in manga; horizontal Latin is English UI/watermark/caption text
    # the reader wants untouched. Guards against horizontal JP SFX via the
    # content (is_japanese) check. The user wants this ON by default.
    english_early_exit_enabled: bool = True

    # Japanese text filter (post-OCR)
    # Filters out non-Japanese text that MangaOCR may hallucinate from English
    japanese_filter_enabled: bool = True
    japanese_filter_min_ratio: float = 0.5  # Min Japanese char ratio (0.0-1.0)
    japanese_filter_katakana_max_length: int = 6  # Max length for katakana-only text

    # OCR-confidence garble gate (pre-translation). Drops a bubble before it
    # reaches the LLM when PARSeq recognition confidence is below this AND the
    # decoded text looks garbled (replacement/bracket scrawl, fails the JP
    # filter, or low JP-char ratio). Stops hallucinated captions on stylized
    # SFX. Conservative: high-confidence text is never dropped. Set <=0 to
    # disable. Tuned on Part13 inspection (dialogue ~0.9+, garbage SFX ~0.5-0.6).
    ocr_confidence_gate_enabled: bool = True
    ocr_confidence_gate_threshold: float = 0.65

    # Join per-block OCR lines with "\n" instead of "" (the default). Changing
    # the LLM input delimiter is format-sensitive (v11) and MUST be holdout-
    # eval'd before flipping; default False keeps the input byte-identical.
    ocr_line_join_newline: bool = False

    # Confidence-gated HYBRID OCR. Default path is the fast non-AR PARSeq model;
    # crops whose recognition confidence is below ocr_confidence_gate_threshold
    # (stylized/handwritten SFX the non-AR model garbles) are re-OCR'd in ONE
    # batch by the higher-quality autoregressive (AR) model, and that result
    # replaces the non-AR one. Only low-confidence crops pay the ~10x AR cost.
    # The existing garble gate then runs on the AR result, so genuinely
    # illegible SFX still drop. Order: non-AR -> AR-retry-on-low-conf -> gate.
    # Set False to use non-AR only.
    hybrid_ocr_enabled: bool = True
    # fp32 AR export with a dynamic batch axis; ORT applies fp16 kernels at
    # runtime. Same input spec/preprocessing/letterbox/normalization as the
    # non-AR model (images: tensor(float) [batch,3,128,512] -> logits
    # [batch,51,4407]); decode is byte-identical to the trusted AR_single model.
    parseq_ar_model_path: str = "models/parseq_manga_ep60_AR_dynbatch.onnx"

    # Vertical-AR-by-default routing. The dominant garble (144-bubble Ikenie-4
    # cohort) is the NAR decode duplicating adjacent kana on dense VERTICAL crops
    # at FALSELY-HIGH confidence (身代わり -> 身身わわ at 0.92), so the conf-gated
    # AR retry above NEVER fires on the worst cases. This routes tall/narrow
    # (h/w >= aspect, the vertical-text signature) crops to the AR model UP FRONT
    # by geometry — independent of confidence and independent of hybrid_ocr_enabled.
    # Horizontal crops stay on the fast NAR path, so only the garble-prone vertical
    # set pays the AR cost. AR is the autoregressive decode that cannot fall into
    # the parallel-decode duplication loops (per-line A/B: vertical CER 24% NAR vs
    # AR clean). Set False to disable (NAR for everything, conf-gated AR retry only).
    ocr_vertical_ar_default: bool = True
    # Aspect trigger: route crops with h/w >= this to AR. Default 1.5 ties to the
    # existing _maybe_rotate_vertical threshold so "rotated-for-vertical" and
    # "routed-to-AR" are the SAME crop set (no surprises). Config-tunable for ablation.
    ocr_vertical_ar_aspect: float = 1.5

    # PER-BUBBLE STREAM EMISSION (WebSocket path only). When True (the
    # default — validated E2E and rolled out), the ws://…/ws/translate/{lang}
    # socket replies with the versioned event-frame protocol (detections ->
    # per-bubble tl -> revise -> plate -> done|error, see src/types/stream.ts)
    # instead of the single monolithic JSON response, so the extension renders
    # each bubble as soon as it is translated. The legacy monolithic reply
    # stays fully supported and is the fallback: set this to False for the
    # WS path to be byte-identical to the pre-streaming behavior. The HTTP
    # POST /translate endpoint is ALWAYS monolithic regardless of this flag.
    translation_stream_events: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def yolo_model_exists(self) -> bool:
        """Check if YOLOv10 model file exists"""
        return Path(self.yolo_model_path).exists()


# Global settings instance
settings = Settings()
