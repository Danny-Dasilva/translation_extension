# Manga-Translation Roadmap: End-to-End Speed + Translation Display Quality

Date: 2026-06-13
Scope: RESEARCH / ROADMAP ONLY (recommend, do not implement)
Evidence: `/tmp/transperf_evidence/` (SUMMARY.md, per-page JSON, *_boxes.csv, *_inpaint.png, *_composite.png, backend.log [authoritative timings], vllm.log)

---

## 1. Executive Summary

- **Baseline (live, 11 pages, backend.log authoritative):** TOTAL median **1276ms** / mean 1353ms / p90 1926ms. Stages are **fully serial** (detect 241ms + fused OCR+translate 690ms + LaMa inpaint 352ms ≈ 1283ms ≈ median). The "pipeline-overlap" mode is a **no-op** for the production PARSeq path — OCR and translation run strictly sequentially.
- **SPEED headline wins (no accuracy gate):** (1) **Overlap inpaint with OCR+translate** → −330ms median (−26%); (2) **parallelize CTD‖YOLO + thread-offload YOLO** → −60-90ms; (3) **WebP plate instead of uncompressed PNG** → response payload −91% (3.38MB → 0.28MB/page), saving ~2.5s of *client-perceived* transfer at 10Mbps that is currently invisible to backend metrics. These three are low-risk and stack.
- **SPEED big bet (gated):** OCR is the dominant stage (690ms median, scales with line count at ~46ms/crop, batch=1 locked). A **dynamic-batch PARSeq export** drops per-crop 46ms→3.4ms (12x) → dense-page OCR 931ms→~65ms. **BLOCKED** on building a labeled per-line OCR eval set; A/B verdict is currently INCONCLUSIVE.
- **DISPLAY headline wins:** **bubbleRect hit-rate 67.1%** (55/82); the 27 misses + **all 9 overspill boxes (11%)** cluster on 2 SFX-heavy pages (637653_045, _090). Root cause is the **null-bubble fallback to the tight OCR bbox → 8px floor**, not the sizing search. Fix the fallback (expand region + use `fontHeightPx` as anchor), couple the mask plate to the layout region, and bundle the comic font locally (currently the @font-face files 404 and the only working glyph source is the Google Fonts CDN, with a measure-in-Arial/paint-in-Bangers race).
- **Cross-cutting:** Fewer/smaller crops help both tracks — batched OCR + delta-crop response + bubble-shape-aware layout all reduce work and bytes. **Two missing labeled eval sets (per-line OCR, bubble-detection GT) gate the highest-value bets** and should be built first.

---

## 2. Current State & Measured Baseline

### Latency (backend.log — AUTHORITATIVE; do NOT sum JSON ocrTimeMs+translateTimeMs, they double-count)

| Stage | Mean | Median | p90 | Notes |
|---|---|---|---|---|
| detect (CTD@1024 + YOLOv10n@640) | 250ms | 241ms | 249ms | ~fixed floor 236-333ms; CTD then YOLO **serial** |
| fused OCR+translate | 753ms | 690ms | 1151ms (max 1326 @10 crops) | **DOMINANT**; scales with #line-crops; OCR & translate actually serial |
| LaMa inpaint | 329ms | 352ms | 507ms | scales with #mask components (1box=75ms → 11box=507ms), runs LAST |
| **TOTAL** | **1353ms** | **1276ms** | **1926ms** | range 827-2117ms; ≈ sum of stages (near-zero overlap) |

### bubbleRect / overspill

- bubbleRect hit-rate **67.1%** (55/82). All 27 misses on 2 pages (637653_045: 0/14, _090: 0/8) — but backend.log shows YOLO *did* fire (1 and 2 bubbles); the **matcher rejected** them (center-containment + 1.15x area + single-winner eviction) and the pages are genuinely borderless SFX.
- Area-expansion (matched, n=55): min 1.31, median **2.00**, mean 2.29, p90 3.16, max 4.22. Every matched bubble enlarges render area; 100% remain taller-than-wide (median W/H 0.38).
- **All 9 overspill boxes (11%)** are on the 2 SFX pages, fall back to tiny tight bbox (e.g. `グ` 16x39 → 31-char gloss), hit the **8px font floor**, render illegibly, and leave JP showing through (mask stays on tight textRegions).

### PARSeq OCR A/B verdict: **INCONCLUSIVE — do NOT swap yet**

- Production `parseq_manga_best_ep60_AR_single.onnx` LOCKED to batch=1 (output hardcoded [1,51,4407] + several decoder reshapes hardcode leading dim 1; **not** a single named node — graph surgery is high-risk). ~46ms/crop.
- Candidate `parseq_manga_large_5p16.fp16.onnx` has dynamic batch, ~3.4ms/crop (12x faster), 1MB fp16.
- Accuracy statistically TIED on clean per-line inputs (A marginally ahead, 10 wins/5 of 249 lines); B regresses on hard kanji (mostly absorbed by `ocr_postprocess.apply_all`). **No labeled per-line GT exists** (bubbles.parquet GT is whole-bubble; ctd-lines mode scores against empty strings). Build the eval set before promoting.

---

## 3. SPEED Track (sorted by impact/effort within each group)

### Quick wins (S effort)

| Recommendation | Where (file:line) | Expected impact | Effort | Risk |
|---|---|---|---|---|
| Encode inpaint plate as WebP q85 not uncompressed PNG | translate.py:100-106 `_encode_png_base64`, call site :337; overlay-renderer.ts:224 | plate b64 3296KB→283KB (−91%, −3.1MB/page); transfer 2.75s→0.23s @10Mbps; +60ms encode | S | low |
| Offload YOLO `model.predict` to thread (prerequisite for parallel detect) | detector_service.py:65 | unblocks event loop; enables CTD‖YOLO | S | low |
| Parallelize CTD + YOLO via `asyncio.gather` | translate.py:165-193 | detect floor 241ms→~150-180ms (−60-90ms median) | S | low |
| Warm up YOLO bubble detector at startup | main.py:42-43 | removes first-request cold-start (333ms→241ms tail) | S | low |
| Warm up LaMa inpaint at startup | main.py lifespan after :58 | removes request-1 cold path (~575ms gap) | S | low |
| Lower max_tokens 128→48-64; raise/remove service semaphore(8) | vllm_openai_translation_service.py:33,40,101 | removes 2nd-wave round-trip on >8-bubble pages (−65-91ms); caps tail risk | S | low |
| Trim fixed prompt wrapper for prefix-cache reuse | vllm_openai_translation_service.py:94-98 | ~24 fixed tok cached after 1st; −10-40ms/page (needs chrF++ holdout) | S | medium |
| Lock MTP γ=2 (do NOT deploy EAGLE3) | serve_v10it_vllm.sh:48 (no change) | already 1.42x vs v9c; EAGLE3 would regress −15-26% | S | low |
| Drop redundant per-box fields (ocrTimeMs/translateTimeMs/fontHeightPx debug-gate) | response.py:25,37-38; translate.py:398,412-413 | removes double-count foot-gun; trivial bytes | S | low |
| Add GZip middleware backstop (AFTER WebP) | main.py:92-101 | marginal once WebP (~2-5%); meaningful only if plate stays PNG | S | low |

### Medium bets (M effort)

| Recommendation | Where (file:line) | Expected impact | Effort | Risk |
|---|---|---|---|---|
| **Overlap inpaint with OCR+translate** (mask depends only on detect, not translation) | translate.py:327-344 (start task ~L199 after build_text_regions, await before response) | hides full inpaint behind 690ms phase; **−330ms median**, −500ms p90; total 1276→~945ms (−26%) | M | low |
| Return only inpaint delta crops (union bbox or per-region) instead of full plate | translate.py:325-344 + response.py:61-65; overlay-renderer.ts:217-230 | union-crop ~200KB b64 (−94%); per-region ~5KB (−99.8%); largest payload win | M | medium |
| Batch all page bubbles into ONE numbered-block vLLM request | translate.py:247-253 + new method replacing vllm_openai_translation_service.py:107-122 | folds N prefill round-trips into 1; dense-page fused stage −300-700ms (needs chrF++ A/B) | M | medium |
| Batch per-component LaMa forwards into one ONNX run (if dynamic batch) | lama_inpaint_service.py:329-332, _forward_one :361-392 | dense pages (10-14 box) −100-200ms; needs batch-axis check first | M | medium |
| Build per-line labeled OCR eval set (gate for OCR swap) | new script reusing compare_parseq_exports.py:236-262 + bubbles.parquet; containment parseq_ocr_service.py:291-297 | converts A/B INCONCLUSIVE→go/no-go; unblocks 12x swap (single-line-only subset = zero hand-labeling) | M | low |
| Swap to dynamic-batch PARSeq export (behind eval gate) | config.py:34,37; translate.py:214 | OCR per-crop 46→3.4ms; dense OCR 931→~65ms; median total −300-600ms | M | high (gated) |
| Re-export model A with `dynamic_axes` instead of graph surgery (if A's accuracy must be kept) | training export script (re-run torch.onnx.export) | same 12x win, keeps A's exact accuracy; eliminates swap risk | M | low |

### Big bets (L effort)

| Recommendation | Where (file:line) | Expected impact | Effort | Risk |
|---|---|---|---|---|
| Stream OCR results per-block into translation (make pipeline-overlap real) | translate.py:211-253 + parseq_ocr_service.py:261-328 | hides shorter of the two phases; dense p90 OCR+trans 1151→~700-800ms | L | medium |
| Collapse CTD-text + YOLO-bubble into one RT-DETR-v2 detector (comic-translate) | detector_service.py:44 create_detector/detect_bubbles | lifts bubbleRect hit-rate >67%, one detector, possible latency cut | L | medium |

---

## 4. DISPLAY Track (text size + placement, sorted by impact/effort)

### Quick wins (S effort)

| Recommendation | Where (file:line) | Expected impact | Effort | Risk |
|---|---|---|---|---|
| Bundle Bangers locally; drop Google Fonts CDN | public/fonts/Bangers-Regular.ttf (create); overlay-renderer.ts:68-88; overlay.css:83-88 | removes CDN request + CSP/offline degradation-to-Arial; @font-face urls currently 404 | S | low |
| `await document.fonts.ready` before measure/layout | overlay-renderer.ts:120-128 before findBestFit | fixes measure-in-Arial/paint-in-Bangers wrap mismatch & first-render clipping | S | low |
| Read `fontHeightPx` as findBestFit seed + 70% downscale floor (gate on bubbleRect==null) | overlay-renderer.ts:379-409; api.ts:26 | caps oversize SFX text; makes dead field load-bearing; faster search | S | low |
| ALL-CAPS transform before measure+draw (setting-gated) | overlay-renderer.ts:341-352 | manga convention; uniform cap-height simplifies centering; better Bangers legibility | S | low |
| Thicken outline: stroke = clamp(size*0.14, 2.5, 6) / multi-offset white halo | overlay-renderer.ts:690 (drawTextBoxText) | readable over LaMa residue & low-contrast; fixes 8px-blob choke; no latency | S | low |
| Strip literal `*...*` markup tokens; italic for emphasis/SFX | overlay-renderer.ts:341-352, buildFontString:518-521 | removes stray asterisks on SFX pages | S | low |
| Proportional padding + tighter wrap before truncation at floor | overlay-renderer.ts:348-349, 405-407, 697-699 | recovers truncated long glosses; less silent line loss | S | low |

### Medium bets (M effort)

| Recommendation | Where (file:line) | Expected impact | Effort | Risk |
|---|---|---|---|---|
| **Fix null-bubble fallback: expand region instead of tight bbox** (aspect-driven enlarge; seed font from fontHeightPx*0.7) | overlay-renderer.ts:297-331 (no-bubble branch); ctd_utils.py emit expanded fallback | targets all 9 overspill + 27 miss boxes; lifts SFX off 8px floor | M | medium |
| Controlled-overflow / ellipsis policy below floor (no silent bleed) | overlay-renderer.ts:405-407 + drawWrappedText:692-696 | eliminates uncontrolled art-bleed on 9 overspill boxes; track "overflowed" flag | M | low |
| Couple white mask plate to layout region (paint over fitted text extent ∩ bubble) | overlay-renderer.ts:267-285 (pass fitted extent) | removes text-over-art halos on 55 matched boxes; no extra detection cost | M | low |
| Re-wrap-per-trial fit + cost-minimizing fallback (comic-translate port) | overlay-renderer.ts:379-409 + 416-458 | removes premature no-fit floor hits; best-effort wrap not clipped overspill | M | low |
| Reflow vertical-JP column → wide horizontal English (rebalance W/H<1 boxes) | overlay-renderer.ts:345-352 + 297-331 | kills word-shredding ("NOW,WHAT/SHOU/LD I/DO"); larger fitted font | M | medium |
| Recompute backend `fontHeightPx` against bubbleRect (raise clamp 50→72) | translate.py:365-374 + image_processing.py:136 | makes hint consistent with render reality on 55 matched boxes | M | medium |
| Dynamic min-font + box-expand for tiny no-bubble SFX (<8000px²) | overlay-renderer.ts:297-331 no-bubble branch + floor | recovers ~22 sub-8000px SFX boxes off 8-17px | M | medium |
| Hyphenation + balanced wrap | overlay-renderer.ts:464-513 wrapTextAtFont | buys 1-2 font-size steps on multi-word bubbles | M | low |
| Backend emit content-aware colors / force resolveColors luminance sampling | translate.py:387-388 (use detect_font_colors); or overlay-renderer.ts:538-543 | fixes black text vanishing into dark art; activates dead auto-contrast path | M | medium |

### Big bets (L effort)

| Recommendation | Where (file:line) | Expected impact | Effort | Risk |
|---|---|---|---|---|
| Bubble-shape-aware center-out layout (MIT manga2eng port: mask centroid + per-line mask collision) | overlay-renderer.ts new layoutCenterOut() augmenting :416-458/:379-409; reuse mask from drawTextBoxBackground:267-282 | structurally prevents overspill; text hugs bubble shape | L | medium |
| Detect free-floating SFX, route to adjacent placement/style + full-glyph mask | translate.py:355-358 (is_sfx flag); overlay-renderer.ts:336-371 | fixes JP-showing-through on _045/_090; covers 27 miss boxes | L | medium |
| True vertical-RL path for CJK retention/SFX (defer unless CJK target) | overlay-renderer.ts drawTextBoxText:319; buildFontString:501-504 | enables faithful vertical CJK/SFX | L | low |

---

## 5. Cross-Cutting (helps BOTH tracks)

- **Fewer/smaller crops** = less OCR compute AND smaller payload. Batched OCR (dense 931ms→65ms) + delta-crop response (−99.8% bytes) + bubble-shape layout all reduce work.
- **`fontHeightPx` is currently dead on the wire** — either make it load-bearing (display: sizing anchor/seed) or stop shipping it (speed: trim bytes + remove foot-gun). Do not leave half-wired.
- **Two missing labeled eval sets gate the top bets:** per-line OCR GT (gates the 12x OCR swap) and bubble-detection GT (gates IoU-fallback + YOLO conf tuning). Both are M effort, low risk, and unlock high-value work. Build first.
- **Bubble matcher quality directly drives display:** IoU/containment-fraction fallback (ctd_utils.py:35-46) + relaxed Pass-2 co-bubble merge (ctd_utils.py:52-63) raise dialogue hit-rate 67%→~80-85% on bubble-bearing pages, shrinking the overspill population the renderer must rescue.

---

## 6. External Best-Practices to Adopt (mapped to our code)

| Source | Technique | Our gap | Where to apply |
|---|---|---|---|
| MIT manga2eng | Center-out mask-aligned line layout (centroid + per-line mask collision) | rectangle-only findBestFit accepts 8px overflow | overlay-renderer.ts layoutAtSize/findBestFit (Big bet §4) |
| MIT standard | Overflow → EXPAND polygon, not shrink font; font_increase when translation longer | we fall back to tight bbox + 8px floor | computeTextRegionBBox null-bubble branch (M §4) |
| MIT manga2eng | downscale_constraint=0.7 hard floor seeded from detected font size | fontHeightPx unread | findBestFit seed (S §4) |
| comic-translate | Re-wrap per trial size + cost-min `(w-roi_w)²+(h-roi_h)²` fallback | single-wrap binary search declares no-fit early | findBestFit/layoutAtSize (M §4) |
| comic-translate | Single RT-DETR-v2 emits text+bubble boxes in one pass | two models (CTD + YOLOv10n) serial | detector_service.py (Big bet §3) |
| comic-translate | 5x5 / multi-offset white halo outline | single thin stroke chokes at 8px | drawTextBoxText:690 (S §4) |
| comic-translate | get_best_render_area: shrink-bubble 0.3 for vertical source | layout uses raw bubbleRect | reflow rec (M §4) |
| BallonsTranslator | ref_src_lines (mirror source line rhythm); soft overflow tolerance | none | wrap rebalance (M §4) |
| koharu | Tight INK bbox from glyph metrics (stops clip); true vertical shaping | fit/draw lineHeight divergence; no vertical | layoutAtSize:431-449 (S §4); vertical (L §4) |
| all four | Finetuned-LaMa inpaint + breathing-room fit (~70% box, 8-12px pad, never touch edge) | we match LaMa; padding fixed 8px | confirms our inpaint; padding rec (S §4) |

---

## 7. Recommended Sequencing

**Phase 0 — Eval infrastructure (unblocks everything, build in parallel):**
1. Per-line labeled OCR eval set (M, low) — single-line-bubble subset for zero hand-labeling. **Gates OCR swap.**
2. Bubble-detection GT set (M, low) — **Gates IoU fallback + YOLO conf tuning + RT-DETR A/B.**

**Phase 1 — Stacked low-risk speed quick wins (no gates):**
3. WebP plate (S) → 4. YOLO thread-offload (S) → 5. CTD‖YOLO parallel gather (S) → 6. YOLO + LaMa warmup (S) → 7. Overlap inpaint with OCR+translate (M, biggest no-gate win, −330ms). *(7 depends on 4-5 only loosely; do after detect refactor for clean async structure.)*

**Phase 2 — Display quick wins (no gates, high user-visible value):**
8. Bundle Bangers + `document.fonts.ready` (S, must pair) → 9. Thicker outline + ALL-CAPS + strip markup (S) → 10. fontHeightPx seed/floor (S, depends on 11 being correct) → 11. Recompute backend fontHeightPx vs bubbleRect (M) → 12. Null-bubble region expansion + controlled-overflow policy (M) → 13. Couple mask plate to layout region (M).

**Phase 3 — Gated/medium bets (after Phase 0 sets land):**
14. PARSeq dynamic-batch swap OR re-export A (M/high → after eval #1 passes) → 15. Numbered-block vLLM batch (M, after chrF++ holdout A/B) → 16. Matcher IoU fallback + max_expand guard + Pass-2 merge (S/M, after eval #2) → 17. LaMa per-component batching (M, after batch-axis check).

**Phase 4 — Big bets (architecture):**
18. Real streaming pipeline-overlap (L) → 19. Center-out mask-aligned layout (L) → 20. RT-DETR-v2 single detector (L, after eval #2) → 21. SFX routing (L) → 22. Vertical-RL (L, defer unless CJK target).

**Hard dependencies:** #14 needs #1; #15 needs chrF++ holdout (MEMORY: chat-template/prompt mismatch → silent ~95% chrF++ collapse); #16/#20 need #2; #10 needs #11; #8 fonts must precede #9-#13 (layout measured against real font).

---

## 8. Open Questions / Risks Needing a Decision

1. **SFX policy (decision required):** leave untranslated (current de-facto), translate with synthesized interiors, or route to a separate typesetter? The 67% metric conflates "SFX correctly skipped" with "dialogue bubble missed" — a **dialogue-only hit-rate** would be more actionable.
2. **Numbered-block batch quality unverified** for v10it Gemma-4-E4B-it. Service comment warns of a small-model few-shot regression (but that was the old HY-MT model). Needs per-bubble vs page-batched chrF++ A/B before shipping. Context-bleed between unrelated bubbles is a risk.
3. **PARSeq swap accuracy** is INCONCLUSIVE until eval #1 exists. Re-exporting A with dynamic_axes is the lower-risk path if A's accuracy must be preserved.
4. **YOLO geometry:** does YOLOv10n return polygons/ellipses or only axis-aligned rects? Center-out/intra-bubble centering will clip on round balloons if rects only.
5. **On the 2 zero-match pages**, did YOLO genuinely find no usable balloons or did the matcher reject good detections? backend.log says YOLO fired (1, 2 bubbles) — leans matcher rejection + genuinely borderless SFX. Per-miss instrumentation needed.
6. **No overspill ground-truth** — 11% is a heuristic from this 11-page sample. Tuning expansion margins needs a real overspill metric.
7. **Composites are APPROXIMATE** (PIL FreeType ≠ browser measureText). Re-measure wrap/line-count in-browser before tuning reflow thresholds.
8. **LaMa batch axis** unknown — per-component batching blocked until verified dynamic.
9. **Whitespace-synthesis fallback** should probe the clean inpaint plate (safer) vs original image (risks adjacent art) — but plate is computed after matching in current ordering.
10. **Translation's isolated share** of the 690ms fused window is unlogged (overlaps OCR). Confirm OCR-vs-translate split via a translate-only micro-benchmark before investing heavily in vLLM batching.
