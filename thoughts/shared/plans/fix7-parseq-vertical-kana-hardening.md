# fix7 — PARSeq Vertical/Stylized Kana Hardening

Branch: `fix7-parseq-vertical`
Status: PLAN (no training runs in this task — setup/planning only)
Date: 2026-06-26
Owner: danny@zenith-technologies.org

---

## 0. TL;DR

The single most damaging failure in the Ikenie-no-Haha-4 page-for-page MT-vs-human
comparison (144 bubbles, avg severity 2.53) is **OCR garble on dense / stylized
VERTICAL kana**: PARSeq's **non-autoregressive (NAR)** decode duplicates and
substitutes adjacent glyphs (身代わり→身身わわ, 吐気→吐吐気, 濯濯, 毎日→少日,
冷蔵庫→冷蔵目) **at falsely-high confidence (0.76–0.92)**. The translation LLM
then hallucinates a fluent-but-wrong English line.

We already shipped a confidence-independent **dup-bigram gate**
(`backend/app/utils/ocr_confidence_gate.py`, `is_implausible_japanese`) that
DROPS these — but a dropped bubble is a *lost line*, not a fixed one. This plan
fixes OCR at the source:

1. **Make the autoregressive (AR) PARSeq path the DEFAULT for tall/narrow
   vertical-text crops** (not just a low-confidence retry). The AR dynbatch ONNX
   already exists and shares weights with the trusted reference model; the
   interior-duplication garble is *intrinsic to the NAR decode* and is absent in
   AR (proven in §2).
2. **Fine-tune PARSeq with targeted synthetic augmentation** (duplicated-stroke,
   vertical-overlap, stylized-font) to reduce residual garble the AR path still
   misses, then re-export AR-dynbatch.
3. **Eval** by re-running the garble-prone Ikenie-4 pages and measuring the
   garble-rate drop, using the existing gate predicates as the garble detector.

---

## 1. Failure signature (what we are fixing)

Source: page-for-page MT-vs-human comparison of *Ikenie no Haha* vol. 4.

| Class | Example (GT → OCR) | Note |
|---|---|---|
| Adjacent doubled kanji | 身代わり → 身身わわ, 吐気 → 吐吐気, 濯濯バサミ | dominant |
| Whole-phrase immediate dup | また昨日みたいな → また昨日みたいなまた昨日みたいな | NAR loop |
| Interior char/phrase dup | 妄想止まらない → 妄..妄ま定れいい妄.想止止らな | corrupt + dup |
| Kanji substitution | 毎日 → 少日, 冷蔵庫 → 冷蔵目 | confident-wrong |

Common properties:
- **Vertical, dense, often stylized** kana/kanji columns (manga reading order:
  right-to-left columns, top-to-bottom within a column).
- **Falsely-high OCR recognition confidence (0.76–0.92)** — the confidence gate
  (`conf < 0.65`) never fires, so only the *linguistic-plausibility* predicates
  in `ocr_confidence_gate.is_implausible_japanese` catch them.
- These exactly match the "intrinsic non-AR decode corruption" documented in the
  per-line A/B (`thoughts/shared/research/translation-perf-display/2026-06-13_parseq-dynamic-batch-proposal.md`,
  "Where B is stuck" §): *interior phrase duplication, doubled kana mid-word,
  substitution errors* — "not safely postprocessable".

### Verified OCR assets on disk (`backend/models/`, main checkout)

The fix7 git worktree is a fresh worktree; the large gitignored model blobs live
only in the main checkout at `/home/danny/Documents/personal/extension/backend/models/`.
Verified present there (`ls`/`find`, 2026-06-26):

| File | Size | Role |
|---|---:|---|
| `parseq_manga_ep60_AR_dynbatch.onnx` (+ `.onnx.data`, `.json`) | 86 MB + 131 MB | **AR, dynamic batch** — the target default for vertical crops |
| `parseq_manga_best_ep60_AR_single.onnx` (+ `.json`) | 133 MB | AR reference, **batch-locked** (legacy prod, batch=1) |
| `parseq_manga_ep60_nonAR_dynbatch.fp16.onnx` (+ `.json`) | 2.5 MB + 65 MB | **current default** (`config.py: parseq_model_path`) — NAR, the garble source |
| `parseq_manga_ep60_nonAR_dynbatch.onnx` (fp32) | 130 MB | NAR fp32 |
| `parseq_manga_ep60_r2_nonAR_dynbatch.fp16.onnx` (+ `.json`) | 68 MB | NAR r2 variant |
| `parseq_manga_large_5p16.fp16.onnx` (+ `.opt.onnx`, `.json`) | 333 MB | "large" stopgap, charset 4400 (different charset) |
| `charset_4407.txt` | — | 4407-char charset used by AR/NAR ep60 models |

`parseq_manga_ep60_AR_dynbatch.json` reports `decode_ar=true`, `img_size=[128,512]`,
charset 4407, `head_dim=4407`, EOS at head index 0 — **byte-compatible decode**
with the NAR model (same charset, same I/O contract `images:[B,3,128,512] → logits:[B,51,4407]`).
This is why hybrid AR-retry can already drop into the same `_decode_with_conf`.

---

## 2. Why AR decoding fixes this (evidence, not assumption)

The dynamic-batch proposal doc ran a formal per-line OCR A/B
(`eval_perline_singleline.py`, real `ParseqOCRService`, real CTD crops) comparing:

- **A** = AR (`parseq_manga_best_ep60_AR_single.onnx`, sequential decode)
- **B** = NAR (`parseq_manga_ep60_nonAR_dynbatch.fp16.onnx`, **same weights**, parallel decode)

Findings (single-line / production regime, the regime we care about):

- **B (NAR) regresses mean CER by +0.67 to +1.04 pp vs A (AR)**, stable across
  seeds and length cuts — *not* a sampling artifact, *not* explained by multi-line
  inputs.
- The regression is driven by **NAR-only artifacts**: interior phrase/char
  duplication (`顔を突き出した顔を突突ままし`), doubled kana mid-word (`突突`,
  `器器`), trailing-fill loops, and substitution corruption — the **exact**
  Ikenie-4 garble signature in §1.
- A postprocess repeat-collapse attempt **could not close the gap**: the surviving
  regression is "intrinsic to B's non-AR decode … No postprocess rule can remove
  it without damaging legitimate Japanese repeats."
- Explicit recommendation: *"Re-export the ep60 weights with `decode_ar=True` +
  dynamic batch … the surest path: it gives A's sequential decode quality and
  batching, eliminating the non-AR hallucination at the source."*

**That re-export already exists** as `parseq_manga_ep60_AR_dynbatch.onnx`. The AR
decode emits one token conditioned on the previously-decoded tokens, so it cannot
fall into the parallel-decode duplication/fill loops that produce 身身わわ. The
only reason AR is not already the default is throughput: NAR is ~10× faster/crop
(3.7 ms vs 37 ms at bs=1). With dynamic batch + GPU, the AR cost amortizes (§3.3).

---

## 3. Proposal: AR-by-default for vertical crops (aspect-ratio routing)

### 3.1 Current wiring (verified)

- `backend/app/config.py`
  - `parseq_model_path = "models/parseq_manga_ep60_nonAR_dynbatch.fp16.onnx"` (NAR default)
  - `hybrid_ocr_enabled = True`
  - `parseq_ar_model_path = "models/parseq_manga_ep60_AR_dynbatch.onnx"`
  - `ocr_confidence_gate_threshold = 0.65` (doubles as hybrid retry threshold)
- `backend/app/services/parseq_ocr_service.py`
  - Loads NAR as the primary session; lazily loads AR (`_ensure_ar_session`) on
    first low-conf hit.
  - `_recognize_batch_with_conf`: runs NAR over **all** crops, then **AR-retry
    only `conf < threshold` crops** (`_ar_retry`), replacing in place.
  - `_maybe_rotate_vertical(crop, thresh_aspect=1.5)`: rotates a crop 90° CCW when
    `h > 1.5 * w` (the existing tall/narrow vertical-text test).
- `backend/app/routers/translate.py`: calls
  `ocr_service.recognize_blocks_with_lines(..., return_confidence=True)`; the
  confidence flows into the garble gate.

### 3.2 The change — route by crop geometry, not confidence

The current hybrid only sends a crop to AR if **NAR already produced low
confidence**. But the failure mode is *falsely-HIGH-confidence* NAR garble — so
the retry never triggers on the worst cases. The fix is to send **vertical crops
to AR unconditionally**, up front, because:

- vertical crops are where the NAR duplication garble concentrates (per-line A/B:
  vertical mean CER 24.2% vs horizontal 17.5%);
- AR's quality win is exactly on these crops;
- horizontal crops (cleaner for NAR, and cheaper) can stay on the fast NAR path.

**Trigger (crop aspect ratio).** Reuse the geometry already used by
`_maybe_rotate_vertical`. Define a routing predicate on the *raw* crop (pre-rotate):

```
is_vertical(crop) := (h / w) >= VERTICAL_AR_ASPECT      # tall & narrow
```

- `VERTICAL_AR_ASPECT` default = **1.5** (same threshold as `_maybe_rotate_vertical`,
  so "rotated-for-vertical" and "routed-to-AR" are the same set — no surprises).
- Make it a config knob `parseq_vertical_ar_aspect: float = 1.5` so it can be
  tuned/ablated without code edits.
- Add a master switch `parseq_vertical_ar_routing: bool` (default decided after the
  eval gate in §5; ship behind a flag so we can A/B in production).

**New batch flow** (replaces "NAR-all → AR-retry-low-conf"):

```
1. Partition crops: vertical_idx (h/w >= aspect)  vs  horizontal_idx (rest).
2. AR-batch  over vertical_idx     -> high-quality results for the garble-prone set.
3. NAR-batch over horizontal_idx   -> fast results for the clean set.
4. (optional, keep) low-conf AR-retry over any horizontal crop still < threshold.
5. Stitch back into original order; downstream garble gate runs unchanged.
```

This keeps the **gate as a safety net** (genuinely illegible SFX still drop after
AR), but the *primary* recognizer for vertical text becomes AR, so most lines are
*recovered* instead of *dropped*.

### 3.3 Latency budget

- Crops/page ≈ 14–17 lines (per-line A/B). A typical manga page is ~90% vertical
  columns, so ~13–15 crops route to AR.
- AR at bs=1 ≈ 37 ms/crop, but the **dynbatch** export batches: AR over a single
  page-batch of ~15 crops on the RTX 5090 amortizes the encoder/decoder passes.
  Budget target: **AR page-batch ≤ ~120–150 ms** (vs NAR ~10–20 ms). This is the
  cost of the fix; acceptable for the quality recovery, and re-measured in §5.
- OOM guard already exists in `_ar_retry` (halve batch on allocation failure);
  reuse it for the AR-default batch.

### 3.4 Where it hooks in

Two options (both captured in the skeleton `backend/scripts/ocr/route_vertical_to_ar.py`):

- **(Preferred) inside `ParseqOCRService._recognize_batch_with_conf`** — partition
  → AR-batch verticals → NAR-batch horizontals → stitch. Self-contained; the
  router (`translate.py`) and the eval harness both pick it up for free.
- **(Config-only fallback)** if we don't want to touch decode logic yet: keep
  NAR-all, but **lower the AR-retry threshold to ~1.0 for vertical crops only**
  (i.e. always retry verticals) by passing a per-crop threshold. Less clean, but
  zero new decode paths.

---

## 4. Synthetic-data augmentation + fine-tune (residual garble)

AR-by-default removes the *decode-induced* duplication. A second tier handles the
*recognition-hard* cases (stylized fonts, overlapping strokes, dense columns) by
fine-tuning the PARSeq weights with targeted synthetic augmentation, then
re-exporting AR-dynbatch.

### 4.1 Augmentation classes (mirror the failure signature)

Generate synthetic single-line crops (128×512, vertical-rendered then
rotated CCW to match `_maybe_rotate_vertical`) with:

1. **Duplicated-stroke / ghosting** — render a glyph, then overlay a low-alpha
   shifted copy (1–3 px) to mimic the stroke-doubling that drives 身→身身, 止→止止.
   Forces the recognizer to map a doubled-stroke glyph back to a single token.
2. **Vertical-overlap / tight leading** — render columns with reduced inter-glyph
   spacing and slight vertical overlap (kerning −10% to −25%), the dense-column
   condition where adjacent glyphs bleed together.
3. **Stylized fonts** — render with the public display fonts already vendored
   (`public/fonts/Anton-Regular.ttf`, `public/fonts/Bangers-Regular.ttf`) plus
   manga-style JP display faces; bold/outline/serif variation; mild rotation
   (±3°), perspective warp, and JPEG recompression to mimic scan artifacts.
4. **Confusable-pair hard negatives** — deliberately render the substitution
   pairs seen in Ikenie-4 (毎/少, 庫/目, 蔵/...) so the model sharpens the
   boundary instead of confidently substituting.

Mix ratio: ~70% real per-line GT (existing `perline_gt.parquet`, 68k rows) +
~30% synthetic augmentation, oversampling vertical orientation.

### 4.2 Training

- Base: the **ep60 checkpoint** (`output/parseq_large/parseq_manga_best_5p16.pt`
  / the ep60 AR weights) on the training box.
- Continue training (LR-warm-restart, small LR) for a few epochs on the
  augmented mix; keep `decode_ar=True`, `img_size=128×512`, charset 4407,
  `refine_iters=1` (must match the export wrapper in
  `backend/scripts/export_parseq_onnx.py`).
- Early-stop on **vertical-split CER** (not overall) so we optimize the failing
  slice.

> Guardrail (from memory `feedback_chat_template_mismatch`, `project_dual_pipeline_paths`):
> serve format must match train format. Any preprocessing change (rotate, resize,
> normalize) must be identical in `_preprocess` and in the synthetic renderer.

### 4.3 ONNX export (AR dynbatch)

Reuse the existing, validated workflow — **do not invent a new exporter**:

- Script: `backend/scripts/export_parseq_onnx.py` (already exports the NAR/AR
  ep60 weights; the `ParseqExport` wrapper, dynamic_axes, opset 17, `dynamo=False`
  TorchScript path are all set up). For an **AR** export, build the model with
  `decode_ar=True` (the checkpoint's `decode_ar` flag is read at load) and confirm
  `dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}}` is honored — the
  current AR-dynbatch export at `backend/models/parseq_manga_ep60_AR_dynbatch.onnx`
  is the reference artifact, so reproduce its config exactly.
- Training box / paths (from memory `reference_ar_dynbatch_ocr_export`,
  `reference_ctd_v26_export`):
  - Host: `danny@100.64.235.63` (ssh key auth), has the dataset + checkpoints.
  - parseq repo: `/home/danny/Documents/personal/manga/comic-text-detector-parseq-v2/parseq_repo`
    (imported by `export_parseq_onnx.py` via `PARSEQ_REPO`).
  - charset: `models/charset_4407.txt`.
- Command sketch (run on training box):

  ```bash
  backend/.venv/bin/python backend/scripts/export_parseq_onnx.py \
      --ckpt  output/parseq_large/parseq_manga_ar_ft_<tag>.pt \
      --out   backend/models/parseq_manga_ep60_AR_dynbatch_<tag>.onnx \
      --opset 17 --batch 1
  ```

  Then (if the torch.onnx dynamo path segfaults on torch 2.11, use the dynamo
  workaround noted in `reference_ctd_v26_export`). Copy
  `.onnx` + `.onnx.data` + `.json` to `backend/models/` and point
  `parseq_ar_model_path` at the new file.

---

## 5. Eval — garble-rate drop on Ikenie-4

The acceptance signal is **fewer garbled bubbles on the exact pages that failed**,
measured with the gate predicates as the garble detector (so the metric is the
same thing we currently mitigate by dropping).

### 5.1 Garble detector = existing gate predicates

Use `app.utils.ocr_confidence_gate.is_implausible_japanese(text)` (and the
sub-predicates `_adjacent_dup_kanji`, `_adjacent_dup_kana`,
`_immediate_substring_dup`, `_repeated_bigram_garble`) as a confidence-independent
**garble classifier** over the OCR output of each bubble. A bubble is "garbled"
if `is_implausible_japanese` returns True. This is the same predicate set that
currently drops these lines, so a drop in its hit-rate == a real OCR-quality win.

### 5.2 Procedure

1. Assemble the Ikenie-4 garble-prone page set (the pages flagged in the
   MT-vs-human comparison). Store crop+page paths under `backend/.bench/` (NAS is
   disposable per memory `reference_nas_cifs_reaping` — write durable output to
   local disk).
2. Run OCR three ways over the **same** CTD line crops:
   - **Baseline**: current NAR-default (`parseq_manga_ep60_nonAR_dynbatch.fp16.onnx`).
   - **AR-default**: vertical→AR routing (§3) with the *existing*
     `parseq_manga_ep60_AR_dynbatch.onnx`.
   - **AR-FT** (after §4): vertical→AR routing with the fine-tuned re-export.
   Reuse `backend/scripts/eval_vision/eval_perline_singleline.py` (Test-2 mode runs
   real CTD crops through the real `ParseqOCRService`) as the harness; add an
   `--ar-vertical` flag and an Ikenie page glob.
3. Metrics per arm:
   - **Garble-rate** = fraction of bubbles where `is_implausible_japanese` is True
     (primary).
   - **Gate-drop count** = bubbles the full gate would drop (lost lines).
   - **Vertical-split CER** vs any available GT (secondary; the Ikenie pages are
     not fully GT-annotated, so garble-rate is the primary signal).
   - **AR page-batch latency** (must stay within §3.3 budget).

### 5.3 Acceptance bar

- **Garble-rate (vertical) drops ≥ 50%** AR-default vs NAR-baseline on the
  Ikenie-4 garble pages (the 144-bubble cohort is dominated by vertical dup).
- **Gate-drop count drops materially** (lines recovered, not just dropped).
- **No regression on horizontal crops** (they stay NAR).
- **AR page latency within budget** (§3.3).
- AR-FT (§4) should push garble-rate further down on the residual recognition-hard
  cases without regressing the clean set.

Only flip `parseq_vertical_ar_routing` ON by default after the AR-default arm
clears the bar; ship the FT model after the AR-FT arm clears it.

---

## 6. Ordered steps

1. **[code]** Implement vertical→AR partition routing in
   `ParseqOCRService._recognize_batch_with_conf` (§3.2/§3.4), behind
   `parseq_vertical_ar_routing` + `parseq_vertical_ar_aspect` config knobs.
   Skeleton: `backend/scripts/ocr/route_vertical_to_ar.py`.
2. **[eval]** Add `--ar-vertical` + Ikenie page glob to
   `eval_perline_singleline.py`; run the §5 baseline vs AR-default A/B on the
   Ikenie-4 garble pages with the existing `parseq_manga_ep60_AR_dynbatch.onnx`.
3. **[gate]** If AR-default clears §5.3, flip `parseq_vertical_ar_routing` default ON.
4. **[data]** Build the synthetic augmentation set (§4.1).
5. **[train]** Fine-tune ep60 AR on the augmented mix (§4.2), early-stop on
   vertical CER (training box, no runs in *this* task).
6. **[export]** Re-export AR-dynbatch via `export_parseq_onnx.py` (§4.3); copy to
   `backend/models/`; point `parseq_ar_model_path` at it.
7. **[verify]** Re-run §5 (AR-FT arm); confirm garble-rate drop and no clean-set
   regression; ship.

---

## 7. Affected / referenced files

| File | Change |
|---|---|
| `backend/app/services/parseq_ocr_service.py` | vertical→AR partition routing in `_recognize_batch_with_conf` |
| `backend/app/config.py` | add `parseq_vertical_ar_routing`, `parseq_vertical_ar_aspect`; later swap `parseq_ar_model_path` to FT export |
| `backend/scripts/ocr/route_vertical_to_ar.py` | **new** — routing logic skeleton / reference impl (this task) |
| `backend/scripts/export_parseq_onnx.py` | reused for AR-dynbatch FT re-export (no change expected) |
| `backend/scripts/eval_vision/eval_perline_singleline.py` | add `--ar-vertical` + Ikenie page glob |
| `backend/app/utils/ocr_confidence_gate.py` | **no change** — its predicates double as the garble detector for eval |
| `backend/models/parseq_manga_ep60_AR_dynbatch.onnx` | existing AR-dynbatch; later replaced by FT re-export |

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| AR page latency regresses E2E | Medium | dynbatch amortization + budget gate (§3.3); keep horizontals on NAR; OOM-halving guard reused |
| Aspect threshold mis-routes (clean horizontal SFX → AR, or genuine vertical → NAR) | Low | tie to existing `_maybe_rotate_vertical` threshold (1.5); config-tunable; ablate in §5 |
| FT regresses clean text / legit reduplication | Medium | 70/30 real/synthetic mix; early-stop on vertical CER but monitor overall; serve==train preprocessing |
| Re-export breaks decode contract (charset/IO) | Low | reuse `export_parseq_onnx.py`; verify `decode_ar`, img_size, charset 4407, IO shapes match existing AR-dynbatch `.json` |
| Dynamo exporter segfault (torch 2.11) | Low | `dynamo=False` TorchScript path already used; dynamo workaround documented (`reference_ctd_v26_export`) |
| Worktree lacks gitignored models | Low | GPU validation runs on the main checkout (`project_dual_pipeline_paths`) |
