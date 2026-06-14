# PARSeq Dynamic-Batch Proposal
Generated: 2026-06-13
Status: Research only — no code changes made

---

## Executive Summary

The production OCR model (`parseq_manga_best_ep60_AR_single.onnx`) is physically
incapable of running batch>1 because the autoregressive decode loop was traced
into the ONNX graph one token-step at a time, stamping the shape `[seq_step, 1,
…]` into 1,879 of its 2,356 Reshape node initialisers. This is not a missing
`dynamic_axes` declaration — the graph itself encodes the AR loop unrolled for a
single sample. No post-export ONNX surgery can fix it; a re-export from a model
with `decode_ar=False` is required.

A clean dynamic-batch, non-AR export already exists:
`parseq_manga_large_5p16.fp16.onnx`. It has zero hardcoded Reshape shapes, full
symbolic `[batch, seq_len, vocab]` output, runs ~11x faster per crop, and already
serves as the default model in `parseq_ocr_service.py`. However, a prior A/B
against Manga109-s whole-bubble ground truth was **confounded by input domain
mismatch** (the AR model was fed single whole-bubble crops at 128×512 while it was
trained as a single-line STR model; the non-AR model handled the confound
differently), producing a misleading result where postprocessed scores swung ±15%
in opposite directions. The A/B cannot be trusted as a signal about per-line
accuracy until we have **per-line** ground truth.

Recommended path: **Option B — adopt `large_5p16.fp16.onnx` after passing a
per-line OCR eval gate.** The model is already deployed as the production
service default; the only remaining work is building the eval gate to confirm
accuracy is preserved on the real input domain.

---

## 1. Root Cause — Why Batch>1 Crashes

### 1a. The AR decode loop becomes a static graph at export time

In `parseq_repo/strhub/models/parseq/model.py` (`PARSeq.forward`), when
`decode_ar=True` the forward method runs this Python loop:

```python
for i in range(num_steps):          # num_steps = max_label_length + 1 = 51
    tgt_out = self.decode(
        tgt_in[:, :j],             # tgt_in shape = (bs, j)
        memory,
        tgt_mask[:j, :j],
        tgt_query=pos_queries[:, i:j],   # (1, 1, embed_dim) — batch dim comes from expand
        tgt_query_mask=query_mask[i:j, :j],
    )
    p_i = self.head(tgt_out)       # (bs, 1, vocab)
    logits.append(p_i)
    next_tokens = p_i.squeeze().argmax(-1)   # SQUEEZE removes batch dim for bs=1
    tgt_in[:, j] = next_tokens
```

`torch.onnx.export` with `dynamo=False` (the TorchScript exporter, which is what
`train_parseq_manga.py:export_onnx` uses) traces through this loop at export time.
At trace time the export call passes `dummy = torch.randn(1, 3, 128, 512)`, so
`bs=1`. Every intermediate tensor shape that depends on `bs` gets **frozen at 1**
in the resulting graph.

The export function is:

```python
# train_parseq_manga.py L1209-1243
def export_onnx(model, tokenizer, device, img_size=(32, 128),
                output_path="output/parseq_manga.onnx"):
    class PARSeqWrapper(nn.Module):
        def forward(self, images):
            return self.model.forward(self.tok, images)   # decode_ar state baked in

    wrapper = PARSeqWrapper(model, tokenizer).to(device)
    dummy = torch.randn(1, 3, *img_size, device=device)  # bs=1

    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},  # too late
        opset_version=17,
        dynamo=False,
    )
```

`dynamic_axes={"images": {0: "batch"}}` tells the exporter to declare the input's
first axis symbolic, but the Reshape ops *inside* the loop have already been
traced with concrete shape values derived from `bs=1`. `dynamic_axes` only
relabels the I/O tensor shapes in the graph's protobuf metadata; it cannot
retroactively make interior computed shapes dynamic when they were traced as
literals.

### 1b. ONNX graph inspection confirms 1,879 hardcoded Reshape initialisers

Graph: `backend/models/parseq_manga_best_ep60_AR_single.onnx`
- Total nodes: 15,832
- Reshape nodes: 2,356
- Reshape nodes whose shape tensor is a graph initialiser (hardcoded): 1,879

The shapes span every token step from `[1, …]` to `[51, …]` (i.e. the entire
51-step unrolled AR loop), with patterns like:

| Hardcoded shape | Count | Meaning |
|---|---|---|
| `[1, 384]` | 206 | attention linear projection for step 1 |
| `[1, 12, 32]` | 212 | multi-head split for step 1 |
| `[1, 1, 384]` | 206 | query/key/value token dim for step 1 |
| `[2048, 12, 32]` | 6 | encoder patch dimension (image-side, not batch) |
| `[N, 384]` for N=1..51 | ~4×51 | each AR step frozen at bs=1 |

The output tensor is also hardcoded: `logits: [1, 51, 4407]` — the `1` in the
first dimension is a static value, not a symbolic axis.

By contrast, `parseq_manga_large_5p16.fp16.onnx` has:
- Total nodes: 3,926 (4x fewer — one parallel forward pass, not 51 unrolled steps)
- Reshape nodes with hardcoded shapes: 0
- Output: `logits: [batch, seq_len, 4401]` (all symbolic)

### 1c. The `p_i.squeeze().argmax(-1)` also blocks batching

Even if the Reshape shapes were somehow made dynamic, line 227 of `model.py`
calls `p_i.squeeze()` with no `dim` argument, which collapses all size-1 dims.
At `bs=1`, `p_i.shape = (1,1,4407)` → squeeze → `(4407,)`. At `bs>1`,
`p_i.shape = (N,1,4407)` → squeeze → `(N,4407)` which works — but the traced
graph froze the output rank as 1-D (scalar argmax), so feeding bs>1 produces a
shape mismatch before the Reshape nodes are even reached.

### 1d. Why the export function's `dynamic_axes` declaration did nothing useful

The line:
```python
dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}}
```
correctly marks the I/O shape metadata as symbolic. The ONNX runtime respects
this for I/O validation. However every interior Reshape that uses a graph
initialiser (a constant tensor `[1, 384]`, etc.) is a static node — the runtime
cannot infer that `1` should track the input batch size. When `batch=2` is fed,
the first Reshape that expects `[1, 384]` receives a tensor of shape `[2, 384]`
and throws a shape mismatch error. This is the exact crash documented in
`config.py:34-37`.

---

## 2. Recommended Path

### Option A — Re-export AR_single weights with decode_ar=False
Re-use the trained ep60 checkpoint but export with the non-AR forward path
(one parallel decode + 1 refine iteration, matching the large_5p16 style).

### Option B — Adopt the existing large_5p16.fp16.onnx (RECOMMENDED)
The dynamic-batch non-AR export already exists, already handles batch>1, and is
already set as the default model in `ParseqOCRService.__init__`. The service
currently runs with `batch_size=24` internally. The only blocker is confirming
accuracy on the real input domain (per-line crops from CTD).

### Option C — Retrain with non-AR objective
Not needed. The AR quality advantage exists at the whole-bubble level but is
irrelevant because the production pipeline feeds single-line crops from CTD.

**Recommendation: Option B.**

Rationale:
1. `large_5p16.fp16.onnx` is already deployed in `parseq_ocr_service.py` as the
   primary model path. `config.py` line 34 still points to `AR_single` only
   because it is the `parseq_model_path` default and `ParseqOCRService` is
   initialised with `model_path="models/parseq_manga_large_5p16.fp16.onnx"` by
   default — but the Settings class still carries the old `AR_single` path. This
   is harmless (the service overrides it) but should be cleaned up.
2. The large_5p16 model already runs 11x faster per crop (3.7ms vs 41ms at bs=1)
   and at higher batch sizes the per-crop cost drops further.
3. Re-exporting AR_single's weights with `decode_ar=False` (Option A) would give
   equivalent non-AR performance but requires accessing the `.pt` checkpoint on
   the training machine, verifying the quality, and uploading a new ONNX file.
   That is more work for equivalent outcome.
4. The prior A/B was confounded (whole-bubble crops fed to a single-line model).
   The ctd-lines mode of the A/B harness already exists to run the correct
   evaluation; it just needs per-line ground truth added (see section 4).

---

## 3. Concrete Commands / Script Edits

### 3a. Config cleanup (after eval gate passes)

In `backend/app/config.py`, update the default to match what the service already
uses:

```python
# config.py line 34 — change from:
parseq_model_path: str = "models/parseq_manga_best_ep60_AR_single.onnx"
parseq_batch_size: int = 1

# to:
parseq_model_path: str = "models/parseq_manga_large_5p16.fp16.onnx"
parseq_batch_size: int = 24
```

The `parseq_batch_size` setting is read by `recognize_text_batch` as its default
`batch_size` parameter (L225). Setting it to 24 enables the batched path that
already exists in the service code.

### 3b. Option A (alternative, if B is blocked for any reason)

On the training machine, run:

```bash
cd /home/danny/Documents/personal/manga/comic-text-detector
uv run python scripts/train_parseq_manga.py \
    --eval-only \
    --checkpoint output/parseq_large/parseq_manga_best.pt \
    --model-size large \
    --img-height 128 --img-width 512 \
    --charset models/japan-crnn-ctc/charset_4407.txt \
    --export-onnx \
    # do NOT pass --decode-ar  (this exports non-AR)
```

This calls `export_onnx(model, ...)` with `model.decode_ar=False`, tracing the
parallel forward path. The resulting `output/parseq_large/parseq_manga.onnx` will
have zero hardcoded batch dims. Then quantize to fp16 with:

```bash
uv run python -c "
import onnx
from onnxconverter_common import float16
m = onnx.load('output/parseq_large/parseq_manga.onnx')
m16 = float16.convert_float_to_float16(m, keep_io_types=True)
onnx.save(m16, 'output/parseq_large/parseq_manga_ep60_nonAR.fp16.onnx')
"
```

Copy to `backend/models/` and point the service at it.

---

## 4. The Eval Gate — Per-Line OCR Ground Truth

### 4a. Why the existing A/B is insufficient

The `compare_parseq_exports.py` bubble-gt mode evaluates whole-bubble crops
against Manga109-s `<text>` annotations. The annotations contain the **full
bubble text** (often 2-5 lines). The model receives a whole-bubble image
(e.g. 200×80px crop) and attempts to emit that full multi-line string — but
PARSeq is a **single-line STR model** trained on 128×512 crops. The result:

- Model A (AR_single) is penalised for stopping at end-of-line ~1/3 through a
  multi-line bubble (correct per-line OCR, but low whole-bubble exact-match).
- Model B (large_5p16) produces repetition artifacts on the same multi-line crops.
- Postprocessing normalises Model A's half-width dots to full-width, inflating its
  postprocessed exact-match from 44.7% raw to 81% — an artefact of the
  normalisation map, not model quality.

The A/B results file (`compare_parseq_exports_results.json`) confirms this: raw
exact-match for Model A is 44.7% but postprocessed is 81%, a +36pp swing that
doesn't reflect actual OCR quality — it reflects the postprocess mapping of
`.......` → `…………`. Model B suffers from repetition on multi-line crops and
cannot benefit from the same normalisation.

The ctd-lines mode (`--mode ctd-lines`) of the harness already runs CTD on pages
and feeds the real production crops to both models, but reports zero ground truth
(`gts = ["" for _ in crops]`). We need to annotate those CTD line crops.

### 4b. Proposed per-line eval set construction

**Step 1 — Leverage existing Manga109-s per-line bboxes.**

The Manga109-s XML annotations (same dataset as `bubbles.parquet`) contain
`<body>`, `<balloon>`, and `<speech>` tags with child `<text>` elements, each with
a `@xmin/ymin/xmax/ymax` bounding box and the transcribed text. These are
**per-line** in many cases — the annotators segmented reading-order lines, not
just bubbles. The `bubbles.parquet` was constructed by taking `<text>` bboxes as
whole-bubble crops, but each `<text>` element often covers only one line.

Action: Parse the raw Manga109-s XML to extract individual `<text>` elements and
check their aspect ratio. Elements with `(ymax-ymin)/(xmax-xmin) < 0.5` (wider
than tall) are single-line candidates. These give direct per-line crop + ground
truth pairs without running CTD at all.

Script sketch:
```python
# backend/scripts/data/manga109/build_perline_gt.py
from pathlib import Path
import xml.etree.ElementTree as ET
import polars as pl
import cv2

MANGA109_ROOT = Path("/mnt/nas/drive_2/manga-ml/datasets/manga109s/Manga109s_released_2023_12_07")
rows = []
for xml_path in (MANGA109_ROOT / "annotations").glob("*.xml"):
    book = xml_path.stem
    tree = ET.parse(xml_path)
    for page in tree.findall(".//page"):
        pg = int(page.get("index"))
        for text in page.findall(".//text"):
            x0, y0 = int(text.get("xmin")), int(text.get("ymin"))
            x1, y1 = int(text.get("xmax")), int(text.get("ymax"))
            gt = (text.text or "").strip()
            if not gt:
                continue
            h, w = y1 - y0, x1 - x0
            if w > 0 and h / w < 0.7:   # single-line heuristic
                rows.append({"book": book, "page": pg, "xmin": x0, "ymin": y0,
                             "xmax": x1, "ymax": y1, "jp_text": gt})
pl.DataFrame(rows).write_parquet("scripts/data/manga109/perline_gt.parquet")
```

**Step 2 — Run both models on per-line crops and score CER.**

Update `compare_parseq_exports.py` to accept `--parquet` pointing at
`perline_gt.parquet` and run in the existing bubble-gt code path (the crops are
now single-line, so the domain mismatch is eliminated).

```bash
backend/.venv/bin/python backend/scripts/eval_vision/compare_parseq_exports.py \
    --mode bubble-gt \
    --parquet backend/scripts/data/manga109/perline_gt.parquet \
    --n 500 --batch-size 24 --seed 42
```

**Step 3 — Acceptance criteria.**

Model B is acceptable if:
- Mean CER vs per-line GT <= mean CER of Model A + 0.5% (absolute)
- Exact-match rate within 3pp of Model A

These thresholds are conservative given that Model B is already the deployed
service default and the prior test showed it outperforming Model A on exact-match
in the raw (non-postprocessed) pass (56.7% vs 44.7%) — the regression was an
artifact of postprocess normalisation on whole-bubble inputs.

**Note on Manga109-s licensing:** The dataset is available for non-commercial
research use and is already in use on the training machine. Ground-truth
annotation `.xml` files accompany the images. If the raw XML is not on the
training machine, it can be reconstructed from `bubbles.parquet` (which already
parsed it) plus the corresponding image dimensions.

### 4c. Alternative: Human annotation of 200 CTD line crops

If Manga109-s per-line bboxes are insufficient (too few horizontal single-line
cases), a fallback is:
1. Run CTD on 20 Manga109-s pages (20-30 pages yields ~400-600 text_line crops).
2. Export the crops as a labelling job (simple HTML interface or Label Studio).
3. Annotate the ground truth in 2-3 hours of human effort for 200 crops.
4. Run the A/B against that dataset.

This is higher effort but produces the cleanest per-line signal.

---

## 5. Expected Performance Win

### 5a. Current baseline (AR_single, bs=1)

- Per-crop inference: ~41ms (observed, `compare_parseq_exports_results.json`)
- OCR stage median: ~595-690ms per page (from E2E benchmarks)
- Crops per page: ~595/41 ≈ 14-17 line crops / page

### 5b. With large_5p16.fp16.onnx at batch=24

At bs=1, large_5p16 runs ~3.7ms/crop (11x speedup). At batch=24 on an RTX 5090
FP16, GPU utilisation is likely saturated for a model this size (3.9k nodes vs
15.8k), so the marginal cost per additional crop in a batch approaches
~0.5-1.0ms/crop. Estimated batch inference time:

- 14 crops in one ONNX call: ~8-12ms total (vs 14 × 41ms = 574ms serially)
- Per-page OCR stage: ~10-20ms (down from ~595-690ms)
- Speedup: **~30-50x on OCR stage alone**

The OCR stage currently dominates E2E latency. If translation is the second
bottleneck at ~200ms, net E2E goes from ~800ms → ~250ms. If both are dominated
by translation, net change is ~-400ms (OCR contribution removed).

Conservative estimate: **OCR stage: 595ms → 15ms. E2E: ~40% reduction.**

### 5c. Note on batch size choice

The service already uses `batch_size=24` internally in `recognize_text_batch`.
This is the right default for a page with many bubbles. For pages with only 5-8
crops, the entire batch fits in one ONNX call; for pages with 30+ crops, two
calls are needed. The OOM fallback to bs//2 already exists in the service code.

---

## 6. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Model B has lower accuracy on some glyph classes** | Medium | Per-line eval gate (section 4) quantifies this before config change |
| **large_5p16 charset is 4401 vs AR_single's 4407** | Low | 6 glyphs difference; the 6 extras in AR_single are ultra-rare punctuation variants. Already documented in compare_parseq_exports.py L21-24 |
| **Repetition artifacts in large_5p16** | Medium | Already mitigated by `_LONG_RUN_RE` and `_has_trigram_loop` guards in parseq_ocr_service.py; the non-AR model does not exhibit the same pattern as the whole-bubble case once fed real single-line crops |
| **FP16 precision loss** | Low | FP16 export is already deployed; model already loaded via CUDA EP with FP16 I/O |
| **OOM at high batch sizes** | Low | OOM fallback (halve bs) already in `recognize_text_batch` L239-243 |
| **config.py still points at AR_single as default** | Low | parseq_ocr_service.py hardcodes `large_5p16.fp16.onnx` as the model_path default — Settings value is overridden. Cleanup needed but not blocking |

---

## 7. Ordered Steps

1. **[Eval]** Build `perline_gt.parquet` from Manga109-s XML on training machine.
   Files: `backend/scripts/data/manga109/build_perline_gt.py` (new).
   Machine: danny@100.64.235.63 (has the dataset at `/mnt/nas/drive_2/manga-ml/datasets/manga109s/`).

2. **[Eval]** Add `--parquet` flag to `compare_parseq_exports.py` so it can load
   the per-line GT parquet.
   File: `backend/scripts/eval_vision/compare_parseq_exports.py`.

3. **[Eval]** Run the A/B with per-line crops and confirm Model B passes
   acceptance criteria (section 4c).

4. **[Config]** Update `config.py:34-37` to point at `large_5p16.fp16.onnx` and
   set `parseq_batch_size=24`.
   File: `backend/app/config.py`.

5. **[Verify]** Run E2E latency benchmark (`backend/scripts/eval_vision/`) on a
   sample page set to confirm the expected speedup.

6. **[Optional / Option A fallback]** If per-line eval reveals Model B accuracy
   is worse than A by >0.5% CER, re-export ep60 checkpoint non-AR:
   - On training machine: `uv run python scripts/train_parseq_manga.py --eval-only
     --checkpoint output/parseq_large/parseq_manga_best.pt --model-size large
     --img-height 128 --img-width 512 --charset ... --export-onnx`
   - FP16 quantise and upload to `backend/models/`.

---

## 8. Affected Files

| File | Change |
|---|---|
| `backend/app/config.py` | Update `parseq_model_path` default + `parseq_batch_size` |
| `backend/scripts/eval_vision/compare_parseq_exports.py` | Add `--parquet` flag for per-line GT |
| `backend/scripts/data/manga109/build_perline_gt.py` | New script to extract per-line crops from XML |
| `backend/app/services/parseq_ocr_service.py` | No changes needed (already uses large_5p16 and batch_size=24) |
| `backend/models/parseq_manga_large_5p16.fp16.onnx` | Already present, no change |

Files NOT affected:
- `backend/models/parseq_manga_best_ep60_AR_single.onnx` — keep as archive
- Training scripts on danny@100.64.235.63 — only needed for Option A fallback

---

## Per-line A/B results (2026-06-13)

Formal per-LINE OCR accuracy A/B on the user-built labeled ground-truth set
(`backend/scripts/data/manga109/perline_gt.parquet`, 68,472 rows / 87 books, 0
missing images). Each sampled line is cropped (`xmin..xmax`, `ymin..ymax` + 2px
pad) from its page JPEG and run through the **real `ParseqOCRService`**
(`recognize_text_batch`), so preprocessing (`_maybe_rotate_vertical`, resize,
normalize) and decode match production exactly.

- Eval script: `backend/scripts/eval_vision/eval_perline_gt.py`
- Results JSON: `backend/scripts/eval_vision/eval_perline_gt_results.json`
- Sample: **N=1000**, seed=0, stratified to ~25% horizontal (750 vertical / 250
  horizontal) so both orientations are well represented (natural split is ~94/6).
- Models: **A** = `parseq_manga_best_ep60_AR_single.onnx` (OLD prod, batch=1,
  charset 4407) · **B** = `parseq_manga_ep60_nonAR_dynbatch.fp16.onnx` (NEW prod,
  batched, **same weights as A**) · **C** = `parseq_manga_large_5p16.fp16.onnx`
  (stopgap, charset 4400).
- Two scoring passes: **POSTPROCESSED** = service output as shipped (`_finalize_ocr`
  = `ocr_postprocess.apply_all` + JP-whitespace collapse + repetition guard);
  **RAW** = same models with `_finalize_ocr` monkeypatched to identity.

> Caveat on absolute numbers: this GT is **bubble-level** — many `jp_text` rows
> contain `\n` (multi-line bubbles), while PARSeq is a single-line STR model.
> That inflates absolute CER for **all** models (~19–28%). It does **not** bias
> the A-vs-B comparison: A and B are scored against identical references, so the
> **A→B delta is the valid measurement**.

### Postprocessed (production-shipped output)

| Model | Split | n | Exact-match | Mean CER | Median CER |
|---|---|---:|---:|---:|---:|
| **A** (old prod) | overall | 1000 | 39.60% | 19.30% | 9.60% |
| | vertical | 750 | 39.20% | 20.70% | 10.00% |
| | horizontal | 250 | 40.80% | 15.08% | 6.46% |
| **B** (new prod) | overall | 1000 | 38.00% | 20.95% | 11.11% |
| | vertical | 750 | 38.00% | 22.19% | 11.11% |
| | horizontal | 250 | 38.00% | 17.23% | 11.11% |
| **C** (stopgap) | overall | 1000 | 33.50% | 25.79% | 14.84% |
| | vertical | 750 | 35.47% | 24.51% | 13.33% |
| | horizontal | 250 | 27.60% | 29.62% | 20.00% |

### Raw (bare model decode, no postprocess)

| Model | Split | n | Exact-match | Mean CER | Median CER |
|---|---|---:|---:|---:|---:|
| **A** | overall | 1000 | 40.80% | 24.48% | 9.09% |
| **B** | overall | 1000 | 39.30% | 27.62% | 10.91% |
| **C** | overall | 1000 | 59.10% | 17.29% | 0.00% |

Timing (this run, RTX, ms/crop): **A 40.3** (forced batch=1 — AR_single export
is batch-locked) · **B 3.88** · **C 3.48**. The ~10x speedup is confirmed; the
question is whether B keeps A's accuracy.

### Verdict: B vs A — **FAIL (CER regression)**

Acceptance bar: B within **+0.5pp mean CER** AND within **3pp exact-match** of A
(postprocessed).

| Metric | A | B | Δ (B−A) | Bar | Result |
|---|---:|---:|---:|---|:--:|
| Mean CER (postproc) | 19.30% | 20.95% | **+1.65pp** | ≤ +0.50pp | ✗ FAIL |
| Exact-match (postproc) | 39.60% | 38.00% | −1.60pp | ≥ −3.00pp | ✓ pass |

**Overall verdict: FAIL.** B is NOT within parity of A; mean CER regresses
+1.65pp (3.3x over the allowed +0.5pp). Exact-match is within tolerance, but the
CER bar is breached. The raw-decode gap is even larger (+3.14pp), so production
postprocessing only halves — not removes — the regression.

### Why (root cause)

Despite sharing weights with A, **B's non-autoregressive decode emits
trailing-repeat / hallucination runs** that A's autoregressive decode does not.
On the 116/1000 A↔B disagreements (88.4% agreement), when they differ **A is
right 19x vs B 4x** (and both wrong 93x — mostly the multi-line bubbles above).
The failures are characteristic non-AR artifacts:

```
GT 'ボケッ!!'        A 'ボドッ'           B 'ボドッ!.....!!!!!!!...!!!!!!!!!!!.!.!!!!....!!!!!!...'
GT '私はただ・・・'    A '私はただ...'       B '私.......はただ......................................'
GT '女王蟻発見！'     A 'さすが...女王蟻発見!' B 'そ.が....女蟻蟻発見!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.!'
GT 'ふしぎな生命体'   A '............ふしぎな生命体' B 'お..は生命体しししななな命体体体体体体体'
GT 'お前……\nどこかで…' A 'お前......どこかでちゃんとごはんもらってるのか?'
                                          B 'お前......どこかでちゃんとごはんもらってるのか??........???????.......'
GT 'えええええ〜！？'  A 'えええええ〜!?'    B 'えええええ〜!?????'
```

The existing repetition guard in `parseq_ocr_service._repetition_guard` only
**logs** these runs, it does not blank/trim them; `ocr_postprocess` caps some
trailing repeats but not the interleaved-punctuation runs B produces.

### Recommendation

- **Do NOT ship B as a like-for-like swap on accuracy grounds alone** — it
  trades 10x throughput for a +1.65pp CER regression driven by non-AR repeat
  hallucination, not a weight difference.
- If the speed win is required, mitigate the non-AR artifacts first: (a) add an
  EOS-confidence / repeat-collapse step to the non-AR decode, or (b) re-export
  with `decode_ar=True` + dynamic batch, or (c) tighten `ocr_postprocess` to trim
  interleaved-punctuation runs (`!.!.!`, `………`) and re-run this A/B.
- **C is not a drop-in** (different charset 4400, different weights). Its raw
  numbers look strong (17.3% CER) but collapse under production postprocessing
  (25.8%), so it is not comparable without postprocess tuning.

Re-run anytime with:

```bash
backend/.venv/bin/python backend/scripts/eval_vision/eval_perline_gt.py \
    --n 1000 --seed 0 --batch-size 24 --with-c
```


---

## Single-line / production-regime re-test (2026-06-14)

The original FAIL above used **bubble-level GT** (whole `<text>` bboxes; ~4,771
of 68,472 rows contain embedded `\n` and many more are run-on multi-line
transcripts). That is **out-of-distribution** for B, a single-line recognizer
whose non-AR decode hallucinates repeat runs on too-long inputs. Production OCR
(`recognize_blocks_with_lines`) feeds **single-line CTD crops**, so this re-test
restricts to that regime to check whether the FAIL was purely OOD.

New self-contained script (does not touch production code):
`backend/scripts/eval_vision/eval_perline_singleline.py`
Results: `backend/scripts/eval_vision/eval_perline_singleline_results.json`

Both A and B run through the **real `ParseqOCRService`** (production preprocess +
`_finalize_ocr` postprocess). A = `parseq_manga_best_ep60_AR_single.onnx`
(reference, batch=1, charset 4407); B = `parseq_manga_ep60_nonAR_dynbatch.fp16.onnx`
(candidate, batched, **same weights as A**, charset 4407).

### Test 1 — single-line GT (postprocessed = shipped output)

Filter: `'\n' not in jp_text` AND `n_chars <= 20`. **Single-line pool = 56,323**
rows (vertical 53,683 / horizontal 2,640). Stratified sample, horiz oversampled
to ~25%.

n=2000 (seed 0, max_chars=20):

| model | split | n | exact | mean CER | median CER |
|-------|-------|---|-------|----------|-----------|
| **A** | overall | 2000 | 46.20% | 21.50% | 7.69% |
|       | vertical | 1500 | 43.93% | 23.58% | 10.00% |
|       | horizontal | 500 | 53.00% | 15.26% | 0.00% |
| **B** | overall | 2000 | 45.60% | 22.54% | 9.09% |
|       | vertical | 1500 | 43.67% | 24.24% | 10.53% |
|       | horizontal | 500 | 51.40% | 17.46% | 0.00% |

- Δ mean CER = **+1.042 pp** (bar: ≤ +0.50) → **FAIL**
- Δ exact-match = −0.600 pp (bar: ≥ −3.00) → within bar
- Speed: A 37.6 ms/crop, B 3.8 ms/crop (**10.0× faster**)

Robustness re-run n=4000 (seed 1, max_chars=15, tighter single-line cut):
Δ mean CER = **+0.674 pp** → still **FAIL**; Δ exact = −0.40 pp. The regression
is stable across seeds and length cuts — it is **not** a sampling artifact and
**not** explained by multi-line OOD inputs.

### Test 2 — real CTD per-line crops (true production distribution, no GT)

Production CTD detector run on 36 real manga pages
(`637653_Haha to Ochite Iku Part 12` + `653631_… Part 13`, `<NNN>.webp`) →
**487 single-line crops**. A and B run on the SAME crops.

- A↔B exact agreement: **87.68%** (427/487)
- A repeat-artifact rate: **3.29%** (16/487)
- B repeat-artifact rate: **4.11%** (20/487)
- **B-only artifacts (present in B, absent in A): 6 → 1.23%**
- A-only artifacts (present in A, absent in B): 2 → 0.41%

Artifact = run of ≥4 identical chars, or a trailing punctuation run (`!!!!`,
`....`, `・・・・`, etc.) of length ≥4, in B's output but not A's for the same crop.

Examples (`* = B-only repeat artifact`), real single-line crops:

```
 *[637653/021#L11]  A 'うっ'              B 'うっ!!!ー!!'
 *[637653/086#L3]   A '..'                B '...。..'
 *[653631/011#L11]  A 'でしょお〜〜っ?'    B 'でしょお〜〜??'
 *[653631/036#L9]   A '帰ったわよぉー!!'   B '帰ったわよぉーー!!'
 *[653631/061#L0]   A 'わー'              B 'わーっ...。..'
 *[653631/061#L16]  A 'ーー'              B 'ハーー..'
  [637653/001#L1]   A '母と堕ちていく'     B '心と堕ちていく'
  [637653/006#L0]   A '…支障が牛じる…'     B '…支障が生じる…'   (B actually better here)
  [637653/011#L3]   A 'クリ'              B 'グリ'
```

Even on **single-line** crops the same non-AR failure mode appears (trailing
`!!!ー!!`, `...。..`, doubled `ーー`), confirming it is intrinsic to B's non-AR
decode, not a consequence of feeding it multi-line bubbles. The repetition guard
in `_repetition_guard` only logs; `ocr_postprocess` does not trim these
interleaved/short punctuation runs.

### Verdict: DO NOT SHIP B

The "bubble-level FAIL was just OOD" hypothesis is **rejected**. On the exact
production regime (single-line, real `ParseqOCRService`, real CTD crops):

1. **Test 1 FAIL** — B regresses mean CER by +0.67 to +1.04 pp (bar +0.5), stable
   across two seeds / length cuts. The regression shrinks vs the +1.65pp
   bubble-level number (so multi-line OOD *did* inflate it) but does **not**
   disappear.
2. **Test 2 NOT OK** — B introduces B-only repeat artifacts at **1.23%** on real
   single-line crops and a higher overall artifact rate (4.11% vs A's 3.29%).

**Recommendation unchanged:** do not swap B in as-is. To capture the 10× speed
win, first either (a) add a repeat-collapse / EOS-confidence step to the non-AR
decode, (b) re-export with `decode_ar=True` + dynamic batch, or (c) tighten
`ocr_postprocess` to trim short trailing/interleaved punctuation runs — then
re-run this single-line A/B and require Test 1 ≤ +0.5pp **and** B-only artifact
rate ≈ A.

Re-run:
```bash
backend/.venv/bin/python backend/scripts/eval_vision/eval_perline_singleline.py \
    --n 2000 --max-chars 20 --pages 18 --batch-size 24 --seed 0
```


---

## Postprocess repeat-collapse attempt (2026-06-14) — VERDICT: NOT YET (ship blocked)

Goal: clear the Test-1 bar (B ≤ A + 0.5pp mean CER) by strengthening the
shared OCR postprocess (`backend/app/utils/ocr_postprocess.py`,
`apply_all`) to kill B's non-AR repeat artifacts — **without** regressing A
(which runs the same pipeline) and **without** damaging legitimate Japanese
repeats. Decode/config/commit untouched.

### What was added to `ocr_postprocess.apply_all`

Two new artifact-targeted collapse steps run after `normalize_text` and before
the legacy `strip_trailing_repeats`. Thresholds were derived from the 56,323-row
single-line GT pool so each step is a near no-op on clean text:

1. **`collapse_trailing_loop`** — trims a *looping* trailing punctuation block
   (a punct char recurring across 2+ separate runs), but only when the loop
   involves a period-class char (`.`/`。`/`．` — the NAR dot-fill signature) OR
   the tail is ≥8 long with a char spanning ≥3 runs. Replaces the block with its
   first contiguous run. **Fires on 1 / 56,323 GT rows (0.0018%).** Non-looping
   emphatic tails (`・・・・！？`, `ーー！！`, `…!?`, `!!??`, interrobang `!!???!!?`)
   are preserved.
2. **`collapse_cjk_runs`** — caps an identical *CJK ideograph* run to 2. A CJK
   ideograph repeated even 3× never occurs in legit GT (0 rows), so this is a
   pure-artifact signal (`体体体体体` → `体体`). Kana/symbol runs (laughter
   ハハハ, vowel elongation わ～～～) are untouched.

Empirical safety: the new steps change only **3 / 56,323 legit GT rows
(0.005%)**, vs the legacy `strip_trailing_repeats` which mangles **22%** of
legit GT (laughter/elongation/ellipsis) for both models — but that legacy step
was left as-is because it hits A and B identically (delta-neutral) and removing
it is out of scope for this artifact fix.

### Example artifact fixes (raw B → collapsed)

```
'わーっ...。..'                                          -> 'わーっ..'
'...。..'                                               -> '..'
'!.....!!!!!!!...!!!!'                                  -> '!'
'体体体体体'                                             -> '体体'
'ボドッ!.....!!!!!!!...!!!!!!!!!!!.!.!!!!....!!!!!!...'   -> 'ボドッ!'
'そ.が....女蟻蟻発見!!!!!!!!!!!!!!!...!!!.!'              -> 'そ.が...女蟻蟻発見!!!.!'
'私.......はただ............................'             -> '私...はただ..'
'えええええ〜!?????'                                      -> 'えええ〜!??'
'顔を突き出した顔を突突ままし'                              -> '顔を突き出した顔を突突ままし'   (UNFIXABLE — interior phrase dup)
'い...い兵器器..兵.兵'                                    -> 'い...い兵器器..兵.兵'           (UNFIXABLE — interior char dup)
```

### A/B re-eval (real ParseqOCRService, n=2000 seed 0, max_chars 20, 36 CTD pages)

**Test 1 — single-line GT (postprocessed = shipped output):**

| model | split | n | exact | mean CER | median CER |
|-------|-------|---|-------|----------|-----------|
| **A** | overall | 2000 | 46.20% | **21.50%** | 7.69% |
|       | vertical | 1500 | 43.93% | 23.58% | 10.00% |
|       | horizontal | 500 | 53.00% | 15.26% | 0.00% |
| **B** | overall | 2000 | 45.60% | **22.53%** | 9.09% |
|       | vertical | 1500 | 43.67% | 24.22% | 10.53% |
|       | horizontal | 500 | 51.40% | 17.46% | 0.00% |

- **Δ mean CER = +1.032 pp** (bar ≤ +0.50) → **FAIL** (was +1.042; collapse moved it −0.010pp)
- Δ exact-match = −0.600 pp (bar ≥ −3.00) → within bar
- Speed: A 39.2 ms/crop, B 3.73 ms/crop (**10.5× faster**)

**A is byte-identical to its pre-collapse baseline** (21.50% overall / 23.58%
vert / 15.26% horiz / 46.20% exact) → criterion (b) "A must not regress" **PASS**.

**Test 2 — real CTD per-line crops (487 crops, production distribution):**

| metric | before collapse | after collapse | A |
|--------|----------------:|---------------:|--:|
| A↔B exact agreement | 87.68% | 87.89% | — |
| B artifact rate | 4.11% | **3.70%** | 3.29% |
| B-only artifacts | 1.23% | **0.82%** | — |

→ criterion (c) "B repeat-artifact rate drops to ~A's level" **PASS** (B-only
0.82% ≤ 1%; B 3.70% within A+1%). The eval's own Test-2 gate reports **OK**.

### PASS / FAIL on the three ship criteria

| # | criterion | result |
|---|-----------|:------:|
| (a) | B mean CER within +0.5pp of A | **FAIL** (+1.032pp) |
| (b) | A mean CER does not regress | **PASS** (21.50% unchanged) |
| (c) | B repeat-artifact rate ≈ A on real CTD crops | **PASS** (B-only 0.82%, B 3.70% vs A 3.29%) |

### Where B is stuck (why postprocess alone can't clear the bar)

A per-case CER decomposition on the 2000-row set (B-worse CER mass = 26.0,
A-worse = 5.4, net ≈ +20.6 → ~+1.03pp) shows the regression is **NOT** the
trailing/CJK artifacts that the collapse fixes (those are a small slice and now
caught). The dominant surviving classes are **intrinsic non-AR decode
corruption woven into the content**, which is not safely postprocessable:

- **Interior phrase duplication / loops:** `顔を突き出した顔を突突ままし`,
  `こ...こら待..こ...こら待て〜〜`, `い...い兵器器..兵.兵`. Cannot dedup without a
  language model; immediate-substring collapse damages **9.8%** of legit GT
  (legit reduplication ドクンドクン / ごちゃごちゃ / ドキドキ).
- **Doubled kana mid-word:** `突突`, `器器`, `300のの`, `子子`. Indistinguishable
  from legit doubled kana / laughter, so uncappable.
- **Substitution errors** (unrelated to repeats): `母`→`心`, `クリ`→`グリ`,
  `牛`→`生`, `どき`→`はき` — pure recognition differences from the non-AR pass.
- **Trailing same-char punct runs** that overlap legit emphasis: `?????` vs
  legit `・・・・・・` (len-6 ellipsis appears in thousands of GT rows), and doubled
  `ーー` (legit per GT `ーー！！`). Capping these would regress A.

### Verdict: **NOT YET — do not ship B on postprocess alone**

Postprocess tightening closed the **artifact** gap (Test 2 now OK, A untouched)
but the **+1.03pp CER regression is intrinsic to B's non-AR decode** (interior
phrase/char duplication + substitutions), not trailing fill. No postprocess rule
can remove it without damaging legitimate Japanese repeats (laughter,
reduplication, ellipsis) and thereby regressing A.

To capture the 10× speed win, a **decode-time fix is required**, in order of
preference:
1. **Re-export the ep60 weights with `decode_ar=True` + dynamic batch** (Option A
   in §2/§3b). This is the surest path: it gives A's sequential decode quality
   *and* batching, eliminating the non-AR hallucination at the source.
2. Add an **EOS-confidence / repeat-aware truncation inside `_decode`** for the
   non-AR logits (cut the tail at the first low-confidence step after a
   high-confidence content run), and re-run this same A/B.

The postprocess changes are safe to keep regardless (A unchanged, B artifacts
down) but are **not sufficient** to gate B in as a like-for-like swap.

Re-run:
```bash
backend/.venv/bin/python backend/scripts/eval_vision/eval_perline_singleline.py \
    --n 2000 --max-chars 20 --pages 18 --batch-size 24 --seed 0
```
