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

