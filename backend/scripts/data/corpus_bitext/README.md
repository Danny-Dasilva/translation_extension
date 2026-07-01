# corpus_bitext — curated JP→EN bitext mining from page-aligned scanlations

Converts page-aligned **(JP raw page image, EN scanlation page image)** pairs
into curated JP→EN training rows in the project's **v11 page-context parquet
schema** `[prompt, en, src, register_tag, gold_flag]`. This is the data half of
retraining the manga translator on freshly-mined translated pairs.

The English scanlation is a **redraw** of the JP page (layout approximately
preserved, text replaced), and **no text is stored anywhere** — both sides must
be OCR'd from the images, then JP source bubbles aligned to their EN translations
**across the two different images**.

```
(JP page, EN page)
   │  detect + OCR each side                         ── ocr_adapters.py
   ▼
JP bubbles {text,bbox,conf}        EN bubbles {text,bbox}
   │            └──────── align across pages ────────┘   ── align.py
   ▼   matched (jp_src, en_tgt, jp_bbox, en_bbox, match_dist, jp_ocr_conf, page)
   │  curate (precision-favoring filters + quality score)── curate.py
   ▼
v11 page-context rows  [prompt, en, src, register_tag, gold_flag]  ── format_rows.py
```

## Per-side OCR decision

| Side | Engine | Why | Runs on |
|------|--------|-----|---------|
| **JP** (raw) | CTD v26 detect (`comictextdetector_v26_round9_onofix_20260622.onnx`) + **PARSeq** recognize (`parseq_manga_ep60_*`) | The PARSeq recognizer is the production JP path and—critically—emits the **`ocr_conf`** (mean-softmax) signal the garble gate consumes. CTD gives bubble boxes. | **CPU-capable** (onnxruntime CPU EP, auto fallback). GPU ~100–130 FPS detect + ~7 ms/crop recognize. |
| **EN** (typeset redraw) | **VLM** (Qwen2.5/Qwen3-VL) per-bubble text+bbox, reusing `scripts/eval/transcribe_gt_vision.py` | Typeset English is read far more robustly by a grounding VLM than by a JP recognizer, and the VLM **doubles as the EN detector** (one pass → text + bbox per bubble). Matches the eval-gold transcription path exactly. | **GPU / remote** (`100.64.235.63:8001`). **Deferred** until the GPU frees from the training run. Use `--vlm-coord-norm 1000` for Qwen3-VL (0–1000 grid coords). |

The JP detect+OCR half therefore **smoke-tests on CPU today**; only the EN VLM
half waits on the GPU.

## Alignment algorithm (`align.py`)

JP and EN bubbles live on **different images**, so matching is on **normalized**
coordinates (each page normalized by its own pixel size → the unit square).

1. Build a cost matrix over (jp_i, en_j):
   `cost = centroid_dist + 0.25·|Δw|+|Δh| + 0.10·|reading_order_rank_gap|`
   - `centroid_dist` = Euclidean distance of normalized bubble centroids (primary
     cue: the redraw preserves relative position).
   - size term = normalized width/height agreement.
   - reading-order term = the column-major **RTL** rank gap (the exact training
     reading order from `build_v11_dataset.manga_reading_order`) — a tie-break
     that stabilizes ambiguous columns.
2. Solve **optimal bipartite assignment** (`scipy.linear_sum_assignment`; greedy
   mutual-nearest fallback if scipy is absent).
3. **Keep** a pair only when its **normalized centroid distance ≤ `tol`** (default
   0.08). Count mismatches (merged/split bubbles, untranslated SFX present on one
   side only) fall out as unmatched → **precision over recall**.

Output per matched bubble: `{jp_src, en_tgt, jp_bbox, en_bbox, match_dist,
jp_ocr_conf, page}`.

## Curation thresholds (`curate.py`, all tunable via `CurationConfig`)

| Filter | Default | Drops |
|--------|---------|-------|
| `max_match_dist` | 0.06 | loose/ambiguous alignments |
| `min_ocr_conf` + garble gate | 0.65 | low-conf / linguistically-implausible JP OCR (reuses `app.utils.ocr_confidence_gate`) |
| SFX glossary | — | onomatopoeia bubbles (`sfx_pre_translate` hit) |
| EN language | — | EN target with no Latin letters or residual JP glyphs (untranslated/SFX leak, e.g. `ズキッ`) |
| `len_ratio` (en_words / jp_glyphs) | 0.10–2.50 | merged/split/mis-paired bubbles |
| min JP glyphs | 2 | fragments |
| exact dup `(jp, en)` | — | duplicates |
| `min_coverage` (page-level) | 0.35 | whole pages whose alignment coverage is poor (redraw mismatch) |
| `keep_threshold` (quality score) | 0.50 | low-quality rows |

**Quality score** (0–1, stored per row so the threshold is re-tunable without
re-OCR): `0.40·pos + 0.25·conf + 0.15·len + 0.20·coverage`.

EN targets are recased from ALL-CAPS scanlation typeset to natural sentence case
(`to_sentence_case`). `register_tag` defaults to `manga_nsfw` (this corpus is
adult-heavy; `nsfw_frac` is reported). `gold_flag = False` (mined, not human gold).

## Output format (`format_rows.py`) — BYTE-EXACT with serving

Each kept JP bubble is rendered in its **full page context** (every JP dialogue
line on the page, in manga reading order) via the **exact** training builder
`build_context_prompt(PAGE_INSTR, jp_lines, k)` from `build_v11_dataset` — the
same template the v11 LoRA trained on and serving mirrors. A format mismatch here
is the documented **~95 % chrF++ collapse** risk, so the prompt is never
hand-assembled (parity asserted in validation). Context is windowed to 12 lines
around the target (mirrors v11fix7). Schema: `[prompt, en, src, register_tag,
gold_flag]`.

## Validation (known-answer, no GPU) — `validate_ikenie.py`

Uses per-bubble gold for two chapters (`ikenie4/gold_q3.jsonl`,
`ikenie5/gold_q3.jsonl`; 220 pages, 1,352 pairs) as the "OCR output" and a known
JP↔EN pairing. To make cross-page matching realistic, EN boxes are **perturbed**
(global affine = different scan resolution + per-bubble Gaussian centroid jitter)
and **shuffled**, then `align_pages` must recover the gold row↔row pairing.

**Measured alignment precision/recall vs redraw-jitter** (σ as fraction of page
dimension; 3 seeds; tol=0.08):

| jitter σ | precision | recall | f1 |
|---------:|----------:|-------:|---:|
| 0.00 (sanity floor) | 0.998 | 0.998 | 0.998 |
| 0.01 | 0.991 | 0.991 | 0.991 |
| **0.02** (realistic redraw) | **0.965** | **0.963** | **0.964** |
| 0.04 | 0.938 | 0.787 | 0.856 |
| 0.06 | 0.912 | 0.557 | 0.692 |
| 0.08 | 0.875 | 0.401 | 0.550 |

Count-mismatch (drop 20 % EN + 10 % spurious boxes, σ=0.02): **P=0.928, R=0.629**.

**Key property:** precision stays high (0.87–0.99) across all jitter while recall
degrades — the tolerance gate drops uncertain matches rather than mis-pairing,
exactly what curation wants. Ikenie's real EN redraw is near-pixel (gold
IoU≈0.95), i.e. the σ≈0.00–0.01 end. **The realistic jitter magnitude for the
broader 375k corpus (mixed scan sources, hamming up to 14) needs GPU verification
on real JP/EN OCR pairs** — see Assumptions.

End-to-end on the gold (σ=0): **920 curated rows** from 1,352 pairs, quality
concentrated in 0.9–1.0, NSFW frac 1.0. Outputs under `validation_out/`:
`validation_report.json`, `sample_ikenie_bitext.parquet`, `sample_rows.jsonl`.

A **real CPU OCR** smoke test (`smoke_cpu.py`) runs CTD+PARSeq on Ikenie JP
source pages with CUDA disabled and feeds the real bubbles through the full
pipeline (e.g. page 5 → 8 JP bubbles OCR'd @ conf 0.93–1.0 → 5 curated rows).

## Full-corpus GPU cost estimate (201,845 good+partial pages)

Density: ~11 detected regions/page (measured on Ikenie box-insp).

| Stage | Per-page | 201,845 pages |
|-------|----------|---------------|
| JP CTD detect (~130 FPS) | ~7.5 ms | ~0.4 GPU-hr |
| JP PARSeq recognize (~7 ms/crop × 11) | ~77 ms | ~4.3 GPU-hr |
| **JP side total** | | **~5 GPU-hr** (or CPU-only ≈ 50× slower) |
| **EN VLM (7B-VL, image-prefill heavy)** | ~0.5–2 s batched | **~25–60 GPU-hr** (dominant; throughput-dependent) |
| **Total** | | **~30–65 GPU-hr on one GPU** (~1.5–3 days single-GPU; hours on a multi-GPU box) |

The EN VLM pass is the bottleneck and the **least certain** number — flag for
GPU verification. Yield: Ikenie gives ~4.2 kept rows/page; at scale expect
~1–3/page after noisier real OCR + stricter thresholds → **~200k–600k raw rows**.
Since translation-SFT plateaus at ~1–10k pairs, the constraint is **precision,
not volume**: subsample the top quality-score rows (and dedup across galleries)
down to the highest-quality ~10–50k for training.

## How to run over the corpus (once images land in staging)

Images for the good+partial subset are being extracted to
`/home/danny/manga_corpus_staging/` (preserving `galleries/<gid>_<lang>/<file>`).
Output is **LOCAL only** — never `/mnt/nas` (the CIFS share silently reaps output
dirs ~9 min after write).

```bash
cd backend

# 1. (now, GPU busy) JP-OCR half only, on CPU, dumps detected JP bubbles/page:
.venv/bin/python scripts/data/corpus_bitext/run_gallery.py \
    --staging-root /home/danny/manga_corpus_staging \
    --status good,partial --jp-only --cpu-only \
    --out-dir scripts/data/corpus_bitext/shards --resume

# 2. (GPU free) full mine — JP CTD+PARSeq + EN VLM + align + curate + format:
.venv/bin/python scripts/data/corpus_bitext/run_gallery.py \
    --staging-root /home/danny/manga_corpus_staging \
    --status good,partial \
    --vlm-endpoint http://100.64.235.63:8001/v1/chat/completions \
    --vlm-coord-norm 1000 \
    --out-dir scripts/data/corpus_bitext/shards --resume
```

Per-gallery shards `<en_gid>_<jp_gid>.parquet` + stats are written incrementally
(resumable), then merged into `data_corpus_bitext_pagecontext.parquet`. Fold that
into the training mix the same way `build_v11fix7_corrective.py` folds corrective
rows (concat onto `data_v11_pagecontext.parquet`, modest upweight, track
`nsfw_frac`).

## Assumptions needing GPU verification

- **Realistic redraw drift σ** for the broader corpus (Ikenie is near-pixel;
  other scan sources with hamming up to 14 may drift more → lower recall, but
  precision holds). Re-measure alignment P/R on a handful of real JP/EN OCR pairs
  once the VLM is up; raise `tol`/`max_match_dist` only if precision stays high.
- **EN VLM throughput** (the dominant cost term) — benchmark pages/sec on the
  actual served model.
- **EN VLM coord convention** — confirm pixel vs 0–1000 (`--vlm-coord-norm`) for
  the served checkpoint before a full run (wrong coords destroy alignment).
- **NSFW sub-tagging** — currently a blanket `manga_nsfw`; a per-work classifier
  could refine `register_tag` / `nsfw_frac` if needed.
```
