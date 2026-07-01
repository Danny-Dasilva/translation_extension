# CTD Detector Retrain — Scoping Plan: Unbubbled On-Art Vertical SFX/Moan Text

Date: 2026-06-14
Status: SCOPING ONLY (no training, no code changes)
Author: research/scoping pass

## Problem statement

Unbubbled, hand-drawn vertical kana SFX / moan text drawn directly over body
artwork (the イク/ダメ/あっ class — low contrast, irregular stroke, no bubble
background) produces near-zero detection signal from the production
comic-text-detector. Concrete measurement: Part13 page 012 bottom-left column
(イクッ×4 / ダメッ / あっ) has **DBNet line-prob max = 0.031** in that ROI —
i.e. effectively no signal — so the column is never cropped, never OCR'd, never
translated.

A naive fix (lowering `ctd_text_threshold`, default `0.3` —
`backend/app/config.py:65`) does not work: an earlier threshold sweep flooded
false positives (18 → 30 lines) without recovering the SFX column, because the
prob map simply has no mass there. The fix has to come from the **model**, not
post-processing.

---

## Part 1 — Current production CTD (what serving runs)

### Serving service
`backend/app/services/ctd_service.py` (`ComicTextDetectorService`).

- Input: letterboxed 1024×1024 RGB, `/255`, NCHW
  (`_preprocess`, `_letterbox` — `ctd_service.py:77-116`).
- Config defaults (`backend/app/config.py:63-68`):
  `ctd_model_path=models/comictextdetector.onnx`, `ctd_input_size=1024`,
  `ctd_text_threshold=0.3`, `ctd_block_confidence=0.4`, `ctd_min_text_area=100`.

### Model file & identity (VERIFIED)
`backend/models/comictextdetector.onnx` — 94.6 MB,
`md5 165141f9…` — **identical** to
`training/comic-text-detector/data/comic-text-detector.onnx`.
This is the **original upstream dmMaze `comic-text-detector`** model, NOT either
of the locally-retrained variants. Confirmed output heads via onnxruntime:

| Output | Shape | Role |
|--------|-------|------|
| `blk`  | `[1, 64512, 7]`        | YOLO-style block detections (cx,cy,w,h,obj,2×cls) |
| `seg`  | `[1, 1, 1024, 1024]`   | 1-class binary text segmentation mask |
| `det`  | `[1, 2, 1024, 1024]`   | DBNet text-line maps (ch0 = prob/shrink, ch1 = threshold) |

### How text_lines vs blocks are produced
- `blk` → `_parse_blocks` (`ctd_service.py:273-354`): conf prefilter
  `obj*cls ≥ block_confidence`, scale back, NMS, contained-box filter.
- `det` → `_extract_text_lines` (`ctd_service.py:356-409`): takes `det[0,0]`
  prob map, **binary = prob > text_threshold (0.3)**, `findContours`,
  bounding-rects ≥ `min_area`. **This is the head that fails on SFX** — the prob
  map has ~0 mass over hand-drawn on-art kana, so no contour is ever found.
- When `blk` yields nothing, blocks are derived from lines via koharu merge
  (`_derive_blocks_from_text_lines`, `:663`). So if `det` is empty for the SFX
  column, there is no fallback — the column is fully invisible.
- `seg` → `_process_mask` (`:411`) only refines the inpainting mask; it is
  clipped to block bounds, so it cannot rescue an undetected region either.

**Conclusion:** the gap is specifically the **DBNet line head (`det`)** (and
secondarily the block head) having no learned response to unbubbled on-art
vertical kana. The architecture (1024² UNet seg + DBNet line + YOLO block) is
adequate; the *training distribution* is the problem — upstream was trained on
bubbled / printed text.

---

## Part 2 — Training infrastructure (what exists)

### 2a. LOCAL (`training/comic-text-detector/`, untracked at repo root)
A self-contained **YOLOv11-backbone reimplementation** (not the upstream repo).
- Train entry points: `train_db.py`, `train_seg.py`;
  datasets `db_dataset.py`, `seg_dataset.py`; `basemodel.py` (`TextDetector`).
- Config `configs/default.yaml` (YOLOv11 backbone, seg+detection modes).
- `TRAINING_PROGRESS.md`: produced `runs/segmentation/best.pt` (seg 100ep) +
  `runs/detection/best.pt` (DBHead 148ep), exported to a `mask[1,1,1024,1024] +
  lines[1,2,1024,1024]` ONNX (`scripts/export.py`).
- Data: `data/merged_train`, `data/merged_val`, `data/generated_annotations`
  (auto-labeled), `data/yolo_blocks`. Annotations = masks (`*.png`) + YOLO
  `*.txt` + JSON bbox.
- **Assessment:** older / experimental. Single-class line head, no
  onomatopoeia-specific class. Usable as a fallback but superseded by remote.

### 2b. REMOTE (`danny@100.64.235.63:/home/danny/Documents/personal/manga/comic-text-detector/`)
This is the **mature, actively-developed** CTD training repo (git, NAS-backed,
last commits 2026-06-13/14). It is the source of truth and already contains most
of what an SFX retrain needs. Key pieces:

**Model definition** — `basemodel.py`:
- `TextDetectorV26` (`:262`): YOLO26 backbone (frozen) → `UnetHead` (seg) +
  `DBHead` (line polygons). `initialize_db(unet_weights)` warm-starts the DBHead
  `upconv3/4` from the trained UNet (`:300-308`).
- `UnetHead` (`:79`) supports **per-class branches** at upconv5/6 — i.e. it can
  emit a **2-channel seg mask (ch0 = text, ch1 = onomatopoeia)**.
- `DBHead` (`:183`): 3-channel output (shrink, threshold, binary),
  `shrink_with_sigmoid=True`.

**Train scripts** (`scripts/`):
| Script | Head | Notes |
|--------|------|-------|
| `train_seg_v26_2class.py` | UNet seg, **2-class (text + onomatopoeia)** | WeightedRandomSampler `--sampler-mode=ono_weighted` oversamples by ono pixel content; `--resume-1ch` transfers a 1-ch checkpoint into ch0 and inits ch1 |
| `train_db_v26.py` | DBHead (DBNet line) | trains the previously-empty `text_det` head; labels = YOLO-OBB polys → MakeShrinkMap/MakeBorderMap; `DBLoss(use_bce=False)` (Dice) |
| `train_obb.py` / `train_obb_nano.py` | YOLO26s-**OBB** lines | 2 classes: `horizontal_text_line`, `vertical_text_line` (`data/manga_lines_obb.yaml`) |
| `train_detector*.py` | YOLO26 block | 3-class eng/ja/unknown |
| `train_multitask.py`, `train_shared_neck.py` | combined | shared-neck export |

**Launch wrappers:** `scripts/launch_v26_seg_retrain.sh`,
`launch_v26_obb_retrain.sh`, `launch_large_training.sh`,
`launch_optionc_finetune.sh`.

**Datasets** (`datasets/ -> /mnt/nas/drive_2/manga-ml/datasets/`):
- `manga_unified` (bubble/block det, 3-class; train≈6996/val≈836/test≈452).
- `coo_masks` — **Comic-Onomatopoeia (COO) glyph polygons → ono masks, 4779
  mask PNGs already built** (3-ch BGR, G=ono channel). `scripts/build_coo_masks.py`.
- `manga_text_seg_zenodo` / `manga_text_seg_zenodo_built`, `manga_seg`,
  `manga_segmentation_masks` (+`_addon3`) — text seg masks.
- `obb_consensus_labels_20260505` — OBB line labels (≥10k train) for vertical/
  horizontal text lines.
- `manga109s` extracted.

**Existing trained runs** (`runs/`): `seg_v26_2class_*` (multiple
bundle A/B runs through 2026-05-15), `manga_obb` (Bundle E OBB retrain —
git log: "V5-independent recall@0.5 0.433 → 0.971"), `finetune_obb_neckhead`.

**Exported ONNX** (`/mnt/nas/drive_2/manga-ml/onnx/`):
`unified_v26.onnx`, `unified_v26_2class.onnx`, `unified_v26_all_heads.onnx`,
`unified_v26_shared_neck.onnx` (+ `.engine`).

### 2c. Production-vs-remote GAP (the most important finding)
**Production serving still runs the ORIGINAL upstream model.** The remote box
has already trained:
- a **2-class seg head with a dedicated onomatopoeia channel**, and
- an **OBB head with an explicit `vertical_text_line` class**,
neither of which has been shipped to `backend/models/`. A large part of the
"retrain" may actually be **(a) evaluate the existing `unified_v26_all_heads.onnx`
/ 2-class seg on the SFX failure cases, and only then (b) targeted-augment +
fine-tune** to close whatever residual gap remains. Do not assume a from-scratch
retrain is required until the existing v26 ono/OBB heads are measured on the
イクッ-class pages.

---

## Part 3 — Retrain plan

### Step 0 (gate): measure existing v26 heads on the failure set — DO THIS FIRST
Before any training, pull `unified_v26_all_heads.onnx` and the latest
`seg_v26_2class` / `manga_obb` weights and run them on a curated set of
unbubbled-SFX pages (Part13 p012 and siblings). Two outcomes:
- If the ono-seg channel or the `vertical_text_line` OBB head already fires on
  the イクッ column → the work is mostly **integration/serving** (Part 4),
  not training. Likely the cheapest win.
- If they still miss → proceed to fine-tune below, now with a concrete baseline.

### Data — sourcing the SFX class
Target text class: hand-drawn vertical kana SFX/moan over artwork, low contrast.
Sources, in priority order:

1. **COO (Comic Onomatopoeia)** — already built: 4779 ono masks in
   `datasets/coo_masks`. This is the single best existing source for the SFX
   *class* (tight glyph polygons, vertical kana). Primary signal.
2. **Mine the in-house manga corpus** — page images at
   `/mnt/nas/drive_2/manga-ml/ehentai_corpus/nhentai` (galleries; the
   Haha-to-Ochite-Iku-class chapters live here, NOT in
   `corpus/` which is text-only LM data). These contain exactly the adult-manga
   moan-SFX distribution that production sees. Semi-automated labeling:
   - run the v26 ono-seg + OBB heads to propose regions,
   - human-correct in a fast polygon tool (the repo already has consensus /
     `annotate_all_heads.py`, `annotate_pairs.py`, `build_*_masks.py` flow),
   - this is the **domain-match** data that COO lacks (COO is shōnen/seinen).
3. **Synthetic augmentation** — render vertical kana SFX strings
   (イク/ダメ/あっ/んっ… lexicon — note `corpus/raw/sfx.txt` already exists as
   a SFX string source) over real manga pages from manga_unified, with
   hand-drawn-style fonts (`fonts/` on remote), random rotation, low-alpha /
   low-contrast blending, stroke jitter, outline-only variants. This is the
   cheapest way to get *volume* and to specifically teach low-contrast-over-art.
   `text_rendering.py` (local) / synthtiger helpers (`scripts/convert_synthtiger.py`,
   `_synthtiger_v2_helper.py`) already exist.
4. **Manga109 / Zenodo / MS92** existing seg datasets — for the mix-in
   regression anchor (bubbled/printed text), already built.

**Volume estimate (fine-tune, not from-scratch):**
- Real SFX (COO + mined): aim **~800–1500 annotated pages** with on-art vertical
  SFX regions (COO 4779 masks ≈ a few hundred pages already; mine ~300–500 more
  domain pages).
- Synthetic: **5k–15k** rendered SFX-over-art composites (cheap, generate to taste).
- Regression mix-in (bubbled text): keep at **~50–60%** of each batch from
  `manga_unified` / zenodo so bubbled detection does not regress.

### Approach — fine-tune from current weights (NOT from scratch)
- **Heads to emphasize:** the **DBNet line head (`det`)** is the one that fails
  in serving — prioritize `train_db_v26.py` with OBB-poly labels that include the
  vertical SFX regions; secondarily the **2-class seg head** (`train_seg_v26_2class.py`,
  `--sampler-mode=ono_weighted`, `--resume-1ch`) so the ono channel is populated.
  The OBB `vertical_text_line` head (`train_obb.py`) is the most direct structural
  match for vertical columns and should be (re)trained with mined SFX in the mix.
- **Warm-start:** resume from the existing v26 weights (backbone frozen, per the
  v26 recipe — `train_db_v26.py` sets backbone+seg to eval/no-grad, trains
  dbnet+adapters). For the seg head use `--resume-1ch` to preserve text knowledge
  in ch0.
- **Augmentation for the SFX class specifically:**
  - vertical-orientation emphasis (do NOT flip_lr/ud — reading direction matters;
    the v26 hyp already sets `flip_lr/ud=0.0`),
  - low-contrast-over-art: alpha-blend / multiply text onto busy art crops,
    reduce text-vs-background luminance delta, add screentone backgrounds,
  - stroke/scale jitter, slight rotation (±10–15°) for hand-drawn feel,
  - `ono_weighted` sampler so SFX-bearing pages are oversampled despite being a
    minority of pixels.
- **Anti-regression (mix-in ratio):** ~50–60% bubbled/printed pages per batch;
  validate bubbled recall every epoch and early-stop on regression.

### Eval
Two fixed held-out sets, never in train:
1. **SFX-recall set** — ~50–100 pages with hand-drawn on-art vertical SFX
   (incl. Part13 p012). Metric: **per-line recall on the SFX class** at fixed
   `text_threshold=0.3` (and a small sweep), **gated on precision not collapsing**.
   Reuse `scripts/bench_human_gt.py` (already has `--ono-skip-very-sparse-gt`
   semantics, the G=ono / R=text 3-ch mask convention, and the
   false-positive-flood accounting that the earlier 18→30 sweep exposed).
2. **Bubbled-regression set** — normal bubbled pages; metric: recall/precision
   must stay ≥ current production. Reuse the existing `manga_unified` val/test
   and OBB human-GT test (`manga_lines_obb_humangt_test`,
   `build_obb_humangt_test.py`).
- Primary success criterion: **SFX-class line recall goes from ~0 to a usable
  level (target ≥0.6) WITHOUT bubbled-set recall dropping or page-level
  false-positive count exploding** (the 18→30 flood is the failure mode to beat —
  measure FP/page, not just recall).

### Effort / sequence (GPU: remote RTX 5090, free)
1. **(0.5 day) Step 0 gate** — eval existing `unified_v26_all_heads.onnx` +
   latest `seg_v26_2class`/`manga_obb` weights on the SFX failure set. Decide
   train-vs-integrate.
2. **(1–2 days) Data** — curate SFX-recall + regression eval sets; mine ~300–500
   domain SFX pages from `ehentai_corpus/nhentai` + pseudo-label + human-correct;
   wire COO masks; generate synthetic SFX-over-art composites.
3. **(1 day each, parallelizable on 5090)** Fine-tune:
   - `train_db_v26.py` (line head — the serving-critical one),
   - `train_seg_v26_2class.py` (ono channel),
   - optionally `train_obb.py` (vertical_text_line).
   Each is hours, not days, since backbone is frozen and we warm-start.
4. **(0.5 day) Eval + threshold/FP sweep** via `bench_human_gt.py`.
5. **(0.5 day) Export + serving integration** (Part 4).

Total: ~4–6 days wall, most of it data curation, not GPU time.

---

## Part 4 — Serving integration (after a winning checkpoint)
The serving `ctd_service.py` already auto-assigns outputs by name/shape heuristic
(`_assign_outputs`, `:163`) and handles `[1,1,…]`/`[1,2,…]` maps, so a v26 export
that keeps `blk`/`seg`/`det` (or adds a 2-ch seg) is **drop-in**. Steps:
- Export the fine-tuned model to the same `blk` + `seg`(+ono) + `det` ONNX layout
  (existing `scripts/export_unified.py`); the remote already produces
  `unified_v26_all_heads.onnx`.
- If a 2-class seg is shipped, ch1 (ono) likely needs a small handling tweak in
  `_process_mask`/`_assign_outputs` (currently assumes 1-ch seg) — minor.
- Copy ONNX to `backend/models/comictextdetector.onnx` (+ optional `.fp16.onnx`,
  which serving auto-prefers — `ctd_service.py:36-41`).
- Re-validate the SFX page (Part13 p012) end-to-end through the pipeline.

---

## Key file references
**Serving:** `backend/app/services/ctd_service.py`, `backend/app/config.py:63-68`,
`backend/models/comictextdetector.onnx`.
**Local train:** `training/comic-text-detector/{train_db.py,train_seg.py,db_dataset.py,seg_dataset.py,basemodel.py,configs/default.yaml}`.
**Remote train (`danny@100.64.235.63`):**
`/home/danny/Documents/personal/manga/comic-text-detector/`
- `basemodel.py` (`TextDetectorV26`, `UnetHead`, `DBHead`)
- `scripts/{train_db_v26.py,train_seg_v26_2class.py,train_obb.py,build_coo_masks.py,bench_human_gt.py,annotate_all_heads.py}`
- `scripts/launch_v26_{seg,obb}_retrain.sh`
- `data/{manga_unified_clean.yaml,manga_lines_obb.yaml,train_seg_v26_hyp.yaml}`
**Remote data/NAS:**
`/mnt/nas/drive_2/manga-ml/datasets/{coo_masks,obb_consensus_labels_20260505,manga_unified,manga_text_seg_zenodo_built}`,
`/mnt/nas/drive_2/manga-ml/onnx/unified_v26_all_heads.onnx`,
`/mnt/nas/drive_2/manga-ml/ehentai_corpus/nhentai` (mining source),
`/mnt/nas/drive_2/manga-ml/corpus/raw/sfx.txt` (SFX string lexicon).
