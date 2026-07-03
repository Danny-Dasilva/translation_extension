# Detection-Recall Gold Set & Harness

Measures **CTD detection recall** against ground-truth bubble boxes that were
drawn **independently of our own detector**.

## Why this exists (audit §5)

Our POV and OCR gold sets are seeded from our own `ComicTextDetectorService`
output — every gold bbox has `IoU == 1.0` against a detection **by
construction** (all 650 rows). A detector **false-negative** — a bubble the
detector never proposes — is therefore invisible to every existing harness.
Recall is unmeasurable.

This harness fixes that by scoring against boxes the detector never saw.

## Gold source (NOT detector-seeded)

`detection_recall_gold.json` is built from the **AnimeText** dataset (HuggingFace),
via `training/comic-text-detector/scripts/prepare_animetext.py` →
`training/comic-text-detector/data/yolo_blocks/val/`.

- Labels are the **dataset authors' own** block annotations (class `0 =
  text_block`), in YOLO normalized `cxcywh`, converted here to pixel `xyxy`.
- They are the upstream training/eval labels for the CTD model — produced
  **before and independently of** our production detector. No box in this gold
  set came from `ComicTextDetectorService`.
- Contrast: the sibling `data/val/annotations/*.json` files carry `confidence`
  fields — those ARE detector outputs and are deliberately **not** used here.

Current set: **20 pages, 165 boxes**, spread across bubble-count buckets
(1 → 22 boxes/page) for a meaningful sparse-and-dense recall test.

### Regenerating / resizing the gold set

```bash
cd backend
python3 scripts/eval/build_recall_gold.py \
  --src /home/danny/Documents/personal/extension/training/comic-text-detector/data/yolo_blocks/val \
  --out scripts/eval/detection_recall_gold.json \
  --n 20
```

The gold JSON bakes in each page's boxes + image dimensions, so it is
self-contained. The page images themselves are gitignored and stay on disk;
the harness reads them at run time via `--images-dir`.

## Running the harness

**Scoring math (no GPU, no model):** covered by unit tests —

```bash
cd backend && python3 -m pytest scripts/eval/test_detection_recall_eval.py -q
```

**Real detector run (needs onnxruntime + the CTD model, ideally a GPU):**

```bash
cd backend && uv run python scripts/eval/detection_recall_eval.py \
  --gold scripts/eval/detection_recall_gold.json \
  --images-dir /home/danny/Documents/personal/extension/training/comic-text-detector/data/yolo_blocks/val/images \
  --out scripts/eval/detection_recall_result.json
```

Output, per IoU threshold (default `0.5, 0.75`):

- `recall` = matched_gold / total_gold (`1 − miss rate`)
- `precision` = matched_det / total_det
- `false_negatives` = gold boxes with no detector match — the previously
  invisible misses
- `false_positives` = detections matching no gold box

## Status

- [x] Harness `detection_recall_eval.py` — pure scoring layer + detector-backed `run()`.
- [x] Independent gold set `detection_recall_gold.json` (20 pages / 165 boxes, real AnimeText labels — **not stubbed**).
- [x] Unit tests `test_detection_recall_eval.py` — 8 passing, validate IoU/recall/FN math on a synthetic 2-vs-2 fixture.
- [ ] **DEFERRED:** actual CTD run. Needs `onnxruntime` + `models/comictextdetector.onnx`
      + GPU — not available in this worktree (gitignored models absent; no GPU).
      Wiring is done; run the command above on the GPU box to get real numbers.

## Caveats

- AnimeText `text_block` boxes are semantically "text region within a bubble",
  which matches CTD's `blocks` output. Annotation-style differences (e.g. tight
  text vs full balloon) can depress IoU without being true misses; read
  `false_negatives` alongside a spot-check overlay before treating a low recall
  as a regression. The IoU=0.5 tier is the primary recall signal; 0.75 is a
  localization-tightness check.
- Greedy 1-1 matching by descending IoU. A gold box counts as recalled iff some
  detection clears the threshold and isn't already claimed by a tighter match.
