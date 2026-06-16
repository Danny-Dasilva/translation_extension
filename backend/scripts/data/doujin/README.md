# Doujin JP↔EN Pair Mining Pipeline (scaffold)

Mines NSFW manga parallel translation pairs from **raw images** on the NAS
(no text/translation files exist — every pair is produced by OCR). Follows the
Hinami et al. (AAAI'21) method: pair → page-align → bubble-detect+OCR →
emit → QE-filter.

> **Status: SCAFFOLD.** Modules + CLI work on a small-scale path and have been
> smoke-tested on real NAS data. Do **not** run across the full 983 GB corpus
> as-is — scale-up notes below.

> **Internal research use only.** Source material is NSFW doujinshi. This
> pipeline exists to build a private MT training corpus; outputs are silver
> (`gold_flag=False`) and must not be redistributed.

---

## NAS output volatility — READ THIS

The `/mnt/nas/drive_2` CIFS share **empties `_translated_*` / output dirs ~9 min
after write** (verified, see project memory). **All durable output is written to
LOCAL disk** under this directory (`backend/scripts/data/doujin/`). Never point
`--out-dir` at the NAS.

---

## The verified pairing assumption (IMPORTANT)

The task assumed `{id}_en` ↔ `{id}_jp` galleries share a numeric work id.
**This is FALSE** — verified on the live NAS:

```
$ ls galleries/ | sed -E 's/_(en|jp)$//' | sort | uniq -d | wc -l
0          # zero ids have BOTH an _en and a _jp sibling
$ # 35,712 _en  +  34,716 _jp  — every id is single-language
```

Gallery dirs contain **only images**, no title/artist metadata. So gallery
pairing **must be content-based** (cover/page perceptual-hash matching), not
id-based. The `archive_ubuca_v5_*` zips *do* carry rich filename metadata
(`[English]`, circle, artist, title, parody) and are paired by normalized title.

---

## src format contract (consumed by the v12 builder)

```
src = "doujin:{workid}:p{page}:b{idx}"
```

| field    | meaning |
|----------|---------|
| `workid` | slug for the matched work, **no `:` chars**. gallery pair → `g{jp_id}-{en_id}`; ubuca → `title(+artist)` slug |
| `page`   | 0-based aligned page index within the work |
| `idx`    | 0-based bubble index within the page, in **manga reading order** (right-to-left, column-major, top-to-bottom) |

Downstream: group rows by `(workid, page)`, order by `idx` to reconstruct page
context. `register_tag="nsfw_doujin"`, `gold_flag=False`.
Round-trip helpers: `doujin_common.format_src` / `parse_src`.

---

## Latin-OCR dependency choice

**easyocr** (English) is the chosen EN-bubble OCR engine — pip-installable,
GPU-capable (reuses the torch already in `pyproject`), handles curved/handwritten
scanlation fonts better than tesseract. It is **not yet in `pyproject.toml`**.

- Not installed → `doujin_vision.get_latin_ocr()` returns `StubLatinOCR`
  (returns `""`); the JP side stays fully functional and the orchestrator
  reports `empty_en` counts.
- To enable: `backend/.venv/bin/pip install easyocr`
- Swap-in alternative: PaddleOCR weights already ship under
  `app/weights/paddleocr-vl/` — implement the `LatinOCR` Protocol and drop in.

---

## What's functional vs stubbed (this env)

| Stage | Status |
|-------|--------|
| 1 pairing — ubuca filename parse | ✅ functional |
| 1 pairing — gallery cover phash + greedy match | ✅ functional (self-contained DCT phash; `imagehash` dep not required) |
| 2 page alignment (phash NN) | ✅ functional |
| 2 homography (AKAZE+RANSAC) | ✅ functional (cv2 present) |
| 3 bubble detect (YOLO/CTD) | ✅ functional (DetectorService) |
| 3 JP OCR (PARSeq) | ✅ functional (ParseqOCRService) |
| 3 EN OCR (Latin) | ⚠️ **stubbed** — needs `pip install easyocr` |
| 4 pair emission (src + reading order) | ✅ functional |
| 5 QE — LaBSE cosine gate | ✅ functional (LaBSE loads in this env) |
| 5 QE — COMET soft score | ◻️ not wired (intentionally — never hard-gate slang) |

---

## Files

| file | stage |
|------|-------|
| `doujin_common.py` | pure logic: src format, filename/gallery parsing, manga reading order |
| `doujin_vision.py` | phash, page alignment, homography, Latin-OCR interface (lazy heavy deps) |
| `pair_galleries.py` | **stage 1** — scan + pair → `galleries_index.parquet`, `ubuca_index.parquet`, `candidate_pairs.parquet` |
| `align_and_ocr.py` | **stages 2-4** for a single pair / small batch |
| `build_doujin_pairs.py` | **orchestrator** (stages 1-5) → `doujin_pairs.parquet` |
| `tests/` | pytest for the pure logic (36 tests, no GPU) |

---

## Usage

```bash
PY=backend/.venv/bin/python

# Stage 1a — index ubuca zips (fast, filename-only)
$PY scripts/data/doujin/pair_galleries.py --source ubuca --limit 200

# Stage 1b — index galleries + cover phash + greedy cross-lang match (slow CIFS)
$PY scripts/data/doujin/pair_galleries.py --source galleries \
    --match-galleries --limit 200 --max-distance 12

# Stages 2-4 on one pair (debug)
$PY scripts/data/doujin/align_and_ocr.py \
    --jp-dir <NAS>/galleries/<id>_jp --en-dir <NAS>/galleries/<id>_en \
    --workid g<jp>-<en> --max-pages 4

# Full orchestrator over N matched pairs (stages 1-5)
$PY scripts/data/doujin/build_doujin_pairs.py --limit 2 --max-pages 4 \
    --labse-threshold 0.6        # add --keep-empty-en while Latin OCR is stubbed

# Tests
$PY -m pytest scripts/data/doujin/tests/ -q
```

Every stage is `--limit`-able, idempotent (skips existing output unless
`--force`), and resumable.

---

## Scaling up (do NOT do this blindly)

1. **Install easyocr** first — otherwise every EN bubble is empty.
2. **Gallery matching is O(en × jp)** greedy phash. 35k×35k cover comparisons is
   feasible (64-bit hamming) but the CIFS *reads* to compute 70k cover phashes
   are the bottleneck — phash once, persist `galleries_index.parquet`, then
   match purely from the parquet (no NAS).
3. **Copy a working pair's pages to local disk before OCR** — CIFS random reads
   are slow and the share can reap files mid-run.
4. Lower `--max-distance` (try 10-12) to cut false-positive page/work matches.
5. Validate phash work-matches on a labelled sample before trusting them —
   cover phash alone will mismatch reprints/recolors; add an AKAZE inlier-count
   sanity check (already available via `estimate_homography`) as a second gate.
6. Turn the LaBSE gate on (default) and inspect the `qe_score` distribution
   before committing to a threshold for NSFW/slang register.

## Smoke-test artifacts

`*_index.parquet`, `candidate_pairs.parquet`, `doujin_pairs.parquet` in this dir
are small smoke-test outputs (a few galleries / a synthetic self-pair). They are
git-ignored and safe to delete/regenerate.
