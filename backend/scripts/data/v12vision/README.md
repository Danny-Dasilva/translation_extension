# v12vision — multimodal POC dataset (Phase 2 vision-LoRA seed)

Proof-of-concept **multimodal** training dataset for a future **vision-LoRA**
fine-tune of the manga translator. **Nothing is trained here** — this is just the
dataset + a reusable builder.

## Why

Production translator (Gemma-4 E4B, v11 page-context) is served **text-only**:
OCR'd JP + numbered page context → EN line. The MT roadmap verdict is that adding
the page **image** as context is the #1 lever to fix the pronoun / speaker-collapse
ceiling. The **Ikenie gold pages are the one place we have BOTH the page image
AND high-quality human JP→EN pairs**, so they are the natural seed corpus.

## Files

| file | what |
|------|------|
| `build_v12vision_poc.py` | reusable, chapter-agnostic builder (gold jsonl + image dir → dataset) |
| `data_v12vision_poc.jsonl` | the built POC dataset — **220 page rows** (ikenie4 + ikenie5) |
| `stats.json` | row counts, image found/missing, per-chapter counts |
| `README.md` | this file |

## Row schema (one JSON object per line, **one line per page**)

```json
{
  "image_path":   "<absolute local path to the page image (01_source.webp)>",
  "jp_ocr":       "<full-page OCR'd JP lines, reading order, newline-joined>",
  "page_context": "<v11 serve-format page block: instruction + numbered Page>",
  "en_target":    "<human EN for the page's gold bubbles, reading order, \\n-joined>",
  "meta": { "chapter": "ikenie4|ikenie5", "page": <int>, "n_bubbles": <int> }
}
```

`meta.n_bubbles` = number of gold (human-EN) target bubbles on that page.

## How it was built

For each page, the per-bubble gold rows are grouped by page (`src` =
`"ikenieN:pPP:idxII"`). The `idx` in `src` is **the same index** as the page's
pipeline `bubbles.json` — verified: **100% of gold rows' JP exactly equals
`bubbles.json` `ocr_jp` at the same idx** (1352/1352 across both chapters), and
bbox/page also match. The box-inspection page-dir name is the zero-padded page
number (`p5` → `005`, `p132` → `132`).

Per field:

- **`jp_ocr` / `page_context`** — built from the **full-page** OCR in the
  pipeline's per-page `bubbles.json` (`ocr_jp`, sorted by `idx` = reading order).
  These are the exact OCR lines the v11 **text** model was served, so the JP side
  is byte-faithful to serve format. `page_context` replicates
  `vllm_openai_translation_service.build_v11_context_prompt` **byte-for-byte**
  (the `V11_PAGE_INSTR` string + the `Page:\n1. …\nN. …` block — a documented
  train/serve contract) **except** the trailing per-line `Translate line k: …`
  suffix is omitted, because a **page-level** row has no single marked target.
- **`en_target`** — the **human** EN from `gold_q3.jsonl`, joined in reading
  order (idx order). `gold_q3.jsonl` is a curated **"worst-issues" subset** of each
  page's bubbles (the bubbles a judge flagged + corrected against Qwen3-VL vision
  gold), so this is the highest-quality supervised signal available. Rows with a
  duplicate `idx` (same bubble flagged for >1 issue — 3 pages in ikenie4) are
  deduped keeping the first occurrence.
- **`image_path`** — absolute path to the page's `01_source.webp` (the 1280×1791
  source page the pipeline ingested). **Local, no NAS dependency.**

### Inputs (verified paths)

- Gold pairs: `backend/scripts/eval/data/ikenie4/gold_q3.jsonl`,
  `backend/scripts/eval/data/ikenie5/gold_q3.jsonl`
- Page images + full-page OCR:
  `backend/.bench/ikenie4_v11fix6_box_insp/<NNN>/{01_source.webp,bubbles.json}`,
  `backend/.bench/ikenie5_v11fix6_box_insp/<NNN>/{…}`

> **Image source = LOCAL `01_source.webp`**, *not* the NAS raws. Per project
> memory the `/mnt/nas/drive_2` CIFS share silently reaps `_translated_*` output
> ~9 min after write, so a stable local image source is strongly preferred. The
> NAS galleries (`583875…`, `616137…` for ch4; `628187`, `1782491042389120…` for
> ch5) were therefore **not** used. The builder refuses to write under `/mnt/nas`.

## Counts (see `stats.json`)

| chapter | page rows | images found | images missing | target bubbles (human-EN) |
|---------|-----------|--------------|----------------|---------------------------|
| ikenie4 | 124       | 124          | 0              | 641                       |
| ikenie5 | 96        | 96           | 0              | 711                       |
| **total** | **220**  | **220**      | **0**          | **1352**                  |

## Known gaps / caveats

- **Serve-format match:** the JP side (`jp_ocr` / `page_context`) is **byte-exact
  to v11 serve format** for the page block; the only intentional difference is the
  dropped `Translate line k:` suffix (this dataset is page-level, not marked-line).
- **`en_target` is the gold subset, not every bubble.** `jp_ocr`/`page_context`
  cover the **whole** page (the model still sees full context + the image), but
  `en_target` only covers the gold-curated worst-issue bubbles for which we have
  trustworthy **human** EN — so on most pages `n_bubbles` < number of lines in
  `page_context` (e.g. ikenie4 p5: 8 context lines, 6 EN targets). The non-gold
  bubbles only have *machine* `translation_en` in `bubbles.json`, which is
  deliberately excluded to keep the target high-quality. A future full-page
  variant would need human EN for the remaining bubbles.
- **0 pages missing images / 0 fallbacks** in this build (all 220 gold pages have
  both `01_source.webp` and `bubbles.json` locally). The builder still handles
  missing images (`image_path:""`, tracked in stats) and missing `bubbles.json`
  (falls back to gold JP lines for `jp_ocr`/`page_context`).

## Rebuild

```bash
cd backend/scripts/data/v12vision
python build_v12vision_poc.py                 # both chapters, default paths
python build_v12vision_poc.py --chapters ikenie4
python build_v12vision_poc.py --out /local/dir # LOCAL only; refuses /mnt/nas
```
