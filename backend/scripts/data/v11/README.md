# v11 — Context-Augmented Single-Line SFT dataset

Fresh-from-base SFT data for the manga JP→EN translator (v11). Each example
asks the model to translate **ONE** line while *seeing its surrounding page or
conversation* as context, and is mixed with plain single-line examples so the
model stays strong on isolated lines.

Build: `.venv/bin/python backend/scripts/data/v11/build_v11_dataset.py`

## Files

| file | what |
|------|------|
| `data_v11_pagecontext.parquet` | final training parquet — **291,780 rows** |
| `data_v11_pagecontext.sample.jsonl` | 40-row readable sample |
| `eval_pagecontext_heldout.jsonl` | 2,362 page-context eval rows over **317 disjoint manga pages** (0 overlap with train) for chrF++ |
| `build_v11_dataset.py` | reproducible builder (seed=42) |

## Schema (ONE unified schema)

```
[prompt, en, src, register_tag, gold_flag]
```

`prompt` is the **full user message**. The training `prompt_user_template`
becomes a passthrough `"{prompt}"`. Two prompt shapes share the column:

**(A) Page / conversation context** (manga pages, VN/novel windows):
```
Translate the marked line of this manga page from Japanese to English. Use the
page context for speakers, pronouns, and continuity. Output only the translation
of the marked line.

Page:
1. {jp1}
2. {jp2}
...
N. {jpN}

Translate line {k}: {jpk}
```
`assistant = en` of line k.

**(B) Plain single line** (byte-for-byte the v10 prompt, materialised into `prompt`):
```
Translate the following Japanese to English. Output only the translation.

Japanese: {jp}
```

## Reading order signal — VERIFIED bbox, right-to-left

The manga109 `src` is `manga109:BOOK:pN:hash` where **`hash` == the manga109
annotation `text_id`**. Joining `(book, page, text_id)` against
`scripts/data/manga109/bubbles.parquet` recovers the bbox
`(xmin,ymin,xmax,ymax)` for **all 114,704** manga109 rows. The join is exact:
`jp` matches `bubbles.jp_text` at **100%** (1.0 match rate).

Each page's bubbles are then ordered by **manga reading order**:
right-to-left column-major, top-to-bottom within a column
(`manga_reading_order()` buckets bubbles into vertical columns by x-center with
a 6%-of-page-width tolerance, orders columns rightmost-first, then top→bottom).

Bubble-line collapse: manga109 stores **one `<text>` element per vertical line
inside a bubble**, so a multi-line bubble yields several consecutive `text_id`s
with identical normalised jp+en. The builder collapses consecutive identical
`(jp,en)` runs into one logical bubble so the page context is clean and the
target line is not duplicated.

VN/novel context uses a sliding window: `vntl_v31_1k:rowN:turnK` grouped by
`rowN`, `nilane_small:recN:pairM` grouped by `recN`, ordered by numeric
turn/pair index, window = up to 6 preceding lines + the target.
`vntl_raw` is a single flat group (all under `row0`) → **not windowable**, kept
plain. VN/novel lines already embed speaker tags like `[Kotarou]:` in both jp
and en, so the conversation context anchors speakers directly.

## Composition (291,780 rows, seed=42)

| slice | rows | % |
|-------|------|---|
| **manga page-context** (`*:pagectx`, manga109, 2× upweight) | 112,426 | 38.5% |
| **conversation-context** (`*:convctx`, vntl_v31_1k + nilane) | 26,341 | 9.0% |
| plain vntl_raw | 28,246 | 9.7% |
| plain parallelfiction | 21,000 | 7.2% |
| plain vntl_v31_1k | 20,674 | 7.1% |
| plain manga109 | 19,696 | 6.8% |
| plain open_mantra_train | 12,000 | 4.1% |
| plain gemma_anchor | 8,538 | 2.9% |
| plain sfx_jp_ono | 8,040 | 2.8% |
| plain aratako_synth | 8,031 | 2.8% |
| plain uuf_sfx | 7,500 | 2.6% |
| corrective (v11 seed, 3×) | 5,928 | 2.0% |
| plain sfx_yuiseki_flat | 4,797 | 1.6% |
| plain nilane_small | 4,438 | 1.5% |
| plain sfx_giongo | 2,163 | 0.7% |
| plain ocr_garbage | 1,962 | 0.7% |

**Context-augmented: 138,767 (47.6%)  ·  Plain single-line: 153,013 (52.4%)**

Design choices:
- **Manga page-context upweighted 2×** (`MANGA_CTX_REPEAT=2`) — it is the
  flagship new skill, so it is a dominant but not overwhelming slice (38.5%).
- **Plain rows keep v10 multiplicity** (no dedup) — `ocr_garbage` (6 unique
  pairs ×1,962), `gemma_anchor`, and the sfx sources are *deliberately*
  repetitive (they teach `garbage→"..."`, `sfx→sfx`). Deduping them would
  remove the robustness signal the v10 mix intentionally carries.
- **Context rows ARE deduped** on `(prompt,en)` — accidental dupes there are
  real noise.
- **Corrective seed 3×** so the 1,976 corrective pairs are not drowned out.
- `~35%` of manga bubbles and `~60%` of VN/novel lines are *also* emitted as
  plain single-line, so every register is represented in both modes.

Token budget: with `max_seq_length=1024`, sampled 3,000 rows through the
`unsloth/gemma-4-E4B-it` chat template → **max 508 tokens, 0% over 1024**. No
truncation. The 12-bubble context cap (`MAX_BUBBLES_CONTEXT`) + bubble-collapse
keep prompts well within budget.

## EXACT training-config changes (sft_gemma4_e4b_v10it.py + yaml)

The current stack formats `{jp}` and reads a `jp` column. v11 ships a `prompt`
column instead. Two minimal edits:

### 1. `training/configs/gemma4_e4b_v10it_sft.yaml`

```yaml
data:
  train_path: backend/scripts/data/v11/data_v11_pagecontext.parquet   # was data_v10.parquet
  eval_size: 1000
  prompt_user_template: "{prompt}"   # was the multi-line "Japanese: {jp}" block — now PASSTHROUGH
  completion_field: en
  use_chat_template: true
```

(Optionally point `output.dir` at a v11 run dir, e.g.
`backend/training/runs/manga-bubbles/gemma4_e4b_v11it`.)

### 2. `backend/scripts/train/sft_gemma4_e4b_v10it.py` — `build_chat_dataset()`

Change the per-row field read from `jp` to `prompt` (the user message is now
pre-rendered in the column, so `.format()` is a passthrough):

```python
# BEFORE
jp = (r.get("jp") or "").strip()
en = (r.get("en") or "").strip()
if not jp or not en:
    skipped += 1
    continue
user_msg = user_template.format(jp=jp).rstrip()

# AFTER
prompt = (r.get("prompt") or "").strip()
en = (r.get("en") or "").strip()
if not prompt or not en:
    skipped += 1
    continue
user_msg = user_template.format(prompt=prompt).rstrip()   # template is "{prompt}" → passthrough
```

Everything downstream (chat-template wrapping, completion-only `-100` masking,
LoRA scope, Trainer) is unchanged. Held-out eval still uses the tail slice; for
the dedicated chrF++ page-context eval use `eval_pagecontext_heldout.jsonl`
(disjoint pages) — each row has `prompt`, `en`, plus `jp`/`kind` for reference.

> If you prefer zero script edits: rename the parquet column `prompt`→`jp` and
> set `prompt_user_template: "{jp}"`. The unified-schema approach above (a
> `prompt` column + passthrough template) is cleaner and is what this dataset
> ships, so the one-line `build_chat_dataset` edit is recommended.

All other hyperparameters (LR 2e-4, 1 epoch, eff-batch 16, r=16/α=32,
language-only LoRA regex, max_seq 1024) are unchanged from v10.
```
