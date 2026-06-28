# fix6 — Corrective fine-tune for "mistranslation on CLEAN OCR"

Status: PLAN (no training run started)
Branch: `fix6-clean-ocr-finetune`
Author: fix6 setup task
Date: 2026-06-26
Related prior work:
- `thoughts/shared/plans/2026-06-14_translation-finetune-scoping.md` (the v11it corrective-SFT plan this extends)
- Memory `feedback_v12_nsfw_oversampling_regression` (DPO-toward-NSFW → euphemism regression — the central risk here)
- Memory `feedback_cpo_pitfalls` (preference-pair training pulls toward terse/bland)
- Memory `project_v11_pagecontext_model` (v11 is PRODUCTION; train/serve format must match)
- Memory `feedback_chat_template_mismatch` (chat-template / prompt-format mismatch → silent ~95% chrF++ collapse)

---

## 0. TL;DR / recommendation

The page-for-page audit of **Ikenie no Haha 4** (134 pages, 497 flagged bubbles) found the single
most frequent model fault is **mistranslation on CLEAN OCR**: 158 bubbles, avg severity 2.25. The
Japanese was *read correctly* (OCR is not at fault) but the English is wrong. This is the
highest-leverage training-data fix available.

Recommendation: build a **corrective SFT slice** (NOT DPO) of `(jp → human_en)` rows that
**replace** our wrong outputs, in the existing v11 `[jp, en, src, register_tag, gold_flag]` seed
schema, fold it through `build_v11_corrective_seed.py` → `build_v11_dataset.py` (which already
3×-upweights the corrective seed and re-expresses it as plain single-line rows in the unified v11
prompt format), and gate on the existing `eval_pagecontext_heldout.jsonl` chrF++ harness plus a new
adult-domain vocab probe set. Keep the contrastive `(jp, our_wrong_en, human_en)` triple only as a
**curation/audit artifact and held-out probe**, not as a DPO training signal — see §4.

---

## 1. Scope

### In scope
- The 158 clean-OCR mistranslation bubbles surfaced by the Ikenie 4 audit, expanded to the three
  documented sub-modes:
  - **(a) wrong sense / over-literal idiom** — 頂きましょうか→"take this in" (should be "shall we
    eat"); 締まる→"it's closing" (should be "tight"); 搾り取られる→"held back" (should be "milked dry").
  - **(b) polarity / negation flips** — 口に含んでんなよ (neg imperative) → "Keep it in your mouth"
    (opposite); 立派な尻尾生えて → "shame you didn't grow".
  - **(c) adult-domain vocab** — 尻(ass)→"breasts"; マンコ(pussy)→"panties"/"butthole";
    バイブ(vibrator)→"the vibe"; 洗濯バサミ(clothespin)→"scissors"; 風俗(brothel)→"customs";
    顔射(facial)→euphemism.
- Sourcing the same three sub-modes from the **broader manga corpus** (not just this title) so the
  fix generalizes and we don't overfit one author/series register.

### Out of scope (do NOT put in this fine-tune)
- Anything an orchestration / prompt fix can reach (笑-marker, short-utterance kana normalize,
  SFX handling) — already handled in `vllm_openai_translation_service.py`.
- OCR fixes — by construction these bubbles have **correct** OCR.
- Detector / inpaint / layout work.
- NSFW *register oversampling* as a percentage knob — the v12 negative result (§6) shows that
  backfires. We fix *specific wrong senses*, not "more NSFW data".

---

## 2. The contrastive-pair concept

For each flagged bubble we have, or can recover, a triple:

```
jp          = ocr_jp           (correct Japanese, from our pipeline)
our_wrong_en= translation_en   (our wrong output, from bubbles.json)
human_en    = typeset English  (the human scanlation GT, from the .webp page image)
```

- `human_en` is the **chosen / gold** target.
- `our_wrong_en` is the **rejected** output — what the model currently produces.

Two distinct uses of the triple, and they must not be conflated:

1. **Training signal (what we actually train on): SFT on `(jp → human_en)` only.** The rejected
   side is *not* fed to the trainer. We want the model to *produce* `human_en`; an SFT row that
   maps `jp → human_en` directly teaches that.
2. **Curation + eval (where the rejected side earns its keep):**
   - During curation, `our_wrong_en` is the trigger that flags the row as worth fixing (it diverges
     from `human_en`). It also lets us bucket by sub-mode (sense / polarity / vocab).
   - As a held-out probe, the triple powers a *contrastive* eval: for each held-out jp, does the
     fine-tuned model now score chrF++ closer to `human_en` than to `our_wrong_en`? This is a
     sharper signal than absolute chrF++ for "did we actually fix the known failure".

---

## 3. SFT vs DPO — decision and rationale

**Decision: corrective SFT (continuation of v11it), NOT DPO.**

Rationale:
- **The documented v12 regression.** The last time we trained a preference signal toward the
  NSFW/explicit register (DPO/chat NSFW oversampled to ~36%), the model learned to **suppress /
  euphemize** explicit terms — the *opposite* of intent (`オチンチン気持ちいい？` "Does your cock feel
  good?" → "Does it feel good?"). Held-out NSFW chrF++ regressed −0.64 (dialogue −1.18) and the
  eroge-vocab probe dropped 7/9→4/9. Sub-mode (c) here is squarely adult-domain vocab — exactly the
  axis that regressed. A DPO objective that pushes *away from* `our_wrong_en` is the same shape of
  pressure and risks the same coyness collapse.
- **CPO/preference pitfalls (memory).** Length-normalized preference loss with teacher-chosen rows
  + high β pulls outputs toward terse/bland; preference-pair margins are a poor proxy for
  translation quality. We rely on holdout chrF++, not margins.
- **SFT is the lower-variance fix for "produce this exact better string".** Our problem is not
  "the model can't rank good vs bad" — it's "the model emits the wrong sense". A direct
  `jp → human_en` SFT row is the most direct correction and is already the mechanism the v11
  pipeline supports (the corrective seed is plain SFT, 3×-upweighted).
- If a *second* iteration later shows residual ranking errors that SFT alone can't move, DPO can be
  reconsidered — but only with a euphemism audit of every preference pair first (per the v12
  lesson) and with the eroge-vocab probe as a hard gate.

---

## 4. Data format (must match the v11 pipeline exactly)

The fix6 rows enter through the **existing** v11 corrective-seed path. Do not invent a new format —
a train/serve prompt-format mismatch causes silent chrF++ collapse (memory
`feedback_chat_template_mismatch`).

### 4.1 Seed schema (what `extract_mistranslation_pairs.py` emits, then is promoted to)
The trainer-facing seed schema is identical to `v11_corrective_seed`:

```
[ jp: str, en: str, src: str, register_tag: str, gold_flag: bool ]
```

- `jp`   = `ocr_jp` (the correct Japanese).
- `en`   = `human_en` (the scanlation GT — the corrected target).  **`our_wrong_en` is dropped here.**
- `src`  = `corrective_v11:cleanocr_<submode>:<id>` where `<submode>` ∈ {sense, polarity, vocab}.
  This keeps the fix6 rows attributable in the composition report (`kind_of` will bucket them as
  `corrective`).
- `register_tag` = matched to the bubble register (`manga_dialog` for SFW dialog,
  `vn_eroge` for adult-domain). Must be a value already present in `data_v10`.
- `gold_flag` = `True` (these are human-verified targets).

### 4.2 How it becomes training rows
`build_v11_dataset.py` → `corrective_rows()` wraps each seed row with
`build_plain_prompt(jp)` = `"Translate the following Japanese to English. Output only the
translation.\n\nJapanese: {jp}"`, sets `en` as the assistant target, and the main composer emits the
corrective seed **3×** (`corr_rows + corr_rows + corr_rows`, no dedup) so it is not drowned out. We
therefore add the fix6 rows to the corrective seed parquet and re-run `build_v11_dataset.py`
unchanged — no schema or pipeline change is required.

### 4.3 Serve-side parity (already correct)
At serve time, `build_v11_plain_prompt` / `build_v11_context_prompt` in
`vllm_openai_translation_service.py` reproduce the same two prompt shapes byte-for-byte (verified:
`V11_PAGE_INSTR` / `V11_PLAIN_INSTR` are the same strings as the dataset builder). Page-context
serving wraps the target line in the numbered Page block; our corrective rows train the plain shape,
which the model also sees at serve time for single-bubble cases. No change needed.

---

## 5. Adult-domain noun glossary (bake into the corrective set)

A curated JP→EN gloss for sub-mode (c). Every gloss term should appear in **multiple** corrective
rows with **varied phrasing** (per the existing seed builder's "teach the PATTERN, not the string"
principle) so the model learns the sense, not a memorized line. Treat this as the seed list; expand
from corpus mining.

| JP | wrong (observed) | correct sense | register_tag |
|----|------------------|---------------|--------------|
| 尻 / お尻 | "breasts" | ass / butt / rear | vn_eroge / manga_dialog |
| マンコ | "panties" / "butthole" | pussy / cunt | vn_eroge |
| ちんこ / チンポ / オチンチン | (euphemism) | cock / dick | vn_eroge |
| バイブ | "the vibe" | vibrator | vn_eroge |
| ローター | "rotor" | (egg) vibrator | vn_eroge |
| 洗濯バサミ | "scissors" | clothespin | manga_dialog |
| 風俗 | "customs" | brothel / sex work | vn_eroge |
| 顔射 | (euphemism) | facial / cumshot | vn_eroge |
| 中出し | (euphemism) | creampie / cum inside | vn_eroge |
| アナル / 尻穴 | "anal hole" mistranslations | ass / anus | vn_eroge |
| 搾り取られる | "held back" | milked dry / drained | vn_eroge |
| 締まる | "it's closing" | tight / clamps down | vn_eroge / manga_dialog |
| 頂きましょうか | "take this in" | shall we eat / let's dig in | manga_dialog |

Notes:
- The gloss is **curation guidance**, not a runtime dictionary. We do NOT add a serve-time term
  substitution (that fights the model and breaks register). We bake the senses into SFT rows.
- Audit each adult-domain target for **euphemism** before including it (the v12 failure mode):
  the GT must use the explicit word where the source does. If the human scanlation itself
  euphemized, drop the row — a euphemistic target reinforces the exact regression we're avoiding.

---

## 6. Risks

| Risk | Mitigation |
|------|-----------|
| **Euphemism regression (the v12 trap).** Training toward "better" adult vocab teaches coyness instead. | SFT not DPO (§3); euphemism-audit every sub-mode (c) target (§5); gate on eroge-vocab probe (must not drop below v11). |
| **GT extraction is from images.** `human_en` lives in the typeset `.webp`, not in any structured field — extracting it needs OCR/vision or a manual pass, and OCR'd English will itself contain errors. | §7 makes human-EN extraction an explicit, reviewed step; vision-OCR output is a *draft* for human verification, never auto-promoted to a gold target. |
| **Overfit to one title.** Ikenie 4 register/author bias. | Source the same three sub-modes from the broader manga corpus; cap per-title share of the corrective set. |
| **Catastrophic forgetting / register drift.** Corrective slice over-corrects and degrades general lines. | Keep the v11 mix ratios; corrective seed stays 3× (not higher) of a small absolute count; run the FULL `eval_pagecontext_heldout.jsonl` chrF++ as a regression gate, not just the targeted probe. |
| **Train/serve format mismatch → silent collapse.** | Reuse the existing seed→`build_v11_dataset` path unchanged; do not hand-format prompts (§4). |
| **Mislabeled "clean OCR".** A garbled/romaji-leak `ocr_jp` slips through and we train on garbage JP. | The extraction script filters on `ocr_conf` threshold + non-garbled heuristic + `filtered`/`ocr_gate_dropped` flags, and the seed gets a human review pass before promotion. |
| **Polarity flips are subtle.** Sub-mode (b) negation errors are easy to mis-pair against GT. | Flag sub-mode (b) rows for mandatory human review; do not auto-include. |

---

## 7. Held-out chrF++ eval plan

Reuse the existing harness and add a targeted probe:

1. **Regression gate (must not drop):** the existing `backend/scripts/data/v11/eval_pagecontext_heldout.jsonl`
   (disjoint-page page-context chrF++). v11 is the baseline; v11.fix6 must be ≥ v11 here.
2. **Targeted clean-OCR probe (the fix):** carve a **held-out** slice of the fix6 triples
   (`jp, our_wrong_en, human_en`) — these pages are NOT in the training corrective seed. Two metrics:
   - absolute chrF++ of model output vs `human_en` (should rise vs v11),
   - **contrastive margin** = chrF++(out, human_en) − chrF++(out, our_wrong_en) (should go positive;
     v11 is typically negative — closer to its own wrong answer).
3. **Eroge-vocab probe (the guardrail):** the existing `model_compare_eval` eroge-vocab probe class
   from `project_v11_pagecontext_model`. v11.fix6 must NOT regress below v11 (the v12 canary).
4. Per-sub-mode breakout (sense / polarity / vocab) so we see which sub-mode moved and which didn't.

---

## 8. Data volume & sourcing

- **This title (Ikenie 4):** 1486 bubbles total; 1379 non-filtered with a translation; ~1113 pass a
  conservative clean-OCR filter (`ocr_conf ≥ 0.85`, non-garbled). The audit flagged **158** as
  clean-OCR mistranslations — these are the high-value seed rows. After human GT extraction +
  review, expect ~120–158 verified `(jp, human_en)` corrective rows from this title.
- **Broader manga corpus:** mine the same three sub-modes from other audited / available titles to
  reach a robust slice **without** overfitting. Target on the order of **300–600** verified
  corrective rows total (this title + corpus). Keep it small and high-precision; the 3× upweight in
  `build_v11_dataset` multiplies the effective count, and the v12 lesson warns against bulk.
- **Adult-domain gloss rows:** generate multiple varied-phrasing rows per gloss term (§5) the same
  way `build_v11_corrective_seed.py` already does for Class 3 — these are author-written corrective
  rows, not extracted, and complement the extracted pairs.
- Cap any single title at ≤ ~40% of the corrective slice.

---

## 9. Sequence (effort)

1. Run `extract_mistranslation_pairs.py` over the Ikenie 4 bench dirs → seed JSONL of
   `(jp, our_en, human_en)` triples (human_en is a vision-OCR DRAFT or empty placeholder).
   **[DONE — extraction half functional + run.]** The `(jp, our_en)` extraction half is
   implemented (clean-OCR filter: `ocr_conf ≥ 0.85`, not `filtered`/`ocr_gate_dropped`,
   `translation_en` present, garble rejected via `is_garbled` + the production
   `is_implausible_japanese` gate) and produced **1085 clean-OCR candidate rows** across all
   134 pages → `backend/scripts/data/v11/seed_cleanocr_pairs.jsonl`. Each row also carries a
   coarse `submode_guess` (vocab 48 / negation 45 / idiom 2 / other 990) to pre-sort the review
   queue. **`human_en` is left EMPTY** — it is TYPESET into the GT page `.webp` and is NOT in any
   structured field; **GT human-EN recovery (vision-OCR draft or manual transcription, then human
   verification) is the remaining step before these become training rows.** No `human_en` is
   fabricated.
2. Human pass: fill/verify `human_en`, assign sub-mode + register_tag, euphemism-audit (c). Drop
   bad pairs (garbled JP, euphemized GT, ambiguous polarity).
3. Expand from the broader corpus + author the gloss rows (§5).
4. Convert verified triples → v11 corrective seed schema (`jp, en=human_en, src, register_tag,
   gold_flag`), append to the corrective parquet (or a fix6-specific seed merged in).
5. Re-run `build_v11_dataset.py` (unchanged) → new training parquet + `eval_pagecontext_heldout.jsonl`.
6. Train v11.fix6 (corrective SFT continuation; same recipe as v11it). **Not part of this task.**
7. Eval gates §7. Ship only if all gates pass and no eroge-vocab regression.

---

## 10. Open questions

- Human-EN extraction method: vision-OCR (which model?) vs manual transcription for the 158 lines.
  Manual is feasible at this volume and avoids OCR'ing the English wrong.
- Do we keep a separate `fix6` corrective seed parquet or merge into `v11_corrective_seed.parquet`?
  (Leaning: separate file, both loaded by `build_v11_dataset` — keeps provenance clean.)
- Exact `ocr_conf` threshold for "clean" — 0.85 is a starting point; tune against false-positive
  garbled rows during the human pass.
