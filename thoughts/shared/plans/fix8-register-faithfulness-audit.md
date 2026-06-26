# fix8 — Register Faithfulness Audit & Remediation Plan

**Status:** PLAN / SETUP only (no training runs in this branch)
**Branch:** `fix8-register-audit`
**Priority:** LOW (severity 1.1) — strictly below the OCR / mistranslation work. Register
softening loses nuance but the bubble is still readable & roughly on-topic; OCR garble and
hard mistranslations are higher-severity and should land first.

---

## 1. Problem

The MT-vs-human comparison on *Ikenie no Haha 4* (hardcore NSFW title) found a **consistent,
one-directional register failure** across ~29 bubbles: the v11 model **softens crude/explicit
Japanese into polite/euphemistic English**. The direction is **always softer, never cruder**:

| JP source | Faithful (human) | v11 model (softened) |
|-----------|------------------|----------------------|
| 精液 (semen) | "cum" / "load" | "the liquid" |
| 喘ぎ声 (moaning) | "moaning" | "calling out" |
| 顔射 (facial) | "facial" / "cum on her face" | euphemistic "Spray" |
| おかわり (blunt "refill"/seconds) | "Again." / "Refill." | over-polite "Another one, please" |
| マンコ (cunt) | "pussy" / "cunt" | "butthole" / euphemism |

This is the same failure class the existing `register_glossary.py` already documents for
潮 (squirt) → "seawater". The new finding is that the softening is **broader than the one term
currently handled**.

## 2. CRITICAL constraint — do NOT repeat the v12 mistake

> **MEMORY: v12 NSFW oversampling regression.** A prior v12 attempt **OVERSAMPLED NSFW DPO/chat
> data to ~36%** to push register. It **BACKFIRED**: the model learned *euphemism* and regressed
> the explicit register further. Audit NSFW pairs for euphemism before retrying.

**Therefore this plan explicitly forbids re-oversampling NSFW data.** The hypothesis here is the
opposite of "we need more NSFW": we believe the *existing* preference pairs already contain
**euphemistic CHOSEN translations** that actively *train the softening*. The fix is to **clean /
repair the chosen side**, not to add volume.

### Why euphemistic CHOSEN rows get in (root-cause, verified)

The preference dataset is built by
`backend/scripts/data/cpo/build_preferences_from_scored.py`:

- Per source line, candidates are `gold` (C1 reference), `teacher` (Gemma 31B), `onpolicy`
  (the model's own sample).
- **`chosen = argmax(COMET-Kiwi score)`, `rejected = argmin`.**
- COMET-Kiwi is a fluency/adequacy QE metric. **It does not reward explicit register.** A
  *fluent euphemism* ("Another one, please") can score **higher** than a *crude-but-faithful*
  rendering ("Refill.") — so the euphemism wins the `chosen` slot and the model is trained
  toward it. This is the softening pump.

Verified on disk (`v10_5_preferences.parquet`, 11,901 rows): explicit source terms are present
(精液×11, 潮×10, 射精×9, おっぱい×34, チンコ×2, おかわり×1, …), and `chosen_kind` splits
`gold 6964 / onpolicy 3999 / teacher 938`. A spot scan shows many golds *are* faithful
(精液→"cum", おっぱい→"tits"), so the fix must be **surgical (flag & repair specific euphemistic
rows)**, not a blanket transform.

## 3. Documented CPO pitfalls to respect (from memory)

When/if a remediation retrain is eventually run (out of scope for *this* branch), honor:

- **Shuffle the preference parquet BEFORE any tail-split.** Unshuffled tail-splits stratify by
  source and corrupt the holdout.
- **Teacher-chosen rows + high β pull the length-normalized loss toward terse/bland outputs.**
  Watch the `chosen_kind=teacher` fraction; crude-faithful refs are often *longer*, and an
  aggressive β will punish them.
- **Judge on a register-specific holdout TRANSLATION eval, NOT on preference-pair margins.**
  Margins can improve while real outputs get blander. The eval must measure faithfulness on held
  examples, not pair separation.

---

## 4. Deliverables & scope of THIS branch

THIS branch ships **planning + a skeleton audit tool only**:

1. This plan doc.
2. `backend/scripts/data/audit_register_euphemism.py` — skeleton (signature + docstring + TODOs)
   that loads a preference parquet and flags rows whose **chosen** translation euphemizes an
   **explicit source term**.

No training, no data mutation, no model changes are performed here.

---

## 5. Audit methodology (what the script will do)

**Input:** a preference parquet (default
`backend/scripts/data/cpo/v10_5_preferences.parquet`). The JP source lives inside the `prompt`
column after `"Japanese: "`; the candidate English is in `chosen` (and `rejected` for contrast).

**Glossary-driven flagging.** Reuse the *pattern* of `register_glossary.RegisterEntry`
(jp_terms + jp_excludes guards + whole-word EN matching). For each explicit term, define:

- `jp_terms` — the crude source token(s): 精液 / 喘ぎ / 顔射 / マンコ / おかわり / 潮 / 射精 / ザーメン …
- `crude_en` — the set of faithful English renderings ("cum", "moan", "facial", "pussy"/"cunt", …)
- `euphemisms` — the softened renderings we treat as a FAILURE
  ("the liquid", "calling out", "spray", "another one please", "butthole", …)
- `jp_excludes` — disarm guards for false positives. **Required**, because naive substring match
  is noisy: in a quick scan イク matched マイク (microphone) and ジェイク (Jake); 潮 matches
  潮干狩り (clam digging). Guard every term.

**Flag rule (a row is suspect when ALL hold):**
1. source (extracted JP) contains a `jp_terms` entry **and not** a `jp_excludes` entry, AND
2. `chosen` contains a known `euphemism` **OR** `chosen` contains **none** of the term's
   `crude_en` faithful renderings (i.e. the explicit concept was dropped/softened), AND
3. optionally: `rejected` *did* contain a faithful rendering (strong signal — the crude
   candidate existed and was demoted by the QE score). This subset is the highest-value repair
   set.

**Output:** a CSV/JSONL report — `term, jp_excerpt, chosen, rejected, chosen_kind, score_gap,
flag_reason` — sorted so the "rejected was faithful, chosen is euphemistic" rows surface first.
Print summary counts per term and per `chosen_kind` (gold vs onpolicy vs teacher tells us whether
the softening comes from the human reference, the model, or the teacher).

**Precision over recall.** Same philosophy as `register_glossary.py`: better to miss some than to
mislabel a clean line. Start the glossary SMALL (the 5 confirmed terms from the *Ikenie* report)
and grow only on confirmed (jp, euphemism→crude) mappings.

## 6. Remediation (DESIGN — executed in a later branch, NOT here)

1. **Repair, don't oversample.** For each flagged row:
   - if a faithful candidate exists in the same source row (often the `rejected` side, or the
     `teacher`/`onpolicy` candidate), **swap chosen↔rejected** or re-pick `chosen` to the
     faithful candidate; OR
   - if every candidate is euphemistic, **drop the row** (don't train softening either way) or
     hand it to the small crude-faithful reference set below.
2. **Add a small crude-faithful reference set.** A *targeted, NOT oversampled* set of correctly
   explicit references for the confirmed failure terms, sized to **correct** the bias, not to
   dominate the mix. Keep the NSFW fraction at/below its current level — the v12 lesson is that
   *fraction*, not *faithfulness*, was the failure.
3. **Verify with a register-specific holdout TRANSLATION eval** (not preference margins): a fixed
   list of crude source lines with faithful gold; measure how often the model emits a euphemism.
   This eval is the go/no-go gate.

## 7. Complementary LIGHT serve-time option (cheap, no retrain)

A near-zero-cost mitigation while the data fix is pending. **There is already a gated system
prompt:**

- `LIGHT_SYSTEM_PROMPT` at `backend/app/services/translation_text_utils.py:67`, filled via
  `.format(target=...)`, gated by `settings.translation_system_prompt_enabled`
  (`backend/app/config.py:192`, default **False**).
- It is currently applied **only** on the numbered-block path
  (`backend/app/services/vllm_openai_translation_service.py:521`). The single-line v11 path
  (`translate_single`, ~line 311) sends a **bare user message with no system prompt**.

**Option:** extend `LIGHT_SYSTEM_PROMPT` with one register-faithfulness sentence, e.g.
*"Render crude/explicit anatomy and acts with the same bluntness as the source — do not soften
精液→'the liquid', マンコ→euphemism, 顔射→'spray'; use direct English (cum, pussy, facial)."*

**CAVEATS (must respect):**
- **The v11 single-line model is acutely train/serve format-sensitive** — a documented ~95%
  chrF++ collapse occurs when the served format drifts from training. v10it also *collapses* on
  the heavier prompt. So any system-prompt change is **A/B-gated behind the existing flag** and
  validated on the holdout before defaulting on. Do **not** inject this into the bare single-line
  prompt without an A/B; it may regress more than it fixes.
- This is a **band-aid**, not the fix. The durable fix is the data repair in §6. Prefer the
  post-edit glossary (§8) for terms with a clean wrong→right mapping.

## 8. Where any deterministic post-edit goes (no retrain, highest precision)

For terms with an unambiguous (jp, euphemism → crude) mapping, the safest fix is the existing
post-edit glossary, **not** a prompt change:

- Add `RegisterEntry` rows to `backend/app/services/register_glossary.py` (today it only handles
  潮→squirt). Both pipelines already call it through
  `backend/app/services/translation_postedit.py` (`restore_register` → `postedit_one`), so a new
  entry fixes the API router **and** the batch benchmark with no further wiring.
- Note the honesty limit from that module: a post-edit can only swap a wrong WORD that is
  present; it **cannot re-insert dropped content** (e.g. 精液→"the liquid" is swappable, but a
  fully flattened bubble is not). Those un-swappable cases are exactly what the data fix in §6 is
  for.

---

## 9. Explicit DO-NOT list

- **DO NOT re-oversample NSFW DPO/chat data** — the v12 regression proved oversampling teaches
  euphemism. Fix the chosen side; keep the NSFW fraction flat.
- **DO NOT** change the bare single-line v11 prompt without an A/B behind
  `translation_system_prompt_enabled` (train/serve sensitivity / ~95% collapse risk).
- **DO NOT** judge a remediation retrain on preference-pair margins — use the register holdout
  translation eval.
- **DO NOT** tail-split the preference parquet without shuffling first.

## 10. File map (verified on disk)

| Concern | Path |
|---------|------|
| Canonical preference parquet (gitignored, main checkout) | `backend/scripts/data/cpo/v10_5_preferences.parquet` (11,901 rows) |
| Preference-pair builder (argmax-QE chosen) | `backend/scripts/data/cpo/build_preferences_from_scored.py` |
| Upstream candidate+score pipeline | `backend/scripts/data/build_v10_5_preferences.py` |
| Older CPO triplet builder (gemma-chosen) | `backend/scripts/data/build_cpo_triplets.py` |
| Existing register post-edit glossary (潮→squirt) | `backend/app/services/register_glossary.py` |
| Shared post-edit chain (wires both pipelines) | `backend/app/services/translation_postedit.py` |
| Gated serve-time system prompt | `backend/app/services/translation_text_utils.py:67` (`LIGHT_SYSTEM_PROMPT`) |
| System-prompt flag | `backend/app/config.py:192` (`translation_system_prompt_enabled`) |
| System-prompt apply site | `backend/app/services/vllm_openai_translation_service.py:521` |
| Audit tool (this branch) | `backend/scripts/data/audit_register_euphemism.py` |
