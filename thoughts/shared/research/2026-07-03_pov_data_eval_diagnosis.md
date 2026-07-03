---
date: 2026-07-03
topic: "Gendered-pronoun (POV) failure — DATA vs EVAL vs MODEL diagnosis"
model_under_test: "v1 = Qwen3-VL-8B text-SFT (data_v13ship_v1_messages.jsonl)"
scorer: backend/scripts/eval/pov_probe.py
testset: backend/.bench/pov_ab/testset_large.json (148 rows, Furube p1–p9)
status: diagnostic — PROPOSE only, nothing trained/modified
verification: numbers below are ✓VERIFIED by re-running the scorer's own regexes over the on-disk data
---

# POV failure diagnosis: it is primarily an EVAL problem, secondarily a DATA-DISTRIBUTION problem, and NOT a model gender-capability problem

## TL;DR (the one-paragraph answer)

The "20% gendered-POV / 48% gate" number is **not measuring gender**. Of the 105
gendered-resolvable rows in the holdout, the production model commits a genuine
he↔she inversion on **1** — and that 1 is a scorer false-positive (the JP `ふふふっ`
= laughter, rendered "HE--HE--HE--HE", trips the `\bhe\b` regex). The other **102**
"failures" are the model rendering a Japanese pro-drop line as **1st/2nd-person
dialogue ("I"/"you")** where the human scanlator wrote a **3rd-person narration
caption ("she"/"he")**. That is a *person / register* disagreement, not a gender
error, and on most of those rows the model has the **right referent** — it just
didn't spell it with a gendered pronoun. So: the metric conflates person-choice
with gender, is dominated by adversarial 3rd-person captions, carries a 79%-"she"
class prior, and contains label artifacts. Fix the eval first; the model is not
inverting gender.

---

## 1. Is the EVAL measuring the right thing?  → NO. This is the core finding.

**Scope of the testset** ✓VERIFIED (`.bench/pov_ab/testset_large.json`): 148 rows,
**all Furube** (`Counter({furube_p9:29, p1:26, p8:24, p4:23, p6:12, p7:12, p5:11,
p3:6, p2:5})`). The scorer's `FURUBE_37` (`pov_probe.py:70`) = p1/p2/p3 only; the
"All-148" scope is still 100% Furube. Ikenie4/5 gold exists but is **not** in this
POV testset.

**Gendered-resolvable rows** (`required_family`, exactly one family in `human_en`,
`pov_probe.py:107-117`): **105 / 148**. Family prior: **83 she / 22 he = 79% she.**
A model that blindly says "she" on every gendered row scores 79% on the gendered
denominator — well above the 48% gate. The gate is not prior-robust.

**What the production model (`our_en_v11fix8`) actually does on those 105 rows**
✓VERIFIED (re-ran `detect_families`/`pov_pass`):

| v11fix8 status on gendered row | count | share |
|---|---|---|
| no gendered pronoun (person-mismatch: model used I/you) | **102** | 97.1% |
| correct family present | 2 | 1.9% |
| **INVERSION** (opposing family present) | **1** | 1.0% |

The single "inversion" is `furube_p8:p11:idx22`: JP `ふふふっ`, human_en
`HE--HE--HE--HE...` (laughter), v11 `Hehe, I wonder what kind of face she'll make?`.
The "he" required-family is **laughter mis-parsed as the pronoun "he"**; the model's
"she" is arguably the *more* correct referent. **→ genuine he↔she inversions in the
entire holdout: 0.**

**The 102 "person-mismatch" rows, by the person the model chose** ✓VERIFIED:
57 use 1st-person "I", 21 use 2nd-person "you", 24 mixed. And the *references*
themselves: **65 / 105 human_en are pure 3rd-person narration** (contain no I/you at
all — author caption boxes like `SHE HEADED STRAIGHT TO THE OFFICE WITHOUT ANY
SLEEP.`), 40 are mixed.

This is exactly the caveat flagged in `feedback_image_context_poc_result.md`
("the Furube gendered holdout is dominated by 3rd-person NARRATION captions;
both dialogue-trained arms floor") — now quantified. The probe's presence-of-family
semantics (deliberately chosen at `pov_probe.py:120-129` to stop pronoun-evasion
from scoring 100%) **over-corrected**: it now scores a *legitimate person choice*
(rendering a pro-drop caption as internal monologue "I") as a POV failure. The old
`probes.check_pronoun_gender` absence-of-wrong logic was too lenient; this one is too
strict. Neither isolates gender.

**Label-quality artifacts inside the 105 "gendered" rows** ✓VERIFIED:
- **1** laughter false-positive (`ふふふっ` → "HE-HE-HE").
- **2** mega-afterword blobs: whole author afterwords crammed into one "bubble"
  (`furube_p4:p66:idx23` = 1564 chars, `furube_p7:p79:idx26` = 1631 chars). A single
  `required_family` over a 1600-char essay is meaningless.
- **7** rows with JP < 6 chars where the human_en clearly doesn't align to the JP
  fragment (`娘だ!!` → `*HE'S DRUNK.`; `だって` → `HE SAYS SEX WITH MOM IS OVER.`) —
  bbox/OCR misalignment, not translation error.

**Verdict (eval):** the metric is measuring "how often does the model emit a
3rd-person gendered pronoun on a pro-drop caption," heavily confounded by a she-prior,
narration/dialogue register, and ~10% junk labels. It does **not** measure he↔she
resolution, which is the stated target.

---

## 2. Does the TRAINING DATA carry learnable, correct gender signal? → YES for labels; NO in *distribution*.

Sampled the actual v1 training file `scripts/data/v13ship/data_v13ship_v1_messages.jsonl`
(**72,098 rows**, `has_image=0` for all — v1 is text-only, ✓VERIFIED).

**Gendered-pronoun distribution of the assistant targets** ✓VERIFIED:

| target contains | count | share |
|---|---|---|
| no gendered pronoun | 66,121 | **91.7%** |
| she | 3,271 | 4.5% |
| he | 2,516 | 3.5% |
| both | 190 | 0.3% |

Only **8.3%** of training targets carry any gendered pronoun. The model's
overwhelming learned prior is **"Japanese pro-drop → I/you, no gendered pronoun"** —
which is *correct* for dialogue and is exactly what it does on the Furube captions.
The eval then punishes it for obeying its training distribution.

**Label correctness/consistency** (sampled the 5,977 gendered training rows): the
gendered labels are **clean and contextually grounded, not coin-flips** — `ケンジ`→"him",
`彼`→"he", `ナハト`(Nacht)→"she", `[Nishikujou]:…`→"her". Gender is carried by names,
`彼/彼女`, honorifics, or speaker tags. (~40% carry a role/honorific marker in the JP
prompt by a *loose* regex that omits bare `彼`; true recoverable share is higher.) The
labels are not the problem. **The problem is the 92/8 split**: there is almost no
training pressure to produce 3rd-person captions, so the model never learned the
scanlator's narration register.

**Why the v2 POV-contrastive SFT regressed** (`feedback_pov_contrastive_sft_regressed.md`):
sampled `scripts/data/pov_mine/pov_contrastive.sample.jsonl` — balanced 16 he / 14 she,
but drawn from **clean named-entity prose** (`薫`/Kaoru, `エル`/Elle, `ヴィル`/Will,
`カミナル様`/Caminal-sama). That is the *wrong distribution*: Furube's failure is
person-choice on **nameless pro-drop manga captions**, which these rows don't
represent. Forcing 24.9k terse 3rd-person rows pulled fluency down (chrF −3.08) without
touching the real failure — consistent with `project_mt_finetuning_roadmap` /
`project_v11fix9_dataclean_result` (more text pairs is the wrong lever).

---

## 3. Is the gender signal recoverable from the input at all? → Mostly yes from the page, but it is the wrong question for the *current* failure.

For the 105 Furube gendered rows, where is a gender/role/honorific marker
(`姉/兄/母/父/息子/娘/彼/彼女/ちゃん/くん/さん/…/トモキ/桂谷`) available? ✓VERIFIED:

| gender recoverable from… | count | share |
|---|---|---|
| the target JP line itself | 26 | 25% |
| only the page CONTEXT (other OCR lines) | 66 | 63% |
| NEITHER — needs cross-page character memory | 13 | 12% |

So ~88% is in-principle recoverable from the page, 12% genuinely needs cross-page
memory (e.g. `またイキやがった` → "SHE CAME AGAIN" — subject known only from prior pages).

**But this recoverability is moot for the observed failure.** The model is *not*
guessing gender wrong; it is choosing 1st person on pro-drop. Whether the page
"contains" the mother's gender doesn't change that the correct-referent-but-wrong-
person rendering scores 0 under this probe. Recoverability matters only for a *fixed*
probe that actually tests gender (see §5).

---

## 4. Is upstream gender signal being discarded? → YES, three concrete losses.

The serve prompt is text-only and the "context" is a **numbered bag of raw OCR JP
lines** (`build_v11_context_prompt`, `vllm_openai_translation_service.py:282-307`;
mirrors `build_v11_dataset.build_context_prompt:109`). The instruction literally says
"Use the page context for speakers, pronouns, and continuity" (`build_v11_dataset.py:96-100`)
but the model is handed no structured help:

1. **No character roster / name→gender→role map.** A `Cast:` anchor scaffold exists
   (`build_cast_anchor_line`, `vllm_openai_translation_service.py:171-179`; flag
   `translation_cast_anchor`, `app/config.py:355`) but is **default OFF** and only emits
   a **static `DEFAULT_CAST_ANCHOR`** — it is *not* populated from a per-work roster.
   The 63%-context-only + 12%-cross-page rows are precisely what a real roster would fix.
2. **No caption-vs-dialogue (box-type) signal.** Whether a bubble is a narration
   caption or a speech balloon is available upstream (CTD detection / bbox geometry) but
   is flattened into the same numbered list. That flag is exactly what disambiguates
   "render 3rd-person" vs "render I/you" — i.e. the dominant failure axis — and it is
   thrown away.
3. **Honorifics/role nouns survive in raw OCR but are neither normalized nor
   surfaced.** `姉ちゃん/母さん/息子/-ちゃん/トモキ` appear in the context bag but the model
   gets no signal that these bind the referent's gender; OCR garbling further erodes them.

---

## The diagnosis, weighted

| Bucket | Weight | Evidence |
|---|---|---|
| **EVAL** (metric mismeasures) | **Primary** | 0 true he↔she inversions in 105 rows; 102/105 "failures" are person-choice not gender; 79% she prior; laughter FP + afterword blobs + misaligned short-JP rows; narration-dominated (65/105 pure 3rd person) |
| **DATA (distribution)** | Secondary | training is 91.7% no-pronoun → model never learns caption 3rd-person register; upstream roster/box-type/honorific signal discarded; v2 mined pairs were wrong-distribution |
| **MODEL (gender capability)** | **Not the problem** | model does not invert gender; it picks the right referent and a defensible person; labels it "fails" are legitimate monologue renderings |

This also explains the two failed fixes: image-context POC gave +2.4pt but **0 he↔she
corrections** and text-POV-contrastive SFT regressed — because *there were essentially
no he↔she errors to correct.* Both levers were aimed at a failure the probe hallucinated.

---

## 5. Proposed better POV eval (design — not built)

**Principle: separate the two axes the current probe fuses — GENDER (he↔she) and
PERSON/REGISTER (1st/2nd/3rd) — and score each on rows where it is well-posed.**

**A. Clean the row set first (applies to any metric):**
- Drop laughter/onomatopoeia: strip `ふふ+/ﾌﾌ/くくく` JP and `HE-HE`/`HA-HA` runs before
  `detect_families`.
- Drop mega-captions: `len(human_en) > ~300` (afterword blobs).
- Drop misaligned rows: `iou < 0.7` OR `ocr_clean == False` OR `len(jp) < 6`
  (all fields already present in the furube gold jsonl).
- Report **class-balanced** he/she accuracy (equal weight per family) to kill the
  79%-she prior.

**B. GENDER metric — a curated "gender-obligatory, gender-recoverable" subset:**
- Keep a row only if (i) a faithful EN rendering *must* commit to a gender AND
  (ii) gender is inferable from target+context (name/`彼`/`彼女`/honorific/role, i.e.
  the §3 "target" + "context" buckets, ~88%). Hand-verify; this yields a *small*
  (tens of rows) but *honest* set.
- Score **inversion-only**: FAIL iff the prediction asserts the **opposing** gender
  (this is the old absence-of-wrong semantics, now safe because the set is
  gender-obligatory). Evasion is scored separately, not as a gender fail.
- Ideal: build **minimal contrastive pairs** (same JP frame, swapped referent gender
  via name/honorific) and measure whether the model flips with the cue.

**C. PERSON/POV metric — score it as what it is:**
- On narration captions, the reference commits to 3rd person; measure person-agreement
  (1/2/3) against a reference that reflects the *desired product register*, and feed the
  model the caption/dialogue box-type flag (§4.2) so the task is well-posed. Do not call
  this "gender."

**D. Referent-level scoring (stretch):** resolve both ref and prediction to a
*referent id* (the mother / the son) and check the model didn't assert the wrong
person's gender — person-invariant, so "I"/"she" for the same correct referent both pass.

**E. Broaden beyond Furube:** ikenie4/5 gold is even thinner (34/644 and 9/711 gendered)
and also 53–67% narration — so more Furube-style pages won't help. Curate a *dialogue*
he↔she set (the register the model is actually trained for and shipped on).

---

## 6. Concrete data-fix proposals (in ROI order)

1. **Fix the metric before touching training** (§5). The "POV ceiling" is largely a
   measurement artifact; a corrected probe likely shows the model already resolves
   gender well and the real gap is narration *register*, not gender.
2. **Wire a real per-work character roster into the existing `Cast:` anchor.** The
   scaffold exists (`build_cast_anchor_line`, flag `translation_cast_anchor`); populate
   `DEFAULT_CAST_ANCHOR` per work from corpus mining (name→gender→role) and A/B it. This
   directly targets the 63% context-only + 12% cross-page rows and the discarded name/
   honorific signal — an *input* fix, not a weights fix.
3. **Pass the caption-vs-dialogue box-type flag through to the prompt.** It disambiguates
   the dominant person-mismatch failure and is currently discarded upstream.
4. **Preserve honorifics/role nouns end-to-end** (OCR→prompt): ensure `-san/-kun/-chan`
   and `母/息子/姉` reach the target line and context uncorrupted; consider a light
   normalization pass rather than dropping them.
5. **Do NOT add bulk text POV-contrastive pairs** — proven to regress (chrF −3.08) and
   wrong-distribution (named LN/VN prose vs nameless manga captions). If any targeted
   data is added, it must be *manga narration-caption* rows with the box-type/roster
   context, not clean bitext.
6. **Re-scope image-context / teacher distillation to the curated gender set (§5B).**
   The current metric cannot see the image lever (POC showed 0 he↔she fixes because
   there were ~0 to fix). Only a gender-obligatory set can tell whether the image/teacher
   actually helps.

---

## Appendix — exact commands used (all read-only, reproducible)

- Scope/family prior, person-mismatch table, artifacts, recoverability: re-ran the
  scorer's own regexes (`_HE_RE`/`_SHE_RE`/`required_family`/`pov_pass`,
  `pov_probe.py:90-129`) over `.bench/pov_ab/testset_large.json`.
- Training distribution: iterated `scripts/data/v13ship/data_v13ship_v1_messages.jsonl`
  (72,098 rows).
- v2 mined data: `scripts/data/pov_mine/pov_contrastive.sample.jsonl`.
- Ikenie comparison: `scripts/eval/data/ikenie{4,5}/gold_q3.jsonl`.
- Prompt/roster wiring: `vllm_openai_translation_service.py:171-336`,
  `scripts/data/v11/build_v11_dataset.py:96-117`, `app/config.py:355`.
