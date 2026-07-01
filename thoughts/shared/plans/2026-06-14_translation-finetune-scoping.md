---
title: Targeted fine-tune to fix recurring JP→EN error classes in gemma4 "v10it"
date: 2026-06-14
status: SCOPING / PLAN ONLY — no training in this doc
author: research session (Opus)
related:
  - thoughts/shared/handoffs/kraken-v10it-deploy/
  - thoughts/shared/handoffs/kraken-v10_5-cpo/
  - memory: feedback_chat_template_mismatch.md, feedback_cpo_pitfalls.md
---

# Targeted fine-tune to fix 4 recurring error classes in v10it

## 0. TL;DR / recommendation

- **Method: a second SFT corrective pass (call it `v11it`)**, NOT a new CPO/DPO round, for error classes 1–3. The pronoun, comparative-grammar, and vocab errors are **deterministic mistakes with a single correct answer** — corrective (input → fixed output) SFT pairs teach the right token sequence directly. Preference learning is the wrong tool: it only nudges relative probability and (per our own CPO post-mortem) the length-normalized SimPO reward pulls toward terse/bland outputs and regressed every metric. Error class 4 (confabulation) is best addressed by *not* adding length, plus a handful of negative-style examples — also achievable in SFT by including faithful short references.
- **Train as a continuation LoRA on the same `unsloth/gemma-4-E4B-it` base**, on a **mix of [new corrective pairs ~3–5k] + [a downsampled slice of the existing v10 general data ~40–60k]** to prevent catastrophic forgetting / register drift. Keep the exact v10it recipe and chat template.
- **Hard constraint (from prior 8-hour near-miss): the corrective data AND every eval MUST use the identical Gemma-4 `-it` chat template** that v10it trained with. See §3.
- **A large slice of the "errors" is actually a serving/orchestration gap, not a model-capability gap** — flag and split these out before building data (§2.4). The model lacks page context at inference because the deployed path translates a page as a flat numbered list with no neighbor context. Only the residual, context-independent errors belong in this fine-tune.

---

## 1. Existing training setup (investigated, with paths)

### 1.1 What v10it is and how it was trained
- **Training script:** `backend/scripts/train/sft_gemma4_e4b_v10it.py`
- **Config:** `backend/training/configs/gemma4_e4b_v10it_sft.yaml`
- **Base model:** `unsloth/gemma-4-E4B-it` (instruction-tuned, **NOT** `-pt`). Config line 5.
- **Method:** **SFT** (plain HF `Trainer`, not `SFTTrainer`, because Gemma4ForConditionalGeneration is detected as a VLM and Unsloth silently sets `skip_prepare_dataset=True` — script lines 326–331, 59–84). Completion-only loss is baked in by pre-tokenizing and setting `labels=-100` on prompt tokens (lines 86–127).
- **LoRA:** r=16, alpha=32, dropout=0.0, bias=none. Target modules are an **anchored regex scoped to `language_model.*` only = 258 modules** (config line 21; runtime asserts the count at lines 235–247). This scoping is required so the Gemma-4 MTP drafter stays compatible at deploy. Any new fine-tune MUST keep the same 258-module regex.
- **Hyperparams:** lr 2e-4, cosine, warmup 0.03, weight_decay 0.01, **1 epoch**, per-device batch 4 × grad-accum 4 = **effective batch 16**, bf16, adamw_8bit, max_seq 1024 (config lines 36–55).
- **Data:** `backend/scripts/data/manga109/data_v10.parquet` — **258,981 rows**, schema `[jp, en, src, register_tag, gold_flag]`.
- **Output adapter:** `backend/training/runs/manga-bubbles/gemma4_e4b_v10it/final`; merged (k-norm-patched, vLLM-ready) bundle at `.../gemma4_e4b_v10it/merged`.

### 1.2 The training prompt + chat template (CRITICAL)
- **User-message body** (config lines 28–31, and `build_chat_dataset` lines 95–103):
  ```
  Translate the following Japanese to English. Output only the translation.

  Japanese: {jp}
  ```
- This bare body is then wrapped with `tok.apply_chat_template([{role:user, content:...}], add_generation_prompt=True)` — i.e. the **Gemma-4 `-it` chat template** (`<bos><start_of_turn>user … <end_of_turn> <start_of_turn>model`). The completion is the `en` string plus the closing turn token.
- **Single source line in → single English line out.** No page context, no neighbor bubbles, no system prompt during training.

### 1.3 Dataset composition (`data_v10.parquet`)
By `register_tag`: manga_dialog 114,704 · vn_eroge 34,500 · novel 28,500 · vn 28,246 · sfx 22,500 · manga 15,000 · synthetic 8,031 · garbage 7,500.
By `src` prefix: manga109 114,704 (Manga109 teacher-translated, 1.5× weighted) · vntl_v31_1k 34,500 · vntl_raw 28,246 · parallelfiction 21,000 · open_mantra_train 12,000 · … · ocr_garbage 1,962.
Provenance: `backend/scripts/data/manga109/data_v10.mix-summary.json` = v7.1 base (144,277 rows) + Manga109 unique (76,469 rows) up-weighted ×1.5 → 114,704 sampled rows. `gold_flag`: 194,204 true / 64,777 false.
**Note:** there is already a `vn_eroge` register slice (34,500) — the intimate/adult register the inspection complains about *exists* in the data but is under-represented vs. 114k clinical manga_dialog. This matters for error class 3.

### 1.4 The CPO experiment (v10.5) — DO NOT REPEAT BLINDLY
- **Script:** `backend/scripts/train/cpo_gemma4_e4b_v10_5.py`; config `backend/training/configs/gemma4_e4b_v10_5_cpo.yaml`.
- Merged-then-re-LoRA chain on top of v10it merged. CPO-SimPO joint: `loss_type=simpo, cpo_alpha=1.0, simpo_gamma=0.5, beta=2.0, lr=1e-6`, effective batch 64.
- Preference builder: `backend/scripts/data/cpo/build_preferences_from_scored.py` → `v10_5_preferences.parquet` (11,901 pairs; schema `[prompt, chosen, rejected, chosen_score, rejected_score, margin, chosen_kind, rejected_kind, src]`). Candidates = gold / teacher / on-policy, scored by CometKiwi-XL; chosen=argmax, rejected=argmin, margin≥0.05, len-ratio∈[0.5,2.0].
- **Outcome (from memory `feedback_cpo_pitfalls.md`):** v10.5 **regressed** Gemma-EM −8.56pp, chrF++ −7.02, BLEU −12.37 vs v10it. v10it (no CPO) remains production. Two documented failure modes: unshuffled tail-split biased the eval slice (44% teacher-chosen in eval vs 4% train), and teacher-chosen rows + high β pulled the policy terse/bland. **These are hard constraints below.**

### 1.5 Eval harness
- **Held-out sets:** `backend/training/eval_held_out/` — `MANIFEST.json`, `vntl128.jsonl` (128), `open_mantra_test.jsonl` (631), `regression_canary.jsonl` (500, news_commentary — the "don't regress general MT" canary), `probes.jsonl` (31 hand-built, categories: names/honorifics/curly_quotes/sfx/idiom/repetition/refusal/length). flores is MISSING (row_count 0); `custom_manga_1500` is absent-by-design.
- **Scoring:** `backend/scripts/eval/score_summary_metrics_v2.py` — sacrebleu **chrF++** (`word_order=2`), BLEU, teacher-fidelity EM, CometKiwi-XL (reference-free QE), XCOMET-XL, MetricX-24. Per-bubble alignment against a Gemma teacher reference jsonl.
- **Significance:** `backend/scripts/eval/paired_bs_chrf.py` / `paired_bs_metric.py` (paired bootstrap).
- **Chat-template-correct generation:** `backend/scripts/eval/diag_v10it_chat_template.py` and `inference_v10it_quality.py` (these are the scripts that apply `apply_chat_template` — use these patterns, NOT the old raw-prompt bench).
- **vLLM serve for eval:** `backend/scripts/eval/serve_v10it_vllm.sh` (merged + MTP drafter, `/v1/chat/completions`).

### 1.6 How the model is actually SERVED (key gap)
- `backend/app/services/vllm_openai_translation_service.py:182` `translate_numbered_block`: a whole page's bubbles are packed into ONE `/chat/completions` call as a plain numbered list ("1. text"→"1. translation"), temperature 0.0. There is **no neighbor-context window and no system prompt** (lines 203–217 explicitly note v10it was *not* trained on the few-shot batched system prompt and collapses if given it).
- **Implication for error class 1 (self-reference):** to translate お母さん先にイッちゃう correctly you need to know the *speaker is the mother*. Per-line training and flat-list serving both strip that. Some of these errors are unreachable by any per-line fine-tune and need a context-bearing prompt change instead — see §2.4.

---

## 2. The 4 error classes → which are model-capability gaps

For each: the linguistic root cause, whether SFT corrective data can fix it, and the data strategy.

### Class 1 — 3rd-person self-reference (お母さん/母さん → "my mom")
- **Example:** お母さん先にイッちゃうわねぇ♡ → wrong "My mom's going to cum first" / correct "Mommy's gonna cum first ♡". Also corrupts the SON's lines mentioning 母さん.
- **Root cause:** Japanese uses kinship nouns (お母さん, 母さん, お姉ちゃん) as 1st/2nd-person pronouns depending on speaker. The correct rendering depends on **who is speaking**, which is not in the single source line. This is **partly a context gap, partly a capability gap**:
  - *Capability part (FIXABLE by SFT):* a mother referring to herself as お母さん in an utterance with 1st-person verb morphology (イッちゃう**わ**ねぇ, sentence-final feminine わ/の) should default to "Mommy/I", not "my mom". The model can learn this default from corrective pairs that pin the mapping `お母さん + feminine-self-predicate → Mommy/I`.
  - *Context part (NOT fixable per-line):* when the same noun is genuinely 3rd-person ("姉ちゃんがトイレに行ってる間に母さんから…" → "I got a message from **Mom** while she…" — which the current output actually got right) you need the discourse to disambiguate. Route these to the serving-prompt change (§2.4), not the fine-tune.
- **SFT verdict:** YES for the self-reference default. Build pairs where お母さん/母さん in self-referential utterances map to Mommy/Mom/I, and a *contrastive* set where they are genuinely 3rd-person, so the model learns the cue (sentence-final feminine particles, verb person) rather than a blanket rule.

### Class 2 — Comparative grammar より / の方がいい inverted
- **Examples:** 姉ちゃんよりいいですぅ → wrong "I'm better than the big sister" / correct "This is better than [with] big sister". 母さんの方がいいですぅ → wrong "I'd rather be the mother" / correct "Mom's is better".
- **Root cause:** Pure **grammatical-capability gap**, fully context-independent. `XよりY (がいい)` = "Y is better than X" / "better than X". `Xの方がいい` = "X is the better one / X is better". The model is mis-assigning the comparison's subject and inverting the direction. This is deterministic — exactly the case where corrective SFT pairs are far more direct than preference pairs.
- **SFT verdict:** YES, highest-confidence win. Generate a **grammar-pattern drill set**: many `Xより…いい` and `Xの方がいい` templates across nouns/registers with correct English, so the construction is over-represented relative to its ~natural frequency.

### Class 3 — Adult/manga vocab + register
- **Examples:** ブラ → "top" (should be "bra"); バカ♡ (affectionate) → "Kid!" (should be "Silly♡/Dummy♡"); hallucinated "breast milk" from お母さん+匂い (smell); register too clinical for an intimate genre.
- **Root cause:** Mix of (a) **lexical gaps** (ブラ=bra, specific adult vocab) — capability, SFT-fixable; (b) **register/politeness mapping** (affectionate バカ♡, casual feminine endings) — capability, SFT-fixable with register-matched references; (c) the clinical tone is partly a **data-balance** problem (114k clinical manga_dialog vs 34.5k vn_eroge). The hallucinated "breast milk" overlaps with Class 4.
- **SFT verdict:** YES, with a curated **vocab/register lexicon set** + up-weighting the existing `vn_eroge` register slice in the mix. Pull genre-appropriate references from the existing `vn_eroge`/`open_mantra` corpus rather than inventing tone.

### Class 4 — Confabulation (adds content absent from source on long emotive lines)
- **Root cause:** The model pads long emotive/short-input lines with invented content (e.g. "breast milk"). This is a **faithfulness/decoding tendency**, the hardest to fix with data alone.
- **SFT verdict:** PARTIAL via SFT — include faithful, *non-padded* references for long emotive lines and short interjections (短い→短い), so the training distribution rewards staying on-source. Do **not** chase this with preference pairs (the SimPO length term already biases short; combining that with our terse-collapse history is dangerous). If residual after SFT, address at decode time (lower max_new_tokens budget per line in the serving path, and/or a light repetition/length guard) rather than another training round.

### 2.4 SPLIT OUT what prompt/orchestration can fix (do NOT put in the fine-tune)
Before building data, triage every flagged bubble. **These belong to a serving change, not the model fine-tune:**
- Genuinely 3rd-person kinship references that need neighbor context (Class 1 context-part).
- Speaker attribution in general.
- The numbered-list serving path (`vllm_openai_translation_service.py:182`) strips intra-page context; a *context-bearing per-line prompt* (e.g. include the 1–2 neighbor bubbles as context, or a lightweight speaker tag) would fix a chunk of Class 1 **without any training**. Recommend a parallel, separate task to add an optional context window to the serving prompt and re-measure — only the errors that survive that change are true capability gaps for the fine-tune.
- OCR-garble lines (e.g. the duplicated/garbled bubble idx 7 in `Part13_inspection_v6/006/bubbles.json`) are upstream OCR failures — exclude entirely.

---

## 3. Hard constraints (from prior sessions — non-negotiable)

1. **Chat-template parity (memory: `feedback_chat_template_mismatch.md`).** v10it was trained with `apply_chat_template` on `unsloth/gemma-4-E4B-it`. The corrective dataset MUST be built with the **exact same user-message body** (`Translate the following Japanese to English. Output only the translation.\n\nJapanese: {jp}`) and wrapped with the same chat template. Every eval MUST go through `/v1/chat/completions` (vLLM) or `apply_chat_template` (Unsloth) — never raw `prompt=`. A converging loss is NOT evidence of success; always run the 20-prompt chat-template smoke test (`diag_v10it_chat_template.py`) before trusting any number. Prior cost of getting this wrong: ~8 hours and a false "model is broken" conclusion (chrF++ 20.9 vs 70.4).
2. **If any preference/CPO is ever used:** shuffle the parquet (`df.sample(fraction=1.0, shuffle=True, seed=42)`) before the tail-split; do not use preference-pair `rewards/margins` as the early-stop metric — use holdout chrF++ on a translation set; drop teacher-chosen rows or lower β to ≤0.5 and lr to ≤2e-7. (We are recommending SFT specifically to sidestep all of this.)
3. **Keep the 258-module language-only LoRA regex** (`^model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$`) so the MTP drafter stays compatible at deploy.
4. **Catastrophic-forgetting guard:** never train the corrective set alone. Always mix with a downsampled slice of `data_v10.parquet` and gate the run on the `regression_canary.jsonl` (general MT) not regressing.

---

## 4. Dataset design

### 4.1 Schema (must match the existing SFT pipeline exactly)
Produce a parquet with the **same `[jp, en, src, register_tag, gold_flag]` schema** as `data_v10.parquet` so it drops straight into `build_chat_dataset`. New rows: `gold_flag=true`, `register_tag` set to the genre (`vn_eroge`/`manga_dialog`/etc.), `src` prefixed `corrective_v11:<class>:<id>` for traceability and ablation.

### 4.2 Per-class data sourcing and target counts

| Class | Source strategy | Target count | gold source |
|---|---|---|---|
| 1 — self-reference (capability part) | Mine `お母さん/母さん/お姉ちゃん`-bearing lines from `data_v10.parquet` + the Manga109/vn_eroge corpus; curate corrected EN. Add a **contrastive 3rd-person** subset. | ~1,200 self-ref + ~400 contrastive | Hand-curate the high-value seed (~150) from `.bench/Part13_inspection_*/bubbles.json`; expand with a stronger teacher (gemma-4-12B-it or translategemma-27b-it) then human-spot-check. |
| 2 — comparative grammar | **Templated drill set**: enumerate `Xより(も)…(が)いい` and `Xの方がいい` over nouns/pronouns/registers; correct EN per template. Mine real instances from corpus for naturalness. | ~1,500 (1,000 templated + 500 mined) | Teacher-generate then verify by rule (direction must match the construction). Highest-confidence, cheapest class. |
| 3 — vocab + register | Curated **lexicon list** (ブラ=bra, affectionate バカ♡, etc.) as short pairs; plus up-weight existing `vn_eroge` register rows (already in corpus). | ~800 lexicon + reuse existing 34.5k vn_eroge via mix-up-weighting | Lexicon: hand-built + teacher for sentence context. Register: reuse existing gold, no new generation needed. |
| 4 — confabulation | Faithful references for long emotive lines + short interjections (no padding). Mine the long-line / short-line tails of the corpus. | ~600 | Prefer existing gold; for the inspection-surfaced lines, hand-curate faithful EN. |

**Total new corrective pairs ≈ 3,500–4,500.** (Round numbers; tune after the §2.4 triage tells us how many flagged bubbles are truly capability gaps.)

### 4.3 Teacher-model use (guard against the CPO teacher-terseness trap)
- Use a stronger reference model (**gemma-4-12B-it** or **translategemma-27b-it**) only to *draft* candidate corrected EN, then **human-spot-check** (especially Class 1/3 where tone matters). Do NOT auto-accept teacher output as gold — the CPO post-mortem showed a 31B teacher's compact outputs degrade voice/faithfulness. Class 2 can be auto-verified by a direction-checking rule and needs little human review.
- The inspection `bubbles.json` files give `ocr_jp` + the bad `translation_en` — perfect for seeding "this JP, NOT this EN, the correct EN is …" curation, but we only store the **(jp, corrected_en)** SFT pair (no rejected field — this is SFT, not preference).

### 4.4 Mix ratio (catastrophic-forgetting / register-drift guard)
- Final training parquet = **new corrective (~4k) + a downsampled general slice from `data_v10.parquet`**.
- Recommended general slice: **~40,000–60,000 rows** sampled from `data_v10.parquet`, **up-weighting `vn_eroge`** (e.g. take all 34.5k vn_eroge + ~20k stratified across manga_dialog/vn/novel) to shift register toward the intimate genre without forgetting general MT.
- Corrective fraction ≈ **6–9% of the mix** — enough to move the targeted behaviors, small enough to avoid overfitting the drills. (Mirrors the ~8% teacher-chosen ratio that *did* move behavior in CPO — but here it moves it toward correctness, not terseness.)
- Train **1 epoch** like v10it (the corrective rows will be seen once; if drills underfit, raise their weight by duplication ×2–3 rather than adding epochs, to avoid forgetting).

### 4.5 What we ALREADY have vs. need to CREATE

| Already have | Need to create |
|---|---|
| Full v10it SFT pipeline + config (reusable verbatim) | Triage of flagged bubbles → capability vs. orchestration (§2.4) |
| `data_v10.parquet` 258k rows incl. 34.5k vn_eroge register | ~4k curated corrective pairs across the 4 classes |
| Inspection `bubbles.json` with ocr_jp + bad EN (seed corpus) | Templated grammar drill generator (Class 2) + lexicon list (Class 3) |
| Chat-template-correct eval scripts (`diag_v10it_chat_template.py`, `inference_v10it_quality.py`) | A **targeted held-out eval set per error class** (§5) |
| chrF++/CometKiwi scoring + paired-bootstrap significance | The mixed training parquet (corrective + downsampled general) |
| Held-out general sets incl. `regression_canary.jsonl` | (Optional, parallel) serving-prompt context window for Class-1 context-part |
| Merge + vLLM-MTP serve scripts | — |

---

## 5. Method recommendation + eval plan

### 5.1 Method: corrective SFT continuation (`v11it`)
- Re-run `sft_gemma4_e4b_v10it.py` **unchanged in mechanism**, pointing `data.train_path` at the new mixed parquet. Same base (`unsloth/gemma-4-E4B-it`), same 258-module regex, same chat template, same effective batch 16, 1 epoch.
- **One hyperparam to reconsider:** because this is a *corrective continuation* and we want to avoid over-writing v10it's general competence, consider lr **1e-4** (half of v10it's fresh-train 2e-4) — a continuation-style rate. Decide via the dry-run + a 200-step mid-train quality eval (the CPO post-mortem showed quality regressions only appear after step ~200, so a step-100 smoke is insufficient — add a step-200+ check).
- **Why not DPO/CPO:** Classes 1–2 are single-correct-answer mistakes → SFT teaches the exact target sequence directly; preference learning only shifts relative likelihood and risks the documented terse/bland collapse. Class 4 is actively *hurt* by SimPO's length term. Class 3 register is a data-balance problem solved by mix-weighting, not preferences. (If, after SFT, a specific behavior needs sharpening, a *tiny* targeted DPO with the §3.2 guards could be a follow-up — but it is explicitly out of scope for the first pass.)

### 5.2 Targeted eval sets (build alongside the data)
Create one held-out JSONL per class (`{jp, en_ref, category, src}`, ~40–80 lines each, **disjoint from training**):
- `eval_class1_selfref.jsonl` — self-referential kinship + contrastive 3rd-person.
- `eval_class2_comparative.jsonl` — `Xより…いい` / `Xの方がいい` held-out templates and mined lines.
- `eval_class3_vocab_register.jsonl` — lexicon + register lines.
- `eval_class4_faithfulness.jsonl` — long emotive + short interjection lines.
Score with `score_summary_metrics_v2.py` (chrF++ + CometKiwi). For Classes 1–2 add a **rule-based accuracy check** (does the output contain the correct comparison direction / 1st-person rendering?) — chrF++ alone is noisy on short lines.

### 5.3 Regression gates (must pass before shipping v11it)
- `regression_canary.jsonl` (general MT): chrF++ must **not** drop more than ~0.5 (paired-bootstrap, `paired_bs_chrf.py`) vs v10it.
- `vntl128.jsonl` + `open_mantra_test.jsonl`: chrF++/CometKiwi non-regression (the v10.5 failure was caught exactly here).
- `probes.jsonl`: names/honorifics/curly-quotes behaviors unchanged.
- Per-class targeted sets: **improvement** over v10it on the 4 sets, with no regression on the general gates.
- **Mandatory chat-template smoke** (20 prompts via `diag_v10it_chat_template.py`) before any scoring run.

### 5.4 Decode/serving follow-ups (separate, not part of the fine-tune)
- Add an optional neighbor-context window (or speaker tag) to `translate_numbered_block` in `vllm_openai_translation_service.py` and re-measure Class-1 context-part — this is the cheapest fix for a chunk of the self-reference errors and should be tried in parallel.
- If Class-4 confabulation persists, tighten per-line `translate_max_tokens` budget and keep the existing repetition guard.

---

## 6. Effort / sequence

1. **Triage (0.5 day):** Walk all `.bench/Part13_inspection_*/bubbles.json` (and any other inspection runs); label each flagged bubble capability vs. orchestration vs. OCR-garble (§2.4). Output a tagged seed CSV. *Gate: how many true capability gaps per class — this re-sizes §4.2.*
2. **Class-2 drill generator (0.5 day):** Highest-confidence, mostly automatable with rule verification. Produces ~1,500 pairs.
3. **Class-1/3/4 curation (1.5–2 days):** Mine corpus + teacher-draft + human-spot-check. ~2,500 pairs. Build the per-class held-out eval sets (disjoint).
4. **Assemble mixed parquet (0.5 day):** corrective + downsampled-up-weighted general slice; write to `backend/scripts/data/manga109/data_v11.parquet`; mirror the `mix-summary.json` bookkeeping.
5. **Dry-run + mid-train eval (0.5 day):** 100-step smoke + chat-template 20-prompt check + a 200-step quality probe.
6. **Full train `v11it` (~3–5 h GPU):** reuse `sft_gemma4_e4b_v10it.py` with the new config (`gemma4_e4b_v11it_sft.yaml`, lr 1e-4 candidate).
7. **Merge + eval (0.5 day):** merge LoRA, serve via `serve_v10it_vllm.sh`, run §5.2/§5.3 gates with paired-bootstrap significance.
8. **Ship or iterate:** if a class underfits, duplicate its drills ×2–3 (not more epochs) and re-run; if a class is purely context-dependent, route to the §5.4 serving change instead.

**Rough total: ~5–7 working days + 1 GPU train cycle**, dominated by curation (step 3). Class 2 is the fastest, highest-confidence win and could ship first as a mini-ablation to validate the whole pipeline before investing in Classes 1/3/4.

---

## 7. Open questions to resolve before building data
- After §2.4 triage, what is the true capability-gap count per class? (Re-sizes §4.2.)
- Is the parallel serving-prompt context change in scope now, or strictly after v11it? (Affects how much of Class 1 the fine-tune must carry.)
- Confirm a teacher is available locally (gemma-4-12B-it / translategemma-27b-it) for drafting, and budget human spot-check time accordingly.
