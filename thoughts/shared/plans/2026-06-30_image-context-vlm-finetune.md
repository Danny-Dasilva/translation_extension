# Image-Context JP→EN Manga-Translation LoRA on a 7–8B VLM — Scoping Runbook

**Date:** 2026-06-30
**Status:** SCOPING (research-only; nothing trained, no pipeline code touched)
**Owner thread:** mt-finetuning-roadmap → Phase 2 (image-as-context, the validated #1 lever)
**TL;DR decision:** Prototype on **Qwen3-VL-8B-Instruct** (Apache-2.0, already cached+served on the box, the model that scored **46% POV vs our 4B's 17%** in the validated A/B). Train a page-image-context LoRA using the **PBP-VIS-NUM** input format (numbered bubbles on a redacted page image + numbered JP line list), on a **Manga109 + NSFW-corpus + Ikenie/Furube-gold** mix, gated on the 44-case POV harness + a refusal-rate=0 probe + no chrF regression. The Gemma-4-E4B v12vision scaffold is reused as the **data builder + recipe pattern**, not the base.

This runbook scopes a multi-week investment. It is decision-forcing: each section ends with the concrete choice and its kill criterion.

---

## 0. Evidence baseline (do not re-derive — see memory)

| Fact | Number | Source |
|---|---|---|
| Our 4B (v11fix8), text-only, POV-correct on 44 hard swap bubbles | **17%** (wrong 77% of the time) | `project_mt_finetuning_roadmap` UPDATE 2026-06-30b |
| Qwen3-VL-8B **text-only**, same 44 cases | **46%** (~3× our 4B) | same |
| Qwen3-VL-8B **+ page image, zero-shot** | 50% (only +2; fixed I↔you, **zero** he↔she) | same |
| Furube human-gold e2e: dominant failure | speaker/referent POV swaps + short-bubble hallucination on CLEAN jp | `project_furube_human_eval` |
| v11fix9 data-clean-only retrain | **FAILED** to beat v11fix8 → data cleaning is not the lever | `project_v11fix9_dataclean_result` |
| Lippmann COLING 2025 PBP-VIS-NUM (GPT-4 Turbo, zero-shot) | chrF **36.8**, page-level optimal, **beyond-page HURTS**, biggest gain on short bubbles | arXiv 2411.02589 |

**Two interpretations that drive the whole plan:**
1. The **4B→8B model upgrade is the high-confidence, validated lever** (+29 POV points, text-only). Even if the image bet fails, an 8B base is a real win — so the base-model choice carries the program.
2. Zero-shot image adds ~nothing because the untrained model can't *use* low-res manga panels for gender; **only a fine-tuned image model tests the image hypothesis**. This is the experiment that doesn't exist yet.

---

## 1. BASE MODEL DECISION

### Decision: **Qwen3-VL-8B-Instruct** to prototype. Fallback ladder below.

| Criterion | Qwen3-VL-8B-Instruct | Qwen2.5-VL-7B-Instruct | Gemma-4-E4B vision scaffold (existing) |
|---|---|---|---|
| **Quality ceiling (POV)** | **46% text-only (MEASURED on our harness)** | not measured here; Qwen3 dense ≥ Qwen2.5 per Qwen | 4B = 17% (the discourse bottleneck we're escaping) |
| **NSFW refusal** | refuses explicit even w/ hardened prompt → needs FT suppression | same class (Instruct-tuned) | base barely refuses (4B, lightly safety-tuned) — but weak |
| **vLLM serving** | supported (`Qwen3-VL-Dense` in vLLM registry/ascend docs) | supported, very mature | gemma4 **vision** serving UNCONFIRMED on this rig (serve stub warns) |
| **5090 32GB LoRA feasibility** | ~24GB bf16-LoRA reported; fits 32GB, QLoRA-4bit fallback | ~20GB, fits easily | fits (4B), already scaffolded |
| **Latency (~2–4s/page budget)** | acceptable; lose Gemma MTP draft | acceptable | fastest (4B) but lowest quality |
| **License** | **Apache-2.0** | Apache-2.0 | Gemma license (more restrictive) |
| **Already cached on box?** | **YES — served at `100.64.235.63:8001`, used for our EN-OCR gold + the POV A/B** | no | yes (base `unsloth/gemma-4-E4B-it`) |
| **unsloth FastVisionModel FT** | YES (official Qwen3-VL-8B vision notebook exists) | YES | YES (scaffold present) |

**Why Qwen3-VL-8B-Instruct wins:**
- It is the **only base whose quality lever is measured on our own harness** (46% POV). Choosing it banks the validated +29-point model-size gain regardless of how the image bet lands.
- It is **already on the box and serving** — zero acquisition cost, and the EN-OCR corpus pipeline (`corpus_bitext/`) already calls it, so the serving + processor plumbing is proven.
- **Apache-2.0** removes any redistribution/derivative friction (matters for a commercial extension).
- Qwen3-VL ≥ Qwen2.5-VL on reasoning/STEM per Qwen's own claims, and Qwen3-VL's dynamic-resolution + 32-px patch rounding gives finer control of the image-token budget than Qwen2.5's 28-px.

**Why NOT the Gemma-4-E4B v12vision scaffold as the base:** the base is the **weak 4B** that *is* the discourse bottleneck (17% POV). The image LoRA on a 4B can at best recover the ~33% image-only cases while still failing the ~67% capacity cases the 8B already fixes text-only. The scaffold's value is its **data builder + LoRA-section audit pattern + box-hardening**, which port to Qwen.

**Fallback ladder (if the prototype stalls):**
1. **Qwen3-VL-8B LoRA won't fit / vLLM vision-serve breaks** → switch to **Qwen2.5-VL-7B-Instruct** (lighter, most mature vLLM path; re-run the 44-case A/B text-only first to confirm its POV ceiling ≈ Qwen3's before committing).
2. **Image FT shows no lift over 46% text-only** (image bet dead for this base) → ship **text-only Qwen3-VL-8B-Instruct SFT** as the production translator upgrade anyway (still +29 POV vs the 4B — the validated win).
3. **Qwen refusals prove incorrigible** (see §2) → fall back to the **Gemma-4-E4B v12vision scaffold** (weak base, but no refusal wall and known-good infra) for the image experiment, while keeping text-only Qwen for the model-size win.

---

## 2. NSFW-REFUSAL MITIGATION

**The constraint:** Qwen3-VL-8B-Instruct refuses explicit content ("I can't translate that…") even with a hardened system prompt. The training corpus is NSFW manga, and production must translate it. **A refusal in production = a blank/garbage bubble.**

**Why SFT suppresses it (mechanism, not hope):**
- Refusal is a **repurposed direction that already exists in the base model**, *installed* by instruct safety-tuning, not a deep capability (Arditi et al. 2024, "Refusal … Mediated by a Single Direction"). Instruct models refuse ~25–30% where base models refuse ~7%.
- Our SFT is **completion-only**: the assistant turn is *always* a faithful EN translation, *never* a refusal. Every NSFW training row is a direct counter-example to the refusal behavior on exactly the input distribution that triggers it. This is the most reliable known suppressant — we are not jailbreaking at inference, we are removing the behavior at the weights.
- Literature corroborates direction: ~1,000 in-distribution counter-examples materially move refusal behavior ("LLMs Can Unlearn Refusal with Only 1,000 Benign Samples"). Our NSFW share will be thousands of rows.

**Base-vs-instruct question:** Qwen3-VL ships **Instruct / Thinking only** — there is no released *base* (pre-instruct) VLM checkpoint to sidestep the refusal. So we **use Instruct and suppress via SFT**. Do **not** use the *Thinking* variant — its CoT both leaks into output (seen in the hi-res A/B) and gives the safety policy a reasoning surface to refuse on; Instruct is the cleaner target.

**Concrete suppression recipe:**
1. Make NSFW rows a **first-class, always-present slice** of the SFT mix (not a tail) so the gradient sees refusal-triggering inputs every few steps — but **cap the share** (see §3 guardrail; the v12 36%-oversample regressed register into euphemism).
2. Keep the **system/instruction prompt identical** between train and serve (the byte-exact contract; a mismatch is the documented ~95% collapse risk). Train the model to translate *given this exact prompt*, so the prompt itself stops being a refusal cue.
3. Optionally seed ~200–500 **explicit "former-refusal" rows**: inputs that the OOTB model refused (mine them by running Qwen3-VL-8B over the NSFW eval set and capturing refusals), paired with the correct human/teacher EN. Highest-value counter-examples.

**Verification — refusal-rate probe (a HARD GATE):**
- Build `backend/scripts/eval/refusal_probe.jsonl`: ~150–300 explicit NSFW bubbles/pages drawn from the Ikenie/Furube gold + a refusal-prone slice of the corpus (run OOTB Qwen3-VL over the NSFW eval set; keep the ones it refused).
- Metric: `refusal_rate = fraction of outputs matching a refusal classifier` (regex/keyword first pass: `i can('?t| not)|i'?m (sorry|unable)|inappropriate|as an ai|cannot (assist|help|translate)`; then a small LLM-judge confirm pass to avoid false positives on legitimately apologetic dialogue).
- **Gate: post-FT refusal_rate == 0** on this probe. Measure OOTB Qwen3-VL on the same probe first to quantify the starting refusal rate (expected double digits) — that delta is the proof the FT worked.

---

## 3. DATA RECIPE

### 3.1 Training-row schema (PBP-VIS-NUM, adapted for fine-tuning)

One row = **one page**. Following Lippmann PBP-VIS-NUM (the best-scoring visual method, chrF 36.8):

```
user turn:
  [IMAGE block]  = the page image with bubble text REDACTED and each target
                   bubble overlaid with its number (1..N) at the bubble centroid.
  [TEXT block]   = byte-exact serve prompt:
                   <V11/Qwen page instruction>
                   Page (numbered JP lines, reading order):
                   1. <jp_ocr line 1>
                   2. <jp_ocr line 2>
                   ...
                   N. <jp_ocr line N>
assistant turn:
  1. <en line 1>
  2. <en line 2>
  ...
  N. <en line N>
```

Key fidelity points (from the COLING ablation + our train/serve contract):
- **Numbers drawn ON the image**, co-indexed with the numbered JP text list — this is the mechanism that lets the VLM bind a bubble's *visual* gender/speaker cue to a *specific* line. Redacting the JP glyphs (replace with the number) prevents the model from leaning on in-bubble OCR instead of the scene.
- **Page-level context ONLY.** Beyond-page context measurably HURTS (Lippmann) and matches our own page-context design. Do not add prev/next pages.
- **Reading-order** = the production column-major RTL order (`build_v11_dataset.manga_reading_order`) — reuse it so train==serve.
- Output as a **numbered list** so line↔bubble assignment is supervised (not a blob). The serve path already splits per-bubble.

**Adapt the existing builder:** `backend/scripts/data/v12vision/build_v12vision_poc.py` already emits `{image_path, jp_ocr, page_context, en_target, meta}` per page from the Ikenie gold. Two additions for the Qwen recipe:
- a **numbered-bubble image renderer** (redact bubble polygons from `bubbles.json` boxes, draw the index at each centroid) → writes a derived `*_numbered.webp` alongside the source;
- emit **numbered** `jp_ocr` and **numbered** `en_target` (the POC currently newline-joins). The page bboxes are already in `bubbles.json` / the POV testset's `page_bboxes`.

### 3.2 Dataset mix

| Source | Role | Path / status | Approx scale | Notes |
|---|---|---|---|---|
| **Manga109-s** | SCALE + clean in-domain JP+bbox | images `/mnt/nas/drive_2/manga-ml/datasets/manga109s/...`; JP+bbox parquet `backend/scripts/data/manga109/bubbles.parquet`; machine-EN `bubbles_translated_qe_deduped.parquet` | 114k bubble rows / ~10k pages | **License: images NOT redistributable, but ML *outputs* publishable w/ acknowledgement.** Train locally only; never ship the images. EN is machine (QE-filtered) — adequate for scale, not gold. |
| **NSFW corpus** | register + **refusal suppression** | images `/home/danny/manga_corpus_staging/galleries` (327k local); pairs via `corpus_bitext/` pipeline + `manifest_pages_375k.jsonl` | scalable to ~tens of k pages (OCR-GPU-bound) | EN = Qwen3-VL-mined (machine). This is the ONLY NSFW visual signal — load-bearing for §2. |
| **Ikenie/Furube human-gold** | in-domain GOLD, the eval-adjacent supervision | `backend/scripts/eval/data/ikenie{4,5}/gold_q3.jsonl`, `.../furube/gold_furube_p{1,2,3}.jsonl`; POC already built (220 pages) | ~220 pages / ~1.3k gold bubbles | Highest-quality target. Upweight ×3 (v11fix-style). **Hold Furube out of TRAIN — it's an eval set.** Use Ikenie for train, Furube for eval (or split). |
| **v11 page-context parquet (TEXT, image-absent)** | preserve the register/fluency backbone | `backend/training/runs/manga-bubbles/data_v11fix8_pagecontext.parquet` (302k rows) | subsample ~50–100k | Keeps the text translation quality the 4B already has; prevents the VLM SFT from forgetting register. **Mixed in the SAME SFT** (see collator note). |

**Mixing the image-absent text rows in the same SFT (collator requirement):**
The v11 text rows have **no image**. Qwen's processor / `UnslothVisionDataCollator` must accept rows where the image content block is absent and pad them as text-only in the same batch. Concretely: build conversations where text rows have `content = [{"type":"text", ...}]` (no image block) and image rows have `[{"type":"image",...},{"type":"text",...}]`. **Verify on the box (`--inspect`) that the collator handles mixed image/no-image batches** — if it chokes, the fallback is to (a) bucket batches so a batch is all-image or all-text, or (b) two-phase: text-SFT warm-up → image-SFT. Keeping them in one SFT is preferred (single train==serve register).

**Mix ratios (starting point, tune on POC):**
- ~40% Manga109 (scale, clean JP, machine EN)
- ~25% NSFW corpus (register + refusal suppression)
- ~10% Ikenie gold ×3 (in-domain gold)
- ~25% v11 text rows (register backbone, image-absent)
- **NSFW share guardrail: keep total NSFW ≤ ~18%** of rows. The **v12 36%-oversample REGRESSED** the model into euphemism/coy register (`feedback_v12_nsfw_oversampling_regression`). 36% is the documented danger zone; v11/v11fix held ~16%. Audit NSFW pairs for euphemism before inclusion.

### 3.3 Distillation option (frontier-VLM teacher labels)

To get **gold-grade EN with correct POV at scale** (the machine EN is the quality ceiling), label ~10k pages with a frontier VLM teacher (e.g. Claude or Gemini vision) reading the actual page → numbered EN.
- **Cost:** ~$65–215 for ~10k pages (per roadmap estimate; varies by teacher/price).
- **Why it matters:** the teacher *sees the image*, so its EN already resolves the POV the student needs to learn — this is the single highest-leverage data upgrade for the pronoun axis.
- **ToS CAVEAT (decision-forcing):** the corpus is **NSFW**, and most frontier VLM ToS prohibit explicit-content processing AND prohibit using outputs to train competing models. → **Use the frontier teacher ONLY on the SFW Manga109 slice** (clean, redistributable outputs, no ToS conflict) to lift the *scale* slice's EN quality and POV; use the **self-hosted Qwen3-VL-8B** (already on box, no ToS issue) as the teacher for the **NSFW** slice. Two teachers, partitioned by content rating. Defer distillation to Phase 3 — it's an amplifier, not needed to test the core hypothesis.

---

## 4. TRAINING CONFIG

Base: `Qwen/Qwen3-VL-8B-Instruct` via unsloth `FastVisionModel`. Port the v12vision trainer (`sft_gemma4_e4b_v12vision.py`) — its LoRA-section audit-guard and dry-run/inspect modes transfer directly; swap the model class + module-name expectations.

| Knob | Value | Rationale |
|---|---|---|
| **LoRA rank / alpha** | r16 / α32 (start) | proven v10/v11 recipe; drop to r8 if vision-tower loss is unstable in the dry-run |
| **Dropout / bias** | 0.0 / none | matches text recipe |
| **LR** | **1e-4** | vision towers are more LR-sensitive than text; the v12vision config already chose 1e-4 (text used 2e-4). Bump to 2e-4 only if dry-run loss is healthy |
| **Towers adapted** | **language + vision** (attn + MLP). **+ projector/merger if it exposes nn.Linear** | the whole point is conditioning the vision path. Qwen3-VL's vision merger IS Linear (unlike Gemma's Parameter-only projector), so LoRA likely attaches → adapt it too. **AUDIT-GUARD: bail if vision_tower LoRA count == 0** (the scaffold already does this) |
| **Vision soft-token budget** | cap image to ~**1024–1280 px** long side → bound to a few hundred image tokens (Qwen3-VL rounds patches to mult-of-32) | bounds seq len + VRAM + latency. The hi-res A/B (1810px) helped little; don't pay for native res. Tune `max_pixels` in the processor |
| **max_seq_len** | **2048** (raise to 4096 only if page p99 overflows) | image tokens + numbered JP list + context + numbered EN target |
| **Batch / accum** | per-device **1–2**, grad-accum to **eff-batch 16** | images are heavy; halve batch vs text, double accum. Tune down if OOM |
| **Epochs** | **1** | translation SFT plateaus fast; 1 epoch is the proven recipe |
| **Optim / precision** | adamw_8bit, bf16 LoRA (QLoRA-4bit fallback if >32GB) | |
| **completion_only_loss** | true | mask the prompt; loss only on the numbered EN |
| **packing** | false | never pack multimodal samples |

**Box-hardening flags (MANDATORY — from `reference_gemma4_training_quirks`):**
- `train.in_training_eval: false` (sm_120 in-training-eval build SIGSEGVs; real eval is the disjoint harness). **Set per-config AND on the box clone** (box edits don't propagate to local).
- Env: `TOKENIZERS_PARALLELISM=false`, jemalloc preload, `MALLOC_ARENA_MAX=2` (heap-corruption guard), `TORCH_CUDA_ARCH_LIST=12.0`.
- **Do NOT stack serve+train+retrain** back-to-back for hours — the box 5090 thermally faulted (Xid hang needing reboot) under sustained multi-hour load. Let the card cool; power-limit; **add `resume_from_checkpoint`** to the trainer FIRST (the SFT script lacks it — a mid-run GPU fault = full restart otherwise). Launch via `setsid nohup` for ssh durability.
- **Dry-run first** (`--inspect` then `--dry-run`, ~4 min) — catches env/API breaks (unsloth↔transformers version pins) before a multi-hour run. Verify the unsloth vision-collator schema + module names on the actual box build (the scaffold flags every `?? VERIFY` line).

**Merge + serve:**
- Merge with unsloth **`save_pretrained_merged(save_method="merged_16bit")`** — keeps the vision tower + merger folded in. The language-only clean-merge does NOT cover the vision branch. **No k_norm refold needed** (that's a Gemma artifact; Qwen has no tied-k_norm dedup issue).
- Serve via vLLM Qwen3-VL multimodal: `--limit-mm-per-prompt image=1`, **drop MTP** (Qwen3-VL has no Gemma MTP drafter anyway), `--enforce-eager` to start (disable vision graph capture while debugging). Feed the **page image as a shared prefix** per page (N bubble queries reuse one image → prefix-cache hit keeps latency in the 2–4s/page budget). Adapt `serve_v12vision_vllm.sh` (currently a Gemma stub) to the Qwen model dir.
- **Quantization: ship bf16.** INT4 **erased the fine corrective signal** on the v11fix corrective models (`reference_gemma4_vllm_quant`) — don't quantize a discourse-corrective model. 8B bf16 ≈ 16GB weights + KV fits the 32GB serve card.

**Rough wall-clock / VRAM (5090 32GB):**
- LoRA train VRAM: ~24–28GB bf16 (QLoRA-4bit drops to ~14–18GB if needed).
- Wall-clock: text recipe was ~5h/290k rows; vision is ~1.5–2× heavier per step → estimate **~8–12h for a ~100k-row mix**. **De-risk with the POC first** (220 pages ≈ 1–2 GPU-hours).
- Serve VRAM: ~18–22GB (weights + KV + image activations), fits with headroom; watch co-tenancy with the videonest job.

---

## 5. EVAL & GATING (multi-signal, go/no-go)

Reuse the existing harnesses; **chrF is a deflated, contaminated FLOOR** (human ref is ALL-CAPS/fragmented; 6.5% of gold `en` has leftover JP; IoU joins cross-match boxes) — use it only as a no-regression tripwire, never as the headline.

| Signal | Harness | GATE |
|---|---|---|
| **POV correctness (the headline)** | `.bench/pov_ab/` — 44 hard swap bubbles, judge-scored, POV-correct-rate excl. neutrals | **MUST beat 46%** (Qwen3-VL text-only) **AND** fix ≥ some **he↔she gender inversions** (zero-shot image fixed none — the FT must). If FT-image ≤ 46% AND no gender fixes → image bet is dead for this base |
| **Refusal rate** | `refusal_probe.jsonl` (§2), classifier + LLM-judge | **== 0** (hard gate). Report OOTB rate as the delta-proof |
| **In-domain adequacy** | Furube human-gold (`gold_furube_p{1,2,3}`) + a **VLM page-adequacy judge** reading the actual page (robust to garbled OCR & noisy refs; doubles as a visual metric) | adequacy ≥ v11fix8 on the held-out Furube pages; no increase in silent-omission / hallucination rate |
| **No chrF regression** | `run_ikenie4_regression.sh` paired-bootstrap vs v11fix8 baseline | chrF++ Δ **not significantly negative** (CI95 may include 0; just must not regress). Re-OCR both sides with the **same AR-GPU OCR** (CPU-OCR confounds the probe gate — the documented v11fix8 lesson) |
| **Corrective probes** | reverse_sense / name_invention / number_romaji / sfx_meta_leak | no probe regresses (the strict v11fix gate) |

**Wire all of the above into one multi-signal panel** (the existing paired-bootstrap gate + the new VLM-adequacy judge + the refusal probe). A single chrF number would have washed out v11fix9 — the panel catches that. **Promotion requires: POV > 46% with ≥1 gender fix, refusal_rate=0, no chrF regression, no probe regression, adequacy ≥ baseline.**

---

## 6. PHASED PLAN

### Phase 0 — DONE
Zero-shot A/B established Qwen3-VL-8B text-only = 46% POV, image zero-shot uninformative, NSFW refusal confirmed. (`project_mt_finetuning_roadmap` 2026-06-30b.)

### Phase 1 — THE FIRST CONCRETE EXPERIMENT (cheapest decisive run)
**Goal:** answer the three program-defining questions in one cheap run.
1. Does a **fine-tuned** image model beat the 46% text-only POV ceiling (does the image bet live)?
2. Does corpus SFT drive **refusal_rate → 0**?
3. Can the box even **train + merge + serve** a Qwen3-VL-8B vision LoRA?

**Run:**
- Rebuild the 220-page Ikenie POC with **numbered-bubble images + numbered JP/EN** (adapt `build_v12vision_poc.py`); add ~2–3k NSFW corpus rows (already mineable via `corpus_bitext/`) + ~5–10k v11 text rows for register. Keep it small — a few-GPU-hour run.
- Port `sft_gemma4_e4b_v12vision.py` to Qwen3-VL-8B (`FastVisionModel`, swap module-name expectations, keep the audit-guard). **`--inspect` → `--dry-run` → 1-epoch LoRA.**
- Merge (`save_pretrained_merged`) → serve on vLLM → run the **44-case POV harness + refusal probe + a quick Ikenie chrF spot-check**.

**Effort:** ~1–2 days (most of it is the numbered-image renderer + the Qwen port + box verification). **GPU:** ~2–4 hours.
**Risk:** medium — unsloth/vLLM Qwen3-VL API drift on the box build; collator mixed-batch behavior; box thermal fault (mitigate: small run, cool between).
**KILL CRITERIA (Phase 1):**
- FT-image POV ≤ 46% **AND** zero gender inversions fixed → **image bet dead** → pivot to Fallback #2: ship text-only Qwen3-VL-8B SFT (bank the +29 model-size win) and stop the image program.
- refusal_rate stays > 0 after corpus SFT → escalate suppression (former-refusal rows, higher NSFW slice within the ≤18% cap); if still > 0 → Fallback #3 (Gemma base for the image experiment).
- Can't train/merge/serve Qwen3-VL on the box at all → Fallback #1 (Qwen2.5-VL-7B) or revert to the Gemma scaffold infra.

### Phase 2 — SCALE THE WINNING RECIPE
If Phase 1 clears the POV + refusal gates: build the full mix (§3.2, ~100k rows, Manga109 scale + NSFW + Ikenie gold ×3 + v11 text backbone), 1-epoch LoRA, full multi-signal gate (§5).
**Effort:** ~1 week (data build dominates — Manga109 numbered-image rendering + EN association). **GPU:** ~8–12h train.
**Risk:** medium — NSFW euphemism regression (mitigate: ≤18% cap + euphemism audit); register drift from machine-EN scale (mitigate: gold ×3 + text backbone).
**KILL:** full-gate fails to beat v11fix8 on the panel → keep v11fix8 in production; the image LoRA stays R&D.

### Phase 3 — DISTILLATION AMPLIFIER (optional, only if Phase 2 wins)
Frontier-VLM teacher labels ~10k SFW Manga109 pages (POV-correct EN), self-hosted Qwen for the NSFW slice (§3.3). Retrain. Targets the POV ceiling further.
**Effort:** ~1 week + ~$65–215. **Risk:** low-medium (ToS partitioning is the gating concern — already designed around it).
**KILL:** no adequacy/POV lift over Phase 2 → distillation not worth the cost; freeze Phase 2 as production.

---

## Appendix — key paths

- Trainer scaffold (port to Qwen): `backend/scripts/train/sft_gemma4_e4b_v12vision.py`
- Config: `backend/training/configs/gemma4_e4b_v12vision_sft.yaml`
- Data builder + POC: `backend/scripts/data/v12vision/{build_v12vision_poc.py,data_v12vision_poc.jsonl}` (220 pages, Ikenie4+5)
- NSFW bitext pipeline: `backend/scripts/data/corpus_bitext/` (JP=CTD+PARSeq, EN=Qwen3-VL); manifest `data/manga_datasets/merged/export/manifest_pages_375k.jsonl`; images `/home/danny/manga_corpus_staging/galleries`
- Manga109: images `/mnt/nas/drive_2/manga-ml/datasets/manga109s/...`; JP+bbox `backend/scripts/data/manga109/bubbles.parquet`; machine-EN `bubbles_translated_qe_deduped.parquet`
- Gold/eval: `backend/scripts/eval/data/{ikenie4,ikenie5}/gold_q3.jsonl`, `.../furube/gold_furube_p{1,2,3}.jsonl`
- POV harness: `backend/.bench/pov_ab/{testset.json,results.json,results_hires.json}` (44 cases)
- Regression: `backend/scripts/eval/run_ikenie4_regression.sh`
- Serve stub (port to Qwen): `backend/scripts/eval/serve_v12vision_vllm.sh`; text serve ref: `serve_v10it_vllm.sh`
- Box: `100.64.235.63` (RTX 5090, Qwen3-VL-8B served at `:8001`); local RTX 5090 32GB

## Appendix — external sources
- Lippmann et al., COLING 2025, "Context-Informed MT of Manga using Multimodal LLMs" — arXiv 2411.02589 / aclanthology.org/2025.coling-main.232 / github.com/PLippmann/multimodal-manga-translation (PBP-VIS-NUM: numbers ON redacted page image, page-level optimal, beyond-page hurts, chrF 36.8 GPT-4 Turbo zero-shot)
- Unsloth Qwen3-VL-8B vision fine-tuning notebook + docs (FastVisionModel, ~24GB LoRA, select vision/language/attn/mlp layers)
- vLLM Qwen3-VL-Dense serving support (vLLM model registry / vllm-ascend Qwen-VL-Dense tutorial)
- Qwen3-VL-8B-Instruct — huggingface.co/Qwen/Qwen3-VL-8B-Instruct (Apache-2.0)
- Arditi et al. 2024, "Refusal in LMs Is Mediated by a Single Direction" — arXiv 2406.11717; "LLMs Can Unlearn Refusal with Only 1,000 Benign Samples" — arXiv 2601.19231

---

## 7. PRE-MORTEM MITIGATIONS (2026-06-30, deep, before implementation)

Folded in before kickoff. Tigers #1/#2 + the uncensored-base change are Phase-1 fixes; #3/#4 are Phase-2 gates.

**#1 [TIGER, was invisible] Add a fine-tuned TEXT-ONLY control arm.** The Phase-1 gate "beat 46%" compares FT-image to ZERO-SHOT Qwen text — any fine-tune beats that, so it can't attribute a win to the image. FIX: Phase 1 trains TWO LoRAs on identical data (image-on and image-off) and reports **image value = FT-image − FT-text**, not FT-image − 46%. Promotion needs the image arm to beat the text arm (esp. on he↔she), not just beat zero-shot.

**#2 [TIGER] Kill Phase-1 eval leakage.** The 44-case POV harness = 7 ikenie4 + 37 Furube, and Phase 1 trains on Ikenie. FIX: score Phase-1 POV on the **37 Furube-only** held-out cases (report all-44 too, but the Furube-37 is the honest number); do NOT use the on-training-data Ikenie chrF as a signal.

**#3 [TIGER, Phase-2 gate] Machine-EN labels carry the POV errors we're fixing.** Manga109 (40%) + NSFW-corpus (25%) EN targets are machine-generated by POV-flawed models → training on them can reinforce wrong POV. Phase 1 dodges this (POC = Ikenie GOLD) so it can WIN then Phase 2 FAILS. FIX: treat machine-EN as REGISTER/SCALE supervision only; POV supervision comes from GOLD + distilled (Phase 3). Before Phase 2, either POV-filter the machine EN or upweight gold/distilled so the POV gradient isn't dominated by machine labels. Do NOT scale on unfiltered machine EN and expect POV to improve.

**#4 [TIGER, Phase-2 gate] Box thermal fault + no resume.** The box 5090 Xid-hung mid-run during v11fix8 training; the SFT script lacks `resume_from_checkpoint`. FIX: BUILD resume support before the 8–12h Phase-2 run (Phase-1's 2–4h is short enough to risk without it); power-limit + cool between runs; `setsid nohup`.

**UNCENSORED BASE (user direction, 2026-06-30):** switch the base to an **abliterated Qwen3-VL** — kills the refusal tiger (§2) at the weights instead of relying on SFT suppression. Primary candidate: `huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated` (30B-A3B MoE, uncensored, higher capacity than the 8B at ~3B active). CAVEATS to validate FIRST (in progress): (a) a 30B MoE must be QUANTIZED (FP8/AWQ ~16–30GB) to serve on the 32GB 5090 — bf16 ≈60GB won't fit; (b) QLoRA-training a 30B MoE on ONE 32GB card is at the edge — it may be TEACHER-ONLY with the **8B-dense abliterated as the trainable student**; (c) abliteration can dent capability + community license/provenance varies. VALIDATION EXPERIMENT (running): serve the abliterated model, re-run the 44-case POV A/B (all-44 + Furube-37) + a refusal count. GO only if refusals≈0 AND POV ≥ the censored 8B's 46%. If the 30B won't serve/train on the box → fall back to `Huihui-Qwen3-VL-8B-*abliterated` (fits + QLoRA-trainable + the drop-in uncensored twin of the measured-46% base).

**ELEPHANT [latency]:** the VLM swap is ~2–4s/page vs the current ~1.4s (a browser extension). No SLA is stated. Before Phase-2 promotion, confirm the latency is acceptable to the product (image-as-shared-prefix per page + low soft-token budget are the levers).

**Revised Phase-1 order:** (0) validate the abliterated base [running] → (1) build the numbered-image POC on Ikenie GOLD → (2) train image-on AND image-off LoRAs → (3) score on Furube-37 POV + refusal probe → image value = image−text arm, refusals=0. KILL: image arm ≤ text arm on POV with no gender fixes → ship the text-only abliterated SFT (still banks the +29 model-size win) and stop the image program.
