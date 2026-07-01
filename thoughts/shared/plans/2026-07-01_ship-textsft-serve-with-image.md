# SHIP Runbook — Text-Only SFT on Abliterated Qwen3-VL-8B, Served WITH the Page Image

**Date:** 2026-07-01
**Status:** SHIP (decision made; this runbook executes it — do NOT re-litigate the approach)
**Forks:** `thoughts/shared/plans/2026-06-30_image-context-vlm-finetune.md` (that plan scoped the KILLED image-TRAINING approach; this one keeps its §2 refusal / §3 data / §4 config / §5 gate / box-hardening and rewrites the training objective to TEXT-ONLY and the serve path to image-as-inference-prefix)
**Owner thread:** mt-finetuning-roadmap → Phase 2 (image-as-context), post-experiment

---

## THE DECISION (already made — this runbook only executes it)

> **Ship:** base = `huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated` (uncensored, 0 refusals OOTB, QLoRA-trainable, the POV-neutral drop-in twin of the measured-46%-POV Qwen3-VL-8B) + **TEXT-ONLY SFT** on a refusal-stripped corpus + **serve WITH the page image as a shared per-page prefix at INFERENCE ONLY**.
>
> **Image-context TRAINING is KILLED** — measured on the 148-case held-out POV set: FT-image+image **43.9%** < FT-text+image **48.0%**. Adapting the vision tower *hurt*. The page image still helps *at inference* on a text-trained model (text-SFT + page-image = **48% POV**, +16pt over the 32% text-only baseline), so we serve with it but do not train on it.
>
> **`v11fix8` bf16 (the current 4B Gemma prod translator) STAYS PRODUCTION** until this clears the full gate below.
>
> **Quantization (GPTQ-INT4) REJECTED** — it erased the corrective signal on the v11fix line. Serve bf16.

Everything below is **decision-forcing**: each section ends with the concrete choice and its kill criterion.

---

## 0. Evidence baseline (do NOT re-derive)

| Fact | Number | Source |
|---|---|---|
| Abliterated Qwen3-VL-8B, refusals OOTB | **0** | POV A/B, `.bench/pov_ab/results_ablit_8b.json` |
| FT-image arm + image, 148-case held-out POV | **43.9%** | `.bench/pov_ab/results_large_fton.json` |
| FT-text arm + image, 148-case held-out POV | **48.0%** | `.bench/pov_ab/results_large_ftoff.json` |
| Text-only serve (no image), same set | **~32%** (baseline) | `.bench/pov_ab/results_large_base.json` |
| Net: **image at INFERENCE on a text-trained model** | **+16pt POV** | fton vs base comparison |
| Net: **image at TRAINING** | **−4.1pt POV** (worse than text arm) | fton vs ftoff |
| FT reintroduced refusals (machine-EN NSFW targets mined by Qwen3-VL) | **~2%** | Phase-1 refusal count |
| Current 4B (v11fix8) text-only E2E pipeline | **≈ 1.4s** (already > 1s) | prod telemetry |

**Load-bearing reads of the evidence:**
1. The **4B→8B model-size upgrade is the validated win** (+29 POV text-only, measured). It carries the ship regardless of the image.
2. The **image helps only as an inference-time prefix, not as a training signal.** Training the vision tower degraded POV — do not adapt it.
3. The **refusal leak is a DATA-side artifact** (Qwen3-VL-mined machine-EN targets contain refusal strings), not a base-model behavior. Fix it in the corpus, not the weights.

---

## 1. BASE + OBJECTIVE DECISION

### Decision: `huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated`, **TEXT-ONLY SFT**, serve-with-image-prefix, bf16.

- **Base:** already wired into `backend/training/configs/qwen3vl_8b_imagectx_poc*.yaml` (`model.name_or_path`) and served on the box at `:8001` (alias `qwen3vl_ablit8b`). Abliteration removes the refusal direction at the weights → no NSFW refusal wall.
- **Objective = TEXT-ONLY:** set `finetune_vision_layers: false`. Train language + attention + MLP only. The 06-30 audit-guard *bails if `vision_tower==0` under a vision banner* — for the ship it must be inverted: **assert `vision_tower==0` and `language_model>0`** (a text-only LoRA under a text banner). Fix the guard in `sft_qwen3vl_8b_imagectx.py` (`~L212-L230, L424-L430`) before the run.
- **Serve bf16, no quant.** ~16GB weights + KV + image activations fits the 32GB 5090 with headroom.

**Kill criterion:** if the abliterated base cannot serve+train+merge on the box (Phase 0), fall to `Qwen2.5-VL-7B-Instruct`-abliterated (lighter, most mature vLLM path); if no uncensored VLM trains/serves at all, keep v11fix8 and stop.

---

## 2. #1 RISK — HARD LATENCY GATE: E2E pipeline MUST be < 1 second

**This is the load-bearing section. It can sink the ship on its own.** The current 4B text-only pipeline is already **≈1.4s** — over budget *before* any 8B swap. An 8B VLM with a page image at <1s E2E is aggressive and may be infeasible. Treat <1s as a **PROMOTION GATE equal in weight to quality.**

### 2.1 Honest per-stage E2E budget (single page, N≈6–12 bubbles)

Stages: **detect → OCR → inpaint → translate → typeset.** detect/OCR/inpaint are page-parallel-izable; **translate is the long pole** and the only stage the 8B swap touches.

| Stage | Impl | 4B pipeline (now) | 8B text-only serve | 8B + image prefix |
|---|---|---|---|---|
| detect | CTD ONNX @1280 (`detector_service.py`) | ~120–200ms | same | same |
| OCR | PARSeq/manga-ocr AR-GPU, batched (`manga_ocr_service.py`) | ~150–300ms | same | same |
| inpaint | LaMa (`routers/inpaint.py`, `total_ms`) | ~150–250ms | same | same |
| **translate** | vLLM chat (`vllm_openai_translation_service.py`) | **~500–700ms** | **~900–1300ms** | **~1500–2500ms** |
| typeset | client render | ~50–100ms | same | same |
| **E2E (stages serialized)** | | **≈1.4s** | **≈1.8–2.4s** | **≈2.5–4.0s** |
| **E2E (detect/OCR/inpaint overlapped, translate is long pole)** | | ~0.9–1.1s | ~1.3–1.7s | ~1.9–3.0s |

**Blunt finding:** even a **text-only 8B is > 1s** at the current pipeline structure. The `<1s` target pressures the *whole ship*, not just the image. Hitting <1s with **any** 8B likely requires overlapping detect/OCR/inpaint with each other AND shaving the translate long pole hard. Hitting <1s with **image-on** is the aggressive case and may be out of reach.

### 2.2 Latency levers (apply in order; measure after each)

1. **Image soft-token budget** — cap long side ~**1024px**, patches rounded to mult-of-32 (`max_pixels: 1048576` already in config) → a few hundred image tokens, not native-res. The 1810px hi-res A/B barely helped; do not pay for resolution.
2. **Shared-prefix KV-cache reuse** — the page image is ONE prefix shared across the page's N bubble queries. Image prefill is paid **once per page**, then N decodes hit the prefix cache. This is the single biggest image-latency lever; without it, image cost is paid N× and the ship is dead on latency.
3. **Request batching** — batch the N bubble queries of a page into one vLLM step (`--max-num-seqs` ≥ N); amortize scheduler + prefix overhead.
4. **`--enforce-eager` OFF once stable** — enable CUDA-graph capture after debugging; recovers the graph-capture overhead the stub disables by default.
5. **Draft / speculative decoding** — EN targets are short, high-overlap; a small drafter can cut decode wall-clock materially. (Qwen3-VL has no Gemma MTP drafter — would need a separate draft model.)
6. **Smaller image** — drop to ~768px if quality holds; re-measure POV.
7. **Drop the image entirely** — text-only serve of the 8B. Banks the +29 model-size POV win, pays zero image latency. This is lever-of-last-resort AND fallback rung (a).

### 2.3 FALLBACK LADDER if <1s is infeasible with serve-with-image

Take the highest rung that clears; each rung is a live ship outcome.

- **(a) Text-only serve of the abliterated-8B.** Drop the image; bank the +29 POV model-size win, no image latency. Still likely ~1.3–1.7s → may itself miss <1s; if so, needs stage-overlap + decode-shaving to squeak under. **Fastest 8B option.**
- **(b) Keep the image, relax the target with product sign-off.** If image POV (+16pt) is worth 2–3s to the product, get an explicit SLA change in writing and ship serve-with-image at the measured latency. Requires product decision, not an engineering one.
- **(c) Reject the 8B upgrade on latency, keep v11fix8.** If neither (a) nor (b) is acceptable, the 4B stays production and the 8B stays R&D. This is a legitimate outcome — the quality win does not override a hard product latency constraint.

### 2.4 Decision + kill criterion

**Choice:** attempt serve-with-image, measure E2E on real pages, then take the highest fallback rung that meets the product's latency bar.
**KILL:** serve-with-image E2E cannot be brought under the product-accepted bar AND product will not relax the SLA AND text-only 8B also misses the bar → **rung (c): keep v11fix8.** `<1s` (or the product-signed-off number) is a **hard promotion gate** — a model that wins every quality signal but blows latency does NOT ship.

---

## 3. REFUSAL-STRIP METHOD (open question — concrete proposal)

**Problem:** the abliterated base has 0 refusals, but FT reintroduced ~2% because the machine-EN NSFW targets were **mined by Qwen3-VL (which itself refuses)** → refusal strings leaked into the SFT **targets**. The fix is on the **target side of the training data**, not the weights.

**Method — refusal classifier over the EN TARGET side, DROP matching rows:**
1. **Regex first pass** over every `en_target`:
   `i can('?t| not)|i'?m (sorry|unable)|inappropriate|as an ai|cannot (assist|help|translate)|i (won'?t|will not) (translate|provide)|not able to`
2. **Small LLM-judge confirm pass** on regex hits (avoid dropping legitimately apologetic in-story dialogue, e.g. a character saying "I'm sorry"). Judge prompt: "Is this a TRANSLATION output or an AI ASSISTANT REFUSAL?" → drop only confirmed refusals.
3. **DROP** confirmed-refusal rows from the training set (do not attempt to repair — a repaired target is unverified).
4. Log drop count + rate per source (Manga109 / NSFW-corpus / gold). A high NSFW drop rate is expected and is the signal the leak was real.

**Where:** a new filter stage in `backend/scripts/data/corpus_bitext/` (target-side, runs after EN mining, before row formatting in `format_rows.py`), applied to every machine-EN source. Gold rows are human EN → exempt but still regex-scanned as a tripwire.

**HARD GATE — `refusal_probe`:** post-FT `refusal_rate == 0` on `backend/scripts/eval/refusal_probe.jsonl` (~150–300 explicit NSFW bubbles/pages; the classifier from steps 1–2 scores outputs). Report the OOTB abliterated rate (≈0) and the pre-strip FT rate (≈2%) as the delta-proof the strip worked.

**Decision + kill:** target-side strip + probe gate. **KILL:** if `refusal_rate > 0` persists after stripping (i.e. the base itself refuses, not just leaked targets) → escalate to a former-refusal counter-example slice; if still >0 → the abliteration is incomplete, fall to a different uncensored base or rung (c).

---

## 4. DATA RECIPE (TEXT-ONLY — drop the numbered-image rows)

**Fork of 06-30 §3, with the PBP-VIS-NUM image format REMOVED.** No numbered-bubble redacted images, no image blocks in training rows. Every training row is text→text: `{JP page context + numbered JP lines} → {numbered EN lines}`. Reuse `data_poc_imageoff.jsonl` schema (text-only, byte-identical text to the image-on set with the image block removed) as the row format; scale it.

| Source | Role | Path / status | ~Scale | Notes |
|---|---|---|---|---|
| **v11 page-context parquet (TEXT)** | register/fluency backbone | `backend/training/runs/manga-bubbles/data_v11fix8_pagecontext.parquet` (302k) | subsample ~80–120k | The proven text backbone; keeps register the 4B already has. |
| **Manga109-s (TEXT)** | scale + clean in-domain JP | `backend/scripts/data/manga109/bubbles.parquet` + `bubbles_translated_qe_deduped.parquet` | ~40–80k | Machine EN = register/scale supervision ONLY (carries POV noise — do not treat as POV gold). Train locally; never ship images. |
| **NSFW corpus (TEXT)** | register + the content prod must translate | `corpus_bitext/` pairs, `manifest_pages_375k.jsonl` | ~tens of k, **≤18% of mix** | **Refusal-stripped (§3).** v12 36%-oversample regressed into euphemism — hold ≤18%, audit for euphemism. |
| **Ikenie human-gold (TEXT)** | in-domain GOLD, POV supervision | `backend/scripts/eval/data/ikenie{4,5}/gold_q3.jsonl` | ~gold, **×3 upweight** | The real POV signal. **Hold Furube OUT of train — it is the eval set.** |

**Mix (starting point, tune on POC):** ~45% v11 text backbone · ~25% Manga109 · ~15% NSFW (stripped, capped) · ~15% Ikenie gold ×3.
**Format contract:** byte-exact serve prompt == train prompt (the documented ~95%-collapse risk on mismatch). Numbered JP list in production column-major RTL reading order (`build_v11_dataset.manga_reading_order`); numbered EN target list (per-bubble supervised, not a blob).

**Note on train/serve asymmetry (deliberate):** we train text-only but serve with an image prefix. This is a controlled asymmetry — the image is *added context at inference*, and the measured result (48% > 32%) proves the text-trained model uses it. Do NOT "fix" the asymmetry by adding image rows to training — that is the killed approach.

**Decision + kill:** text-only mix, gold ×3 for POV, NSFW ≤18% stripped. **KILL:** if the text-SFT does not preserve v11fix8 register (chrF regression at gate §7) → machine-EN scale is diluting register; cut Manga109 share, raise gold/backbone weight, retrain.

---

## 5. TRAINING CONFIG (text SFT)

Fork `backend/training/configs/qwen3vl_8b_imagectx_poc_imageoff.yaml` → `qwen3vl_8b_textsft_ship.yaml`. Trainer: `backend/scripts/train/sft_qwen3vl_8b_imagectx.py` (already has dry-run/inspect, LoRA-section audit, `resume_from_checkpoint`).

| Knob | Value | Rationale |
|---|---|---|
| base | `huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated` | already in config |
| **finetune_vision_layers** | **false** | image-TRAINING is killed; text-only objective |
| finetune_language_layers | true | the POV/register lever |
| finetune_attention/mlp | true | q/k/v/o + gate/up/down |
| LoRA r / α | 16 / 32 | proven v10/v11 recipe |
| LR | **2e-4** | text tower (not vision) — use the text recipe's 2e-4, not the vision 1e-4 |
| max_seq_len | 2048 | numbered JP list + page context + numbered EN (no image tokens in training) |
| batch / accum | per-device 2 / accum 8 → eff 16 | text rows are lighter than the image POC's 1×16 |
| epochs | 1 | translation SFT plateaus fast |
| completion_only_loss | true | mask prompt; loss on numbered EN only |
| packing | false | keep line↔bubble alignment |
| in_training_eval | **false** | sm_120 in-training-eval SIGSEGVs |
| resume_from_checkpoint | true (auto) | box thermal-fault insurance |

**Audit-guard flip:** with `finetune_vision_layers: false`, invert the guard to **assert `vision_tower==0` AND `language_model>0`** (text LoRA under a text banner) — currently it bails when vision==0.

**BOX-HARDENING (MANDATORY — box = `100.64.235.63`, RTX 5090 sm_120):**
- Env: `TOKENIZERS_PARALLELISM=false`, jemalloc `LD_PRELOAD`, `MALLOC_ARENA_MAX=2`, `TORCH_CUDA_ARCH_LIST=12.0`.
- `in_training_eval: false` **per-config AND on the box clone** (box edits don't propagate local).
- Launch via `setsid nohup` (ssh durability). `resume_from_checkpoint` wired **before** any long run.
- **Do NOT stack serve+train+retrain back-to-back** — the 5090 thermally faulted (Xid hang) under sustained load. Cool/power-limit between runs.
- Dry-run first: `--inspect` (module-name + LoRA-section dump) → `--dry-run` (~4 min, catches unsloth↔transformers pin breaks) → full run.
- Merge: unsloth `save_pretrained_merged(save_method="merged_16bit")`. No k_norm refold (Gemma artifact; Qwen has none). Serve bf16.

**Rough VRAM / wall-clock (5090 32GB):** text LoRA ~18–24GB bf16 (QLoRA-4bit → ~12–16GB fallback). Text SFT ~5h for ~290k rows → **~3–5h for a ~150k-row mix**. Serve ~16–20GB.

**Decision + kill:** text SFT, vision frozen, 1 epoch. **KILL:** dry-run reveals the audit-guard/collator can't produce a clean text-only LoRA on this base build → fix the guard/collator or fall to Qwen2.5-VL.

---

## 6. SERVE-WITH-IMAGE (page image as a shared per-page inference prefix)

**This is the rewritten serve section (06-30 served for a vision-trained model; here the model is text-trained but we feed the image at inference).**

- Adapt `backend/scripts/eval/serve_v12vision_vllm.sh` (currently a Gemma stub) → Qwen3-VL merged dir. Verify `--limit-mm-per-prompt image=1` syntax on the box vLLM build; `--enforce-eager` to start.
- **Per-page shared prefix:** construct the page prompt as `[IMAGE block (≤1024px)] + [numbered JP page]` ONCE; issue the N bubble queries against it so the image prefill is cached and reused (§2.2 lever 2). This is what keeps image latency bounded.
- The extension calls into `vllm_openai_translation_service.py` — add an optional image-prefix path (page image passed alongside the JP page); the text-only path stays as the fallback rung (a).
- **Serve bf16.** `--max-num-seqs ≥ N`, drop MTP (none for Qwen3-VL).

**Decision + kill:** serve-with-image via cached page prefix. **KILL:** prefix caching does not actually reuse the image across bubble queries on the box vLLM build (image paid N×) → image latency is uncontrollable → fall to text-only serve rung (a).

---

## 7. EVAL GATE (multi-signal go/no-go — ALL must pass to promote)

| Signal | Harness | GATE |
|---|---|---|
| **POV (headline)** | `.bench/pov_ab/` on the **Furube-37 HELD-OUT** cases (`testset_large.json` Furube subset) | **≥ 48%** (the FT-text-arm serve-with-image mark) with ≥1 he↔she fix; report all-148 too |
| **Refusal** | `refusal_probe.jsonl` (§3), classifier + LLM-judge | **== 0** (hard) |
| **No chrF regression** | `run_ikenie4_regression.sh` paired-bootstrap vs v11fix8 | chrF++ Δ not significantly negative. **Re-OCR BOTH sides with the SAME AR-GPU OCR** (CPU-OCR confounds the gate — documented v11fix8 lesson) |
| **In-domain adequacy** | Furube gold `gold_furube_p{1,2,3}.jsonl` + VLM page-adequacy judge reading the actual page | ≥ v11fix8; no rise in silent-omission / hallucination |
| **Corrective probes** | reverse_sense / name_invention / number_romaji / sfx_meta_leak | no probe regresses |
| **LATENCY (§2)** | E2E measured on real pages | **< 1s** (or product-signed-off number) — hard gate |

**Wire all six into one panel** (paired-bootstrap gate + VLM-adequacy judge + refusal probe + latency meter). A single chrF number washed out v11fix9 — the panel is what caught that.
**PROMOTE iff:** POV ≥ 48% (Furube-37) with ≥1 gender fix **AND** refusal_rate==0 **AND** no chrF regression **AND** adequacy ≥ v11fix8 **AND** no probe regression **AND** latency gate met.

**Decision + kill:** six-signal panel, all-pass to promote. **KILL:** any hard signal fails and cannot be recovered within the fallback ladder → keep v11fix8 production.

---

## 8. PHASED PLAN (each phase ends with its kill criterion)

### Phase 0 — Validate the abliterated-8B serves + trains + merges on the box
Serve the abliterated base at `:8001` (done — POV A/B ran). Confirm a **text-only** LoRA dry-run (`--inspect`→`--dry-run`) attaches to language+attn+mlp only (vision_tower==0, guard flipped) and merges cleanly.
**VRAM/wall-clock:** trivial (dry-run ~4 min).
**KILL:** cannot dry-run/merge a text-only LoRA on the abliterated base → Qwen2.5-VL-abliterated or rung (c).

### Phase 1 — Text-only SFT LoRA on the refusal-stripped set (1 epoch, dry-run gated)
Build the §4 text mix, run §3 target-side refusal strip, `--inspect`→`--dry-run`→1-epoch LoRA (§5 config), merge (`save_pretrained_merged`).
**VRAM/wall-clock:** ~18–24GB, **~3–5h** for ~150k rows. `setsid nohup`, `resume_from_checkpoint`, cool between runs.
**KILL:** refusal_rate stays >0 after strip → escalate (§3); train/merge fails on box → §5 kill.

### Phase 2 — Serve-with-image + full six-signal gate INCLUDING <1s latency
Serve merged model with the per-page image prefix (§6). Run the full §7 panel on Furube-37 + refusal probe + Ikenie regression + adequacy judge + **E2E latency on real pages**.
**VRAM/wall-clock:** serve ~16–20GB; eval ~1–2h.
**KILL (decision-forcing):**
- All six pass, latency < 1s → **PROMOTE serve-with-image.**
- Quality passes, latency fails → walk the §2.3 ladder: (a) text-only 8B if it clears latency, else (b) product SLA sign-off, else (c) **keep v11fix8.**
- Quality fails → **keep v11fix8**; the 8B stays R&D.

---

## Appendix — key paths (all absolute-resolvable from repo root `/home/danny/Documents/personal/extension`)

- Trainer (flip guard to text-only): `backend/scripts/train/sft_qwen3vl_8b_imagectx.py`
- Configs to fork: `backend/training/configs/qwen3vl_8b_imagectx_poc_imageoff.yaml` → `qwen3vl_8b_textsft_ship.yaml`
- Text-only row schema: `backend/scripts/data/v12vision/data_poc_imageoff.jsonl`
- Refusal strip target: new stage in `backend/scripts/data/corpus_bitext/` (after EN mine, before `format_rows.py`)
- Data sources: `backend/training/runs/manga-bubbles/data_v11fix8_pagecontext.parquet`, `backend/scripts/data/manga109/bubbles*.parquet`, `corpus_bitext/` + `manifest_pages_375k.jsonl`, `backend/scripts/eval/data/ikenie{4,5}/gold_q3.jsonl`
- Serve stub (port to Qwen): `backend/scripts/eval/serve_v12vision_vllm.sh`; text ref `serve_v10it_vllm.sh`
- Translation service (add image-prefix path): `backend/app/services/vllm_openai_translation_service.py`
- Pipeline stages: `detector_service.py`, `manga_ocr_service.py`, `routers/inpaint.py`
- Eval: `.bench/pov_ab/{testset_large.json,results_large_{base,ftoff,fton}.json}`, `run_ikenie4_regression.sh`, `gold_furube_p{1,2,3}.jsonl`, `refusal_probe.jsonl` (build)
- Box: `100.64.235.63` (RTX 5090 sm_120), abliterated Qwen3-VL-8B served `:8001`
