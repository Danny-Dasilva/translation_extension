# v2 Training Plan — 30B-A3B (Qwen3-VL MoE, abliterated)

**Date:** 2026-07-02
**Status:** FUNDED experiment. Planning only — no code changes, no launches (box hardware
check still pending, §4).
**Author:** planning agent (feat/ship-textsft)
**Owns:** this file.

Candidate: `huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated` (MoE, ~3B active / ~30B
total). The v2 bet: transfer the v1 text-SFT recipe (which added +5–7pt POV to the dense 8B —
`project_mt_finetuning_roadmap` UPDATE 2026-07-01/01b) onto the larger MoE base, which already
posts the **best untuned** all-148 gendered POV (15.2% img-on, +7.6pt image delta) and table-best
all-rows 39.9% zero-shot (synthesis §11, MEASURED 2026-07-02).

---

## 0. TL;DR feasibility verdict

**QLoRA-on-a-quantized-MoE, as literally phrased, is NOT the path — but a supported adjacent
path exists and the stack has the MoE machinery.**

- You **cannot** QLoRA the cached **AWQ** checkpoint (`JinRiYao2001/…-AWQ`). AWQ/W4A16 is a
  *serve-only* packing; the training stack cannot attach trainable LoRA to AWQ-packed weights,
  and `autoawq` is **not installed** in the training venv (verified — `pip` reports NOT
  INSTALLED). The AWQ artifact on the box is for vLLM serving, not training.
- The **supported** path is standard **unsloth QLoRA (bitsandbytes NF4)** on the **bf16 base**
  `huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated` with `model.load_in_4bit: true`.
  The installed stack supports this arch:
  - transformers **5.12.0** ships the `qwen3_vl_moe` architecture
    (`transformers/models/qwen3_vl_moe/{configuration,modeling,modular}_qwen3_vl_moe.py` — ✓ VERIFIED present).
  - unsloth **2026.5.2** `models/vision.py` **explicitly lists `"qwen3_vl_moe"`** in its
    supported-arch list (line 247) and in `VLLM_SUPPORTED_VLM` (line 244–248) — ✓ VERIFIED.
  - unsloth has **MoE-aware LoRA**: `FastVisionModel.get_peft_model` auto-invokes
    `get_moe_target_parameters(model, target_modules)` (`vision.py:1416–1418`) which detects the
    MoE via `is_moe_model`, reads `num_experts`/`num_local_experts` (incl. from
    `config.text_config`), and populates PEFT `target_parameters` for the fused expert
    `nn.Parameter` weights (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`) —
    `_utils.py:3211`. This is exactly the machinery A3B needs (experts are `nn.Parameter`, not
    `nn.Linear`, so plain `target_modules` would miss them). ✓ VERIFIED.
- **Verdict: SUPPORTED IN PRINCIPLE, UNPROVEN HERE.** Every 8B run was dense; no MoE / no
  30B / no `qwen3_vl_moe` has ever been trained in this repo. The synthesis §11 "unproven here"
  flag stands — but it is unproven-not-unsupported. The de-risking work is a preflight dry-run
  (G0 below), not a research port.
- **Serve implication (the hard constraint):** you QLoRA on bf16 → merge to bf16 (~60GB, does
  **not** fit the 32GB 5090) → **must re-quantize to AWQ/W4A16 to serve**. Per
  `reference_gemma4_vllm_quant`, 4-bit serve-quant **erased the fine corrective SFT signal** on
  the Gemma corrective models (v11fix7/8 regressed below bf16 under both RTN and calibrated
  GPTQ). Different arch, but this is the **#1 go/no-go risk**: if AWQ serve-quant washes out the
  POV-contrastive SFT gain, the 30B is **not servable on current hardware** at the quality it
  was tuned for (bf16 30B won't fit the 5090). Gate G4 exists specifically to catch this.

---

## 1. Feasibility of QLoRA on a quantized MoE (detail)

### 1.1 What "QLoRA on quantized MoE" actually means here

Two different "quantized" things get conflated:

| Format | Role | Trainable in our stack? |
|---|---|---|
| **bnb NF4** (`load_in_4bit=True`) | unsloth QLoRA base packing | **YES** — this *is* QLoRA |
| **AWQ / W4A16** (cached `…-AWQ`) | vLLM serve packing | **NO** — serve-only, autoawq absent |

QLoRA = quantize the base to NF4 on load, freeze it, train LoRA adapters in bf16 on top. That is
supported for this MoE. Training *on the AWQ file* is not a thing this stack does.

### 1.2 The trainer we reuse

`backend/scripts/train/sft_qwen3vl_8b_imagectx.py` (the v1 text-SFT trainer) is the right harness
— it is arch-generic via `FastVisionModel`:
- Loads with `FastVisionModel.from_pretrained(base, dtype=bf16, load_in_4bit=cfg[...])`
  (`:347–353`).
- Attaches LoRA via `FastVisionModel.get_peft_model(...)` (`:403–415`) — which is where the MoE
  expert-parameter auto-detection fires. **For v1 (dense 8B) this path returned no MoE params;
  for the 30B-A3B it must return the expert `nn.Parameter` paths.** This is the single most
  important thing to assert in the dry-run (G0): that `get_moe_target_parameters` returns a
  non-empty list, i.e. the experts are actually being adapted and not silently skipped.

### 1.3 The masking landmine (already fixed in v1 — must re-verify on 30B)

Per `reference_qwen3vl_completion_only_masking`: the YAML `completion_only_loss` key is **INERT**
on this trainer (trl's masking bypassed by `skip_prepare_dataset=True` + `_is_vlm`); masking is
100% owned by `UnslothVisionDataCollator`, which defaults `train_on_responses_only=False` → bare
run = **full-sequence loss** (~85% of gradient on the JP prompt, invisible in the loss curve).

**Good news: the v1 trainer already wires this correctly** (`:606–625`, ✓ VERIFIED):

```python
data_collator = UnslothVisionDataCollator(
    model, processor,
    train_on_responses_only=_completion_only,
    instruction_part="<|im_start|>user\n",     # Qwen3-VL ChatML
    response_part="<|im_start|>assistant\n")
# runtime probe (built-in, hard assert):
assert _masked > 0.4 and _kept > 0
```

The ChatML delimiters are identical for Qwen3-VL-30B-A3B (same `<|im_start|>{role}\n…<|im_end|>`
template family), so no change is expected — **but the probe MUST be confirmed to fire and pass
on the 30B tokenizer** before the full run (v11fix8 precedent: 95.4% masked, 41 target tokens
kept). This is a required G0 pre-flight gate, not optional.

### 1.4 Version tension to resolve at G0

- This repo's local `.venv`: unsloth 2026.5.2 / transformers **5.12.0** / trl 0.23.0 / peft
  0.19.2.dev0 / bitsandbytes 0.49.2 — **has** `qwen3_vl_moe`. (This is the serve/local venv.)
- The **box training venv** (`~/manga-translate-train/.venv-training`,
  `reference_gemma4_training_quirks` gotcha #3) is unsloth **2026.6.7 / transformers 5.5.0**.
  **transformers 5.5.0 predates `qwen3_vl_moe`** (it's a newer arch) → the box training venv
  **very likely needs a transformers/unsloth bump** before it can even load this base. Confirm
  `from transformers.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration` imports on the
  box venv; if not, upgrade to a transformers that has it (the local 5.12.0 works) with a
  matching unsloth (≥2026.5.2). Note the Gemma-4 caveat (5.12 breaks Gemma4 unsloth load) is
  **irrelevant here** — we are not loading Gemma. ? INFERRED (box venv not directly inspected
  this session) — verify on box at G0.

---

## 2. Training data

Reuse the **v1 v13ship text-SFT mix** + fold in the **new POV-contrastive pairs**.

### 2.1 v1 mix (baseline, proven)

`scripts/data/v13ship/` → `data_v13ship_v1_messages.jsonl` (box: `/home/danny/ship_v1/…`).
Stats (`v13ship_v1_stats.json`, ✓ VERIFIED): **72,098 rows** =
- `v11_backbone` 63,503 (the v11 page-context backbone)
- `ikenie_gold` 7,098 (Ikenie gold ×3 upweight)
- `nsfw_corpus` 1,497
- Register mix: manga_dialog 28,679 / vn_eroge 9,808 / manga_nsfw 9,313 / novel 6,529 / vn 6,349
  / sfx 4,863 / manga 3,221 / synthetic 1,740 / garbage 1,596.
- **NSFW frac 26.5%** — above v1's ideal but **below the 36% danger zone** that caused the v12
  euphemism regression (`feedback_v12_nsfw_oversampling_regression`). Keep at/under this; do not
  oversample NSFW for the 30B.
- **Refusal-strip is built in** (`compose_training_mix.py` refusal classifier, self-test passes,
  0 dropped in v1). This is **mandatory** — SFT on machine-EN targets reintroduced 1–3 refusals
  on the 8B (`project_mt_finetuning_roadmap` UPDATE 2026-07-01b, catch #3). The abliterated base
  is 0-refusal; do not let the data undo that.

### 2.2 New POV-contrastive pairs (DEPENDENCY — parallel agent)

Synthesis §6.1: POV-contrastive pairs mined from the 375k-page corpus, pre-filtered by the
presence-scorer gender-marker logic, emitted through the **byte-identical builder** (precedent
`scripts/data/v11/build_voice_addressee_probe.py`). **This mine is being produced by a parallel
agent — it is a hard dependency for G1.** When delivered:
- Fold into the mix via `scripts/data/compose_training_mix.py` (synthesis §6.3 wants the v13ship
  spec ported here as a declarative weight-spec — do that so the 30B ablation is a one-YAML diff).
- Upweight modestly (Ikenie-gold precedent ×3; POV slice is the *targeted* lever for the 30B's
  measured Furube-37 0/15 gap). Keep total NSFW ≤ 26.5%.
- Run it through the refusal-strip and the byte-exact prompt shape (train==serve; format
  mismatch = the ~95% chrF collapse class, `feedback_chat_template_mismatch`).

**Do NOT** add bulk bitext or more DPO (`project_mt_finetuning_roadmap`: plateaus past ~10k;
DPO-for-register caused the euphemism regression). The lever is composition + capacity, not volume.

### 2.3 Image-context training — explicitly OUT of scope

The decisive N=148 result (`project_mt_finetuning_roadmap` UPDATE 2026-07-01b): image-context
*training* is **not worth funding** (FT-image+image 43.9% < FT-text+image 48.0%). v2 stays
**text-only SFT, vision frozen** (`finetune_vision_layers: false`), served **with** the page
image as a shared per-page prefix (the validated win). This keeps the 30B QLoRA cheap (no vision
LoRA, bounded image tokens) and matches the v1 recipe exactly.

---

## 3. The masking probe gate (required pre-flight, every launch)

Per `reference_qwen3vl_completion_only_masking`, gate **every** launch:

1. Build one batch through `UnslothVisionDataCollator`.
2. Assert `0.4 < (labels == -100).mean() < 1.0` **and** `(labels != -100).sum() > 0`.
3. Log the exact numbers (expect ~90–96% masked, tens of EN target tokens kept).

This is **already wired** into `sft_qwen3vl_8b_imagectx.py:612–625`. Additional 30B-specific
assertions to add for G0 (MoE-specific, not in the 8B path):

4. Assert `get_moe_target_parameters(model, target_modules)` returned a **non-empty** list (the
   experts are actually LoRA'd). Log the expert param paths.
5. Assert the LoRA param count is materially larger than the dense-8B run (experts add adapters).

A dry-run (`--dry-run`, 100 steps on 1%, ~4 min per the Gemma precedent) that passes 1–5 is the
G0 gate. The loss will start **higher** than a (broken) full-sequence run — that's the honest
EN-only loss, expected.

---

## 4. VRAM / box constraints & the hardware incident

### 4.1 VRAM (QLoRA, single RTX 5090 = 32GB)

- **All 30B params must be resident** — MoE's ~3B-active reduces *compute per token*, not memory.
  In bnb NF4: 30e9 × ~0.5 B ≈ **~15GB weights**.
- + gradient-checkpointing activations (unsloth "unsloth" mode, low), + LoRA adapters incl. expert
  adapters (small), + 8-bit adam optimizer state on trainable params only (small), + VL image
  activations **bounded** by `max_pixels: 1048576` (~1024px long side, few-hundred image tokens).
- Realistic peak at batch-1 / grad-accum-16: **~20–26GB**. **Fits 32GB but tight.** The v1 8B
  QLoRA peaked ~18GB; the 30B adds ~2× weights + expert adapters. Keep `per_device_batch_size: 1`,
  `max_seq_length: 2048`, image cap as-is. If OOM: drop max_pixels, or shard experts — but the
  first lever is simply ensuring the box GPU is **free** (below).

### 4.2 Box contention & serve exclusivity

- The box GPU is co-tenanted with a variable `videonest` job (7–30GB,
  `reference_gemma4_vllm_quant`). Training needs it **free** — a 30GB videonest spike + 26GB
  train = boot OOM.
- **Training and serving cannot overlap.** AWQ serve peaks **28.3GB at 0.90 util** (synthesis
  §11) — there is **no co-serve headroom**. You cannot serve v1 (or the 30B) on the box while
  training the 30B on the same card. Schedule exclusively, or serve v1 from the local 5090 during
  box training.

### 4.3 The hardware incident (HARD BLOCKER for on-box training)

Synthesis §7 + §7.1: on **2026-07-02 ~04:00** the box had **filesystem-level corruption** — 7
corrupted `.so` files across 5 unrelated packages — consistent with its **thermal-fault history
under sustained load**. **memtest + fsck are PENDING** (need interactive sudo).
`reference_gemma4_training_quirks` gotcha #4: the box RTX 5090 **faulted (Xid-class hang)** under
exactly the pattern we'd repeat here — sustained multi-hour SFT stacked with serve/retrain.

**A 30B QLoRA is longer and hotter than any 8B run to date.** Training on the box **before the
hardware check** risks: (a) a mid-run GPU hang (recoverable only by reboot — no passwordless sudo
on the box), and (b) a **silently corrupted checkpoint** written under load that looks healthy in
loss but tanks eval. **Mandatory before G2:** run the §7.1 runbook (dmesg/EDAC/MCE scan, `smartctl
-a /dev/nvme0` media_errors, schedule fsck, ≥1 full memtest86+ pass). Do **not** stack train +
serve + merge in one session. Checkpoint every 25% (`save_steps_pct: 0.25`, `resume_from_checkpoint`
supported in the v1 config) so a fault is resumable, not a restart. Hash-verify the merged
artifact after write (the venv hash-check pattern from §7).

**Cloud fallback** (recommended if the box hardware check slips): rent a single 40–48GB GPU
(A100-40GB / L40S-48GB) — 30B QLoRA fits comfortably with headroom, no thermal risk, no
contention. ~$1–2/hr.

---

## 5. Serve path

- **bf16 30B does not fit the 32GB 5090** (~60GB merged) → serve **must** be AWQ/W4A16
  (~16–19GB weights, **28.3GB peak** with FP8 KV at 0.90 util, synthesis §11). **No co-serve
  headroom** — the box can serve the 30B **or** run the OCR/inpaint backend, not comfortably both;
  plan the same Tailscale split as v1 (translate on box, detect/OCR/inpaint on local 5090,
  synthesis §9.1).
- The cached `…-AWQ` is the **base** quant — after SFT you must **re-quantize the merged model**
  (fresh AWQ/W4A16 of the tuned weights). Budget a quant pass + a cert.
- **RISK (from `reference_gemma4_vllm_quant`):** 4-bit serve-quant **erased the corrective SFT
  signal** on the Gemma corrective models — RTN and calibrated GPTQ both regressed v11fix7/8 below
  bf16 ("4-bit erases the fine corrective signal regardless of calibration faithfulness"). The MoE
  arch differs (no PLE embeddings), and AWQ ≠ GPTQ, so it may not repeat — but it **must be
  measured** (G4): cert the AWQ-served 30B against its **own** bf16 (rented-GPU) reference on the
  POV-contrastive slice. If AWQ washes the POV gain, the 30B bet does not close on current
  hardware.
- Merge gotchas to expect (from the 8B/Gemma precedents): unsloth `save_pretrained_merged` omits
  processor/tokenizer JSON → copy from base; VL feature-extractor load needs `processor_config.json`;
  quant may drop tied norm tensors → verify vLLM loads the merged+quant without
  "weights not initialized". Validate merge integrity behaviorally (canonical smoke translation
  byte-check, synthesis §9).

---

## 6. Go/No-Go decision points, cost & time

| Gate | Criterion | Est. effort | Kill / fallback |
|---|---|---|---|
| **G0 — preflight (de-risk the unproven bit)** | Box (or cloud) training venv imports `qwen3_vl_moe`; dry-run loads bf16 30B in NF4, `get_moe_target_parameters` returns non-empty experts, masking probe passes (0.4<masked<1, kept>0). | ~½ day | If unsloth can't LoRA the MoE experts here → **STOP**, escalate (upgrade unsloth or bf16-full-FT on rented 80GB — expensive). |
| **G1 — data ready** | POV-contrastive mine delivered (parallel agent), folded via `compose_training_mix.py`, refusal-strip verified, NSFW ≤26.5%, byte-exact prompt shape. | ~½ day after mine lands | Dependency slip → train on v1 mix alone as a capacity-only baseline first. |
| **G2 — train** | 1 epoch QLoRA, r16/α32, lr 2e-4, eff-batch 16, language-only + MoE experts, checkpoints @25%. **Box hardware cleared first** (§4.3). | **~6–12h walltime** (30B > 8B's 4–5h/290k; conservative for 72k+POV rows on 5090; MoE compute ≈ ~7B-forward but backprop touches all targeted experts) | GPU fault → resume from checkpoint. Corruption suspected → move to cloud. |
| **G3 — model cert** | Merge→bf16 serve (rented/box)→ turnkey cert: `pov_probe` all-148 + Furube-37, `refusal_eval`, `run_ikenie4_regression.sh` chrF. **Must:** beat zero-shot 30B (15.2% all-148 img-on) AND lift Furube-37 above 0/15, 0 hard refusals, chrF ≥ v1-8B. | ~½ day | No lift over zero-shot 30B → SFT didn't transfer → **STOP** (keep v1-8B). |
| **G4 — serve-quant cert** | AWQ/W4A16 quant of merged 30B; cert AWQ vs its own bf16 on POV slice + chrF + refusal. | ~½ day | AWQ erases POV gain (the §5 risk) → 30B not servable at tuned quality on 32GB → **fall back to v1-8B**, or provision a ≥48GB serve GPU for bf16. |

**Total wall time:** ~1 week (½d preflight + data dep + ~1d train + ~1d cert × 1–2 iterations).
**GPU cost:** trivial on the owned box; **cloud fallback ~$30–60** (2–3 runs at ~$10–20 each on a
40–48GB rental). The expensive resource is *box-hours during the hardware-uncertain window*, not
dollars — which is why the cloud fallback is cheap insurance.

**Overall go/no-go:** proceed to G0 immediately (cheap, de-risks the one genuinely unproven
thing). The **two kill points that matter** are **G0** (does unsloth actually adapt the MoE
experts here) and **G4** (does the POV-contrastive gain survive the mandatory serve-quant). If
both clear, 30B-A3B+SFT is a viable v2 base; if either fails, v1-8B remains the ship and the 30B
is shelved without much spend.

---

## 7. Open items / dependencies

- **[DEP]** POV-contrastive mine (parallel agent) — blocks G1.
- **[BLOCKER]** Box memtest + fsck (§4.3 / synthesis §7.1) — blocks on-box G2; not required if
  training on cloud.
- **[VERIFY]** Box training venv `qwen3_vl_moe` import + unsloth version (§1.4) — first G0 check.
- **[PORT]** v13ship spec → `compose_training_mix.py` declarative weight-spec (synthesis §6.3) —
  makes the 30B ablation a one-YAML diff; do at G1.
- **[NEW CONFIG]** `training/configs/qwen3vl_30b_a3b_textsft_v2.yaml` — fork of
  `qwen3vl_8b_textsft_v1.yaml`, swap `model.name_or_path` to the 30B, `load_in_4bit: true`, keep
  vision frozen, keep masking wiring. (Not written — planning only.)
</content>
</invoke>
