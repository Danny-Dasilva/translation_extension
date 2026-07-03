# POV / Model-Compute Options for the Gendered-Pronoun Ceiling

**Date:** 2026-07-03
**Status:** RESEARCH / PROPOSAL ONLY — no code, no launches. Feasibility on **owned hardware
(2× RTX 5090, 32GB)** is the deciding factor; cloud is a costed fallback.
**Author:** research agent (feat/ship-textsft)
**Owns:** this file.

Question: the gendered-pronoun / POV ceiling (he↔she, speaker↔addressee) in the JP→EN manga
translator looks like a **capacity** problem. 4B→8B gave +29 POV points (text-only); the shipped
8B (v1) still resolves only ~13–20% of hard gendered cases. The obvious next step — 30B-A3B — is
**BLOCKED**: bitsandbytes NF4 cannot quantize Qwen3-VL-MoE's fused-expert `nn.Parameter`s, so it
needs an 80GB GPU ([[project_v2_30b_a3b_readiness]], commit 3a0ad57). This doc costs the
alternatives.

---

## 0. TL;DR — the finding that reframes the whole question

**The prior audit's blocker is dodgeable.** Synthesis §11 rejected "Qwen3-VL-32B-dense" on the
grounds that *no abliterated variant existed*. **That is no longer true (re-checked 2026-07-03):**
huihui-ai now ships **both** a dense-32B Instruct *and* a dense-32B Thinking abliterated Qwen3-VL:

- `huihui-ai/Huihui-Qwen3-VL-32B-Instruct-abliterated` (33B, dense, bf16, ~66GB) [HF]
- `huihui-ai/Huihui-Qwen3-VL-32B-Thinking-abliterated` (33B, dense, bf16, reasoning) [HF]
- Refusal: "tops the UGI Leaderboard with a perfect W10 score… first model to not refuse at all"
  (text abliterated; **image path NOT abliterated** — a risk to measure, see §1). [HF]

Because it is **dense** (all `nn.Linear`), it **does not hit the fused-MoE-expert quantization
blocker** that killed the 30B-A3B — bnb NF4 swaps its Linears to `Linear4bit` exactly as it does
for the working 8B. And a **32B DENSE model has ~32B ACTIVE params per token vs the 30B-A3B's ~3B
active** — for a capacity/reasoning-bound problem like POV, that is the *whole point*. The measured
30B-A3B zero-shot (15.2% all-148) barely beat the 8B (13.3%) precisely because its active capacity
is 3B-class. Dense-32B is both **more feasible on owned hardware** and **more likely to move POV**.

**Ranked verdict (POV benefit × feasibility on 32GB-owned):**

1. **Qwen3-VL-32B-DENSE-abliterated + text-SFT (v1 recipe) + serve-with-image** — the 30B plan
   retargeted onto a base that actually QLoRA-trains on the 5090 and carries more active capacity.
2. **Bounded / distilled reasoning step** (short gender-attribution prefix, or CoT-distilled SFT)
   on the 8B or the 32B base — strongest *direct* evidence for this exact failure mode, but full
   open "thinking" is latency-incompatible with a browser extension, so it must be capped.
3. **Strong dense TEXT model + OCR + a VLM-derived gender/speaker tag (hybrid)** — cheaper, easier
   to quantize/serve, but forfeits the image-only POV cases; a fallback if VLM serve-quant (G4)
   washes the gain.
4. **Agentic translate→critique→fix at inference** — near-zero training cost, but self-bias +
   2–3× latency; unreliable when the critic is the same 8B that erred.
5. **Cloud 30B-A3B QLoRA (80GB)** — now *dominated* by #1: weaker active capacity, not servable
   on owned hardware (needs perpetual 80GB or the same AWQ wash risk), poor ROI.

---

## 1. Option A (NEW, TOP) — Qwen3-VL-32B-**dense**-abliterated + text-SFT + serve-with-image

The v1 recipe (text-only SFT, vision frozen, served WITH the page image as a shared per-page
prefix) that gave **48.0% POV @ N=148** on the 8B ([[project_mt_finetuning_roadmap]] UPDATE
2026-07-01b), transferred onto a dense-32B abliterated base — the same bet as the 30B plan, but on
a base that trains on the owned card.

- **Expected POV benefit — HIGH.** 4B→8B was +29pt; the ceiling is capacity-bound and ~67% of POV
  errors had the referent *in text* (a pure capacity case) (roadmap UPDATE 2026-06-30). Dense-32B
  is the first jump that actually **quadruples active capacity** over the 8B (vs the 30B-A3B which
  adds almost none). Zero-shot dense-32B should clear the 30B-A3B's 15.2%; +SFT (proven +5–7pt on
  the 8B) + serve-with-image (+11–16pt measured) stack on top. This is the single best-evidenced
  capacity lever available.
- **Feasibility — TRAINS on 32GB-local (tight); cheap cloud insurance.** Dense ⇒ **no MoE
  blocker** (✓ high-confidence: the 3a0ad57 root-cause is that *only* fused `nn.Parameter` experts
  fail to quantize; a dense model has none — bnb NF4 handles its Linears exactly as it does the
  working 8B). 33B in NF4 ≈ ~17–18GB weights + bounded VL image tokens (`max_pixels` cap) + LoRA +
  8-bit-adam ⇒ realistic peak **~22–28GB at bs1/seq2048/grad-ckpt** — fits the 32GB 5090 with
  headroom the 30B-A3B never had. Unsloth already publishes `unsloth/Qwen2.5-VL-32B-Instruct-
  unsloth-bnb-4bit`, i.e. a supported 32B-VL QLoRA path. `? INFERRED` on exact peak — gate with the
  **same G0 GPU load probe** used for the 30B, which *should pass here* (unlike the 30B). If it
  runs hot on the box or spikes over 32GB, a **48GB L40S / A100-40GB rental (~$0.7–1.2/hr, ~$5–10
  per run)** clears it comfortably.
- **Cost / latency.** Train: owned-box hours (or ~$5–10 cloud). **Serve latency ≈ 8B class**
  (dense-32B AWQ single-stream is in the same ballpark as the 30B-A3B's measured ~185 tok/s;
  serve-with-image prefix is nearly free per [[project_serve_with_image_latency]]). No extra
  inference passes, no thinking tokens — extension-compatible.
- **Biggest risk — the SAME G4 serve-quant wash, unchanged.** bf16 33B (~66GB) will not fit the
  32GB 5090, so serving **must** be AWQ/W4A16 (~18–20GB + FP8 KV, fits). Per
  [[reference_gemma4_vllm_quant]], 4-bit serve-quant *erased* the corrective SFT signal on Gemma
  (RTN and GPTQ both regressed below bf16). Dense AWQ is far better-trodden than the exotic MoE
  quant and *may* preserve signal better — **but it is the #1 go/no-go and must be certed** (AWQ
  vs its own bf16 reference on the POV slice). Secondary risk: **image path is not abliterated** —
  when served WITH an NSFW page image the base could refuse; the 8B huihui is also text-only
  abliterated and serves at 0 refusals with the image prefix, so precedent is reassuring, but
  measure it (refusal_eval with image on).

**Why this beats the shelved 30B-A3B directly:** same serve constraint (AWQ), but (a) it *trains
on owned hardware* instead of needing a rented 80GB host, and (b) 32B dense active capacity >> 3B
active — more of the exact thing the POV ceiling needs.

Sources: [Huihui-Qwen3-VL-32B-Instruct-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-32B-Instruct-abliterated),
[Huihui-Qwen3-VL-32B-Thinking-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-32B-Thinking-abliterated),
[unsloth Qwen2.5-VL-32B bnb-4bit VRAM discussion](https://huggingface.co/unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit/discussions/3),
[Unsloth requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements).

---

## 2. Option 1 — Reasoning / CoT for POV ("who speaks, their gender, from honorifics/context")

Two forms. Direct evidence that CoT helps the *exact* failure mode is real, but the naive form is
latency-incompatible with a browser extension.

### 1a. Reasoning as a bounded / distilled step (RECOMMENDED form)

- **Expected POV benefit — HIGH for the failure mode.** CoT that explicitly resolves subject/verb
  and pronoun antecedents before translating is the most on-target intervention in the literature:
  gender-bias CoT drove Telugu 80%→4% and Kannada 40%→0% (arXiv 2405.19701), and CoT improves
  *context-aware* translation with explicit pronoun-resolution steps (arXiv 2510.18077). Not
  JP-specific, but the mechanism (resolve pro-drop referent → commit gendered pronoun) is exactly
  our gap. Qwen3-VL ships native **Thinking** variants at 8B and 32B (dense) tuned for stepwise
  reasoning — so the base is available.
- **Feasibility — same as Option A** (the 32B-Thinking-abliterated is dense; the 8B-Thinking fits
  trivially). The design lever is *bounding* the reasoning, not the model.
- **Cost / latency — the crux.** Naive "thinking" is a non-starter for the extension: Thinking
  variants emit up to ~40k reasoning tokens, and reasoning MT *generates more* tokens for
  low-token-activation targets (arXiv 2605.07533) — a 5–40× latency blow-up per page.
  Speculative-CoT recovers only ~1.6–2.3× (arXiv 2504.19095), still multiples. **Mitigation:**
  (i) SFT-distill the reasoning into a *terse* rationale (LightThinker / CoT-compression, arXiv
  2502.15589), or (ii) train a **bounded gender/speaker prefix** — the model emits a ≤1-line tag
  ("speaker=female, addressee=male, cue=わたくし + skirt") then the translation — capping thinking
  to tens of tokens. Either keeps latency near baseline while capturing the reasoning gain.
- **Biggest risk.** Uncontrolled thinking length blows the latency budget; and CoT can *leak* into
  the output (already observed once — the "CoT leak" fixed in roadmap UPDATE 2026-06-30b). Must
  train the rationale to be masked/bounded and stripped at serve, and cert latency.

### 1b. Agentic translate→critique→fix at inference (see also §4/ranking #4)

- **Benefit — LOW–MEDIUM, uncertain.** TEaR / Self-Refine / Reflexion improve MT via a
  feedback→refine loop with no retraining. But **self-bias** dominates for open/weaker models: they
  amplify their own errors across iterations and self-correct unreliably (arXiv 2402.11436) — i.e.
  the same 8B that emitted the wrong gender is a poor judge of its own gender choice. Gains are
  reliable mainly with a *stronger* critic (GPT-4/Gemini class).
- **Feasibility / cost.** Zero training; but **2–3× inference latency** (translate + critique +
  refine passes) — heavy for a per-page extension. Best value if the critic uses a signal the
  translator lacked (the page image, or a gender tag) rather than pure self-reflection.
- **Biggest risk.** On our own 8B/32B, self-critique likely won't reliably flip he↔she and triples
  latency for little gain — negative ROI unless paired with an external/stronger critic.

Sources: [CoT gender-bias MT (arXiv 2405.19701)](https://arxiv.org/html/2405.19701),
[CoT context-aware translation (arXiv 2510.18077)](https://arxiv.org/html/2510.18077),
[reasoning-MT token dynamics (arXiv 2605.07533)](https://arxiv.org/pdf/2605.07533),
[Speculative CoT (arXiv 2504.19095)](https://arxiv.org/html/2504.19095v2),
[LightThinker compression (arXiv 2502.15589)](https://arxiv.org/pdf/2502.15589),
[self-bias in self-refinement (arXiv 2402.11436)](https://arxiv.org/pdf/2402.11436),
[Qwen3-VL Thinking variants](https://github.com/qwenlm/qwen3-vl).

---

## 3. Option 3 — Cloud path for the 30B-A3B (80GB A100/H100)

- **Expected POV benefit — LOW–MEDIUM.** The A3B is only **~3B active** per token; measured
  zero-shot 15.2% all-148 is barely above the 8B (13.3%). SFT *might* add +5–7pt if it transfers,
  but you are paying for total-param count you cannot use per token on a capacity-bound problem.
  Dense-32B (Option A) delivers the capacity the A3B does not.
- **Feasibility — train only in cloud; NOT servable on owned hardware.** QLoRA needs 80GB
  (~65–72GB peak, per [[project_v2_30b_a3b_readiness]]). Rental is cheap: **A100-80GB ~$1.19–1.99/hr
  (RunPod), H100 ~$1.87–2.69/hr; a ~6–12h run ≈ $10–25.** But **serving** bf16 30B needs an 80GB
  host too (~$860/mo continuous on an A100-80GB) — infeasible for an extension backend on owned
  cards — or you AWQ it down to fit the 5090 and eat the **same G4 wash risk** as Option A while
  keeping *less* active capacity.
- **Cost / latency.** Train ~$10–25 one-off; serve either $860+/mo (bf16 cloud) or AWQ-on-owned
  (free hw, wash risk). Decode ~185 tok/s (MoE fast).
- **Biggest risk.** You rent to train a 3B-active model you then can't serve on your own hardware.
  **Dominated by Option A** on every axis that matters. Renting is *worth it only* as a one-off to
  cheaply de-risk a bf16 reference (e.g. to cert Option A's AWQ against a bf16 gold), not as the v2
  base.

Sources: [H100 rental comparison 2026 (IntuitionLabs)](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison),
[GPU pricing 2026 (Spheron)](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/),
[Vast.ai A100 pricing](https://vast.ai/pricing/gpu/A100-SXM4).

---

## 4. Option 4 — Stronger dense TEXT model + OCR (+ rendered gender/context summary) vs a VLM

- **Expected POV benefit — MEDIUM, capped below the VLM+image path.** A larger dense text-reasoning
  model (e.g. `huihui-ai/Qwen3-32B-abliterated` text, or a 30–70B dense reasoner) fed OCR-JP + a
  synthesized gender/speaker/character summary recovers the **~67% "referent-in-text" capacity
  cases** — the same cases the 4B→8B jump fixed. **But it structurally cannot see the ~33% of POV
  cases whose gender cue is only in the image** (roadmap UPDATE 2026-06-30), and the MEASURED win
  was exactly the image: FT-text served **+image 48.0% vs text-only 37.2%** at N=148. A pure-text
  model forfeits that delta. To recover it you must *derive* the gender summary from something — if
  from a VLM, you've re-introduced the VLM (with extra hops/latency); if from text alone, you're
  capped at the text-visible cases.
- **Feasibility — EASIEST to scale/quantize/serve.** No vision tokens; dense text models AWQ/GPTQ
  cleanly and serve small (a 32B text AWQ ~18GB fits the 5090; QLoRA-trains like Option A). This is
  its one genuine advantage.
- **Cost / latency.** Cheapest to serve; low latency. A **hybrid** (cheap VLM vision pass emits a
  per-bubble gender/speaker tag → strong text model translates) captures image cues *and* keeps a
  quantizable text translator — but it is two models and re-derives most of what serve-with-image
  already does in one.
- **Biggest risk.** Net-below the VLM+image path on POV (loses the measured image lever); the
  hybrid's value hinges on the tagging VLM being accurate and cheap, which is just the VLM problem
  again. **Best role: the fallback if Option A's G4 serve-quant washes the VLM gain** — a
  quantization-friendly path that still beats today's 8B on the text-visible majority.

Sources: [Huihui Qwen3-32B-abliterated (text)](https://huggingface.co/huihui-ai/Qwen3-32B-abliterated),
plus the internal MEASURED N=148 image delta ([[project_mt_finetuning_roadmap]] UPDATE 2026-07-01b).

---

## 5. Ranked summary

| # | Option | Expected POV | Feasible on 32GB-owned? | Cost / latency | Biggest risk |
|---|---|---|---|---|---|
| **1** | **32B-DENSE-abliterated + text-SFT + serve-with-image** (Option A) | **HIGH** (first real capacity jump over 8B; +SFT +image stack) | **YES — trains on 5090** (dense ⇒ no MoE blocker; ~22–28GB peak; cloud 48GB ~$5–10 fallback) | Train owned/~$10; serve AWQ ~8B-class latency | **G4 AWQ wash** (bf16 66GB won't fit → must serve AWQ); image path un-abliterated |
| **2** | **Bounded/distilled reasoning step** (Thinking base or gender-prefix SFT) | **HIGH** for the exact failure mode | YES (8B trivial; 32B-Thinking = Option A cost) | **Latency is the crux** — must cap thinking (full CoT = 5–40× blow-up) | Uncontrolled thinking length / CoT leak into output |
| **3** | **Dense TEXT model + OCR + gender tag (hybrid)** | MEDIUM (capped; loses image-only cases) | **YES — easiest to quantize/serve** | Cheapest serve, low latency | Below VLM+image on POV; hybrid re-introduces a VLM |
| **4** | **Agentic translate→critique→fix at inference** | LOW–MED, uncertain | YES (no training) | **2–3× latency**, no train cost | Self-bias: our 8B is a poor judge of its own gender error |
| **5** | **Cloud 30B-A3B QLoRA (80GB)** | LOW–MED (~3B active) | **Train cloud-only; NOT servable on owned hw** | Train ~$10–25; serve $860/mo bf16 or AWQ-wash | Dominated by #1 on every axis; unservable on owned cards |

**Recommendation.** Pursue **Option A** as the v2 base: it is the 30B plan's intent (bigger
capacity + the proven text-SFT + serve-with-image recipe) redirected onto a base that (a) **actually
QLoRA-trains on the owned 5090** because it is dense, and (b) carries the **active capacity** the
30B-A3B lacked — the exact lever a capacity-bound POV ceiling needs. Fold **Option 2** in as a
*bounded* gender-attribution prefix in the SFT targets (cheap, on-target, latency-safe). Keep
**Option 4** as the quantization-friendly fallback if the G4 AWQ cert fails. Use **cloud (Option 3
infra, not the A3B model)** only as a one-off to produce a bf16 reference for the G4 cert. Shelve
the **30B-A3B** — it was never the capacity win it looked like, and the dense-32B dominates it.

**Cheapest first move (de-risk, ~1 day, no train):** zero-shot the **dense-32B-Instruct-abliterated**
(AWQ, via vLLM on the box) through the existing turnkey harness (`pov_probe` all-148 + Furube-37,
`refusal_eval` with image-on) exactly as the 30B-A3B was evaluated (synthesis §11). If dense-32B
zero-shot clears the 30B-A3B's 15.2% and holds 0 content refusals with the image on, Option A is
green to plan; the existing 30B training scaffold (`sft_qwen3vl_8b_imagectx.py`, the v2 mix,
`validate_30b_g0.py`) retargets to it with a one-line base swap.
