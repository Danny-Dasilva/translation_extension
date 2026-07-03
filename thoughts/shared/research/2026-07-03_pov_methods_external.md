# Research: What Actually Moves Pronoun / Gender / POV Correctness in MT (JP→EN, manga)
Generated: 2026-07-03
Scope: external literature 2023–2026. PROPOSAL only — no implementation.

## The core diagnosis (why your two cheap fixes failed)

Your he↔she inversions persist because **the disambiguating gender signal is genuinely
absent from the local input you condition on.** Japanese is pro-drop AND under-marks
referent gender, so:

- **Text-only POV-contrastive SFT regressed** — you taught the model to *prefer* a
  pronoun pattern without giving it any new *evidence* to decide he vs she. It optimizes
  a signal that isn't in the input → it guesses more confidently but not more correctly,
  and pays a fluency tax. This matches the literature: blind gender/preference tuning
  produces "ambiguous results… not well suited for sentences with multiple entities"
  (Savoldi et al. 2025; Vanmassenhove line of work).
- **Image-as-context LoRA gave +2.4pt but ZERO inversions fixed** — because the *panel
  image rarely encodes who the pronoun refers to* either. Visual context fixes deixis,
  layout, tone (Lippmann COLING-2025's actual wins) but a manga panel almost never
  disambiguates a third-person referent's gender. Context conditioning only helps when
  the antecedent's gender is *recoverable from the context you added* — the context-
  utilization literature is explicit that models only benefit from gold disambiguating
  context, and degrade/ignore when it's absent or noisy (arXiv 2402.01404; 2509.14031).

**Consequence for ranking:** the highest-ROI levers are the ones that *inject a gender
signal the input lacks* — an external character→gender cast map, and the JP source's own
register cues (pronoun forms / sentence-final particles / keigo). Everything that merely
reshapes context format or applies preference pressure without new evidence is predicted
(and empirically shown, by your own two failures) to underperform on he↔she specifically.

---

## Ranked candidates (by expected ROI for he↔she)

### #1 — External character→gender cast map, injected as entity-level control (GoE-style)
- **What it is:** Maintain a per-work `character → {gender, 1st-person pronoun, register}`
  table; at inference resolve the speaker/referent of each bubble and inject explicit
  gender for each named/anaphoric entity ("Speaker: Rin (female); referent: Kaito (male)").
- **Evidence it fixes pronouns:** "Fine-grained Gender Control in MT with LLMs" (GoE
  prompting, Naszadi et al., NAACL 2024 / arXiv 2407.15154) shows *entity-level* gender
  info given to an LLM reaches SOTA on **multi-entity** controlled translation — exactly
  the manga case (multiple characters per page). WMT24 speaker-listener test suite
  (Dawkins et al., arXiv 2411.06194) shows **meta-context about the characters
  significantly determines gender agreement of referents inside dialogue** — i.e. the map
  is the causal lever. Earlier, Stanovsky/Saunders "NMT doesn't translate gender
  coreference right unless you make it" (arXiv 2010.05332) shows explicit coref/gender
  supply is what flips it.
- **Applicability (8B VLM, LoRA, manga, 2×32GB):** Very high. Works **prompt-only** first
  (no training) to measure ceiling, then LoRA to internalize the tag format. The map can
  be built once per series (cheap: NER + a few human/LLM confirmations, or bootstrapped
  from the first-person pronouns/SFP in #2). This is the one method that gives the model
  information it currently does not have.
- **ROI: HIGHEST.** Directly attacks the missing-evidence root cause; cheap; composable.

### #2 — Exploit JP source register cues as explicit gender/role features
- **What it is:** Per bubble, extract the speaker's self-reference pronoun (俺/僕/私/あたし),
  sentence-final particles (わ/かしら vs ぜ/ぞ/だ), and keigo level; emit them as feature
  tags feeding both speaker-gender inference and the cast map in #1.
- **Evidence:** "Character-Aware English↔Japanese Translation" (Nagato & Matsuzaki, WMT
  2025, aclanthology 2025.wmt-1.10) is built on the premise that **personal pronouns,
  sentence-final particles and honorifics are the primary carriers of speaker gender /
  personality** in Japanese fictional dialogue, and shows speaker-embedding conditioning
  preserves them better than plain fine-tuning. Sociolinguistic grounding: ore/boku/ze/zo
  ≈ masculine; atashi/wa/kashira ≈ feminine (see Japanese-gendered-language studies,
  arXiv 2006.15935; Gender-differences-in-Japanese). Unlike the image, these cues **are
  in your OCR text already** — currently unused.
- **Applicability:** High and cheap — rule/lexicon extractor over OCR text, no GPU. Best
  disambiguates **the SPEAKER's** gender (I↔you and self-reference), which is a large slice
  of your POV confusion; combine with #1 for third-person referents.
- **ROI: HIGH.** Free signal already present in-source; directly targets speaker POV.

### #3 — Inference-time structured reasoning ("who speaks / their gender / referent" before translating)
- **What it is:** A short reasoning pre-pass that names the speaker, their gender, the
  referent and its gender, THEN translates — as an agentic step, NOT as CoT-SFT.
- **Evidence (positive):** "Chain-of-Thought Reasoning Improves Context-Aware Translation"
  (arXiv 2510.18077, Oct 2025) — on DiscEvalMT (pronominal anaphora + lexical cohesion),
  reasoning prompts reach ~90% discrimination / ~92 COMET; gains *concentrate on the
  discourse/pronoun cases*. R1-T1 (arXiv 2502.19735) shows RL-incentivized human-aligned
  CoT (incl. context-aware paraphrasing) improves MT broadly.
- **Evidence (critical caveat):** "LLM Reasoning for MT: Synthetic Data over Thinking
  Tokens" (arXiv 2510.11919) finds **distilling CoT into a 'thinking' MT model does NOT
  beat plain I/O fine-tuning** — the payoff is from *inference-time* reasoning and from
  teacher-refined targets, not from CoT-LoRA. So do this as a runtime reasoning/agent pass,
  don't bake a "think-then-translate" LoRA (that path is a known dead end and echoes your
  own regressed contrastive-SFT).
- **Applicability:** Medium-high. 2× latency per bubble; gate it to *hard* bubbles only
  (pronoun present + speaker/gender ambiguous). Works with the abliterated 8B as a
  self-prompt or a second small pass; feed it the #1 map + #2 features so the "reasoning"
  has evidence to reason over (reasoning without the map still can't invent gender).
- **ROI: MEDIUM-HIGH** as a *validator/router* on top of #1/#2; LOW as a standalone LoRA.

### #4 — Agentic translate → coreference/gender critic → fix
- **What it is:** Two-stage: draft translation, then a dedicated critic checks pronoun/POV
  consistency against the cast map and rewrites violations.
- **Evidence:** TransAgents (arXiv 2405.11804) multi-agent literary pipeline beats GPT-4
  single-call on book-length text; MAS-LitEval (arXiv 2506.14199) uses a **narrative-
  perspective agent + NER terminology-consistency agent**; SAMAS (arXiv 2602.19840) style
  fidelity. These show a targeted critic pass catches discourse/consistency errors a single
  forward pass misses.
- **Applicability:** Medium. Most valuable as a **page-level POV-consistency critic** (does
  this character's pronoun stay stable across bubbles?) rather than per-bubble. Cost scales
  with pages — gate to chapters where #3's router flags conflicts.
- **ROI: MEDIUM.** Strong for *consistency* (drift across a chapter), moderate for the
  first-instance he↔she call (that's #1/#2's job).

### #5 — Document/discourse-level context conditioning (format & unit tuning)
- **What it is:** Widen the translation unit beyond a single bubble; feed prior N bubbles /
  page transcript in a structured turn format.
- **Evidence:** Context-aware MT survey (arXiv 2506.07583); "You Are What You Train"
  (arXiv 2509.14031) — data composition of context matters a lot and wrong composition
  hurts; context-utilization measurement (arXiv 2402.01404) — models only exploit context
  that actually contains the disambiguator. Lippmann COLING-2025 already gave you the
  manga-specific version of this and it did NOT fix inversions.
- **Applicability:** Medium as *substrate* (needed so #1/#2/#3 have somewhere to live), LOW
  as a standalone fix — you've already demonstrated the null result. Key nuance: prefer a
  **structured transcript with speaker labels** over raw concatenation; unlabeled context
  can *add* ambiguity.
- **ROI: LOW standalone / ENABLING for the others.** Don't expect inversions to move from
  context format alone.

### #6 — Targeted preference optimization on JA-EN pronoun-contrastive pairs (DPO/CPO)
- **What it is:** Build minimal he↔she contrastive pairs and DPO the model toward correct
  pronoun.
- **Evidence:** ContraPro (ZurichNLP, En-De) and the **Nagata & Morishita 2020 JA-EN
  contrastive pronoun benchmark** give the eval + a pair-construction recipe; gender-bias
  MT work suggests DPO pairs *can* improve gender agreement (Savoldi et al. 2025).
- **Why ranked low here:** (a) your text-only POV-contrastive SFT already regressed;
  (b) your own memory notes CPO pitfalls and a v12 NSFW-DPO register regression — blind
  preference pressure has repeatedly hurt this model. DPO only helps if the *chosen* side
  differs because of evidence the model can see — so it must be **conditioned on the #1 map
  / #2 features**, otherwise it repeats the failed-contrastive outcome. Use the JA-EN
  contrastive set as **EVALUATION** first, training second.
- **ROI: LOW-MEDIUM,** and only *after* #1/#2 supply the signal; high regression risk.

### #7 — Coreference-aware architecture / mention-attention decoding
- **What it is:** Architectural coref signals — mention attention, ContraCAT-style coref
  templates, coref-guided decoding.
- **Evidence:** "Mention Attention for Pronoun Translation" (arXiv 2412.14829); ContraCAT
  (COLING 2020); contrastive-coref context-aware NMT (arXiv 2109.05712).
- **Applicability:** Low for your setup — these assume architecture access / encoder-decoder
  NMT, awkward to bolt onto an abliterated decoder-only VLM with LoRA. Ideas (feed coref
  clusters as tags) collapse into #1.
- **ROI: LOW** for a LoRA-on-Qwen3-VL workflow.

---

## Recommended stack for the he↔she failure (composed, cheapest-first)

1. **#2 register-feature extractor** (free, in-source) → bootstraps speaker gender and…
2. **#1 character→gender cast map**, injected as entity-level tags — run **prompt-only
   first** to measure the ceiling before any LoRA. This is the load-bearing fix.
3. **#3 reasoning router** gating a **#4 POV-consistency critic** only on flagged bubbles/
   chapters (latency-bounded).
4. Use the **Nagata & Morishita JA-EN contrastive pronoun set (#6)** as the *evaluation
   harness* for all of the above; defer DPO training until #1/#2 are in and only train
   pairs whose chosen side is justified by injected evidence.
5. Treat **context/document format (#5)** as the container, with **speaker-labelled
   transcripts**, not as an expected fix.

**One-line thesis:** stop trying to make the model *infer* gender it can't see; **supply**
it — from the JP register cues it's ignoring (#2) and an external cast map (#1) — then use
reasoning/critic passes only to *apply and enforce* that supplied signal.

---

## Sources
- [Fine-grained Gender Control in MT with LLMs (GoE) — arXiv 2407.15154 / NAACL 2024](https://arxiv.org/abs/2407.15154)
- [WMT24 Test Suite: Gender Resolution in Speaker-Listener Dialogue Roles — arXiv 2411.06194](https://arxiv.org/abs/2411.06194)
- [NMT Doesn't Translate Gender Coreference Right Unless You Make It — arXiv 2010.05332](https://arxiv.org/pdf/2010.05332)
- [Character-Aware English↔Japanese Translation (speaker embeddings, honorifics/SFP/pronouns) — WMT 2025](https://aclanthology.org/2025.wmt-1.10/)
- [Is Japanese gendered language used on Twitter? (register→gender cues) — arXiv 2006.15935](https://arxiv.org/pdf/2006.15935)
- [Gender differences in Japanese (SFP/pronoun gender lexicon) — Wikipedia](https://en.wikipedia.org/wiki/Gender_differences_in_Japanese)
- [Chain-of-Thought Reasoning Improves Context-Aware Translation — arXiv 2510.18077](https://arxiv.org/abs/2510.18077)
- [LLM Reasoning for MT: Synthetic Data over Thinking Tokens (CoT-SFT null result) — arXiv 2510.11919](https://arxiv.org/pdf/2510.11919)
- [R1-T1: Incentivizing Translation via Reasoning Learning — arXiv 2502.19735](https://arxiv.org/abs/2502.19735)
- [TransAgents: Multi-Agent Literary Translation — arXiv 2405.11804](https://arxiv.org/html/2405.11804v1)
- [MAS-LitEval: Multi-Agent Literary Translation QA (narrative-perspective + NER agents) — arXiv 2506.14199](https://arxiv.org/html/2506.14199)
- [SAMAS: Spectrum-Guided Multi-Agent Style Fidelity — arXiv 2602.19840](https://arxiv.org/pdf/2602.19840)
- [Context-Aware MT with LLMs: A Survey — arXiv 2506.07583](https://arxiv.org/pdf/2506.07583)
- [On Measuring Context Utilization in Document-Level MT — arXiv 2402.01404](https://arxiv.org/pdf/2402.01404)
- [You Are What You Train: Data Composition for Context-Aware MT — arXiv 2509.14031](https://arxiv.org/pdf/2509.14031)
- [A Decade of Gender Bias in MT (Savoldi et al., Patterns 2025)](https://www.sciencedirect.com/science/article/pii/S2666389925001059)
- [ContraPro: Contrastive Pronoun Evaluation (En-De) — ZurichNLP](https://github.com/ZurichNLP/ContraPro)
- [Mention Attention for Pronoun Translation — arXiv 2412.14829](https://arxiv.org/pdf/2412.14829)
- [ContraCAT: Contrastive Coreference Analytical Templates — COLING 2020](https://aclanthology.org/2020.coling-main.417/)
- [Context-Informed MT of Manga using Multimodal LLMs (Lippmann) — COLING 2025](https://aclanthology.org/2025.coling-main.232.pdf)

## Open questions
- Nagata & Morishita (2020) JA-EN contrastive pronoun set: confirm exact coverage of
  he/she vs I/you and license (referenced across the literature; not fetched directly here).
- No paper found that specifically DPO-tunes on JA-EN *manga* pronoun pairs post-Lippmann —
  appears to be a genuine gap (opportunity, but validate as eval before training).
