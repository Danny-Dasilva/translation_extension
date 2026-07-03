# Translation Pipeline Improvements — Historical Record & Decision Context

**Purpose:** onboarding + decision context for the JP→EN manga translation pipeline. Every
material improvement, why it was made, what it measured, and whether it shipped. Chronological
where it aids understanding.

**Scope:** the `backend/` app (detection → OCR → translation → inpaint/typeset → serve), the
translation-model training lineage, the data pipeline, and the eval infrastructure that gates it all.

**Evidence markers**
- ✓VERIFIED — read the file/commit/config directly this pass.
- ?INFERRED — reconstructed from memory notes / handoffs / commit messages, not re-verified against live code.
- Numbers cite the source. chrF deltas are chrF++ unless noted; CI95 = 95% paired-bootstrap CI; "cert PASS" = CI excludes 0 in the win direction AND no corrective probe regresses.

Primary sources traversed: `git log --all`; `backend/training/configs/*.yaml`;
`backend/scripts/eval/serve_v10it_vllm.sh`; `thoughts/shared/handoffs/mt-quality-ikenie4/*.yaml`;
`thoughts/shared/research/2026-07-02_pipeline-audit-synthesis.md`; and the session memory under
`~/.claude/projects/-home-danny-Documents-personal-extension/memory/*.md`.

---

## 1. Current production state (the v1-vs-v10it disconnect)

**What the app actually calls today** ✓VERIFIED (`backend/app/config.py`):
- `vllm_base_url = "http://127.0.0.1:8000/v1"` (config.py:99)
- `vllm_model_name = "v10it"` (config.py:100)

`"v10it"` is only the vLLM **served-model-name string**, not the weights. The script that serves
port 8000, `backend/scripts/eval/serve_v10it_vllm.sh`, has been repointed over time while keeping
`--served-model-name v10it` stable (script line 172) ✓VERIFIED. Its current default weights are
**Gemma-4 E4B v11fix8** (`.../gemma4_e4b_v11fix8_pagecontext/merged_fixed`, bf16), promoted
2026-06-29 (serve script lines 50-57) ✓VERIFIED.

So the production translator the extension hits = **Gemma-4 E4B + v11fix8 LoRA, merged, bf16,
served locally on :8000 with the Gemma-4 MTP drafter (γ=2)**.

**The disconnect.** A newer/better model, **v1** (abliterated Qwen3-VL-8B + text-SFT LoRA), was
trained, merged, and served on the **box at :8001** as `merged_v1`, and it certified **+10.104
chrF++ vs v11fix8** (CI95 [+7.74, +12.87], p=0.0000) plus 0/250 refusals (audit synthesis §9,
2026-07-02) ✓VERIFIED in the synthesis doc. But the app's `vllm_base_url`/`vllm_model_name` still
point at **local :8000 / "v10it" = Gemma v11fix8** ✓VERIFIED. **v1 is validated-but-not-promoted**:
to actually serve it the app would need to target the box endpoint (and the merged-weights serve
config `box_serve_v1_merged.sh`). ?INFERRED that no config change wiring the app to :8001 has
landed — the config default is unchanged.

**Serve-time flags that ARE default-on in prod today** ✓VERIFIED (config.py):
- `translation_v11_pagecontext = True` (:248) — page-context prompt shape (see §2).
- `translation_sentence_merge = True` (:276) — cross-bubble sentence merge.
- `translation_stream_events = True` (:496) — per-bubble WebSocket streaming.
- `translation_serve_image_context = True` (:409) — sends the page image as a shared prefix. NOTE: safe on a text-only served model because the text prompt is byte-identical whether the image is attached or not (guarded by `tests/unit/test_image_context_serve.py`); it only does real work once the served model can consume images (i.e. v1).
- `ocr_confidence_gate_enabled = True`, threshold `0.65` (:448-449) — but see §4 for the 0.80 long-text recalibration that lives in `ocr_confidence_gate.py`.
- `hybrid_ocr_enabled = True` (:464), `ocr_vertical_ar_default = True` (:481) — AR OCR routing.
- `translation_bubble_grouping = False` (:293) — the disabled pre-translation re-segmentation (see §4).
- `translation_cast_anchor = False` (:373) — cast register off by default (see §3).

---

## 2. Translation model lineage

Base recipe that most entries share (the "v11 recipe") ✓VERIFIED from configs + memory:
LoRA r16/α32, dropout 0, **language-model layers only** (vision/audio towers excluded → text-only
by construction), 1 epoch, LR 2e-4 cosine, effective batch 16, completion-only loss, byte-exact
train==serve prompt. Trained on the RTX 5090 (local, then increasingly the "box" 100.64.235.63).

### 2.1 Lineage table

| Model | Base | Date | What changed | Measured result | Status |
|---|---|---|---|---|---|
| v9c | Gemma-4 E4B-pt (llama.cpp Q8_0) | ~May | Early raw-template SFT; llama.cpp serve | 77.8 tok/s baseline; chrF++ ~70 on its own harness | Superseded |
| v10it | Gemma-4 E4B-**it** | 2026-05 | Clean language-only LoRA on `-it` for MTP-drafter compatibility (Path A) | Initially "broke" (chrF 20.9) — was a **chat-template eval bug**, not the model (§8) | Superseded; `"v10it"` name persists as the served-model string |
| v10.5 CPO | v10it + CPO/SimPO | 2026-05 | Preference tuning for register | **REGRESSED**: Gemma-EM −8.56pp, chrF++ −7.02, BLEU −12.37 | Shelved |
| **v11 page-context** | Gemma-4 E4B-it | 2026-06-15 | **Train/serve format fix**: trained on context-augmented single-line (full page as numbered context + "translate line k") to match serving | Fixed self-reference/coreference/comparative; INT4 quant chrF++ 32.21 > bf16 31.28 | Promoted (was prod) |
| v12 NSFW-oversample | Gemma-4 E4B-it | 2026-06-15 | Oversample NSFW/DPO to ~36% for explicit register | **REGRESSED**: NSFW chrF++ −0.64, dialogue −1.18, eroge-vocab 7/9→4/9, page-context −1.38 | Shelved |
| v11fix6 | Gemma-4 E4B-it | 2026-06-27 | Corrective SFT on 530 vision-gold clean-OCR pairs (×3) | chrF++ +1.167 CI95[+0.02,+2.28] **p=0.048 PASS** (box re-eval); all 5 probes up, 0 regress | Certified |
| v11fix7 | Gemma-4 E4B-it | 2026-06-28 | Add on-disk-but-untrained slices: voice/addressee probe, reverse-sense corrective, filled clean-OCR pairs, short-fragment discipline | (interim base for fix8) | Superseded by fix8 |
| **v11fix8** | Gemma-4 E4B-it | 2026-06-29 | fix7 + **2,337 corpus-mined JP→EN pairs** (×3) from the ehentai corpus | chrF++ **+3.001** CI95[+1.74,+4.34] **p=0.0000 PASS** (clean AR-OCR) vs v11fix6; 0 probe regress | **PROMOTED — current local prod** |
| v11fix9 | Gemma-4 E4B-it | 2026-06-29 | Data-clean only (manga109 dedup 132k→75k, '...' cap, fragment drop → 302k→227k) | chrF++ **−1.92** CI95[−3.93,+0.12] p=0.072 (wash-to-worse); name_invention 1.00→0.75 | **FAILED — not promoted** |
| v12vision (POC) | Gemma-4 E4B-it (vision LoRA) | scaffold | Extend LoRA to vision tower + feed page image | Prototype-ready; never a shipped Gemma path (moved to Qwen 8B) | Shelved |
| **v1** | huihui **Qwen3-VL-8B-abliterated** | 2026-07-01 | Model-size jump (4B→8B) + refusal-free base + text-SFT on refusal-stripped mix; served WITH page image | chrF++ **+10.104** CI95[+7.74,+12.87] p=0 vs v11fix8; POV +11pt; 0/250 refusals | **Validated-but-not-promoted (best model)** |
| v2 POV-contrastive | Qwen3-VL-8B-ablit | 2026-07-02 | +24.9k mined POV-contrastive rows | chrF++ **−3.08** vs v1 (p=0); POV did not move; +image collapsed to 0% | **FAILED — not promoted** |
| v2 image-context | Qwen3-VL-8B (2-arm) | 2026-07-03 | Image-on vs image-off control LoRA, PBP-VIS-NUM | image value +2.4pt POV but **0 he↔she fixes** (decisive criterion) | **FAILED — not promoted** |
| v2 30B-A3B | Qwen3-VL-30B-A3B-ablit | 2026-07-02 | Bigger MoE base via QLoRA | **BLOCKED** — bnb can't quantize fused-MoE experts on ≤48GB (§8) | Blocked |

### 2.2 Per-entry rationale

**v11 page-context — WHY (train/serve format fix).** ✓VERIFIED (`project_v11_pagecontext_model`).
v10it was SFT'd on **single-line** pairs (`Translate the following Japanese... Japanese: {jp}`),
but serving sent a **multi-line numbered page block** → off-distribution → silent quality collapse
(meaning inversion, coreference loss, self-reference errors). v11 retrains on
**context-augmented single-line** (full page as numbered context + "Translate line k: {jp}" → one
line) — the exact shape serving uses (`translation_v11_pagecontext`, N calls/page, prefix-cacheable).
This is the single most important architectural lesson (see §8: format sensitivity). v11 validated
fixes: お母さん→"Mommy" self-reference, より comparative direction, 姉ちゃん/母さん kept distinct,
ブラ→bra adult vocab.

**v12 NSFW-oversample — WHY it FAILED.** ✓VERIFIED (`feedback_v12_nsfw_oversampling_regression`).
Goal: boost explicit register by oversampling NSFW (vntl_chat/dpo, opensubtitles, vn_eroge) to
~36%. The model learned to **euphemize** instead (`オチンチン気持ちいい？` v11 "Does your cock feel
good?" → v12 "Does it feel good?"). The DPO/chat sources steer toward coyness. **Lesson: more
in-register data ≠ better register if the data itself is euphemistic; audit for euphemism first.**

**v11fix6/7/8 — WHY (targeted corrective SFT, same v11 recipe).** ✓VERIFIED (configs are
byte-identical to v11 except the DATA; header comment "Only the corrective DATA changes"). Driven
by the Ikenie-no-Haha-4 page-for-page GT audit (§7): a small, upweighted corrective set folded onto
the v11 backbone.
- **fix6**: 530 vision-gold clean-OCR pairs ×3 (0.54% of rows). Near-miss locally (p=0.052) →
  **certified on box re-eval p=0.048**, all 5 targeted probes improved, 0 regressions.
- **fix7**: added data that already existed on disk but had never been trained — voice/addressee
  direction, reverse-sense (締まる sense inversion), short-fragment discipline (terse JP → terse
  literal, to kill page-context fabrication on ≤6-char bubbles), and clean-OCR pairs whose empty
  `human_en` was finally filled from the new Qwen3-VL gold.
- **fix8**: fix7 + **2,337 JP→EN rows mined from the ehentai corpus** (JP via CTD+PARSeq, EN via
  Qwen3-VL-8B, Hungarian centroid alignment). This is the current local production model. **Lesson
  banked: degraded (CPU-non-AR) OCR during the eval regen confounded the probe gate — the first run
  showed a reverse_sense "regression" (0.43) that vanished (0.571) once re-run with the SAME AR OCR
  as the baseline. Always eval with matched OCR.**

**v11fix9 — WHY it FAILED (data cleaning is not the lever).** ✓VERIFIED (`project_v11fix9_dataclean_result`).
Tested whether cleaning the fix8 corpus helps. It didn't (−1.92 chrF, ns; name_invention regressed).
The aggressive full-manga109 dedup (−57k in-domain rows) likely removed useful repetition signal.
**Lesson: text-data cleaning alone does not improve quality; the real lever is capacity/modality,
not corpus hygiene.** (Corroborated by v2 POV-contrastive and v11fix9 both being text-only-data
changes that failed.)

**v1 (Qwen3-VL-8B-ablit + text-SFT) — WHY (model size is THE lever).** ✓VERIFIED across
`project_mt_finetuning_roadmap`, `feedback_image_context_poc_result`, audit §1.2. The Furube human
eval (§7) proved the dominant failure was speaker/referent resolution on JP pro-drop — a **discourse
ceiling** that a 4B model cannot cross. The 4B-deconfound experiment: on the Furube-37 gendered
slice, **both** Gemma-E4B (trained) and Qwen-4B (zero-shot) resolved **0** gendered cases; base-8B
jumped to 6.7–13.3% and v1 (8B SFT) to 13.3–20%. So the 4B→8B jump is the validated win. Two more
findings shaped v1:
  1. **Abliterated base required.** Stock Qwen3-VL-8B *refuses* explicit NSFW even with hardened
     prompts; the huihui abliterated variant gives 0 refusals and is QLoRA-trainable.
  2. **Serve-with-image, don't train-with-image.** On the enlarged N=148 POV set, the **control
     (text-trained) model served WITH the image at inference (48.0%)** beat the image-*trained*
     model (43.9%). The payoff is "Qwen3-VL can read the page" (a base-VLM property), NOT something
     image-context training taught. So the ship plan = **text-only SFT + serve-with-image**, skipping
     an expensive image-context training pipeline. Config: `qwen3vl_8b_textsft_v1.yaml`
     (`finetune_vision_layers: false`, v11 backbone ~65k + Ikenie gold ×3, no mined NSFW) ✓VERIFIED.
  3. **Refusal-strip the training targets.** The abliterated base has 0 refusals but SFT
     *reintroduced* 1–3 (machine-EN NSFW targets mined by Qwen3-VL leaked refusals) → the v13ship
     builder strips refusals from targets first (§3).

**v2 attempts — WHY all three FAILED / blocked.**
- **POV-contrastive SFT** ✓VERIFIED (`feedback_pov_contrastive_sft_regressed`): 24.9k mined POV rows
  regressed chrF −3.08 AND didn't move POV; +image collapsed to 0%. Applying preference pressure
  with no new gender evidence → the model guesses more confidently, not more correctly, + a fluency
  tax. Mined rows were wrong-distribution (clean named LN/VN prose vs nameless manga captions).
- **Image-context LoRA** ✓VERIFIED (`feedback_image_context_poc_result`): clean 2-arm control, image
  value +2.4pt but **0 he↔she fixes**. A manga panel rarely encodes a *third-person* referent's
  gender, so the image can't supply what's missing. The method is sound (vision LoRA attaches, 216
  modules); the signal isn't there.
- **30B-A3B** ✓VERIFIED (`project_v2_30b_a3b_readiness`, §8): architecturally blocked on owned GPUs.

**THE meta-finding: the POV ceiling was a measurement ghost.** ✓VERIFIED
(`feedback_pov_metric_is_broken`, handoff 2026-07-03). All three v2 experiments chased a failure
that the corrected two-axis probe (`pov_probe_v2.py`) proved **does not exist**: v1's genuine
he↔she inversion rate = **0/67** on the curated gender-obligatory subset. The old probe conflated
person-choice (pro-drop → "I/you") with gender, carried a 79%-"she" class prior, and counted
laughter/blob/misalignment artifacts. **Decision: bank v1, stop the POV chase, redirect to real
levers (balloon-fragmentation defect, cast manifest, visual fixes).** Lesson: when repeated well-run
experiments fail the same way, suspect the metric before the method.

---

## 3. Data-pipeline improvements

**v13ship refusal-strip SFT builder** ✓VERIFIED (config `qwen3vl_8b_textsft_v1.yaml` points at
`data_v13ship_v1_messages.jsonl`; rationale in roadmap update 2026-07-01b). The v1 training set =
v11 page-context backbone (~65k) + Ikenie gold ×3, **NO mined NSFW**, with assistant-refusals
stripped from the targets (the abliterated base is refusal-free but SFT reintroduced leaks). Byte-exact
prompt round-trip via `build_v2_messages.py` / `row_to_message` (no prompt drift).

**POV-contrastive miner** ✓VERIFIED commit `d8db35a` + `reference_corpus_pov_mine_engine_gate`.
`pov_mine/mine_pov_contrastive.py` mines gendered rows using the presence-scorer as a pre-filter
(keep rows with independently verifiable gender markers). Proven end-to-end but the mined data
**regressed** the model (§2.2) — the pipeline works, the data/approach was the problem.

**corpus_bitext mining** ✓VERIFIED (`backend/scripts/data/corpus_bitext/`, roadmap fix8 update).
JP side = CTD v26 + PARSeq (CPU to dodge AR-OOM); EN side = Qwen3-VL-8B per-bubble VLM at
coord-norm 1000; Hungarian centroid alignment (P/R ~0.96); curated into byte-exact v11 page-context
rows. This produced the 2,337 rows that made v11fix8 a real win — the first evidence that
**corpus-mined translated pairs are a clean lever** (scaling bounded by OCR GPU-time, not data).
Cost model: EN-VLM serving is the bottleneck at scale (~1–2 s/page batched vLLM vs ~48 s/page
transformers-eager); CPU JP-OCR is the wall for small runs (~4–8 s/page).

**Alignment backlog recovery + hamming pollution filter** ✓VERIFIED commit `26f1457`.
- `align_backlog.py` recovers **34,246 strict JP/EN pairs already on disk but absent from the 375k
  export** (projected ~748k page pairs, ~2× current) — a 4× data lever needing **no downloading**.
- The corpus audit found **41,037 rows (11%) at hamming=0 are likely art-only/untranslated pages
  posing as bitext** — a training-data pollution signal now filterable.
- Background: the manifest grew 134k → **374,688** page pairs by relaxing the `export_manifest.py`
  status/image-present gates (`export_manifest_375k.jsonl`); the source images were reaped off the
  NAS CIFS mount but recoverable from `/mnt/nas/drive_1/manga-ml/ehentai_corpus.tar` (3.3TB).

**The NSFW cap lesson** ✓VERIFIED (`feedback_v12_nsfw_oversampling_regression` + roadmap). NSFW held
**flat at ~16%** across v11/fix6/fix7 after the v12 36%-oversample regression. v11fix8 sits at 0.184.
v1's mix went to ~26.5% but with **quality-audited** (refusal-stripped, non-euphemistic) pairs, not
bulk oversampling. **Rule: register fidelity is a data-quality problem, not a data-quantity problem.**

---

## 4. Detection / OCR improvements

**CTD (comic-text-detector) v26.** ✓VERIFIED commits `961bda4`/`41b742b`/`f544f07`. Custom seg
checkpoint behind a `ctd_v26` flag: 1280 input, obb-driven line extraction, 0.8 threshold, ~24×
better SFX/ono detection with tighter text boxes. Round-9 ono-fix detector wired for real SFX
erasure (feeds the inpaint mask, §5).

**PARSeq AR + dynamic-batch OCR.** ✓VERIFIED (`reference_ar_dynbatch_ocr_export`, config.py:465-485).
A validated autoregressive + dynamic-batch PARSeq export (`parseq_manga_ep60_AR_dynbatch.onnx`,
val_CER 0.0218) exists to fix garble on stylized/vertical text. AR is ~10× slower per crop, so it's
used in a **confidence-gated hybrid** (non-AR fast path; low-conf crops re-OCR'd by AR in one batch;
then the garble gate runs on the AR result).

**Vertical-AR-by-default routing (#7).** ✓VERIFIED (config.py:471-485,
`project_mt_quality_fixes_ikenie4`). The dominant garble class (144-bubble Ikenie-4 cohort) was the
**NAR decoder duplicating adjacent kana on dense vertical crops at falsely-high confidence**
(身代わり → 身身わわ @0.92) — so the conf-gated AR retry NEVER fired on the worst cases. Fix: route
tall/narrow crops (h/w ≥ 1.5) to AR **up front by geometry**, independent of confidence. Measured:
garble on pp.5/45/123 went **5→0** (身身わわ→身代わり, 濯濯→洗濯). Standalone win fixing garble at the
SOURCE, not just dropping it.

**AR/NAR hybrid empty-crop bug + fix.** ✓VERIFIED (`reference_ar_hybrid_empty_bug`, commit
`0ff124b`). `ParseqOCRService._ar_decode_indices` pre-initialized vertical crops to `("", 0.0)` and
the AR-failure branch did `continue` with no NAR result to keep → **every crop silently emptied**
when AR inference OOM'd under VRAM contention (~94% empty per gallery). Invisible because prod sets
`HYBRID_OCR_ENABLED=false` for VRAM co-location and no test exercised the vertical-AR path. Fix: fall
back to NAR (`_nar_decode_indices`) before continuing. **Caveat: the fix makes the AR re-mine SAFE
(no empties) but under contention AR still OOMs → you get NAR-quality output; to actually use AR you
need free VRAM headroom.**

**OCR confidence-gate recalibration (0.65 → 0.80).** ✓VERIFIED commit `4ac7cf7`, audit §3, handoff
2026-07-03. A 650-row labeled table (conf × similarity-to-gold) showed the **0.65–0.75 band is only
~37% correct**; accuracy stabilizes ≥0.85. Meanwhile sub-0.65 short strings (moans/SFX/numbers) were
100% correct but silently dropped. Recalibration: raise the **long-text floor 0.65→0.80**, add a
**short-text carve-out** (<4–6 chars exempt), and make **adjacent-dup-kanji/kana checks
unconditional** (like latin-intrusion already is) so `身身わわ@0.92` is caught despite passing the
`DUP_CONF_CEILING=0.88`. 156 gate tests pass. NOTE: `config.py:449` still shows the base
`ocr_confidence_gate_threshold = 0.65`; the recalibration logic lives in `ocr_confidence_gate.py`.

**Dup-bigram garble gate (#4).** ✓VERIFIED (`project_mt_quality_fixes_ikenie4`, commit `d47300b`).
A confidence-INDEPENDENT detector for duplicated-char/phrase garble that PARSeq emits at falsely-high
conf, whitelisting legit reduplication (様々/段々/ますます). Biggest measured win of the Ikenie fixes:
garble-drops 107→182 (+75 hallucinations eliminated).

**Balloon-fragmentation defect + detection-time grouping fix.** ✓VERIFIED commit `08b8144`, audit §2,
`project_p1_bubble_grouping_resplit`. **The #1 systemic quality defect all four audit lenses
converged on:** CTD emits **one block per text COLUMN**, so a multi-column vertical balloon arrives
as N independent OCR/translation units → the page-context model duplicates the sentence across
columns or blanks all-but-one, and each fragment gets its own render box.
- The **pre-translation** re-segmentation (`translation_bubble_grouping`, `bubble_grouping.py`)
  FAILED validation twice (Ikenie4 corrected-omissions 14→50 with it on). Root cause was deeper than
  membership over-merge: even a *correctly* grouped long balloon loses text in the
  **merge→translate→resplit roundtrip** (model consolidates onto the lead; resplit blanks
  continuations without redistributing). Kept **OFF** (config.py:293).
- The **correct fix (commit 08b8144): fuse side-by-side columns at DETECTION time**, before crop/OCR,
  so OCR sees one crop and translation one JP string per balloon — **no roundtrip to lose text on**.
  Gated. Validate via `.bench/audit_3way_p1.py` (GPU regen + 3-way omission audit), not unit tests
  (which passed while the integration regressed).

---

## 5. Inpaint / typeset

**Solid-rect erase + bubble-gating regression/fix.** ✓VERIFIED commit `2965b92`, audit §4.4. Commit
`22fd106` had removed the bubble gate on the kept-block solid fill ("ALL kept blocks, not only
bubble-matched"), so un-bubbled blocks (SFX/narration over art) got their full bbox painted as a
rectangle → **inpaint scars on artwork**. Fix: solid-fill a block **only when it matched a speech
bubble** (`fit_rects[bi] is not None`) so the rectangle lands inside the balloon interior; un-bubbled
blocks still contribute their tight seg-mask ink to the erase for recall. This was the #1 visual
defect (residual-JP / scars) flagged in the Furube human eval.

**Tail-glyph padding.** ✓VERIFIED same commit (§4.5). Mask tail-glyph clipping (e.g. a trailing "に"
surviving at the end of a column) fixed with asymmetric end-of-column padding.

**SFX/ono erasure.** ✓VERIFIED commits `ca25d6e`/`f373d1a`. The v26 detector's ono/SFX mask channel
feeds real SFX erasure in the inpaint mask (pure-art SFX inpaint is a clean win per Furube).

**Earlier inpaint recall work.** ✓VERIFIED commits `17e214c` (contrast/ink-aware erase — kill
dark-page over-erase), `22fd106` (Phase-0 recall), `6e9ab94` (leave-intact inpaint). The Furube eval
named **inpaint-mask recall (residual Japanese not removed)** the single biggest visual gap → decouple
the erase mask from the keep decision.

---

## 6. Serving / latency

**vLLM + Gemma-4 MTP spec-decode.** ✓VERIFIED (`serve_v10it_vllm.sh`). Path A deploy: merge the LoRA
into the base on CPU (vLLM Gemma-4 LoRA load is silently broken, issue #41754) then serve with
Google's official MTP drafter `google/gemma-4-E4B-it-assistant` (`method=mtp`). **γ=2 wins** on this
corpus (109 tok/s vs γ=4's 104 — per-position accept falls off past pos 1). Baseline was v9c
llama.cpp Q8_0 at 77.8 tok/s.

**GPTQ-INT4 quant — and the exception for corrective models.** ✓VERIFIED (`reference_gemma4_vllm_quant`).
For the **base** v11, GPTQ W4A16 (~12GB, calibrated) actually beats bf16 (chrF++ 32.21 vs 31.28) and
composes with MTP. fp8 is useless (PLE per-layer + tied 262k-vocab embeddings stay bf16). RTN-INT4 is
banned (deterministically truncates speaker-tagged lines). Three arch fixes were needed for the
GPTQ-sequential pipeline: tracing (`sequential_targets` + `project_per_layer_inputs` in
`tracing_ignore`), PLE OOM (`--keep-embeds-on-cpu`), and the KV-sharing `KeyError: 'sliding_attention'`
(persistent store keyed by batch/layer). **BUT INT4 does NOT survive the corrective fine-tunes** —
4-bit erases the fine reverse_sense/name_invention signal regardless of calibration faithfulness
(v11fix8 INT4 cert-FAILED −2.41 chrF). **Rule: ship bf16 for fix7+ corrective models; INT4 only for
the base.** That's why the current serve default is **bf16 merged_fixed** at GPU_UTIL 0.55, not INT4.

**Merged-LoRA + CUDA graphs (the triton fix).** ✓VERIFIED (audit §1.3, §7). The serve script had long
run `--enforce-eager` — which turned out to be **masking a corrupted Triton wheel install**. After
repairing Triton (and the flashinfer sampler, broken on both machines → `VLLM_USE_FLASHINFER_SAMPLER=0`),
merged-weights serve + CUDA graphs became the target config for v1 (`box_serve_v1_merged.sh`, util 0.72).
NOTE: the GPTQ-INT4 Gemma path still REQUIRES `--enforce-eager` (compiled inductor graph crashes with
`CUBLAS_STATUS_INTERNAL_ERROR` on a bf16 GEMM on the first real batch) — that's config.py-independent,
serve-script line 71.

**Streaming protocol.** ✓VERIFIED (config.py:487-496, commits `f4392b9`/`797a35e`/`13a7c6e`/`620e536`).
`translation_stream_events=True` (WebSocket path) emits a versioned event-frame protocol
(detections → per-bubble tl → revise → plate → done, see `src/types/stream.ts`) so the extension
renders each bubble as soon as it's translated; the monolithic reply stays as fallback. First-text
perceived 252–645ms. Frontend adds progressive render + next-page prefetch. 27 protocol tests green.

**Image-context prefix cache.** ✓VERIFIED (`project_serve_with_image_latency`, audit §1.3/§6). The
page image is prefilled ONCE per page (~228ms cold) and reused as a **shared prefix across the N
bubble queries**; measured serve-with-image = **~0.78s/page, only +18ms over text-only** — refuting
the ship-runbook's assumed 2–4s. Multimodal (image-KV) prefix caching verified on box vLLM 0.22.1
(15/16 prefix hits, 0 corrupted outputs). `translation_serve_image_context=True` is safe on the
text-only prod model because the text prompt is byte-identical with/without the image.

**Client concurrency.** ✓VERIFIED commit `47685e0`, audit §1.3. Was hardcoded at 8 (a global
semaphore across pages); now `translation_client_concurrency=32`.

**Measured latencies (E2E, audit §9.1).** ✓VERIFIED. box merged-v1 (8B, CUDA graphs, over Tailscale)
vs old local v11fix8 (4B): mean **1093ms vs 1405ms**, median 1114 vs 1212, p95 1847 vs 2235 —
**the 8B is already ~10–25% faster E2E** than the old 4B baseline, before warm-prefix/streaming/prefetch.
Light pages 571–617ms. Translate is 70–85% of E2E. (Bonus reference: DSpark spec-decode gives ~3.2×
on the 5090 via forced TRITON_ATTN — `reference_dspark_vllm_blackwell`, not yet in the prod path.)

---

## 7. Eval infrastructure

**chrF paired-bootstrap harness.** ✓VERIFIED (`paired_bs_chrf.py`/`paired_bs_metric.py`,
`run_ikenie4_regression.sh`, `score_jsonl_metrics.py`). The cert standard: chrF++ with CI95 paired
bootstrap; **PASS = CI excludes 0 in the win direction AND no corrective probe regresses**. A stable
**bbox-spatial gold join** (commit `a03563b`) was added so every run scores the SAME gold rows — an
earlier "−6.4% / +21 rows" result was a jp-join artifact. Case-insensitive scoring (human typeset is
ALL-CAPS → must lowercase to measure translation not case).

**Ikenie4/5 vision-gold.** ✓VERIFIED (`build_ikenie4_gold.py`, `transcribe_gt_vision.py`,
`project_mt_quality_fixes_ikenie4`). Qwen2.5-VL-7B (later Qwen3-VL-8B-FP8, `reference_gold_ocr_model_selection`)
transcribes the HUMAN English scanlation off GT pages → per-bubble gold. Gold grew 77 → 650 rows.
Distinct from prod OCR (our PARSeq is Japanese-only). **Methodology caveat: the gold `jp` field is
our OWN pipeline OCR, so "ocr_clean %" is a confidence proxy that overstates health — no true CER.**

**Furube human eval.** ✓VERIFIED (`project_furube_human_eval`). First E2E eval vs REAL human EN
scanlations (3 chapters, 183 pages, 1300 bubbles). Translation ~80% acceptable but a hard discourse
ceiling (POV/hallucination on pro-drop); visual ROUGH (residual-JP inpaint recall #1). This eval is
what pointed the roadmap at image-as-context / model-size and named the top visual fixes.

**POV probe → pov_probe_v2 (the metric-ghost correction).** ✓VERIFIED (`pov_probe.py`,
`pov_probe_v2.py`, commit `d042346`, `feedback_pov_metric_is_broken`). The original `pov_probe.py`
`gendered_rate` conflated person-choice with gender, carried a 79%-"she" prior, and scored
laughter/blob/misalignment artifacts as failures — **it hallucinated a ceiling that drove three
failed v2 experiments.** The corrected **two-axis** `pov_probe_v2` scores GENDER inversion (on a
curated gender-obligatory, class-balanced subset) SEPARATELY from PERSON/register, and re-measured
v1's real he↔she rate at **0/67**. This is the durable artifact that stops future ghost-chasing;
`pov_probe.py`'s gendered_rate is deprecated.

**Refusal gate.** ✓VERIFIED (`refusal_eval.py`, `build_refusal_probe.py`, `refusal_probe.jsonl`). A
250-line hard gate (zero regex hits) confirming abliteration holds; v1 = 0/250, v2-POV = 0/250,
image-arm = 0/2840 lines. A 1/250 hit on 30B was a coherence deflection on mojibake, not a safety refusal.

**Cert gate (one-command).** ✓VERIFIED commit `2b1d7a5` (`run_cert_gate.sh`). Aggregates the **6 ship
signals** (POV, refusal, chrF, adequacy, corrective probes, latency) into a single pass/fail run.

**Adequacy judge.** ✓VERIFIED (`page_adequacy_judge.py`, `gemba_mqm_judge.py`, commit `92aa6b0`). A
VLM page-adequacy judge reads the actual rendered page (robust to garbled OCR and noisy refs; doubles
as a visual metric). On the v1 gate it was a statistical tie vs prod (adequacy 0.456 vs 0.462, deltas
≪ noise at n=182) — used as a guardrail, not the promotion driver.

**Detection-recall gold.** ✓VERIFIED commit `b8dabd4` (`build_recall_gold.py`, `detection_recall_eval.py`,
`detection_recall_gold.json`), audit §5. Detection recall was **unmeasurable** — gold bboxes were
seeded from our own detector (IoU==1.0 for all 650 rows), so a CTD false-negative was invisible. An
independently-boxed recall set was built as a prerequisite before ANY detector change (incl. §4's
detection-time grouping).

**The dual/triple-pipeline hazard.** ✓VERIFIED (`project_dual_pipeline_paths`). There are THREE
translate implementations — HTTP router (`translate.py::process_single_image`, canonical),
WebSocket (`websocket_upload.py`), and the batch benchmark (`batch_translate_chapter.py`). New
post-translation logic must land in the shared helper `translation_postedit.py` and be wired into ALL
paths, or benchmark/extension output diverges. (A WS-path bug once sent users the worst output because
it reimplemented the pipeline with the legacy numbered-block format the v11 model degrades on.)

---

## 8. Key lessons / cross-cutting WHY

1. **Train/serve format sensitivity is the master lesson (~95% collapse).** ✓VERIFIED
   (`feedback_chat_template_mismatch`, `project_v11_pagecontext_model`). Feeding a chat-tuned/format-tuned
   model an off-distribution prompt causes SILENT collapse (empty outputs, JP passthrough, ~95% chrF
   drop) that looks like a broken model. It cost ~8h on "broken" v10it (the fix was 4 lines). It's WHY
   v11 exists (train shape = serve shape), WHY every new serve-time prompt lever (cast register,
   image-context) is guarded to be byte-identical when off, and WHY the POC image format landmine
   (`pov_probe.py` byte-matched to v1's format, not the POC's) silently collapsed. **A converging
   training loss is NOT evidence of a working model — only quality-aware inference on real prompts is.**

2. **Model size is THE quality lever for discourse; data volume/cleaning are not.** ✓VERIFIED (audit
   §1.2, `project_mt_finetuning_roadmap`, `project_v11fix9_dataclean_result`,
   `feedback_pov_contrastive_sft_regressed`). The 4B→8B jump tripled POV correctness and banked +10 chrF.
   Meanwhile: bulk bitext plateaus by ~1–10K pairs; data-cleaning (v11fix9) failed; more DPO/POV text
   pairs (v10.5, v12, v2) all regressed. The gains are in **capacity + modality + data QUALITY**, not
   quantity.

3. **The POV metric was broken.** ✓VERIFIED (`feedback_pov_metric_is_broken`). See §2.2/§7. Three
   experiments failed because the eval measured the wrong thing. When repeated well-run experiments
   fail the same way, suspect the measurement.

4. **The MoE-quant blocker.** ✓VERIFIED (`project_v2_30b_a3b_readiness`, handoff). bitsandbytes NF4
   only swaps `nn.Linear`→`Linear4bit`; Qwen3-VL-30B-A3B stores experts as **fused 3-D `nn.Parameter`s**,
   so 0/60 experts quantize → 28.99B params stay bf16 = 54GB → OOMs the 32GB 5090 (and A100-40/L40S-48).
   **Any MoE-VLM QLoRA on ≤48GB is dead this way; dense models are fine.** To run 30B needs an 80GB host
   or a fused-expert-quant stack (none in transformers 5.12 / bnb 0.49.2). If a bigger base is pursued
   it's the **dense** 32B-abliterated, and for general quality, not POV.

5. **Box thermal-fault hardening.** ✓VERIFIED (`reference_gemma4_training_quirks` #3/#4, audit §7).
   The RTX 5090 (local AND box) faults under sustained multi-hour load (Xid-class GPU-handle loss; a
   local 12VHPWR power-delivery fault; filesystem-level `.so` corruption during a 33GB CPU-merge under
   thermal stress). Mitigations now standard: don't stack train+CPU-merge+serve in one session; let the
   card cool; `nvidia-smi -pl 400`; checkpoint often (`resume_from_checkpoint` added after a mid-run
   fault); `in_training_eval: false` (sm_120 SIGSEGVs in the eval-dataset build); hash-verify large
   artifacts written under load. Heavy GPU work goes on the box, not local, where possible.

6. **Register fidelity is a data-QUALITY problem.** ✓VERIFIED (`feedback_v12_nsfw_oversampling_regression`,
   `feedback_cpo_pitfalls`). Oversampling euphemistic NSFW → the model euphemizes. CPO/SimPO's
   length-normalized reward + teacher-chosen rows → terse/bland outputs. Audit pairs for euphemism;
   drop teacher-chosens; gate on held-out translation eval, not preference margins.

7. **Silent-failure traps recur — gate every launch with a probe.** ✓VERIFIED
   (`reference_qwen3vl_completion_only_masking`, `reference_ar_hybrid_empty_bug`). The Qwen3-VL trainer's
   `completion_only_loss` YAML key is INERT (trl masking bypassed) → ~85% of gradient trains on the JP
   prompt unless `train_on_responses_only` is wired into the collator; the dry-run loss looks healthy.
   The AR-OCR path silently emptied every crop. The k_norm drop silently made vLLM reject merged
   checkpoints. Every one was invisible until a downstream gate. **Add a runtime probe (masking
   fraction, empty-rate, smoke test) before any multi-hour run.**

---

## Appendix — key file map

- Production config / flags: `backend/app/config.py`
- Serve (Gemma local, prod): `backend/scripts/eval/serve_v10it_vllm.sh` (serves v11fix8 as "v10it")
- Serve (Qwen v1, box): `box_serve_v1_merged.sh` ?INFERRED path
- Training configs: `backend/training/configs/{gemma4_e4b_v11*,qwen3vl_8b_textsft_v1,qwen3vl_8b_v2_pov,qwen3vl_8b_imagectx_poc*,qwen3vl_30b_v2}.yaml`
- Post-edit shared helper: `backend/app/services/translation_postedit.py`
- OCR gate: `backend/app/utils/ocr_confidence_gate.py`; AR OCR: `backend/app/services/parseq_ocr_service.py`
- Detection-time grouping: `ctd_service.py` + `backend/app/utils/bubble_grouping.py`
- Inpaint mask: `ctd_utils.build_inpaint_mask`
- Eval: `backend/scripts/eval/{run_cert_gate.sh,run_ikenie4_regression.sh,pov_probe_v2.py,refusal_eval.py,page_adequacy_judge.py,detection_recall_eval.py,paired_bs_metric.py}`
- Data mining: `backend/scripts/data/{corpus_bitext/,pov_mine/,v13ship/,align_backlog.py}`
- Handoffs (chronological): `thoughts/shared/handoffs/mt-quality-ikenie4/*.yaml`
- Audit synthesis: `thoughts/shared/research/2026-07-02_pipeline-audit-synthesis.md`
</content>
</invoke>
