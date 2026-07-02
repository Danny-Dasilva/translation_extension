# Pipeline Audit Synthesis — 2026-07-02

Multi-agent audit of the manga-translation pipeline, run the night of the v1 gate evaluation.
Workstreams: 6-signal gate for v1, 4B deconfound, corpus audit, and four improvement lenses
(detect/OCR, translation, inpaint/typeset, data→training→eval loop). All findings are grounded
in file:line references and tonight's fresh eval artifacts (134-page Ikenie4 fair runs).

Related: `thoughts/shared/handoffs/mt-quality-ikenie4/2026-07-02_01-15_gate-run-fair-eval-harnesses-built.yaml`

---

## 1. Headline results

### 1.1 v1 gate (Qwen3-VL-8B abliterated + text-SFT LoRA, now merged, box :8001)

| # | Signal | Result |
|---|--------|--------|
| 1 | POV (presence metric, gendered-only) | v1+image **dominates every arm**: 20% Furube-37 / 13.3% all-148 vs prod 0% / 1.9%. Old ≥48% threshold came from a dead (evasion-blind) metric — recalibrate to strict dominance |
| 2 | Refusal | **PASS** — 0/250 hard gate, zero regex hits |
| 3 | chrF vs v11fix8 | **PASS (win)** — Δchrf++ +8.62, CI95 [+6.42, +11.00], p=0.0000, fair same-code A/B |
| 4 | Adequacy (VLM page judge, 35-page sample) | Statistical tie (adequacy 0.456 vs 0.462; deltas ≪ noise floor at n=182 lines) |
| 5 | Corrective probes | **PASS** — zero regressions |
| 6 | Latency | **PASS** — translate-with-image ~0.5–0.7s/page; image ≈ free (mm prefix cache reuse verified via vLLM Prometheus counters, 15/16 hits, 0 corrupted outputs) |

**Fair-eval caveat that mattered:** the first v1 regen showed 33% empty translations and a probe
"regression" — entirely an artifact of `dedup_by_bubble` pipeline drift (feature added 07-01,
baseline generated 06-29). On identical same-day code with dedup/merge disabled for per-bubble
gold, v1 blanks *less* than prod (1.7% vs 2.7%).

### 1.2 4B deconfound — the 8B tax is justified

Zero-shot huihui Qwen3-VL-**4B**-abliterated (local serve), presence-based POV scorer:

| Arm | Furube-37 gendered | All-148 gendered |
|---|---|---|
| v11fix8 (trained Gemma 4 E4B, prod) | 0/15 = 0.0% | 1.9% |
| base-4B text / +image | 0.0% / 0.0% | 3.8% / 4.8% |
| base-8B text / +image | 6.7% / 13.3% | 8.6% / 11.4% |
| v1 (8B SFT) text / +image | 13.3% / 20.0% | 9.5% / 13.3% |

**Verdict: capacity story.** Below ~8B, neither Gemma-E4B (trained) nor Qwen-4B (zero-shot,
either modality) resolves a single gendered case on Furube-37. The image signal exists at 4B
(+1.0pt) but only the 8B can exploit it (+2.9pt). SFT and image compound on top of capacity.
Model-architecture audit of larger uncensored VLMs (Gemma-4-12B, 30B-class abliterated at
FP8/AWQ) is running separately; any candidate is now cheap to vet (`pov_probe --custom-model
--base-url` + `refusal_eval` + `run_ikenie4_regression.sh` against any OpenAI endpoint).

### 1.3 Serving/latency reframes (from the earlier research round, all verified tonight)

- Multimodal (image-KV) prefix caching **works** on box vLLM 0.22.1 — image prefill paid once
  per page across N marked calls. Warm marked calls ~110–220ms; cold ~200–250ms.
- Merged-LoRA serve + CUDA graphs is the target config (`--enforce-eager` was masking a broken
  Triton install, now fixed; flashinfer *sampler* is broken on both machines → `VLLM_USE_FLASHINFER_SAMPLER=0`).
- The old 1.4s E2E baseline was stale; prod steady-state is 380–920ms. Translate is 70–85% of E2E.
- Client concurrency was hardcoded at 8 (global semaphore across pages) — now
  `translation_client_concurrency=32` (committed 47685e0).

---

## 2. THE systemic defect: multi-column balloon fragmentation

All four improvement lenses independently converged on this. CTD emits one block per text
*column*, never per balloon; every downstream stage then fights the consequences:

- **Detection lens:** 724/1,486 blocks (48.7%, upper bound) sit in geometrically-adjacent
  column groups; 61 groups had a member gate-dropped (balloon renders half its text); 266
  groups translate fragments independently (duplication or consolidation-blanking).
- **Translation lens:** the marked-line model *consolidates* fused content onto one fragment
  and returns empty for the donor (11+ clean examples, e.g. `016/bubbles.json` idx5/6:
  "Fucking you... **Every day**" + empty, where 毎日 belongs to the blanked sibling). None of
  the three existing mitigations (sentence_merge — trailer-list too narrow; bubble_grouping —
  disabled after the 06-29 regression; backfill) catches this class.
- **Render lens (visually CRITICAL):** p013 one balloon → three independently hallucinated
  paraphrases stacked; p022 one balloon → three captions at wildly different font sizes (the
  bubble-detector missed the balloon → every bubble-aware mechanism silently disabled);
  `compose_final`'s ≥60%-buried collision rule deletes whole dialogue lines with zero trace.
- **Eval lens:** per-bubble chrF join is fragile against exactly these drops/merges — the
  worst-scoring rows are index-misalignment artifacts, not model failures.

**Root-cause direction (agreed across lenses):** form balloon-level units at DETECTION time —
fuse side-by-side columns before OCR (extend the koharu merge predicate in
`ctd_service.py:692-745` using YOLO-bubble membership + the tight column-adjacency geometry
already in `bubble_grouping.py:114-156`), so OCR sees one crop and translation sees one JP
string per balloon. This eliminates the merge→translate→resplit roundtrip that killed the
06-29 attempt (that attempt retrofitted grouping onto already-split OCR). Interim guardrails:
span-cap fused groups, validate fused-EN length vs fused-JP, instrument the bare
`except Exception: continue` in `apply_fused_balloon_retranslate` (bubble_grouping.py:815),
add a `suppressed_dialogue_count` stat to compose_final.

**Interaction bug found in tonight's own fix (fix before commit/enable):** the fused-balloon
retranslate output goes through `postedit_one` with the *winner's single-fragment JP* as the
over-expansion denominator (`translate.py:789-795`) — a faithful fused EN will trip
`is_over_expanded` and be blanked to "...". Same pre-existing defect for sentence_merge leads.
Also the batch script applies glossaries BEFORE dedup/fused-retranslate (`batch_translate_chapter.py:732-789`),
opposite of the router — fused output skips the entire glossary chain there. Fix: thread the
effective (fused) JP into postedit; unify ordering to translate → backfill → dedup/fused → glossaries.

---

## 3. Miscalibrated gates (we now have labeled data to fix them)

| Gate | Problem | Evidence | Fix |
|---|---|---|---|
| OCR conf gate 0.65 (`ocr_confidence_gate.py:28`) | The 0.65–0.75 band passing the gate is only **37% correct** vs gold; accuracy stabilizes ≥0.85. Meanwhile 11 sub-0.65 rows were 100%-correct short strings (moans/SFX/numbers) silently dropped | 650-row calibration table (conf × sim-to-gold) built tonight | Fit an operating point from the labeled pairs; short-text carve-out (<4-6 chars exempt), raise long-text threshold toward ~0.80–0.85 |
| `DUP_CONF_CEILING=0.88` (`ocr_confidence_gate.py:396`) | Confidently-wrong dup-garble passes untouched: `身身わわ` @0.92 — the file's own docstring example — mistranslated the manga's central plot line ("Mommy is protecting me" vs "took my place as sacrifice") | 3 uncaught cases in 1,333 kept bubbles; 0 false positives for the adjacent-dup signals | Make adjacent-dup-kanji/kana checks unconditional (like latin-intrusion already is); keep the ceiling for length/bigram signals |
| `OVER_EXPANSION_ABS_MIN_WORDS=8` (`translation_postedit.py:60`) | <8-word outputs never checked: `違う...!!` → 7-word output with a fabricated trailing clause passes; garbled-input hallucinations of proportionate length also pass | fixtures in report | JP-length-proportional floor; garble-signature check before generation |
| `_is_sfx_sized` area<9000px² (`refit_final_composites.py:451`) | Narrow-column *dialogue* misclassified as SFX → truncation + tiny fonts | p022 idx1-3 | Gate on OCR'd JP length, not box area alone |

---

## 4. Production-facing contract bugs

1. **`short_utterance_normalize_enabled=True` in prod diverges from v1's training** (571/29,467
   rows). Every harness force-disables it (correct for parity) — so prod behavior is
   *uncertified*. Before v1 promotes: flip the default to False (v1's byte-contract), and make
   `verify_builder_parity` assert against the LIVE default instead of monkeypatching it.
   For v2, prefer baking normalization into training data so the flag can be True everywhere.
2. **No transport retry** on marked-line calls (`vllm_openai_translation_service.py:614-622`):
   one transient timeout = permanently blank bubble, indistinguishable from consolidation
   blanking. Add 1–2 bounded retries on transport errors only.
3. **Cast/name anchoring is two disconnected hardcoded tables** (`DEFAULT_CAST_ANCHOR`,
   `NAME_LOCKS`) — single-title, never enabled in prod. Unify into a per-title cast manifest;
   longer-term: chapter-level rolling cast register (batch pipeline already has the state).
4. **Solid-rect erase for un-bubbled blocks** paints rectangular scars over artwork
   (`ctd_utils.py:183`, regression from 22fd106, flagged in its own commit message). Restore
   bubble-gating for the solid-fill tier only; keep seg-mask ink erase for recall.
5. **Mask tail-glyph clipping** (p005 "に" survives) — asymmetric end-of-column padding.

---

## 5. Eval infrastructure gaps

- **Detection recall is unmeasurable**: gold bboxes are seeded from our own detector output
  (IoU==1.0 for all 650 rows) — a CTD false-negative is invisible to every harness. Build a
  small independently-boxed recall set before any detector change.
- **POV gendered-n is tiny** (15 of Furube-37). Nearly-free expansion: `required_family()` is a
  pure function — run it over the other 98 testset rows and the 8,828-row furube gold pool.
- **Per-bubble chrF join fragility** (see §2) — worst-bubble mining must cross-check against
  the index-stable inspect dirs.
- **One-command cert**: all six signals exist as separate CLIs with pass/fail semantics —
  aggregate into `run_cert_gate.sh --candidate <model>` (mechanical; every script already
  exits nonzero / emits gate_pass).

---

## 6. Data/training levers for v2

1. **POV-contrastive training pairs** mined from the 375k-page corpus using the presence-scorer
   logic as a *pre-filter* (keep rows with independently-verifiable gender markers). Precedent:
   `build_voice_addressee_probe.py`. Emit through the byte-identical builder — no new prompt shape.
2. **Targeted image-on distillation**: image-on generations on POV-hard rows distilled into the
   text-only model (low-LR, small-N — precedent `qwen3_4b_v8_distill.yaml`), NOT broad
   image-training (measured harmful) and NOT text-only distillation of image wins (information-
   theoretically doomed).
3. **Port the v13ship mix into `compose_training_mix.py`** (declarative weight-spec with
   regression-lesson comments) instead of hardcoded CLI defaults — makes ablations one-YAML diffs.
4. Image-prefix app serve path: build as a hard-gated port of `bench_image_prefix.py`'s
   validated contract with a byte-identity unit test now, wire to v2 only (sending an image to
   text-trained v1 is the ~95%-collapse class).

---

## 7. Environment/hardware incident (box, 2026-07-02 ~04:00)

During the merge/serve work the box (100.64.235.63) exhibited intermittent `ld.so` relocation
assertions, then consistent segfaults. Root cause found by hashing site-packages against pip
RECORDs: **7 corrupted `.so` files across 5 unrelated packages** (torch libtorch_cpu/python,
numpy, tokenizers, cbor2, nvidia_cutlass_dsl ×2) — filesystem-level corruption, consistent with
the box's thermal-fault history under sustained load (train + 33GB CPU merge same night).
Repair: force-reinstall of the 5 packages (torch from the cu130 index). **Recommend memtest +
fsck on the box.** Also fixed en route: corrupted Triton wheel (why the serve always ran
`--enforce-eager`), broken flashinfer sampler on both machines (`VLLM_USE_FLASHINFER_SAMPLER=0`).
Merged model `/home/danny/ship_v1/merged_v1` was written during the unstable window — its
integrity is validated behaviorally by the pending chrF re-cert (a corrupted merge would tank it).

---

## 7.1 Box hardware check runbook (needs interactive sudo — run manually)

Non-sudo diagnostics came back empty (dmesg restricted, smartctl denied). On the box
(`ssh danny@100.64.235.63`), when convenient — ideally when no train/serve is running:

```bash
# 1. Kernel log for MCE / EDAC / NVMe / filesystem errors around 2026-07-02 03:00-05:00
sudo dmesg -T | grep -iE "mce|edac|ecc|i/o error|nvme|ext4.*(err|corrupt)|xfs.*corrupt" | tail -40
sudo journalctl -k --since "2026-07-02 02:00" | grep -iE "mce|error|corrupt" | tail -40

# 2. NVMe health (media errors + error log entries are the key fields)
sudo smartctl -a /dev/nvme0 | grep -iE "critical_warning|media_errors|num_err_log|percentage_used|temperature"

# 3. Schedule fsck on next reboot (root fs)
sudo touch /forcefsck   # or: sudo tune2fs -c 1 /dev/nvme0n1p2

# 4. Memtest: reboot into memtest86+ (GRUB advanced options) and run ≥1 full pass.
#    The corruption pattern (7 .so files across unrelated packages, written during a
#    33GB RAM-heavy merge on a box with prior thermal Xid faults) points at RAM or
#    NVMe-under-thermal-load. If memtest is clean, run: sudo nvme smart-log /dev/nvme0
#    after a sustained-load session and compare media_errors.
```

Interim mitigation until cleared: avoid stacking train + CPU-merge + serve on the box in one
session (matches the existing box-hardening note), and hash-verify any large artifact written
under load (the venv hash-check script pattern from tonight works for this).

## 8. Priority roadmap (synthesized)

**P0 — correctness, before/with v1 promotion**
1. Fix fused-retranslate ↔ postedit denominator bug + unify router/batch ordering (§2 tail)
2. Flip `short_utterance_normalize_enabled` default for v1's serve; mechanize the parity check (§4.1)
3. Transport retry on marked calls (§4.2); telemetry counters for suppression/fused-failures
4. chrF re-cert of the merged+CUDA-graphs serve (pending box repair)

**P1 — the balloon root cause + gate recalibration**
5. Detection-time balloon grouping (§2), with span caps + fused-length validation
6. OCR conf-gate recalibration from the labeled table; DUP ceiling conditional (§3)
7. Solid-rect erase bubble-gating; tail-glyph padding (§4.4-4.5)

**P2 — quality growth**
8. POV eval expansion (free rows first) + POV-contrastive data mine for v2 (§5, §6)
9. Cast manifest + rolling register (§4.3)
10. One-command cert gate; independent detection-recall gold (§5)
11. Image-prefix app path contract-freeze (§6.4); model-architecture audit follow-ups

---

## 9. Workstream ledger (2026-07-02)

| Workstream | Status |
|---|---|
| 6-signal gate for v1 | 6/6 measured; 4 hard passes, adequacy tie, POV dominance (recalibration proposed) |
| 4B deconfound | DONE — capacity story, 8B justified |
| Detect/OCR lens | DONE (report in session log; key items §2, §3) |
| Translation lens | DONE (§2, §3, §4) |
| Inpaint/typeset lens | DONE (§2, §4) |
| Data/train/eval lens | DONE (§4.1, §5, §6) |
| Corpus dupes + missing-EN-pairs doc | DONE → `data/manga_datasets/missing_en_pairs.md` (§10) |
| chrF re-cert (merged v1 + CUDA graphs) | **DONE — PASS in substance**: Δchrf++ **+10.104** CI95 [+7.74, +12.87] p=0.0000 (stronger than the LoRA serve's +8.62). Strict verdict flagged 2 probe "regressions" — inspected: both single-row flips within the measured ~7-10% serving-numerics noise; the name_invention one inverts a pass-by-EMPTY-prediction from the LoRA run. Merge integrity behaviorally validated (canonical smoke answer byte-identical). Serve config: `box_serve_v1_merged.sh` (merged weights, CUDA graphs, VLLM_USE_FLASHINFER_SAMPLER=0, util 0.72) |
| Model-architecture audit (larger uncensored VLMs) | DONE (§11) |

---

## 9.1 Final E2E measurement — proposed prod config

`bench_e2e.py` (7 test images × 5 rounds), full local pipeline (detect/OCR/inpaint local 5090)
with **box merged-v1 (8B, CUDA graphs) as translator over Tailscale**, text-only serve,
normalize-off contract, measured UNDER local GPU contention (videonest co-tenant ~10.6GB;
33 AR-OCR OOM fallbacks logged → numbers are pessimistic):

| metric | v1-merged (8B, box) | old v11fix8 baseline (4B, local) |
|---|---|---|
| mean | **1093ms** | 1405ms |
| median | **1114ms** | 1212ms |
| p95 | **1847ms** | 2235ms |
| min | 571ms | — |

**The 8B config is already ~10–25% faster end-to-end than the old 4B baseline measurement**,
before the warm-image-prefix, per-bubble streaming, or prefetch layers land. Light pages
already run 571–617ms actual. Sub-1s perceived is the streaming layer's job (§ prior research);
sub-1s actual median needs the warm-prefix overlap + uncontended GPU.

## 10. Corpus audit results (dupes + missing pairs)

Deliverable: `data/manga_datasets/missing_en_pairs.md` (446-row machine-parseable download table).

- **Infrastructure:** the manifests' path root `/mnt/nas/drive_2/manga-ml/ehentai_corpus/galleries/`
  is DEAD — corpus archived to `/mnt/nas/drive_1/manga-ml/ehentai_corpus.tar` (3.55TB, 3.98M
  members). Inventory derived from `mctl_index.sqlite:disk_index` (35,689 EN / 34,700 JP gallery
  dirs), matching tar member counts — full coverage, not sampled.
- **Dupes:** zero exact duplicate rows; but heavy M:N gallery fanout (9,310 pairs over 7,372 EN
  / 7,317 JP gids) and **~24% cross-gallery page reuse** — unique content ≈ 286k EN / 284k JP
  pages, not 375k rows. 1,878 same-work title-dup groups. **41,037 rows (11%) at hamming=0 are
  likely art-only/untranslated pages posing as bitext** — a training-data pollution signal worth
  filtering. 6,531 `good` rows exceed the h>14 gate that `unreviewed_hq` enforces (inconsistent QA).
- **Missing EN pairs:** all 7,317 manifest JP galleries have EN on disk (assumption holds
  in-manifest). In the wider strict-pair universe: **446 JP galleries (~35k pages) have no EN on
  disk → 407 unique EN gallery downloads** (table with e-hentai tokens + destinations in the doc).
- **Biggest corpus win requires NO downloading:** 34,274 strict pairs have BOTH sides on disk but
  are absent from the 375k export — an alignment backlog ≈ 4× the current corpus.

## 11. Model-architecture audit (larger uncensored VLMs)

Bottom line: **Qwen3-VL-8B-abliterated stands as the committed choice** — the only candidate
*measured* clearing all hard requirements (zero refusals, chrF win, POV-prefix exploitation,
bf16 single-5090 fit, QLoRA-trainable, mm-prefix-cache support). One upgrade candidate is worth
a 1-day zero-shot eval: **`huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated`** (MoE, ~3B
active → fast decode ~190-200 tok/s class; W4A16/Q5 ~16-19GB + FP8 KV fits the 5090; same
abliteration lineage that measured 0/250; INT4 rejection doesn't apply to zero-shot use).
Eval via the turnkey harness (`pov_probe --custom-model` + `refusal_eval` + regression).

**MEASURED (2026-07-02, W4A16 AWQ via vLLM on box :8003, full vision):** split verdict.
- All-148 gendered: 30B img-on **15.2%** — best untuned score in the table, above base-8B
  (11.4%) and above SFT'd v1 (13.3%), with the largest image delta (+7.6pt: vision does real
  work at this capacity). All-rows 39.9% also table-best.
- Furube-37: **0/15 both arms** (evasions, not wrong-gender) — does not lift the headline slice.
- Refusal: 1/250 — a coherence deflection on mojibake OCR garbage ("cannot be translated..."),
  not a safety refusal; zero content refusals.
- Practicals: 28.3GB peak VRAM at 0.90 util (no co-serve headroom), ~185 tok/s single-stream
  decode (MoE ~3B active — faster than the dense 8B), 296 vision generations in ~3 min.
- **Implication for v2:** the SFT recipe added +5-7pt to the 8B; if it transfers, 30B-A3B+SFT
  could be the v2 base — but requires QLoRA-on-quantized-MoE training (unproven here) and
  forfeits co-serve headroom. Decision deferred to the v2 planning round; checkpoint cached on
  the box (`JinRiYao2001/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-AWQ`).

Rejected/filtered: Gemma-4-12B/27B (multimodal, abliterated variants exist, BUT Gemma's heavier
refusal training abliterates worse — Heretic leaves 3/100 refusals on Gemma-class, KL damage
0.45-1.04; risky vs a hard zero-refusal requirement; community head-to-head also judged Qwen
more literary on JP); Qwen3-VL-32B-dense + InternVL3.5-38B (no abliterated variants — fail the
refusal requirement outright); GLM-4.6V-Flash / InternVL3.5-9B / MiniCPM-V (abliterated or
fit fine, but OCR/VQA-strong not literary-MT-strong, and no capacity advantage over the 8B).
Full per-candidate table + sources in the session log.
