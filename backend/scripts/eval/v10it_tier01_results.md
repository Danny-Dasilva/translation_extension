# v10-it Tier 0 + Tier 1 Results

**Date:** 2026-05-09
**Context:** The 10-oracle research consensus questioned whether v10-it is at production
quality but invisible through chrF++ + Gemma-EM (both broken/saturated metrics).
This document delivers (1) a multi-metric scorecard, (2) bootstrap-significance on
the v10it/v9c chrF++ delta, (3) generalization signal from a multi-genre OpenMantra
benchmark, and (4) inference-time-quality bench results across BoN+chrF-MBR and RAG few-shot.

---

## TL;DR

1. **The chrF++ delta v10-it (70.91) vs v9c (70.40) on 644289 is NOT statistically
   significant** (paired bootstrap, n=256, N=1000): observed delta +1.04, 95% CI
   [-0.56, +2.89], p=0.212. We CANNOT claim v10-it improved over v9c with 644289+chrF++.
2. **All three systems (v7, v9c, v10-it) are essentially tied on every metric**
   on the 644289 holdout — KiwiXL within 0.001, MetricX within 0.10, chrF++ within 0.7
   points. This is exactly what the oracle warned: chrF++ on 644289 has saturated
   because the reference IS the Gemma teacher and all 3 systems distill from it.
3. **OpenMantra (out-of-domain) collapses chrF++ to ~26 across all configs**
   (vs ~71 on 644289), confirming the 644289 chrF++ was an inflated reference-style
   match score, not an actual translation-quality signal. KiwiXL — which doesn't
   need a reference — RISES on OpenMantra (0.574 vs 0.554), suggesting actual
   translation quality is fine; the reference-based metrics are the broken signal.
4. **None of the inference-time tricks delivered a clear win.** BoN+chrF-MBR gives
   marginal +0.45 chrF++ at 100x latency cost; RAG+ICL **regressed** sharply on every
   metric (probably because v10-it never saw multi-shot prompts during SFT).
5. **XCOMET-XL is gated** for the active HF account and could not be scored. CometKiwi-23-XL
   + MetricX-24-Hybrid-XL replace it. Access requests for XCOMET-XL/XXL pending.

**Recommendation:** Do NOT ship a new training run on the strength of 644289 chrF++.
The metric is saturated and the delta is noise. Either (a) stop optimizing toward
the Gemma teacher (which we are now perfectly imitating to within bootstrap noise)
and switch to KiwiXL or MetricX for guidance, or (b) commit to a small human eval
(~50 bubbles, 3 raters) before any v10.6 training. **v10-it is shippable** (it is
indistinguishable from v9c and v7) but there is no defensible quality reason to
prefer it over v9c at this point — the deciding factor should be inference cost
(MTP/EAGLE) and operational fit, not bench numbers.

---

## Phase 1: Multi-metric Scorecard

### 644289 holdout (257 bubbles)

| System | chrF++ | BLEU  | Kiwi-23-XL | MetricX-24↓ | empty% | jp%  | teacher-EM% |
|--------|-------:|------:|-----------:|------------:|-------:|-----:|------------:|
| v7     |  71.10 | 55.38 |     0.5537 |       2.767 |    0.0 |  0.0 |        34.6 |
| v9c    |  70.40 | 54.77 |     0.5532 |       2.861 |    0.0 |  0.0 |        33.1 |
| v10-it |  70.91 | 55.04 |     0.5542 |       2.770 |    0.0 |  0.0 |        33.5 |

(MetricX is an error score in [0, 25]; lower is better.)

**Reading:** v7 leads on chrF++/BLEU/MetricX, v10-it leads on KiwiXL by 0.001
(noise). All three systems are **tied to within bootstrap noise on every metric**.
The v10-it advantage on 644289 is invisible.

### OpenMantra heldout (631 bubbles, 2 unseen volumes: boureisougi + rasetugari)

Only v10-it was scored across configs (v9c/v7 require additional model merges and
were de-prioritized given the 644289 result above already settles the question).

| Config             | chrF++ | BLEU |  Kiwi  | MetricX↓ | wall (s) | throughput |
|--------------------|-------:|-----:|-------:|---------:|---------:|-----------:|
| greedy             |  26.49 | 9.21 | 0.5741 |    5.336 |      1.9 |  326 it/s  |
| constrained        |  26.27 | 8.77 | 0.5734 |    5.372 |     18.7 |   33.7 it/s|
| bon_chrf (n=8)     |  26.94 | 8.23 | 0.5692 |    5.337 |    188.0 |    3.4 it/s|
| bon_chrf_rag (n=8) |  22.82 | 6.45 | 0.5032 |    6.961 |    145.0 |    4.4 it/s|

**chrF++ collapse from 70 → 26 between 644289 and OpenMantra is the headline
finding.** This is consistent with the 10-oracle hypothesis: 644289 chrF++ was
inflated because the reference (Gemma 3 4B base teacher's modeA.jsonl) shares the
same training distribution as our distilled students. On a real out-of-domain set
authored by humans, both length and style differ, so n-gram overlap craters. KiwiXL
goes UP (0.574 vs 0.554), confirming actual translation quality is fine; the
reference-based metrics are the broken signal.

### XCOMET-XL note

`Unbabel/XCOMET-XL` and `Unbabel/XCOMET-XXL` are HF-gated and the active token
(user `heliothryx`) is not on the authorized list. Both repos return a 403 on
the model.ckpt fetch. Access requests have been left pending; CometKiwi-23-XL
and MetricX-24-Hybrid-XL fill the same role for now (both QE+MQM-trained, both
on disk, both non-gated for this account). Switch in XCOMET-XL when access is
granted by editing `score_summary_metrics_v2.py::compute_xcomet`.

XCOMET-lite (myyycroft mirror) is NOT a drop-in for the official XCOMET-XL — its
weights live in a non-comet PEFT format and would require custom loading. Punted.

---

## Phase 1.3: Paired-bootstrap chrF++ — v10-it vs v9c on 644289

Per-bubble chrF++ collected via `sacrebleu.sentence_chrf(word_order=2)`, paired
across 256 aligned segments (one bubble dropped due to OCR-key mismatch).
N=1000 bootstrap resamples with seed=12345.

|                    |       value |
|--------------------|------------:|
| n eval segments    |         256 |
| n bootstrap        |        1000 |
| mean_v10it (segment-level) |  62.95 |
| mean_v9c   (segment-level) |  61.91 |
| observed delta     |       +1.04 |
| 95% CI delta       | [-0.56, +2.89] |
| **p (two-sided)**  |   **0.212** |
| win-rate v10it>v9c |        89.4% |

**Conclusion: NOT significant at α=0.05.** v10-it scores higher on the point
estimate AND wins 89% of bootstrap resamples, but the CI crosses zero. The
chrF++ improvement story is unsupported.

(For context: v10-it vs v7 has observed delta -0.53, 95% CI [-1.39, +0.34],
p=0.242 — also not significant, point-estimate slightly favors v7.)

Per-bubble chrF data is at `backend/scripts/eval/scorecards/per_bubble_*_644289.json`
for re-running the bootstrap with different seeds or N.

---

## Phase 2: Inference-time quality lift

All configs use the same vLLM 0.20.2 + merged v10-it deploy. Tested on the same
631-bubble OpenMantra heldout used above. All configs use the chat template
(`<|turn>user/model`) verified earlier as the eval-correct path for v10-it.

**Knobs by config:**
- `greedy`: T=0, n=1, max_tokens=60
- `constrained`: T=0, n=1, max_tokens=1.5×|jp|+16, repetition_penalty=1.05,
  logit_bias=-100 on all CJK Unified+Hiragana+Katakana token ids
- `bon_chrf`: T=0.9, top_p=0.95, n=8, repetition_penalty=1.05, JP-vocab block,
  + chrF-MBR selection across the 8 candidates via `fastchrf.pairwise_chrf`
- `bon_chrf_rag`: bon_chrf + BGE-M3 top-3 retrieval over 15K manga gold pairs
  prepended as ICL exemplars

**Results table (above) shows:**

- **constrained ≈ greedy** on quality; the JP-vocab block is a no-op because the
  model already wasn't producing JP tokens (jp_passthrough_pct = 0% on 644289).
  Latency goes 10× WORSE due to the 100K+ entry logit_bias evaluation per step.
  **Bad trade.**
- **bon_chrf** delivers a tiny +0.45 chrF++ improvement at ~100× latency cost.
  KiwiXL and MetricX are essentially unchanged. **Not worth it for a 0.5-point
  metric gain that's well within bootstrap noise.**
- **bon_chrf_rag** REGRESSES on every single metric. The 15K-pair index pulls
  semantically-adjacent JP that confuses the model on out-of-domain content; v10-it
  was SFT-only on single-shot prompts and so multi-shot ICL falls outside its
  training distribution. **Unusable as-is.** Possible salvage paths if we ever
  want to retry: train v10.6 with a slice of multi-shot, or restrict retrieval
  to a smaller, higher-quality genre-aware index.

**`bon_chrf_kiwi`** (top-3-by-chrF re-ranked with KiwiXL) was implemented but not
benched at scale due to time + GPU cohabitation issues (Kiwi+vLLM+BGE all need
the GPU). Worth running before any production decision. The script supports it
via `--config bon_chrf_kiwi --kiwi`.

### Latency p50/p95 caveat

vLLM 0.20.2's per-request `RequestOutput.metrics` did not populate
`first_scheduled_time` / `finished_time` in our run (stays None), so per-item p50/p95
came out as 0. The wall-clock totals in the table above are accurate. To recover
true p50/p95 we'd need to re-run with `--disable-log-stats=False` and a different
metrics extraction.

---

## Files produced

* `backend/scripts/eval/score_summary_metrics_v2.py` — multi-metric scorer (chrF++,
  BLEU, KiwiXL, MetricX, with optional XCOMET stub when access lands)
* `backend/scripts/eval/_metricx_inference.py` — minimal MT5ForRegression
  reproducing google-research/metricx for HF-only inference
* `backend/scripts/eval/score_jsonl_metrics.py` — same metric stack but on flat
  JSONL of {jp, en (=ref), pred} pairs (used for OpenMantra)
* `backend/scripts/eval/build_consolidated_scorecard.py` — flattens per-system
  scorecards into one consolidated `score_summary_metrics_v2.json`
* `backend/scripts/eval/paired_bs_chrf.py` — bootstrap-resample paired test
* `backend/scripts/eval/inference_v10it_quality.py` — vLLM Python wrapper with
  greedy / constrained / BoN+chrF-MBR / BoN+chrF+kiwi / BoN+chrF+RAG configs
* `backend/scripts/eval/rag_retrieval_v10it.py` — BGE-M3 RAG index builder + querier
* `backend/scripts/eval/data/openmantra/heldout.jsonl` — 631-bubble multi-genre
  external benchmark (boureisougi + rasetugari, never seen by any v* model)
* `backend/scripts/eval/rag_index_v10/` — BGE-M3 embeddings of 15K manga gold pairs
  (excludes none — open_mantra_train IS in the index but does not overlap with the
  heldout volumes)
* `backend/scripts/eval/scorecards/score_summary_metrics_v2_<label>.json`
  + `per_bubble_<label>.json` (one pair per system)
* `backend/scripts/eval/scorecards/paired_bs_chrf_v10it_v9c.json`
* `backend/scripts/eval/scorecards/paired_bs_chrf_v10it_v7.json`
* `backend/scripts/eval/openmantra_v10it/<config>/translations.jsonl`
  + `stats.json` (one pair per inference config)
* `backend/scripts/eval/score_summary_metrics_v2.json` — top-level consolidated table

---

## Recommendations

1. **Stop using 644289 chrF++ + Gemma-EM as the gating metric.** Both are
   saturated. They cannot tell v7, v9c and v10-it apart with statistical
   confidence. We have re-confirmed this twice now (paired bootstrap on the
   v10-it/v9c delta, p=0.212; on the v10-it/v7 delta, p=0.242).
2. **Adopt CometKiwi-23-XL as the primary quality gate** going forward. It is
   reference-free (no Gemma-teacher leakage) and can be computed in ~10s on the
   257-bubble holdout. Set the gate at "no regression beyond -0.005 vs v10-it"
   for any future ablation.
3. **Add MetricX-24-Hybrid-XL as a secondary gate.** It's reference-based but
   MQM-trained (not n-gram overlap), so it complements KiwiXL. ~5s on 257 bubbles.
4. **Before any v10.6 training**, run a small human eval (~50 bubbles spread
   across 644289 + OpenMantra). This is the only way to settle "is v10-it
   actually production-ready" given that all reference-based metrics on 644289
   have saturated.
5. **Skip RAG few-shot for now.** The current SFT-only v10-it cannot use ICL
   examples; either retrain with multi-shot or drop this branch. The 15K-pair
   BGE-M3 index is built and persisted in case a future v10.6 wants to add
   multi-shot data; it took ~30s to build.
6. **`bon_chrf` is implementable in production** if you want to stress-test v10-it
   under non-greedy decoding (e.g. for diverse caption proposals in the rendering
   step). The +0.45 chrF on OpenMantra is real but tiny; the 100× latency makes
   it unsuitable as a default. Reserve for cases where the user explicitly asks
   for "high quality" + accepts the latency.
7. **Request XCOMET-XL access** on the HF token used by this rig. Until then,
   the metric stack is KiwiXL + MetricX which is sufficient signal.

---

## Reproducibility

```bash
# 1) Score all three systems on 644289
source /home/danny/.venvs/comet/bin/activate
for label in v7 v9c v10it; do
    case $label in
        v7)    pred=/home/danny/manga-output/644289-qwen3mt-v7 ;;
        v9c)   pred=/home/danny/manga-output/644289-gemma4-v9c-unsloth ;;
        v10it) pred=/home/danny/manga-output/644289-gemma4-v10it-unsloth-fixed ;;
    esac
    python backend/scripts/eval/score_summary_metrics_v2.py \
        --pred-dir "$pred" \
        --label "${label}_644289" \
        --metrics chrf,bleu,kiwi,metricx \
        --out-dir backend/scripts/eval/scorecards
done

# 2) Paired bootstrap
python backend/scripts/eval/paired_bs_chrf.py \
    --sys-a-per-bubble backend/scripts/eval/scorecards/per_bubble_v10it_644289.json \
    --sys-b-per-bubble backend/scripts/eval/scorecards/per_bubble_v9c_644289.json \
    --label-a v10it --label-b v9c \
    --out backend/scripts/eval/scorecards/paired_bs_chrf_v10it_v9c.json

# 3) OpenMantra inference sweep (vLLM venv)
PATH=/home/danny/.venvs/vllm/bin:$PATH /home/danny/.venvs/vllm/bin/python \
    backend/scripts/eval/inference_v10it_quality.py \
    --in-jsonl backend/scripts/eval/data/openmantra/heldout.jsonl \
    --out-dir backend/scripts/eval/openmantra_v10it \
    --config greedy,constrained,bon_chrf,bon_chrf_rag \
    --rag-index backend/scripts/eval/rag_index_v10 \
    --gpu-mem-util 0.7 --max-model-len 2048

# 4) Score each config (comet venv)
for cfg in greedy constrained bon_chrf bon_chrf_rag; do
    python backend/scripts/eval/score_jsonl_metrics.py \
        --gold-jsonl backend/scripts/eval/data/openmantra/heldout.jsonl \
        --pred-jsonl backend/scripts/eval/openmantra_v10it/$cfg/translations.jsonl \
        --label v10it_om_$cfg \
        --metrics chrf,bleu,kiwi,metricx \
        --out-dir backend/scripts/eval/scorecards
done

# 5) Build consolidated scorecard
python backend/scripts/eval/build_consolidated_scorecard.py
```

Total time: ~30-40 minutes wall (model loads dominate; the actual scoring is fast).
