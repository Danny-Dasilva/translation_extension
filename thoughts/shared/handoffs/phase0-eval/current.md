# Phase 0 Eval Pipeline - Kraken Handoff

**Task:** Implement Phase 0 of v10.6 plan (eval pipeline build-out, no training)
**Started:** 2026-05-07T00:00:00Z
**Last Updated:** 2026-05-07T01:00:00Z

## Checkpoints
<!-- Resumable state for kraken agent -->

### Phase Status
- 0.1 xCOMET branch in score_jsonl_metrics: ✓ VALIDATED (imports clean; execution deferred)
- 0.2 Lippmann pull: ✓ VALIDATED (clone+parse complete; dataset is JA-PL not JA-EN — dropped)
- 0.3 Groq GEMBA wiring + README: ✓ VALIDATED (dry-run smoke test passed)
- 0.4 Re-evaluate v10it/v9c/v7: PARTIAL (4-metric stack done on 257-bubble holdout for all 3
  systems; xCOMET deferred; OpenMantra cross-system deferred — v9c/v7 OM outputs not generated)
- 0.5 Generic paired bootstrap: ✓ VALIDATED (3 system-pair runs done on 257-bubble)
- 0.6 Final synthesis: ✓ VALIDATED (returned in assistant message; data in score_summary_4metric.json)

### Key finding
**MetricX-24-XL is the ONLY metric where v10it provably beats v9c**: Δ=−0.0917,
CI95=[−0.188, −0.009], p=0.026 SIG. chrF and Kiwi are saturated. v10it ≈ v7 on every
metric (statistical tie). v9c < v7 on chrF and MetricX (borderline trends, p=0.066, 0.062).

### Validation State
```json
{
  "vram_free_mib": 2400,
  "vram_total_mib": 32607,
  "xcomet_can_run": false,
  "xcomet_blocked_by": "specforge (EAGLE-3) using 27.2GB",
  "comet_version": "2.2.7",
  "files_modified": [
    "backend/scripts/eval/score_jsonl_metrics.py",
    "backend/scripts/eval/gemba_mqm_judge.py"
  ],
  "files_added": [
    "backend/scripts/eval/paired_bs_metric.py",
    "backend/scripts/eval/README_free_eval.md",
    "backend/scripts/data/lippmann_love_hina/parse_to_jsonl.py",
    "backend/scripts/data/lippmann_love_hina/heldout_ja_pl.jsonl",
    "backend/scripts/eval/scorecards/v10it_phase0/score_summary_4metric.json",
    "backend/scripts/eval/scorecards/v10it_phase0/paired_bs_v10it_v9c_644289.json",
    "backend/scripts/eval/scorecards/v10it_phase0/paired_bs_v10it_v7_644289.json",
    "backend/scripts/eval/scorecards/v10it_phase0/paired_bs_v9c_v7_644289.json",
    "backend/scripts/eval/scorecards/v10it_phase0/per_bubble_v10it_644289.json",
    "backend/scripts/eval/scorecards/v10it_phase0/per_bubble_v9c_644289.json",
    "backend/scripts/eval/scorecards/v10it_phase0/per_bubble_v7_644289.json",
    "backend/scripts/eval/scorecards/v10it_phase0/per_bubble_v10it_om_greedy.json"
  ]
}
```

### Resume tasks (require GPU free)
1. xCOMET-XL on 3 systems × 257-bubble holdout (and OpenMantra once v9c/v7 generated)
2. Generate v9c + v7 OpenMantra outputs via vLLM/Unsloth
3. Re-run paired_bs_metric with xcomet_xl included
4. Run GEMBA-MQM via Groq (user must export OPENAI_API_KEY + OPENAI_BASE_URL first)

Resume commands are documented in backend/scripts/eval/README_free_eval.md and the
score_summary_4metric.json file under "infrastructure_built".
