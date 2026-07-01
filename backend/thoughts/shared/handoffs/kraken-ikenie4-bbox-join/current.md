# Ikenie4 eval harness: bbox-spatial join (stable before/after)

## Checkpoints
**Task:** Replace jp-join with stable bbox-IoU spatial join in the Ikenie4 regression harness so every run scores the SAME gold-row set.
**Started:** 2026-06-27T00:25:00Z
**Last Updated:** 2026-06-27T00:45:00Z

### Phase Status
- Phase 1 (Tests Written): ✓ VALIDATED (10 tests, failed pre-impl as expected)
- Phase 2 (Implementation): ✓ VALIDATED (10/10 pass; 514 pass overall, 2 pre-existing fails)
- Phase 3 (Harness wiring + validation): ✓ VALIDATED (baseline+merged both score same 44 ocr_clean rows)

### Validation State
```json
{
  "test_count": 10,
  "tests_passing": 10,
  "files_modified": [
    "scripts/eval/build_predictions_for_gold.py (new)",
    "scripts/eval/score_jsonl_metrics.py (--align-key src)",
    "scripts/eval/prep_probe_predictions.py (--inspect-dir page-scoped)",
    "scripts/eval/paired_bs_metric.py (--align-key slug)",
    "scripts/eval/run_ikenie4_regression.sh (--inspect-dir contract)",
    "tests/unit/test_build_predictions_for_gold.py (new)"
  ],
  "last_test_command": ".venv/bin/python -m pytest tests/unit/test_build_predictions_for_gold.py -q",
  "last_test_exit_code": 0
}
```

### Corrected before/after (SAME 44 ocr_clean gold rows, bbox-joined)
- match-rate: baseline 44/44, merged 44/44 (was 33 vs 41 DIFFERENT rows under jp-join)
- corpus chrF++: baseline 4.60 -> merged 5.35
- paired bootstrap (slug-aligned, seed 12345): Δ(merged-baseline)=+0.686 per-sentence chrF++, CI95 [-0.68,+2.19], p=0.31 -> NOT significant
- probes (same 20 set): merged regresses (name_invention, sfx_meta_leak, reverse_sense) but 2 name_invention probes are join artifacts (baseline empty pred)
```
