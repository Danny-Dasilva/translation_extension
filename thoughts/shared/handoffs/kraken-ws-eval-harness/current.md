# Ikenie4 MT Regression-Eval Harness (Tier-0 meta-fix)

## Checkpoints
<!-- Resumable state for kraken agent -->
**Task:** Deterministic, reproducible MT regression-eval harness (Ikenie4 gold set + seeded bootstrap + deterministic probes) to replace stochastic 24-agent LLM-judge comparison.
**Branch:** ws-eval-harness (worktree /home/danny/Documents/personal/ws-eval)
**Started:** 2026-06-26
**Last Updated:** 2026-06-26

### Phase Status
- Phase 1 (Freeze gold set + build script): ✓ VALIDATED (77 rows, 45 ocr_clean / 32 dirty, all matched to bubbles.json; byte-identical across reruns)
- Phase 2 (Vision transcribe skeleton + p41 offset): ✓ VALIDATED (dry-run resolves GT images across p41 boundary correctly)
- Phase 3 (Extend probes.py + unit tests): ✓ VALIDATED (29 probe tests pass; 5 new seedless gold probes)
- Phase 4 (run_ikenie4_regression.sh + prep helper): ✓ VALIDATED (end-to-end run emits chrF on ocr_clean subset + seeded paired bootstrap + probe table + PASS/FAIL verdict)
- Phase 5 (End-to-end validate + determinism): ✓ VALIDATED (self-compare CI~0; simulated-fix Δ=+35.29 CI95=[+21.22,+49.58] p=0 SIG; gold build deterministic)

### Validation State
```json
{
  "gold_rows": 77,
  "gold_ocr_clean": 45,
  "gold_dirty": 32,
  "probe_cases": 20,
  "probe_tests_passing": "29/29",
  "self_compare_ci": "[0.000, 0.000]",
  "simulated_fix_delta": 35.289,
  "simulated_fix_ci95": "[+21.222, +49.580]",
  "simulated_fix_pvalue": 0.0,
  "last_test_command": "PYTHONPATH=. .venv/bin/python -m pytest backend/tests/test_gold_probes.py backend/tests/test_probes.py -q",
  "last_test_exit_code": 0
}
```

### Files
- backend/scripts/eval/build_ikenie4_gold.py        (regenerates gold.jsonl from comparison + bubbles.json; ocr_clean classifier; p41 offset helper)
- backend/scripts/eval/transcribe_gt_vision.py      (one-time vision GT-OCR skeleton; p41 offset baked into resolve_gt_image_path; vision call stubbed)
- backend/scripts/eval/prep_probe_predictions.py    (joins predictions to probe cases by jp)
- backend/scripts/eval/run_ikenie4_regression.sh    (the headline runner: chrf on ocr_clean + seeded bootstrap + probes + verdict)
- backend/scripts/eval/probes.py                    (EXTENDED: reverse_sense, pronoun_gender, name_invention, sfx_meta_leak, number_romaji)
- backend/scripts/eval/data/ikenie4/gold.jsonl      (FROZEN 77-row gold set; un-ignored in .gitignore)
- backend/scripts/eval/data/ikenie4/probes.jsonl    (FROZEN 20 probe cases)
- backend/tests/test_gold_probes.py                 (NEW: 26 tests for the 5 gold probes)
- backend/tests/test_probes.py                      (1-line fix: skip n=0 probes in all-pass assertion)

### Resume Context
- Current focus: COMPLETE. Ready to commit.
- Next action: commit to ws-eval-harness.
- Vision pass still needs: wire a vision-language model into transcribe_gt_vision._vision_transcribe_page (the only stub); offset + alignment + IO are done and dry-run-verified.
```
