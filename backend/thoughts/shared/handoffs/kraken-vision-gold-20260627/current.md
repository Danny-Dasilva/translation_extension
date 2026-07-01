# Kraken: Vision-Gold Pass (Ikenie4)

## Checkpoints
**Task:** One-time vision-gold pass: transcribe human EN from GT scanlation images,
align to OUR bubbles, produce faithful human_en gold; merge with seed77; re-run eval.
**Started:** 2026-06-27
**Last Updated:** 2026-06-27

### Phase Status
- Phase 1 (Tests Written): ✓ VALIDATED (19 unit tests, failed pre-impl)
- Phase 2 (Implementation): ✓ VALIDATED (19/19 green; p5 align verified; SFX salvage added)
- Phase 3 (Full 134-page VLM pass): ✓ VALIDATED (126 pages, 0 failures after salvage; 612 vision rows / 521 clean)
- Phase 4 (Merge gold + re-run eval): ✓ VALIDATED (gold 77->650, 542 clean; eval match_rate=1.0; chrF++=11.60 lowercased on 541 clean; 5 spot-checks OK)

### Result
- gold.jsonl: 77 -> 650 (40 worst_issues + 37 gap_examples + 573 vision_gt); 542 ocr_clean
- gold.seed77.jsonl backup written
- Eval: 541/541 matched a bubble; chrF++=11.60 (case-insensitive; ALL-CAPS gold needed --lowercase)
- Key insight: ALL-CAPS typeset gold needs case-insensitive chrF (2.77 -> 11.60)

### Validation State
```json
{
  "test_count": 17,
  "tests_passing": 17,
  "files_modified": [
    "scripts/eval/transcribe_gt_vision.py",
    "tests/unit/test_transcribe_gt_vision.py"
  ],
  "last_test_command": ".venv/bin/python -m pytest tests/unit/test_transcribe_gt_vision.py -q",
  "last_test_exit_code": 0
}
```

### Resume Context
- VLM live at http://100.64.235.63:8001 model "qwenvl" (do NOT kill).
- p5 spot-check verified: idx2->NO IT'S NOT, idx3->MOM'S BECOME A SUBSTITUTE (correct).
- Key fix: reading-order 2-opt corrects stacked-bubble swaps (manga R-to-L vs EN top-down).
- Next: full pass --pages 1-134 -> gold_full.jsonl; then merge keeping seed77; backup to gold.seed77.jsonl; re-run run_ikenie4_regression.sh --label merged_vg.
