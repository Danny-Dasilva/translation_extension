# Kraken Handoff: PARSeq single-line / production-regime re-test

## Checkpoints
**Task:** Re-test batched OCR model B vs reference A on SINGLE-LINE production regime (Test1 GT single-line, Test2 real CTD crops + artifact scan)
**Started:** 2026-06-14T08:03:02Z
**Last Updated:** 2026-06-14T08:03:02Z

### Phase Status
- Phase 1 (Write eval script): ✓ VALIDATED
- Phase 2 (Test 1 GT single-line run): ✓ VALIDATED (2000 rows; B FAIL +1.04pp CER)
- Phase 3 (Test 2 CTD crops + artifact scan): ✓ VALIDATED (487 crops; B-only artifact 1.23%)
- Phase 4 (Write results JSON + doc section): ✓ VALIDATED

### Verdict
DO NOT SHIP B. OOD hypothesis rejected: B regresses on single-line GT (+1.04pp CER,
must be <=0.5) AND introduces B-only repeat artifacts on real CTD crops (1.23%).

### Validation State
```json
{
  "single_line_rows": 56323,
  "single_line_vertical": 53683,
  "single_line_horizontal": 2640,
  "nhentai_dirs_resolved": true,
  "model_b": "parseq_manga_ep60_nonAR_dynbatch.fp16.onnx"
}
```

### Resume Context
- Current focus: writing single-line retest script
- Next action: run Test 1 then Test 2
- Blockers: none
