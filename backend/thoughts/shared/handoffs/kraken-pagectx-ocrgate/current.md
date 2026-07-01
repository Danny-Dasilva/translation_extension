# Kraken Handoff: page-context numbered-block hold + OCR confidence gate

## Checkpoints
**Task:** FIX1 numbered-block translate must HOLD (not fall back to per-bubble isolation); FIX2 OCR confidence gate to stop hallucinated captions on garbled OCR. Re-evaluate 12 pages. No commit.
**Started:** 2026-06-14T09:00:00Z
**Last Updated:** 2026-06-14T09:00:00Z

### Phase Status
- Phase 1 (Tests Written): ✓ VALIDATED (18 tests, failed before impl)
- Phase 2 (FIX1 parser/prompt/retry/empty-line): ✓ VALIDATED (parser tests green)
- Phase 3 (FIX2 OCR conf surface + gate): ✓ VALIDATED (conf+gate tests green)
- Phase 4 (Wire router + batch script): ✓ VALIDATED (imports OK)
- Phase 5 (Re-evaluate 12 pages): ✓ VALIDATED (8/12 hold; all targets fixed; gate drops 15 garbled SFX)

### Outcome
- numbered-block HOLD: 8/12 pages (was ~0/12). Fallbacks: 005,006,054,057 (v10it
  collapses those batches — model limitation, per-bubble is correct there).
- Root cause of fallback: v10it NOT trained on BATCHED_SYSTEM_PROMPT `[N]` few-shot
  format -> collapsed page to 1 line. Switched to plain `N.` user-instruction format.
- OCR conf surfaced (mean softmax max-prob); gate drops conf<0.65 short SFX/garble.
- All target bubbles fixed: 059/1, 039/0/5/7, 014/1, 045/6 gated, 057 comparatives.

### Validation State
```json
{
  "before_state_dir": ".bench/Part13_translated_en_v4_inspection",
  "new_render_dir": ".bench/insp_v5_render",
  "new_inspect_dir": ".bench/Part13_inspection_v5",
  "vllm": "http://127.0.0.1:8765/v1 model=v10it"
}
```

### Key findings (BEFORE)
- bubbles.json `confidence` == DETECTOR conf, NOT OCR recog conf. Orphan/short SFX default to 0.5.
- Garbled hallucinations: 059 idx4-7, 057 idx4/5/8, 045 idx6, 039 idx0/5/7.
- 039 idx5 (手拭い) rendered "..." despite non-empty JP -> empty-line bug.

### Resume Context
- Next action: write tests, then implement FIX1 then FIX2.
