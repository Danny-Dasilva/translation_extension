# Handoff: vertical-AR-by-default OCR routing (fix7)

## Checkpoints
**Task:** Route tall/narrow vertical crops to AR PARSeq by default (fix dense-kana NAR garble)
**Started:** 2026-06-26
**Last Updated:** 2026-06-26

### Phase Status
- Phase 1 (Tests Written): ✓ VALIDATED (7 tests, failed red as expected)
- Phase 2 (Implementation): ✓ VALIDATED (20/20 OCR tests green; 396 suite pass, 2 pre-existing fails)
- Phase 3 (GPU OCR A/B validation): ✓ VALIDATED (garble 5->0 on p5/45/123)
- Phase 4 (E2E pipeline + He->She probe): ✓ VALIDATED (OCR recovers; He->She does NOT flip — honest negative)

### Validation State
```json
{
  "test_count": 7,
  "tests_passing": 7,
  "suite_total": 398,
  "suite_passing": 396,
  "suite_preexisting_fails": ["test_logging_async.py", "test_unify_schema.py"],
  "files_modified": [
    "backend/app/services/parseq_ocr_service.py",
    "backend/app/config.py",
    "backend/app/registry.py",
    "backend/scripts/batch_translate_chapter.py",
    "backend/tests/unit/test_parseq_vertical_ar.py",
    "backend/scripts/validate_vertical_ar_ocr.py"
  ],
  "last_test_command": ".venv/bin/python -m pytest tests/unit/ -q",
  "last_test_exit_code": 1
}
```

### Resume Context
- Done. The OCR fix lands and recovers garbles; the He->She page-context payoff
  does NOT materialize (v11 model limitation, evidenced).
- Open thread: He->She needs a model/context-reasoning fix, not an OCR fix.
