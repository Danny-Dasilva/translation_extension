## Checkpoints
**Task:** P2-3 inpaint erase-without-replace guard + over-broad mask clamp
**Started:** 2026-06-16T16:10:00Z
**Last Updated:** 2026-06-16T16:20:00Z

### Phase Status
- Phase 1 (Tests Written): VALIDATED (12 tests, failed pre-impl on ImportError)
- Phase 2 (Implementation): VALIDATED (12/12 guard tests pass)
- Phase 3 (Validation): VALIDATED (105 pass, 1 pre-existing unrelated fail: test_unify_schema register tags)

### Validation State
- test_inpaint_guard.py: 12 passed
- full suite: 105 passed, 1 pre-existing failure (test_unify_schema, unrelated to inpaint)
- files: app/services/lama_inpaint_service.py, tests/unit/test_inpaint_guard.py
- UNCOMMITTED per instructions
