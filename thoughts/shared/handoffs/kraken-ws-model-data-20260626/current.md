# Handoff: ws-model-data (4 model-side / data items)

## Checkpoints
<!-- Resumable state for kraken agent -->
**Task:** fix6 page-context corrective shape; reverse-sense corrective data; voice/addressee probe; cast-anchor A/B serve mechanism
**Branch:** ws-model-data (worktree /home/danny/Documents/personal/ws-model)
**Started:** 2026-06-26T00:00:00Z
**Last Updated:** 2026-06-26T00:00:00Z

### Phase Status
- Phase 1 (Tests Written): ✓ VALIDATED (31 tests, collection errors confirmed missing symbols)
- Phase 2 (Implementation): ✓ VALIDATED (31/31 new pass, byte-identity proven)
- Phase 3 (Validation/Refactor): ✓ VALIDATED (385 passed; 2 pre-existing fails unrelated)
- Phase 4 (Commit): → IN_PROGRESS

### Validation State
```json
{
  "test_count": 31,
  "tests_passing": 31,
  "files_modified": [
    "backend/app/config.py",
    "backend/app/services/vllm_openai_translation_service.py",
    "backend/scripts/data/v11/build_v11_dataset.py",
    "backend/scripts/data/v11/build_reverse_sense_corrective.py",
    "backend/scripts/data/v11/build_voice_addressee_probe.py",
    "backend/scripts/data/v11/extract_mistranslation_pairs.py",
    "backend/tests/unit/test_cast_anchor_prompt.py",
    "backend/tests/unit/test_v11_corrective_shapes.py"
  ],
  "last_test_command": "PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_cast_anchor_prompt.py tests/unit/test_v11_corrective_shapes.py -q",
  "last_test_exit_code": 0
}
```

### Pre-existing failures (NOT mine)
- test_ws_ocr_conf_suppression.py: imports app.utils.english_region (absent on this branch; translate.py owned by other workstream)
- test_unify_schema::test_all_register_tags_declared: VALID_REGISTER_TAGS vs hardcoded expected; does not read my files

### Resume Context
- venv: cd /home/danny/Documents/personal/ws-model/backend && PYTHONPATH=. /home/danny/Documents/personal/extension/backend/.venv/bin/python -m pytest ...
- extract_mistranslation_pairs.py is NOT on ws-model-data; it lives on branch fix6-clean-ocr-finetune (commit 8a80366). Saved copy at /tmp/extract_mistranslation_pairs.py.
- Key targets:
  - build_v11_dataset.corrective_rows() (line 332) is plain-only -> fix6 shape bug
  - build_v11_context_prompt() vllm service line 147 -> cast anchor insert point
  - config.py settings flags around line 192-227
