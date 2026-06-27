# Kraken: ws-pipeline-conv — converge 3 pipeline copies + cross-bubble merge + decouple context

## Checkpoints
**Task:** (1) extract shared build_page_translation_units, (2) cross-bubble sentence merge, (3) decouple drop-from-render vs drop-from-context. Worktree /home/danny/Documents/personal/ws-pipeline (branch ws-pipeline-conv). Tests via main venv.
**Started:** 2026-06-26
**Last Updated:** 2026-06-26

### Phase Status
- Phase 1 (new modules + unit tests: sentence_merge.py, page_units.py): ✓ VALIDATED (38 tests passing)
- Phase 2 (wire shared helper into 3 call sites: translate.py pipelined+batch, batch_translate_chapter): ✓ VALIDATED (compiles, WS regression passes against refactored pipelined branch)
- Phase 3 (wire sentence-merge re-split into _run_translation/batch translate): ✓ VALIDATED (merge_req threaded both router + batch)
- Phase 4 (full suite parity + commit): → IN_PROGRESS (396 passed, 1 pre-existing unrelated failure test_unify_schema; ready to commit)

### Validation State
```json
{
  "new_modules": ["backend/app/utils/sentence_merge.py", "backend/app/utils/page_units.py"],
  "new_tests": ["backend/tests/unit/test_sentence_merge.py", "backend/tests/unit/test_page_units.py"],
  "config_added": "translation_sentence_merge (default True) in app/config.py",
  "tests_passing": 38,
  "last_test_command": "PYTHONPATH=. .../python -m pytest tests/unit/test_page_units.py tests/unit/test_sentence_merge.py -q",
  "last_test_exit_code": 0,
  "worktree_dep_fix": "copied app/utils/english_region.py from main into worktree (was untracked-only-in-main, broke translate.py import)"
}
```

### Resume Context
- Current focus: wire build_page_translation_units into translate.py copy (a) pipelined ~488-626, copy (b) batch else ~630-732, batch_translate_chapter ~502-573.
- Next action: refactor pipelined branch to collect raw (blocks,texts,confs) then call helper; preserve inpaint OVERLAP launch in router.
- Blockers: detector v26 ONNX model absent in worktree -> translate.py import fails at module load -> the import-level translate tests (test_ws_ocr_conf_suppression) can't run in worktree; new tests are pure-function so they run. Full GPU/translation validation is main-side.

### Shared-helper contract
build_page_translation_units(blocks, ocr_texts, ocr_confs, text_lines, settings, *, is_japanese_fn, is_leave_intact_fn=None, should_skip_as_english_fn=None, on_drop=None) -> PageTranslationUnits(kept_blocks, kept_texts, kept_confs, page_context_lines, target_positions, erase_only_blocks, kept_indices, merge_plan). .as_tuple() yields the 6-tuple.
Decision order: jp_filter -> leave_intact -> english_exit -> garble gate (erase_only via should_erase_dropped; context re-entry via is_dialogue_context_candidate).
