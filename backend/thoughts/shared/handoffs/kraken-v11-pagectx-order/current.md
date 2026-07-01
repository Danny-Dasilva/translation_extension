# Handoff: v11 page-context serving order + whole-page context

## Task
Fix v11 serving page-context to match the v11 TRAINING format:
1. Column-major RTL block ordering (mirror build_v11_dataset.manga_reading_order) applied to FULL merged block list (real+orphan) in both pipelines.
2. Pass the WHOLE page (all dialogue lines in reading order) as context; only request/return kept lines. Exclude pure-SFX boxes.
3. Confirm both router + batch_translate take this path (no silent fallback).

## Checkpoints
**Task:** v11 page-context order + whole-page context
**Started:** 2026-06-26T18:21:06Z
**Last Updated:** 2026-06-26T18:21:06Z

### Phase Status
- Phase 1 (Tests Written): ✓ VALIDATED (tests failed w/ ImportError)
- Phase 2 (Implementation): ✓ VALIDATED (390 pass, 2 pre-existing unrelated fails)
- Phase 3 (Validation vs live vLLM): ✓ VALIDATED

### Phase 3 results
- before/after Page prompt dumped: column-major RTL + orphan in-position confirmed
- serving build_v11_context_prompt == training prompt: 2362/2362 byte-exact (normalize OFF)
- heldout page-context chrF++ on v10it (n=120): 52.85 (format runs clean, no collapse)
- IK4 p5 "昨日あんな事をしていた" still -> "He": LIVE MODEL IS v10it, not v11. A/B
  proved v10it cannot flip He->She from CROSS-LINE page context (only when お母さん
  is in-line). Serving change is correct+complete; quality win needs v11 served.
- whole-page context now correctly INCLUDES high-conf お母さん dialogue line,
  EXCLUDES conf-0.49/0.53 garble (improved is_dialogue_context_candidate w/ conf).

### Validation State
```json
{"test_count": 390, "tests_passing": 388, "files_modified": ["app/utils/orphan_lines.py", "app/utils/ocr_confidence_gate.py", "app/services/vllm_openai_translation_service.py", "app/config.py", "app/routers/translate.py", "scripts/batch_translate_chapter.py", "tests/unit/test_reading_order_sort.py", "tests/unit/test_page_context_wholepage.py", "tests/unit/test_dialogue_context.py"], "last_test_command": ".venv/bin/python -m pytest tests/unit/ -q", "last_test_exit_code": 1}
```

### Resume Context
- Current focus: dump before/after Page prompt + re-translate IK4 pages vs live vLLM
- Next action: run batch_translate_chapter on IK4 pages 5,8,13,14,16; compare pronoun/split fixes; chrF++ heldout
- Blockers: none (2 failing tests are pre-existing: test_unify_schema register_tag + test_logging_async request_id)
