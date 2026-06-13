# Kraken: manga-translation backend SPEED wins

## Checkpoints
**Task:** Implement all quick+medium speed wins + R1 inpaint-reduction in backend/
**Started:** 2026-06-13
**Last Updated:** 2026-06-13

### Phase Status
- Phase 1 (Config flags): ✓ VALIDATED
- Phase 2 (Quick wins: WebP, parallel detect, warmup, max_tokens, semaphore): ✓ VALIDATED
- Phase 3 (Medium wins: overlap inpaint, R1 interior-fill, numbered-batch, fontHeight, colors): ✓ VALIDATED

### Validation State
```json
{
  "import_app_main": "OK",
  "ast_all_edited_files": "OK",
  "unit_tests": "17 passed, 1 pre-existing unrelated failure (test_unify_schema garbage tag)",
  "lama_helpers_behavioral": "OK (fill flat, reject noisy, trim dark minority)",
  "numbered_parser": "OK",
  "webp_plate_ratio": "0.167 vs PNG",
  "files_modified": [
    "backend/app/config.py",
    "backend/app/routers/translate.py",
    "backend/app/services/lama_inpaint_service.py",
    "backend/app/services/detector_service.py",
    "backend/app/services/vllm_openai_translation_service.py",
    "backend/app/main.py"
  ]
}
```
