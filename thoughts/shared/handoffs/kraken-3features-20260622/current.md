# Kraken: 3 user-facing extension features

## Checkpoints
**Task:** Async logging + On/Off translation toggle + Flag-poor-translation
**Started:** 2026-06-22
**Last Updated:** 2026-06-22

### Phase Status
- Phase 1 (Feature 1 — async logging on hot paths): ✓ VALIDATED (0 console.* left in content-script/overlay-renderer/service-worker; logger used 19/5/20x)
- Phase 2 (Feature 2 — On/Off toggle): ✓ VALIDATED (popup switch listener wired; SW/content/settings already scaffolded)
- Phase 3 (Feature 3 — flag poor translation): ✓ VALIDATED (api-client.flagTranslation + SW handler + overlay ⚑ button + content listener)
- Phase 4 (tsc + build): ✓ VALIDATED (tsc 26→25 errors, 0 new; build:all Chrome+Firefox green)

### Validation State
```json
{
  "tsc_baseline_errors": 26,
  "tsc_new_errors": 25,
  "new_error_types_introduced": 0,
  "build_chrome": "dist-chrome OK",
  "build_firefox": "dist-firefox OK",
  "last_commands": ["pnpm exec tsc --noEmit", "pnpm run build:all"]
}
```

### Files Modified
- src/content/content-script.ts (logging migration + flag-image listener + FlagRequest import)
- src/services/overlay-renderer.ts (logger import+migration; flagged state; ⚑ button; buildFlagPayload/requestFlag)
- src/services/api-client.ts (flagTranslation method + Flag types import)
- src/popup/popup.ts (master switch change listener)
- src/content/overlay.css (flag button styles)

### Already-scaffolded (verified, not duplicated)
- service-worker.ts: logger, applyTranslationToggle, get/setTranslationEnabled, flagTranslation handler, postFlag
- settings-manager.ts: is/setTranslationEnabled
- settings.ts: translationEnabled default true
- api.ts: FlagBox/FlagRequest/FlagResponse
- popup.html: master-switch markup
- content-script.ts: applyEnabledState + toggle/setTranslationEnabled message handlers
