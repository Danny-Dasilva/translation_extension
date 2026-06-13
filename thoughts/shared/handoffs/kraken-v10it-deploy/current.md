# v10-it Deploy + Eval Chain

## Checkpoints
**Task:** Full v10-it deploy + eval (merge → Unsloth eval → vLLM bench → 3-way HTML)
**Started:** 2026-05-07T21:30:00Z

### Phase Status
- Phase 1 (Merge LoRA): → IN_PROGRESS
- Phase 2 (Unsloth eval 257-bubble): ○ PENDING
- Phase 3 (vLLM server up): ○ PENDING
- Phase 4 (vLLM bench): ○ PENDING
- Phase 5 (3-way HTML): ○ PENDING

### Validation State
```json
{
  "v9c_baseline_tps": 77.81,
  "v7_match_gemma": "89/257 (34.6%)",
  "v9c_match_gemma": "85/257 (33.1%)",
  "pareto_gate_target": ">=36% Gemma exact-match"
}
```

### Resume Context
- Adapter: backend/training/runs/manga-bubbles/gemma4_e4b_v10it/final
- Merged out: backend/training/runs/manga-bubbles/gemma4_e4b_v10it/merged
- Eval set: 257 bubbles across 45 pages (slugs from v7/v9c overlap with Gemma teacher reference)
- Reference: /home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl
