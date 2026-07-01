# Handoff: v11fix6 corrective SFT (clean-OCR mistranslation fine-tune)

## Task
Build corrective SFT data from ikenie4 gold.jsonl (jp -> human_en, sentence-cased),
emit in v11 PAGE-CONTEXT shape using bench bubbles.json page context, mix with base
v11 parquet (corrective upweighted ~3x, minority of total, NSFW fraction FLAT), copy
v11 config -> gemma4_e4b_v11fix6_sft.yaml, smoke 20 steps on the box, then launch full
SFT detached. Train ON THE BOX (100.64.235.63), local 5090 busy serving v11.

## Key facts discovered
- gold.jsonl schema is NOT as described in task. Actual keys: jp (=ocr_jp), en
  (=human_en ALL-CAPS), our_en (our model output), src, bbox, ocr_clean, category,
  source_field, judge_note. 542 ocr_clean rows, all register_tag=manga_nsfw.
- gold src (ikenie4:pNN:idxK) joins to .bench/ikenie4_merged_insp/<NNN>/bubbles.json
  by page+idx; jp/our_en/bbox MATCH exactly (verified). bubbles.json gives full
  ordered page bubble list (ocr_jp + translation_en) -> page context.
- source_field: vision_gt=497 (general anchor), worst_issues=26 + gap_examples=19
  (explicit judge-labeled corrective signal, 45 total).
- 509/542 human en rows are ALL-CAPS -> must sentence-case.
- Base v11 parquet: backend/scripts/data/v11/data_v11_pagecontext.parquet (291,780
  rows, schema [prompt, en, src, register_tag, gold_flag]). vn_eroge=45,021 (15.4%
  NSFW-ish). Builder = scripts/data/v11/build_v11_dataset.py.
- Trainer = scripts/train/sft_gemma4_e4b_v10it.py (YAML-driven, prefers `prompt`
  column, chat-template, completion-only). NOT sft_gemma4_unsloth.py.
- BOX env: unsloth 2026.6.7, trl 0.23.1, peft 0.19.1, transformers 5.5.0, torch
  2.10.0+cu128, polars 1.41.2, yaml ok. loguru MISSING (trainer needs it - install).
  GPU free (86MiB/32GB). HF reachable (gemma-4-E4B-it 9 siblings -> will download).
  Box repo ~/manga-translate-train has ONLY venv -> must rsync trainer+config+parquet.
  Memory warning: unsloth wanted transformers 5.8.0; box has 5.5.0 -> VERIFY load.

## Checkpoints
**Task:** v11fix6 corrective SFT
**Started:** 2026-06-27T03:00:00Z
**Last Updated:** 2026-06-27T03:00:00Z

### Phase Status
- Phase 1 (Build corrective data): ✓ VALIDATED (12 contract tests pass; stats below)
- Phase 2 (Config + smoke 20 steps on box): ✓ VALIDATED (loss 1.32, 294 modules, masking ok)
- Phase 3 (Launch full SFT detached): ✓ VALIDATED (PID 2903455 alive, tokenizing 292k rows)

### Phase 3 results (full run LAUNCHED)
- Box: danny@100.64.235.63  PID=2903455 (setsid nohup, survives ssh close)
- LOG: /home/danny/Documents/personal/extension/backend/training/runs/manga-bubbles/v11fix6_full_20260627_031920.log
- OUT: .../training/runs/manga-bubbles/gemma4_e4b_v11fix6_pagecontext/final (adapter)
- config: training/configs/gemma4_e4b_v11fix6_sft.yaml (expected_module_count=294)
- 292,370 train / 1,000 eval; 294 LoRA modules; ~5h expected (v11 ref = 5.0h).
- MONITOR:  sshpass -p danny ssh danny@100.64.235.63 \
    "tail -5 <LOG> | tr '\\r' '\\n' | tail -5; kill -0 2903455 && echo ALIVE"
  loss:    grep -E \"'loss'|epoch\" <LOG>

### POST-RUN GATE (follow-up; do NOT keep SFT unless it passes)
1. MERGE LoRA->merged (on box):
   ~/manga-translate-train/.venv-training/bin/python \
     backend/scripts/eval/merge_gemma4_lora_clean.py \
     --adapter .../gemma4_e4b_v11fix6_pagecontext/final \
     --out     .../gemma4_e4b_v11fix6_pagecontext/merged
2. RESTORE k_norm (memory: merged drops 18 k_norm tensors for KV-shared layers 24-41,
   vLLM aborts without them):
   ~/manga-translate-train/.venv-training/bin/python \
     backend/scripts/eval/patch_v10it_merged_knorm.py \
     --merged .../merged --base unsloth/gemma-4-E4B-it --out .../merged_fixed
   (verify_merged_gemma4.py to sanity-check)
3. SERVE merged_fixed via vLLM, re-render ikenie4, then GATE:
   backend/scripts/eval/run_ikenie4_regression.sh \
     --inspect-dir <new v11fix6 render> --label v11fix6 \
     --baseline-per-bubble backend/scripts/eval/scorecards/ikenie4/per_bubble_merged_vg.json \
     --baseline-probes     backend/scripts/eval/scorecards/ikenie4/probes_*.json
   KEEP ONLY IF: chrF++ Δ CI95 excludes 0 (statistical win) AND no probe regresses.
   Baseline = v11 scorecard per_bubble_merged_vg.json (the NEW vision_gt gold).

### Phase 2 results (smoke)
- COMPAT: model LOADS on transformers 5.5.0 (the 5.8.0 memory note does NOT block load here).
- Arch drift resolved: current unsloth/gemma-4-E4B-it = 42 lang layers -> 294 LoRA
  modules (language-only, no vision/audio leak). v11 config had stale 258; BOTH the
  merged v11 AND v10it checkpoints are num_hidden_layers=42, so 294 is correct full
  coverage = SAME arch v11 trained on. Updated expected_module_count 258 -> 294.
- 20 steps OK: train_loss 1.32, eval_loss 2.34, 36.7M trainable (0.457%), VRAM 19.4GB.
- prompt byte-exact v11 page-context confirmed in mask-check; completion-only masking ok.
- v11 reference full run: 5.0h on 290,780 rows -> v11fix6 (292,370) ~= 5h expected.

### Phase 1 results (corrective_stats.json)
- usable corrective: 530 (542 clean - 12 annotation fragments dropped)
- divergent=514, anchors_kept=16; pagectx=312, plain=218
- 3x upweight -> 1,590 corrective rows = 0.54% of 293,370 total (MINORITY)
- NSFW frac: base 15.43% -> mixed 15.89% (Δ +0.46pp, FLAT, NOT oversampled)
- normalization: 480 recased all-caps->sentence; I-forms + embedded emphasis fixed
- parquet: backend/scripts/data/v11fix6/data_v11fix6_pagecontext.parquet
- tests: backend/tests/unit/test_v11fix6_corrective.py (12 pass)

### Validation State
```json
{"test_count": 12, "tests_passing": 12, "files_modified": ["scripts/data/v11fix6/build_v11fix6_corrective.py", "tests/unit/test_v11fix6_corrective.py"], "last_test_command": ".venv/bin/python -m pytest tests/unit/test_v11fix6_corrective.py -q", "last_test_exit_code": 0}
```

### Resume Context
- Current focus: create gemma4_e4b_v11fix6_sft.yaml, rsync to box, smoke 20 steps
- Next action: scp trainer+config+parquet+build_v11_dataset to box; pip install loguru; smoke
- Blockers: box transformers 5.5.0 (memory wanted 5.8.0) -> VERIFY model load in smoke
