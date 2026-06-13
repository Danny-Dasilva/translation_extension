# kraken-v10_5-cpo handoff

**Task:** Train v10.5 by chaining CPO on top of v10-it (Gemma 4 E4B)
**Started:** 2026-05-07T22:00:00Z
**Last Updated:** 2026-05-08T17:25:00Z
**Status:** COMPLETED — DECISION: DO NOT SHIP v10.5

## Checkpoints

### Phase Status
- Phase 0 (Inventory + plan): ✓ VALIDATED
- Phase 1 (Build preference dataset): ✓ VALIDATED (11,901 pairs)
- Phase 2 (Train v10.5 CPO): ✓ VALIDATED (1995 s, VRAM peak 21.55 GB, adapter saved)
- Phase 3 (Eval v10.5): ✓ VALIDATED — REGRESSION vs v10-it on all 5 metrics
- Phase 4 (4-way HTML): ✓ VALIDATED (`/home/danny/manga-output/v7-v9c-v10it-v10_5.html`, 6.1 MB)
- Phase 5 (Final report): ✓ VALIDATED (returned to parent agent)

### Final Verdict — Pareto Gate FAILED

| Metric          | v10-it | v10.5  | Δ        |
|-----------------|--------|--------|----------|
| Gemma EM%       | 33.46  | 24.90  | **−8.56** |
| chrF++          | 70.91  | 63.89  | **−7.02** |
| BLEU            | 55.04  | 42.67  | **−12.37**|
| empty%          | 0.00   | 0.00   | 0.00     |
| JP-passthrough% | 0.00   | 0.39   | −0.39    |

0 of 5 strictly improved, 4 of 5 strictly regressed. **Do not ship.**

### Phase 2 Training Summary
- Base: `gemma4_e4b_v10it/merged` (15.6 GB, k_norm patched)
- Loss: CPOTrainer with `loss_type=simpo`, `cpo_alpha=1.0`, `simpo_gamma=0.5`, `beta=2.0`
- LR 1e-6, cosine, warmup 0.05, 1 epoch
- Effective batch 64 (per_device 2 × grad_accum 32)
- LoRA r=16, alpha=32, dropout=0.0, 258 lang-only modules ✓
- max_length 768, max_prompt_length 384
- Train rows: 10,901  Eval rows: 1,000 (tail)
- 171 total steps  Wall: 1994.6 s (33.2 min)
- VRAM peak: 21.55 GB
- train_loss final: 3.098
- best_model_checkpoint: checkpoint-120 (eval `rewards/margins`=−0.1834, the least-bad of a flat trajectory)

### Pre-Flight Outcomes
1. Module count: **258** (matches expected 258) ✓
2. Dry-run 100 steps / 500 samples: completed in 788 s, VRAM peak 21.14 GB
3. Dry-run train rewards/margins: noisy positive trend
4. Dry-run eval rewards/margins: 0.228 → 0.258 (clear win signal on randomized 50-row eval)
5. Dry-run smoke: **10/10 English** (no empty / JP-pass / repetition) ✓

### Why Did Eval Regress?
Two compounding causes:

1. **Eval split bias on full run.** The 11,901-row preference parquet wasn't shuffled before tail-splitting. The held-out 1000 rows are 44% teacher-chosens with mean length ~43 chars vs train's 4% teacher / mean ~93 chars. SimPO's length-normalized reward on this biased eval gave **flat negative margins** (−0.1834 to −0.1899 across all 17 evals), and `load_best_model_at_end` rolled back to checkpoint-120 (~70% through training) because the global maximum margin landed there.

2. **CPO loss moved the model away from the manga distribution.** Even setting aside checkpoint selection, the smoke test on real manga prompts showed obvious quality regressions ("Temporary housing?" for "葬儀屋とは" / "What is a funeral director?"). The CPO-SimPO loss with `beta=2.0` + `simpo_gamma=0.5` is aggressive and the preference pairs included a heavy teacher-chosen contingent (8% of train) that pulled the model toward shorter, blander outputs.

### Code Fixes During This Run (kept in repo)
1. `cpo_gemma4_e4b_v10_5.py:223-232` — unwrap multimodal `Gemma4Processor` to inner `GemmaTokenizer` before passing to CPOTrainer (TRL `tokenize_row` calls `processing_class(prompt_str)` which crashes inside the multimodal processor).
2. `cpo_gemma4_e4b_v10_5.py:295-302` — round `save_steps` down to a multiple of `eval_steps` (TRL requires this when `load_best_model_at_end=True`).

### Artifacts (all kept)
| Path | Purpose |
|------|---------|
| `backend/scripts/data/cpo/v10_5_preferences.parquet` | 11,901 pref pairs |
| `backend/scripts/train/cpo_gemma4_e4b_v10_5.py` | training script (working) |
| `backend/training/configs/gemma4_e4b_v10_5_cpo.yaml` | config (locked recipe) |
| `backend/training/runs/manga-bubbles/gemma4_e4b_v10_5/final/` | LoRA adapter (134 MB) |
| `backend/training/runs/manga-bubbles/gemma4_e4b_v10_5/merged/` | merged 15.91 GB |
| `backend/training/runs/manga-bubbles/gemma4_e4b_v10_5/v10_5-summary.json` | train metadata |
| `backend/training/runs/manga-bubbles/gemma4_e4b_v10_5/training.stdout.log` | full log |
| `backend/training/runs/manga-bubbles/gemma4_e4b_v10_5_dryrun/` | dry-run adapter |
| `/home/danny/manga-output/644289-gemma4-v10_5-unsloth/` | holdout translations |
| `/home/danny/manga-output/644289-gemma4-v10_5-unsloth/score_v10_5.json` | scores |
| `/home/danny/manga-output/v7-v9c-v10it-v10_5.html` | 4-way comparison HTML |
| `/tmp/v10_5_dryrun_smoke.jsonl` + `.summary.json` | dry-run smoke |
| `/tmp/v10_5_final_smoke.jsonl` + `.summary.json` | final smoke |

### Next Iteration Hypotheses (if user wants v10.6)
1. **Shuffle the parquet** before splitting to fix the eval-bias artifact (alone, won't change the absolute regression but will give a meaningful eval signal).
2. **Drop teacher-chosen rows** from CPO pairs — those are the ones pulling the model toward terse/bland outputs that don't match the gold-style manga voice. Keep gold-chosen + onpolicy-chosen only.
3. **Lower `beta` from 2.0 → 0.5** and/or **lower LR from 1e-6 → 2e-7**. Current settings are too aggressive and regress fluency.
4. **Disable `load_best_model_at_end`** or save `checkpoint-171` explicitly — the current model selection criterion is meaningless on a biased eval.
5. **Add real holdout-set eval mid-training** (not preference pairs — the actual manga 257-bubble set), and use Gemma EM as the early-stopping metric.
