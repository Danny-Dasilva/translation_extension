# v11fix6 SFT Eval Gate

## Checkpoints
<!-- Resumable state for kraken agent -->
**Task:** Run v11fix6 SFT eval gate, report whether it beats v11 baseline (paired-bootstrap chrF++ + probes)
**Started:** 2026-06-27T20:20:00Z
**Last Updated:** 2026-06-27T20:20:00Z

### Phase Status
- Phase 1 (Merge + k_norm): ✓ VALIDATED (294 LoRA pairs merged, 16.03GB; ALL 42 k_norm keys present under tf5.5.0 -> NO knorm-extra needed, unlike v11 tf5.8.0; merged_fixed built with symlinked shards)
- Phase 2 (Serve on box): ✓ VALIDATED (ready @ http://100.64.235.63:8001/v1, served-model v10it, smoke test clean EN output). SERVE FIX: flashinfer JIT sampler tripped the broken "SM 12.x requires CUDA>=12.9" probe on this cu130/Blackwell box -> set VLLM_USE_FLASHINFER_SAMPLER=0 + TORCH_CUDA_ARCH_LIST=12.0 + VLLM_ATTENTION_BACKEND=TRITON_ATTN + --enforce-eager
- Phase 3 (Re-render Ikenie4 local): ✓ VALIDATED (134/134 pages, all 119 gold pages non-empty. NOTE: first pass OOM'd pages 113-134 because local :8000 v11 vllm holds 20GB on the shared GPU; re-rendered those 22 pages with ORT_GPU_MEM_LIMIT_BYTES=3GB + PARSEQ_BATCH_SIZE=1, 0 errors. Did NOT kill user's local server.)
- Phase 4 (Eval vs v11 baseline): ✓ VALIDATED (paired-bs chrF++ v11=11.60 -> v11fix6=12.75, Δ=+1.152 CI95=[-0.007,+2.277] p=0.052 -> NOT sig; ALL 5 probes IMPROVED, 0 regressions)
- Phase 5 (Verdict + kill box vllm): ✓ VALIDATED (box vllm PID 2958577 killed, GPU back to 86 MiB / 0% util)

### RESULT (MIXED / NO-WIN on strict gate)
- chrF++ paired-bootstrap (seed 12345, n=541): v11=11.601, v11fix6=12.753, Δ=+1.152, CI95=[-0.007,+2.277], p=0.052, win_rate_a=0.974
- Strict gate FAILS: CI95 lower bound = -0.007 just includes 0 (so close it nearly excludes). NOT statistically significant at 0.05.
- Probes (all IMPROVED, zero regressions): reverse_sense 0.000->0.571(+0.571,4/7), pronoun_gender 0.500->1.000(+0.500), name_invention 0.000->1.000(+1.000), sfx_meta_leak 0.333->1.000(+0.667), number_romaji 0.500->1.000(+0.500)
- Likely cause of not-quite-significant chrF: liberal targets in this SFT corpus. Retry = tightened 45-row corrective set.

### Key facts
- v11fix6 adapter (box): ~/Documents/personal/extension/backend/training/runs/manga-bubbles/gemma4_e4b_v11fix6_pagecontext/final
- base: unsloth/gemma-4-E4B-it, r=16 alpha=32 scaling=2.0, lang-only targets (same family as v11)
- base snapshot (box): ~/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it/snapshots/0d5a7f9ba73eda1616e58344f7025fae44914675/model.safetensors
- merge venv (box): ~/manga-translate-train/.venv-training/bin/python (tf 5.5.0, Gemma4ForCondGen imports OK)
- serve venv (box): ~/.venvs/vllm/bin/vllm (tf 5.12.0, torch 2.11 cu130)
- merge scripts copied to box: ~/Documents/personal/extension/backend/scripts/eval/{merge_gemma4_lora_clean,patch_v10it_merged_knorm,verify_merged_gemma4}.py
- v11 reference merged_fixed structure (LOCAL): symlinked 4 shards -> ../merged/, + model-knorm-extra.safetensors (18 k_norm tensors layers 24-41), + index.json updated, + config/chat_template/tokenizer copied
- BASELINE scorecards (v11 vision-gold): scripts/eval/scorecards/ikenie4/per_bubble_merged_vg.json + probes_merged_vg.json (NOT backend/scorecards/ — harness default OUT_DIR)
- regression harness OUT_DIR default = scripts/eval/scorecards/ikenie4

### Resume Context
- Current focus: Phase 1 - merge v11fix6 LoRA -> base on box, then knorm patch -> merged_fixed
- Next action: run merge_gemma4_lora_clean.py on box (training venv), then patch_v10it_merged_knorm.py, build merged_fixed with symlinks
- Blockers: none
