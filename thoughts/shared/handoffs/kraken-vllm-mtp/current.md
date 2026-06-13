# vLLM + Gemma4 MTP deploy infra prep (Path A)

## Task
Pre-stage vLLM with official Gemma 4 MTP support so v10-it serve+bench is unblocked
the moment training finishes (~4-5h). Deliver:
- Isolated venv at `/home/danny/.venvs/vllm/`
- vLLM nightly with PR #41745 (merged 2026-05-06) for `mtp` speculative method
- Pre-staged serve script at `backend/scripts/eval/serve_v10it_vllm.sh`
- Pre-staged bench script at `backend/scripts/eval/bench_vllm_v10it.py`
- All work CPU-only (training holds GPU)

## Checkpoints
**Started:** 2026-05-08T01:00:00Z
**Last Updated:** 2026-05-08T01:15:00Z

### Phase Status
- Phase 1 (Investigate vLLM source/branch):  VALIDATED
- Phase 2 (Create Python 3.11 venv):  VALIDATED
- Phase 3 (Install vLLM nightly + torch cu130):  VALIDATED (~6m, 7.5 GB venv)
- Phase 4 (Verify install + MTP support):  VALIDATED
- Phase 5 (Write serve_v10it_vllm.sh):  VALIDATED
- Phase 6 (Write bench_vllm_v10it.py):  VALIDATED
- Phase 7 (Final report):  IN_PROGRESS

### Validation State
```json
{
  "vllm_pr_41745_state": "MERGED 2026-05-06T14:39:30Z, commit 27e0057ae",
  "vllm_main_HEAD": "09a7cc5ba94c (2026-05-07T23:10Z)",
  "latest_stable_release": "v0.20.1 (2026-05-04, BEFORE the merge — does NOT have MTP)",
  "nightly_index": "https://wheels.vllm.ai/nightly/{cu129,cu130}/",
  "install_command": "uv pip install -U vllm --torch-backend=cu130 --extra-index-url https://wheels.vllm.ai/nightly/cu130 --python /home/danny/.venvs/vllm/bin/python",
  "vllm_version_installed": "0.20.2rc1.dev119+g09a7cc5ba",
  "torch_version": "2.11.0+cu130",
  "torch_compiled_archs": ["sm_75","sm_80","sm_86","sm_90","sm_100","sm_120"],
  "venv_path": "/home/danny/.venvs/vllm",
  "venv_python": "3.11.15",
  "venv_disk_size": "7.5 GB",
  "mtp_files_installed": [
    "vllm/model_executor/models/gemma4_mtp.py",
    "vllm/v1/spec_decode/gemma4.py"
  ],
  "spec_decode_methods_supported": ["deepseek_mtp","draft_model","eagle","eagle3","gemma4_mtp","medusa","mtp","ngram"],
  "Gemma4MTPModel_in_registry": true,
  "Gemma4Proposer_class_present": true,
  "gpu_in_use_by_training": true,
  "training_step": "5671/16124 (35%), ETA ~5h33m (as of 2026-05-07T21:18Z)",
  "v9c_baseline_tps_median": 77.81,
  "merge_script_base_agnostic": true,
  "serve_script": "backend/scripts/eval/serve_v10it_vllm.sh",
  "bench_script": "backend/scripts/eval/bench_vllm_v10it.py"
}
```

### Resume Context
- Current focus: All phases complete; producing final report.
- Next action (post-training, by user/next agent):
  1. Wait for v10-it training to finish (final/ dir written)
  2. `bash backend/scripts/eval/serve_v10it_vllm.sh`  (merges LoRA, then serves)
  3. In another terminal: `uv run python backend/scripts/eval/bench_vllm_v10it.py --out-dir backend/scripts/eval/bench_out/v10it_vllm_mtp`
- Blockers: None.
