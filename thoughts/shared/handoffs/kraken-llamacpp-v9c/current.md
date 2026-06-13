# llama.cpp v9c benchmark setup

## Task
Set up llama.cpp inference for v9c Gemma 4 E4B + LoRA, with Flash-Attn,
benchmark vanilla + speculative-decoding paths, quality compare to Unsloth baseline.

## Checkpoints
**Started:** 2026-05-07T00:00:00Z
**Last Updated:** 2026-05-07T00:00:00Z

### Phase Status
- Phase 1 (Verify llama.cpp toolchain + rebuild w/ FA-all-quants):  VALIDATED
- Phase 2 (Convert v9c LoRA -> GGUF):  VALIDATED (516 tensors, 69.8 MB, alpha=32)
- Phase 3 (Convert + quantize Gemma 4 E4B base GGUF):  VALIDATED (f16, Q8_0, Q4_K_M)
- Phase 4 (Write bench_llamacpp_v9c.py):  VALIDATED
- Phase 5 (Run benchmark sweep):  VALIDATED (5 vanilla combos)
- Phase 6 (Speculative decoding):  VALIDATED (4 ngram variants + 1 model-draft placeholder)
- Phase 7 (Quality spot-check vs Unsloth baseline):  VALIDATED (Q4 broken, Q8/F16 OK)

### Validation State
```json
{
  "llama_cpp_repo": "/home/danny/llama.cpp",
  "llama_cpp_head": "e583f3b ggml : minor coding style (#22308)",
  "cuda_nvcc": "/home/danny/micromamba/envs/cuda128/bin/nvcc",
  "cuda_arch": "120 (sm_120 Blackwell RTX 5090)",
  "existing_build_has": ["llama-server", "llama-mtmd-cli"],
  "existing_build_missing": ["llama-cli", "llama-quantize", "llama-bench"],
  "existing_GGML_CUDA_FA": "ON",
  "existing_GGML_CUDA_FA_ALL_QUANTS": "OFF (need ON)",
  "v9c_adapter_path": "/home/danny/Documents/personal/extension/backend/training/runs/manga-bubbles/gemma4_e4b_v9c/final",
  "v9c_base_model_id": "unsloth/gemma-4-E4B",
  "hf_cache_gemma4_E4B": "/home/danny/.cache/huggingface/hub/models--google--gemma-4-E4B/snapshots/7aa32e6889efd6300124851b164f8b364314c3d8",
  "hf_cache_unsloth_gemma4_E4B": "/home/danny/.cache/huggingface/hub/models--unsloth--gemma-4-E4B/snapshots/5bf6a20911f0f1ae1762c8c42419aad9efa37cbe"
}
```

### Resume Context
- Current focus: Rebuild llama.cpp with full target list + GGML_CUDA_FA_ALL_QUANTS=ON
- Next action: cmake + ninja build, validate llama-cli/llama-quantize/llama-bench produced
- Blockers: none
