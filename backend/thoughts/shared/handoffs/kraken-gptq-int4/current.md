## Checkpoints
<!-- Resumable state for kraken agent -->
**Task:** GPTQ W4A16 quantization for Gemma-4 E4B (PLE) to fix RTN speaker-tag truncation
**Started:** 2026-06-18T10:00:00Z
**Last Updated:** 2026-06-18T12:45:00Z  **STATUS: COMPLETE — GPTQ-INT4 SHIPS (beats RTN+bf16)**

### Phase Status
- Phase 1 (Root-cause the sequential KeyError): ✓ VALIDATED
- Phase 2 (Make sequential trace succeed): ✓ VALIDATED (43 subgraphs)
- Phase 3a (n=2 smoke end-to-end): ✓ VALIDATED (258 modules quant, artifact saved, NO KeyError, layer 24 KV-shared OK)
- Phase 3b (full n=384 GPTQ run): ✓ VALIDATED (258/258 modules, 0 errors, 9.5GB, peak GPU ~2.4GB our proc)
- Phase 4 (Refold k_norm + verify config): ✓ VALIDATED (42 k_norm layers 0-41, embeds BF16, qconfig present)
- Phase 5 (Serve on vLLM, record VRAM): ✓ VALIDATED (EngineCore 12.93GB @ util0.45, KV+, MTP)
- Phase 6 (chrF++ + truncation gate vs RTN/bf16): ✓ VALIDATED -- GPTQ PASSES

### FINAL RESULTS (vntl128, identical eval cmd)
- corpus chrF++:  bf16=31.28  RTN=28.66  GPTQ=32.21  (GPTQ beats both)
- mean sent chrF++: bf16=31.33  RTN=29.86  GPTQ=32.94
- paired bootstrap GPTQ-RTN = +3.08 [95%CI +1.05,+5.15] P(GPTQ>RTN)=0.999; per-seg GPTQ wins 70 / RTN 40 / tie 18
- TRUNCATIONS: RTN 5/128 -> GPTQ 1/128 (the 1 residual idx45 is a line bf16 also mangles)
  fixed: idx2 [運転手]->'Thank you very much'; idx18 [女性]->'...Hm...Um'; idx43/50 [母・都子] full;
  idx85 [旅館の人]->'Excuse me...'  (all match bf16)
- register/NSFW probe: explicit register preserved, NO euphemism collapse (soaking wet / pervert /
  push it deeper); rude/polite/childish/archaic all correct. 葬儀屋さん isolated -> coherent (wrong
  name) NOT degenerate '7777' like RTN.
- served VRAM: bf16 17.5GB / RTN 13.6GB / GPTQ 12.93GB
- VERDICT: SHIP GPTQ-INT4 (artifact: w4a16_gptq, k_norm refolded). Production serve default UNCHANGED (bf16).

### Baselines (chrF++ on vntl128, from prior agent's bench_out/)
- bf16 corpus chrF++ = 31.283 (port 8003)
- RTN  corpus chrF++ = 28.656 (port 8002) -- BASELINE-TO-BEAT
- RTN truncations: [母・都子]草刈り->'[Tsuragi & Haruhi & Mom]'(3.2); お祖父ちゃん->'[Tsururagi & Cowgirl]'(3.6);
  [女性]んーあの->'[Kairyou Member]'(4.9); [運転手]ありがとう->'[Driver]'(17.1); 葬儀屋さん->'7777...'(idx0)
- GPTQ artifact: training/runs/manga-bubbles/gemma4_e4b_v11_pagecontext/w4a16_gptq (k_norm refolded)
- EVAL CMD (mirror RTN): bench_vllm_v10it.py --base-url http://127.0.0.1:8002/v1
  --prompts training/eval_held_out/vntl128.jsonl --out-dir scripts/eval/bench_out/gptq_vntl128
  --limit 128 --max-tokens 128 --label gptq_vntl128 --prompt-style eval

### Smoke artifact verification (n=2, /tmp/gptq_smoke) — CORRECT
- quantization_config: compressed-tensors, 4bit int sym group128
- 258 packed 4-bit weight tensors (= 24 full layers*7 + 18 kv-shared*5)
- embed_tokens + embed_tokens_per_layer stayed BF16; lm_head + PLE proj + vision/audio in ignore
- 24 k_norm (layers 0-23); 24-41 omitted -> refold adds 18 (matches RTN baseline)

### THREE-PART FIX (all required for sequential GPTQ on gemma4 PLE+KV-shared)
1. TRACE: sequential_targets=["Gemma4TextDecoderLayer"] + tracing_ignore includes
   "project_per_layer_inputs" (PLE proj does *shape[:-1] Proxy iteration ->
   TraceError). project_per_layer_inputs is ALREADY in llmcompressor 0.12.0 default
   tracing_ignore. -> trace_subgraphs returns 43 subgraphs (42 layers + head).
   Verified via scripts/quant/trace_repro.py (CPU, no GPU).
2. PLE MEMORY: head subgraph onloads embed_tokens_per_layer ([262144,10752] bf16 =
   5.6GB) to GPU just for a gather -> OOM on contended card. Fix(--keep-embeds-on-cpu):
   pin that embedding to CPU onload + patch get_per_layer_inputs to move the tiny
   gather result to GPU. Peak GPU (our proc) dropped 26GB -> 7.5GB.
3. KV-SHARING (the REAL 'sliding_attention' KeyError): partitioner bakes
   `shared_kv_states = {}` (fresh empty dict literal) into EVERY subgraph's compiled
   forward (verified inspect_subgraphs.py: skv_in/out=False all 43). Producer layers
   write KV into discarded {}, consumer layers 24-41 read fresh {} -> KeyError at
   modeling_gemma4.py:1253. Fix(--fix-kv-sharing): patch Gemma4TextAttention.forward
   to route shared_kv_states through a persistent store keyed by
   (current_batch_idx, layer_type); pipeline runs subgraphs in order per batch so
   producer KV survives to consumer.
All three wired into scripts/quant/quant_w4a16.py (gptq + pipeline=sequential).

### Validation State
```json
{
  "trace_subgraphs_result": "43 subgraphs (42 layers + head)",
  "keyerror_root_cause": "shared_kv_states baked as fresh {} per subgraph (NOT a tracing failure)",
  "peak_gpu_our_proc_after_PLE_fix_MiB": 7500,
  "n2_smoke": "ran 20min through layers, no KeyError (killed by timeout, slow under videonest contention)",
  "gpu_state": "videonest fluctuates 23-29GB used / 3-9GB free"
}
```

### Resume Context
- Current focus: confirm n=2 smoke writes a real artifact, then full n=384 run
- Next action: full GPTQ (n=384) in background no-timeout; then refold k_norm; serve; chrF++
- Blockers: videonest GPU job slows it (CPU offload + contention). Per-layer GPU need ~2-4GB fits.
- FULL CMD: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
  python scripts/quant/quant_w4a16.py --model <merged> --out <w4a16_gptq>
  --method gptq --calib scripts/quant/calib_v11_int4.jsonl --n 384
  --pipeline sequential --device-map cpu --max-seq-len 512 --keep-embeds-on-cpu --fix-kv-sharing
