# EAGLE-3 custom drafter for v10-it (Gemma 4 E4B) — handoff

## Task
Train a custom EAGLE-3 drafter for the merged v10-it Gemma 4 E4B-it model
via SpecForge, then plumb it through vLLM for production deploy.

Target: 80–93% acceptance, ~165–220 tok/s, 2.1–2.8× over v9c llama.cpp baseline.

## Phase status
- Phase 0 recon: VALIDATED (2026-05-09T15:00Z)
- Phase 1 SpecForge install + patch: VALIDATED (2026-05-09T15:18Z, smoke test passed)
- Phase 2 self-distillation data generation: VALIDATED (2026-05-09T16:26Z, kraken-2)
  - Corpus design LOCKED by user: 80k self-distill manga + 30k SFT replay (110k total, manga-only, no generic English)
  - 80k JP inputs sampled from manga_dialog (114704 available) without dedup, seed=42
  - vLLM vanilla served, 80k self-distill translations generated at T=0.7,top_p=0.9 in ~9 min total wall (with 1 mid-run vllm crash + clean resume)
  - 30k SFT replay sampled from data_v10 (all registers, mostly manga-flavored)
  - Final corpus: `/mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/eagle3_corpus.jsonl` (109998 lines)
  - Distribution: 79998 self_distill + 13309 manga_dialog + 4047 vn_eroge + 3277 novel + 3147 vn + 2585 sfx + 1810 manga + 933 synthetic + 892 garbage
- Phase 3 hidden-state dump: SKIPPED (using online mode — data + hidden states generated on-the-fly)
- Phase 4 drafter training: IN_PROGRESS (originally started 2026-05-09T16:31Z, kraken-2;
  RESUMED 2026-05-10T05:24Z, kraken-3)
  - Launcher: `/home/danny/SpecForge/examples/run_gemma4_e4b_v10it_eagle3_prod.sh`
  - Hyperparameters: 3 epochs, batch=1, max_len=4096, lr=1e-4 cosine, warmup_ratio=0.015,
    ttt_length=7, draft_accumulation_steps=32, save_every=500 steps, log_every=10, eval_every=500
  - Output: `/mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/outputs/gemma4-e4b-v10it-eagle3-prod/`
  - Train log: `/mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.log`
  - PID file: `/mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.pid`
  - Per-step time: ~0.1s after warmup; 110k × 3 epochs ≈ 9 hours (faster than user estimate)
  - First ckpt at step 500 (epoch_0_step_500): VERIFIED non-empty, 385MB model
  - Initial loss/acc trajectory (steps 100-700): loss falls from 2.5 to 1.2-2.0 range,
    acc rises from 0.0 to 0.05-0.36 — model is learning, gemma4 chat template fix confirmed working
  - **Resume state (kraken-3, 2026-05-10T05:24Z):**
    - Wrapper PID 2797386 / torchrun 2797418 / **worker 2797536**
    - Resumed from `epoch_1_step_202500` (109998 epoch-0 + 92502 epoch-1 forward steps)
    - Bench kraken's vLLM had already exited cleanly when we polled — no wait needed beyond ~90s
    - Current position (just past resume): forward step 93001/109998 in epoch 1 at ~8.3 it/s
    - GPU: 31 GiB memory, ~70% util in active step, 256 W draw
    - Remaining: 127494 forward steps. ETA: **3.9–4.3 hours** at 8–9 it/s
    - **Required code patch** (NEW): `/home/danny/SpecForge/specforge/modeling/draft/llama3_eagle.py`
      `LlamaForCausalLMEagle3.__init__` now sets `self.all_tied_weights_keys = {}`
      to satisfy transformers >=5.x `_finalize_model_loading`. Without this, `--resume`
      fails with `AttributeError: 'LlamaForCausalLMEagle3' object has no attribute
      'all_tied_weights_keys'`. The drafter has no tied weights so empty dict is
      semantically correct. (Add to `specforge_patches.diff` when convenient.)
    - Resume-flag invocation: launcher takes positional `NUM_GPUS TP_SIZE` first, so
      correct command is `bash run_gemma4_e4b_v10it_eagle3_prod.sh 1 1 --resume`
      (NOT the `-- --resume` form noted in original handoff which was misinterpreted
      as `NUM_GPUS=-- TP_SIZE=--resume` and crashed torchrun arg parsing).
- Phase 5 convert to vLLM-loadable format: VALIDATED (2026-05-10T11:42Z, kraken-4) — drafter loads drop-in via `Eagle3LlamaForCausalLM`, vLLM logs:
  - `Detected EAGLE model without its own embed_tokens — Sharing target model embedding weights`
  - `Detected EAGLE model with distinct lm_head weights. Keeping separate lm_head`
  - No conversion / `speculators` repackaging needed
- Phase 5b register `Eagle3Gemma4ForCausalLM` in vLLM: NOT NEEDED (confirmed — `LlamaForCausalLMEagle3` arch + `target_model_type: gemma4_text` works as-is)
- Phase 6 bench: BLOCKED — step_203500 NOT shippable, see Phase 6 results below

## Phase 6 bench results (2026-05-10T11:43Z, kraken-4)

**Checkpoint tested:** `epoch_1_step_203500` (latest viable; ~313k forward steps in / ~330k total target)

**Setup:**
- vLLM 0.20.2rc1.dev119 on /home/danny/.venvs/vllm
- `serve_v10it_vllm.sh` with `EAGLE3_DRAFTER=...epoch_1_step_203500 GPU_UTIL=0.78 GAMMA=2`
  - Note: had to lower `GPU_UTIL` from 0.85→0.78 because GNOME desktop processes hold ~940 MB of GPU; 0.85 demanded 26.65 GiB but only 26.26 GiB free
- Bench: `backend/scripts/eval/bench_vllm_v10it.py` (25 prompts, manga-style)
- Run via `/home/danny/.venvs/vllm/bin/python` directly (avoids `uv run` transformers dep conflict)
- Output: `backend/scripts/eval/bench_out/v10it_vllm_eagle3/v10it_vllm_eagle3_g2_*.{jsonl,summary.json}`

**Numbers:**
| Metric | Value | vs Target | vs v9c | vs vanilla | vs Google MTP γ=2 |
|---|---|---|---|---|---|
| Median decode tok/s | **101.4** | 150 floor → BELOW | 1.30× | 1.01× (wash) | 0.93× (worse) |
| Mean decode tok/s | 103.4 | — | 1.33× | — | — |
| p95 decode tok/s | 150.7 | — | — | — | — |
| Overall acceptance | **34.4%** (106/308) | 60% floor → FAR BELOW | — | — | comparable to MTP's 31.1% |
| Per-position pos0 | **43.5%** (67/154) | — | — | — | — |
| Per-position pos1 | **25.3%** (39/154) | — | — | — | — |
| Drafts created | 154 (γ=2 confirmed) | — | — | — | — |
| Total accepted/draft ratio | 0.69 acc-tok per request | — | — | — | — |

**CRITICAL: 2/25 prompts produced gibberish:**
- Prompt 1: `you'd probably small""bill"" ush!h!h!h;;h;;h;;h;;h` (64 tok)
- Prompt 25: `h gustilishly` (5 tok)

The other 23 prompts produced reasonable manga translations. The gibberish is consistent with vLLM EAGLE-3 numerical issues (or under-trained drafter producing logits that escape valid token ranges in the rejection sampler).

**Phase 3 decision: DO NOT SHIP.**
- Acceptance 34.4% < 40% floor
- tok/s 101.4 < 120 floor (and effectively a wash with vanilla vLLM at 100.1; we're paying drafter compute for ~zero gain)
- Gibberish output is a hard veto per task constraints

**Why this checkpoint underperforms:**
- Training was at step 203500 of ~330000 target (~62% of full schedule). Early loss/acc trajectory was promising (acc 0.05→0.36 in first ~700 steps) but the drafter clearly needs more epochs to reach the 60-80% acceptance band Google's MTP and SpecForge papers report.
- Comparing to v10-it+Google MTP γ=2 (31.1% accept, 109 tok/s), our custom EAGLE-3 at 34.4% / 101.4 tok/s is in the same ballpark but worse on throughput. Our drafter is essentially a worse Google MTP at this checkpoint.

**What's needed to ship:**
1. **Fix the OOM crash on resume** (see Phase 4 crash analysis below) so training can finish epochs 2-3
2. Resume + complete training to step ~330000 (full 3 epochs)
3. Re-bench at final checkpoint
4. If still <60% acceptance, consider: a) increasing ttt_length from 7 (longer training horizon), b) switching to Google's MTP architecture (which already works reasonably well at γ=2)

## Phase 4 crash root cause analysis (2026-05-10T11:35Z, kraken-4)

**Crash A (05:23:17 UTC, before patch was applied):**
- `AttributeError: 'LlamaForCausalLMEagle3' object has no attribute 'all_tied_weights_keys'` — fixed by patch already in place at `~/SpecForge/specforge/modeling/draft/llama3_eagle.py:1356-1364`. Verified patch IS still applied (`git diff` shows it intact).

**Crash B (05:28:45 UTC, after patch — the real culprit):**
```
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.62 GiB.
GPU 0 has a total capacity of 31.35 GiB of which 1.42 GiB is free.
... this process has 23.32 GiB memory in use.
```
- **Crashed in `ploss.backward()`** (train_eagle3.py:677) on epoch 1 step 93601 (~4 min after resume started)
- Process held 23.32 GiB; only 1.42 GiB free; tried to grow by 1.62 GiB → boom
- Lost ~940 MB to GNOME desktop processes (resources/xdg-desktop-portal/gnome-control-center/ptyxis/loupe/nautilus). They are user processes — cannot kill (per task constraints)
- The crash is a **dataset hot-spot**: a particularly long sequence at max_length=4096 with ttt_length=7 + grad_accum=32 produced a backward pass that needed >24 GiB. Pre-resume training got past this point in epoch 0 because tail-end training (different sample order? different cache state?) happened to avoid the spike, OR because GPU was less crowded with DE processes at that time.

**Mitigation options for next resume (in order of risk/effort):**
1. **Lowest risk:** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (suggested in the OOM message itself; reduces fragmentation, often recovers ~1-2 GB). Just export it before launching the wrapper.
2. **Low risk:** drop `--max-length` from 4096 → 3072. Combined with current corpus length distribution, this would clip a few % of long samples but is safe (no data loss, just truncation). Saves ~25% of activation memory in worst case.
3. **Low risk:** drop `--ttt_length` 7 → 5. Reduces unrolled draft horizon (and slightly reduces training quality at long horizons), but cuts activation memory roughly proportionally.
4. **Medium risk:** drop `--draft-accumulation-steps` 32 → 16 (or 8). Effective batch size drops, may need 2× LR adjustment, trains slower per epoch but smaller per-step memory footprint. **HOWEVER:** the OOM is in a single backward step, not across accum. This won't help.
5. **Higher effort:** add gradient checkpointing in the EAGLE-3 draft model. Would need a SpecForge code change.

**Recommended retry recipe:**
```bash
# 1. Close GNOME apps that hold GPU memory if user wants (saves ~940 MB):
#    Image viewer (loupe), text editor, nautilus, gnome-control-center
# 2. Set memory allocator flag + slightly conservative max_length
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MAX_LEN=3584 \
  bash /home/danny/SpecForge/examples/run_gemma4_e4b_v10it_eagle3_prod.sh 1 1 --resume
```
Note: launcher script may not honor `MAX_LEN` env (it's hard-coded to 4096 inside the script per handoff history). Need to either edit the launcher or pass through. Check before retry.

**Will the crash recur at the same step?**
- Possibly yes — it depends on whether the dataloader shuffle is deterministic on resume. SpecForge's resume reloads RNG state, so if the same long sample lands at the same memory pressure point, yes.
- A **200-step dry-run** (as task suggests) would confirm. Did NOT run this yet — would need to start training and watch for ~10-15 minutes; out of bench budget.

## What's been built

### SpecForge environment
```
venv:           /home/danny/.venvs/specforge   (Python 3.11.15)
torch:          2.9.1+cu129
transformers:   5.8.0  (has Gemma 4 — required)
specforge:      /home/danny/SpecForge (HEAD d5fb617 + local patches, editable install)
```

### SpecForge patches (local, no public fork)
Saved as full diff: `thoughts/shared/handoffs/eagle3-v10it/specforge_patches.diff`

| File | Change |
|---|---|
| `specforge/args.py` | lazy `ATTENTION_BACKEND_CHOICES` import (sglang optional) |
| `specforge/data/template.py` | new `gemma4` chat template (`<\|turn>...<turn\|>` markers — Gemma 4 uses different markers from Gemma 3, this fix is critical) |
| `specforge/distributed.py` | yunchang lazy-imported, gated on `sp_size > 1`; `destroy_distributed` skips None groups |
| `specforge/layers/ring/ring_flash_attn.py` | yunchang.kernels lazy import |
| `specforge/modeling/draft/llama3_eagle.py` | `yunchang.comm.SeqAllToAll4D` lazy import |
| `specforge/modeling/target/eagle3_target_model.py` | (1) sglang imports removed (top-level); (2) `SGLangEagle3TargetModel` reduced to a stub that raises NotImplementedError (sglang 0.5.9 hard-pins transformers==4.57.1 which lacks Gemma 4); (3) `_get_transformer_layers` probes `model.model.language_model.layers` for `Gemma4ForConditionalGeneration`; (4) `set_aux_hidden_states_layers` falls back to `text_config.num_hidden_layers` for multimodal configs |

### New SpecForge files
| File | Purpose |
|---|---|
| `configs/gemma4-e4b-eagle3.json` | EAGLE-3 drafter config: hidden_size=2560, intermediate=10240, vocab=262144, head_dim=256, 8 heads, 2 KV heads, 1 layer; arch `LlamaForCausalLMEagle3`; `target_model_type: gemma4_text` |
| `examples/run_gemma4_e4b_v10it_eagle3_online.sh` | Single-GPU launcher (default uses NAS for output + cache) |

### Project tree changes
| File | Change |
|---|---|
| `backend/scripts/eval/serve_v10it_vllm.sh` | Added `EAGLE3_DRAFTER` env var: when set, switches `--speculative-config` from `method=mtp` to `method=eagle3` and uses the local drafter dir |

## Smoke test result (Phase 1 validation)

Command (4-line dataset, 4 steps, ~1m wall, ~16 GB GPU):
```bash
cd /home/danny/SpecForge && /home/danny/.venvs/specforge/bin/torchrun \
    --standalone --nproc_per_node 1 \
    scripts/train_eagle3.py \
    --target-model-path /home/danny/Documents/personal/extension/backend/training/runs/manga-bubbles/gemma4_e4b_v10it/merged \
    --draft-model-config /home/danny/SpecForge/configs/gemma4-e4b-eagle3.json \
    --train-data-path /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/smoke_test_train.jsonl \
    --output-dir /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/smoke_output \
    --num-epochs 1 --batch-size 1 --tp-size 1 \
    --learning-rate 1e-4 --max-length 512 \
    --chat-template gemma4 \
    --cache-dir /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/specforge_cache \
    --attention-backend sdpa --target-model-backend hf \
    --embedding-key model.language_model.embed_tokens.weight \
    --log-interval 1 --save-interval 100
```

Output:
```
Training Epoch 0:  25%|██▌  | 1/4, loss=2.16, acc=0.00, time=2.68s
Training Epoch 0:  50%|█████| 2/4, loss=1.50, acc=0.00, time=0.25s
Training Epoch 0:  75%|███▌ | 3/4, loss=1.27, acc=0.29, time=0.11s
Training Epoch 0: 100%|████| 4/4, loss=0.91, acc=0.25, time=0.11s
EXIT=0
```

Saved drafter (`smoke_output/epoch_0_step_4/model.safetensors`, 385 MB):
- 15 weight keys total
- `midlayer.self_attn.q_proj.weight: (2048, 5120)` — q is 8 heads × 256 = 2048; input is 2*hidden = 5120 (EAGLE-3 first layer concatenates embeds + hidden_state)
- `midlayer.self_attn.k_proj.weight: (512, 5120)` — k is 2 KV heads × 256 = 512
- `fc.weight: (2560, 7680)` — projects 3*hidden=7680 (EAGLE-3 cat of 3 aux states) → hidden=2560
- `lm_head.weight: (32000, 2560)` — draft vocab 32000
- `t2d`, `d2t` — vocab mapping (262144 → 32000)
- All weights non-zero abs.mean (~0.005-0.02 = standard init scale + 4 steps of update)

After smoke: nvidia-smi reports GPU back to 3.7 GB used (idle, just gnome/slack).

## Critical lesson learned
The biggest landmine in this pipeline: **Gemma 4 uses `<|turn>...<turn|>` chat
markers, NOT Gemma 3's `<start_of_turn>...<end_of_turn>`**. SpecForge's bundled
`gemma` chat template uses Gemma 3 markers. If you train with `--chat-template
gemma`, the loss-mask regex never matches assistant spans, the loss_mask is all
zeros, no gradient flows, and the loss reads `0.00` for the entire run — but
training "succeeds" silently. The patched `gemma4` template fixes this.

## Resume command (for next kraken/operator)

### Phase 2 (DONE by kraken-2 at 2026-05-09T16:26Z)

Corpus design LOCKED: 80k self-distill manga + 30k SFT replay = 110k examples.

What was built (kraken-2):
- `backend/scripts/eval/eagle3_corpus/sample_manga_for_distill.py` — samples
  80k manga_dialog rows from `data_v10.parquet` (no dedup), seed=42.
- `backend/scripts/eval/eagle3_corpus/gen_self_distill_corpus.py` — calls
  vLLM `/v1/chat/completions` to generate sampled translations at T=0.7,
  top_p=0.9, max_tokens=80; concurrent (parallel=32); resumes on partial output.
- `backend/scripts/eval/eagle3_corpus/assemble_eagle3_corpus.py` — merges
  self-distill outputs + 30k SFT replay rows from data_v10 into ShareGPT JSONL.

Final corpus: `/mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/eagle3_corpus.jsonl`
(109998 lines; 22 MB self-distill source). vLLM crashed once mid-run but the
gen script auto-resumed via the `idx` field in already-written rows.

### Phase 4 (LAUNCHED at 2026-05-09T16:31Z by kraken-2 — RUNNING)

Launcher: `/home/danny/SpecForge/examples/run_gemma4_e4b_v10it_eagle3_prod.sh`

```bash
# Launch (already done):
nohup bash /home/danny/SpecForge/examples/run_gemma4_e4b_v10it_eagle3_prod.sh \
  > /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.log 2>&1 &
echo $! > /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.pid
```

Hyperparameters: 3 epochs, batch=1, max_len=4096, lr=1e-4, warmup_ratio=0.015,
ttt_length=7, draft_accumulation_steps=32, save_every=500, log_every=10.

Total steps: 110k examples × 3 epochs = 330k forward/backward passes.
Per-step time: ~0.1s (after warmup). Expected wall: ~9 hours.

Output: `/mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/outputs/gemma4-e4b-v10it-eagle3-prod/`
Each ckpt = 1.9GB (385MB model + 1.6GB optim state). At save_every=500 we'll
get ~660 ckpts = ~1.2TB. NAS has 7.7TB free, so no panic, but consider running
the cleanup helper periodically:

```bash
bash backend/scripts/eval/eagle3_corpus/cleanup_specforge_ckpts.sh \
  /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/outputs/gemma4-e4b-v10it-eagle3-prod
# (keeps last 2 + every 5th by step number)
```

### Phase 4 monitoring

```bash
# Live progress
tail -f /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.log

# Quick stats
grep -oE "[0-9]+/109998" /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.log | tail -1
grep -oE "loss=[0-9.]+, acc=[0-9.]+" /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.log | tail -3

# Process state
ps -p $(cat /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.pid) -o pid,stat,etime,cmd

# Saved checkpoints
ls /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/outputs/gemma4-e4b-v10it-eagle3-prod/

# GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
```

### Phase 4 resume on crash

```bash
nohup bash /home/danny/SpecForge/examples/run_gemma4_e4b_v10it_eagle3_prod.sh \
  -- --resume \
  >> /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.log 2>&1 &
```

(SpecForge `--resume` auto-detects the last checkpoint in `--output-dir`.)

### Phase 4 stopping criterion

Stop when val acceptance plateaus. SpecForge logs eval-time per-position acc
every `--eval-interval` (set to 500). For a 4B target, position-1 acc typically
plateaus at 60-80% by step ~30k.

If you want to stop early at "good enough":
```bash
kill $(cat /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.pid)
# It saves on every save_every interval, so at most you lose <500 steps of work.
```

### Phase 5: deploy via vLLM (drop-in, no conversion needed)

The SpecForge drafter is saved with `architectures: ["LlamaForCausalLMEagle3"]`
and `model_type: llama` + `target_model_type: gemma4_text`. vLLM 0.x has
`LlamaForCausalLMEagle3` registered → maps to `Eagle3LlamaForCausalLM`, and
`gemma4_text` is a known target model type
(`vllm/transformers_utils/model_arch_config_convertor.py`). So no conversion
step is needed — point `EAGLE3_DRAFTER` at the trained checkpoint dir:

```bash
EAGLE3_DRAFTER=/mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/outputs/gemma4-e4b-v10it-eagle3-prod/epoch_X_step_Y \
PORT=8000 SKIP_MERGE=1 GAMMA=2 \
MERGED_DIR=/home/danny/Documents/personal/extension/backend/training/runs/manga-bubbles/gemma4_e4b_v10it/merged \
bash /home/danny/Documents/personal/extension/backend/scripts/eval/serve_v10it_vllm.sh
```

(The serve script already has the `EAGLE3_DRAFTER` knob from kraken-1's Phase 1
work; it switches `--speculative-config method=eagle3`.)

If vLLM rejects the drafter at load time, fall back to Phase 5b below.

### Phase 5b (if Phase 5 fails) — fallback config edits

The drafter saves with `architectures: ["LlamaForCausalLMEagle3"]`. vLLM's
registry maps this directly to `Eagle3LlamaForCausalLM`. If vLLM rejects:

1. Edit drafter `config.json` to change `architectures` to `LlamaForCausalLM`
   (vLLM's older registry format).
2. Or write a `Eagle3Gemma4ForCausalLM` model class in vLLM. Don't go here
   unless option 1 fails — `target_model_type: gemma4_text` should already
   provide enough signal for the existing Eagle3 plumbing.

### Phase 6 bench (helper script ready)

```bash
bash backend/scripts/eval/eagle3_corpus/bench_eagle3_drafter.sh \
  /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/outputs/gemma4-e4b-v10it-eagle3-prod/epoch_X_step_Y
```

This: spawns vLLM with EAGLE-3 drafter → waits for ready → runs the standard
25-prompt bench at single concurrency → kills vLLM → prints summary.

Compare to:
- v9c llama.cpp Q8_0+FA  : 77.8 tok/s baseline
- v10-it vLLM vanilla    : 100.1 tok/s
- v10-it vLLM + Google MTP γ=2 : 109.1 tok/s, 31.1% acc
- v10-it vLLM + custom EAGLE-3: TARGET ≥150 tok/s, ≥60% acc

If acceptance ≥60% AND tok/s ≥150 → ship as production drafter (replace MTP
in `serve_v10it_vllm.sh`).

## Useful files & paths

```
SpecForge clone:        /home/danny/SpecForge
SpecForge venv:         /home/danny/.venvs/specforge
SpecForge patches diff: thoughts/shared/handoffs/eagle3-v10it/specforge_patches.diff
SpecForge prod runner:  /home/danny/SpecForge/examples/run_gemma4_e4b_v10it_eagle3_prod.sh (kraken-2)
NAS cache root:         /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/
  manga_distill_inputs.jsonl   — 80k JP inputs sampled from manga_dialog (kraken-2)
  self_distill_outputs.jsonl   — 80k JP+sampled-EN from v10-it (kraken-2)
  eagle3_corpus.jsonl          — 109998-line ShareGPT corpus for training (kraken-2)
  smoke_test_train.jsonl       — 4-line smoke dataset
  smoke_output/                — Phase 1 smoke test checkpoint
  specforge_cache/             — torch inductor + processed_dataset + vocab_mapping
  outputs/gemma4-e4b-v10it-eagle3-prod/  — production drafter checkpoints (kraken-2)
  train_prod.log               — Phase 4 training log
  train_prod.pid               — Phase 4 PID for monitor/kill

v10-it merged:          backend/training/runs/manga-bubbles/gemma4_e4b_v10it/merged/
Corpus pipeline:        backend/scripts/eval/eagle3_corpus/   (kraken-2)
                          sample_manga_for_distill.py  — sample 80k JP from data_v10
                          gen_self_distill_corpus.py   — vLLM /chat/completions concurrency
                          assemble_eagle3_corpus.py    — combine self-distill + SFT replay
                          bench_eagle3_drafter.sh      — Phase 6 helper
                          cleanup_specforge_ckpts.sh   — disk cleanup helper
vLLM serve script:      backend/scripts/eval/serve_v10it_vllm.sh (with EAGLE3_DRAFTER knob)
vLLM vanilla script:    backend/scripts/eval/serve_v10it_vllm_vanilla.sh
Bench script:           backend/scripts/eval/bench_vllm_v10it.py
Bench prompts:          backend/scripts/eval/bench_prompts.jsonl (25 manga prompts)
```

## Open questions / risks for next operator

1. **Vocab mapping**: SpecForge generates `t2d/d2t` from the training corpus
   tokens (the 32000 most-common). With only 25 manga prompts you'd only see
   maybe ~500 unique tokens, hurting downstream coverage. Use a larger corpus
   for the real run (Phase 2).

2. **Aux hidden state layers**: default for 42-layer Gemma 4 is `[1, 20, 38]`
   (computed via `[1, num_layers // 2 - 1, num_layers - 4]`). These may not be
   optimal. Reasonable to leave as default for first run; tune later if accept
   rate is poor.

3. **Tying embeddings**: drafter has its own `lm_head.weight` separate from
   target's tied `embed_tokens`. The drafter's `embed_tokens` is loaded from
   target via `--embedding-key model.language_model.embed_tokens.weight` and
   frozen. Excluded from saved state_dict (vLLM should re-tie at load).

4. **GPU contention with videonest**: training process runs as `nohup`. If
   videonest grabs the GPU, training will OOM and exit. Detect by checking
   the train.log for CUDA OOM, then resume.

5. **48h budget**: with my conservative 6-8h cap, I stopped before running
   the 18-24h training. The pipeline IS validated; the only remaining work
   is (a) build a real corpus, (b) launch nohup training, (c) wait, (d)
   bench. ~26h serial, but only ~30m of human time.

## Phase 4 resume #3 (2026-05-10T~12:05Z, kraken-5) — OOM fix applied

**Fixes applied to launcher `/home/danny/SpecForge/examples/run_gemma4_e4b_v10it_eagle3_prod.sh`:**
- `MAX_LEN` default: 4096 → **3584** (still covers >97% of training samples per ingest stats)
- TTT length unchanged at 7 (canonical EAGLE-3)
- batch=1, grad_accum=32 unchanged

**Resume command:**
```bash
cd ~/SpecForge
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup bash examples/run_gemma4_e4b_v10it_eagle3_prod.sh 1 1 --resume \
  >> /mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.log 2>&1 &
```

**Process:**
- Wrapper PID **3136147** / torchrun 3136174 / **worker PID 3136263**
- PID file: `/mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/train_prod.pid` (= 3136147)
- Resumed from `epoch_1_step_203500` as expected
- launcher banner shows `max_len: 3584` and `--max-length 3584` in ps output

**60s health check (after launch):**
- Process alive, etime 01:03, RSS 2.3 GiB
- GPU: 21.7 GiB used, 10.4 GiB free, util 14% (in startup/data loading phase)

**5-min stability sample (after training step counter started advancing):**
- Step counter advanced **93601 → 95649** in epoch 1 (training resumed at exact pre-crash position)
- Per-step time logged at ~7.6 it/s (better than 8 it/s estimate)
- New checkpoints written: `epoch_1_step_204000`, `204500`, `205000`, `205500` — all created in the 8-min window
- Loss oscillates 0.5–1.3, acc 0.16–0.23 (consistent with pre-crash trajectory)

**GPU memory observations (15 samples over 5 min, 20s spacing):**
- Min: 26254 MiB (25.6 GiB)
- Max: **31164 MiB (30.43 GiB)**
- Median: ~29.7 GiB
- Samples >31 GiB threshold: **0/15**
- Behavior: oscillates with sample length; expandable_segments allows allocator to grow without fragmentation OOM
- Headroom from 32 GiB total: ~1 GiB at peak, ~6 GiB typical — better than the 200-300 MB headroom that triggered crash B

**ETA:**
- Remaining at sample time (step 95649 / 109998 in epoch 1) = 14349 epoch-1 steps + 2 full epochs (~220k) = ~234k forward steps
- At 7.6 it/s → ~8.5 hours remaining for full 3 epochs
- The user's "~125k forward steps from step 203500" estimate counted only the rest of epoch 1 + 1 epoch — actual remaining is more (3 full epochs targeted, currently 62% through epoch 1).
- **Realistic finish: 8-9 hours from 12:05Z, i.e. ~20:00-21:00Z 2026-05-10**

**Watch points (for next session):**
- If GPU peak crosses 31.5 GiB or another OOM hits, consider dropping `MAX_LEN` to 3072 (next step). Don't drop ttt_length — paper-canonical.
- Step 93601 (where crash B happened pre-fix) was passed cleanly with the new config — strong evidence the fix is sufficient.
- Save cadence is 500 steps; cycle through 205500 → 206000 → ... Watch `/mnt/nas/drive_2/manga-ml/eagle3_v10it_cache/outputs/gemma4-e4b-v10it-eagle3-prod/` directory.

## Last updated
2026-05-10T~12:05Z by kraken-5 (Phase 4 resumed with OOM fix:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True + MAX_LEN 4096→3584. Worker PID
3136263 advancing past prior crash point cleanly; GPU peak 30.43 GiB, headroom OK).
Previous: 2026-05-10T11:43:00Z by kraken-4 (Phase 5/6 — converted+benched
step_203500, decided NOT shippable; diagnosed OOM crash B, recommended fix).
Previous: 2026-05-10T05:24:00Z by kraken-3 (resumed Phase 4 from step_202500).
Previous: 2026-05-09T16:38:00Z by kraken-2 (Phase 2 complete; Phase 4 launched).
Previous: 2026-05-09T15:25:00Z by kraken (Phase 1 + smoke).
