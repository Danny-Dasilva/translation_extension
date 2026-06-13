# Free-tier neural-MQM eval setup

This directory ships two complementary judges:

| Judge | What it does | Cost | Hardware |
|-------|--------------|------|----------|
| `score_jsonl_metrics.py --metrics xcomet` | xCOMET-XL reference-based MQM with per-segment error spans (critical/major/minor) | free | ~7GB VRAM bf16 |
| `gemba_mqm_judge.py --judge openai:llama-3.3-70b-versatile` | GEMBA-MQM via Groq (free tier, no card) | free up to Groq's quota | CPU only |
| `gemba_mqm_judge.py --judge qwen25-72b-4bit` | GEMBA-MQM via local Qwen2.5-72B 4-bit | free | ~40GB VRAM |

xCOMET is reference-based; GEMBA is reference-free LLM-as-judge. Use both for
a triangulated read.

## Groq setup (free, no credit card)

1. Visit `console.groq.com`, log in, navigate to **API Keys** → **Create**.
2. Copy the key (starts with `gsk_…`).
3. Export the env vars in your shell:

   ```bash
   export OPENAI_API_KEY=gsk_...
   export OPENAI_BASE_URL=https://api.groq.com/openai/v1
   ```

   `gemba_mqm_judge.py` honors `OPENAI_BASE_URL` via the `OpenAIJudge` class
   (`backend/scripts/eval/gemba_mqm_judge.py`).

4. Verify wiring with `--dry-run` before spending quota:

   ```bash
   /home/danny/.venvs/comet/bin/python backend/scripts/eval/gemba_mqm_judge.py \
     --predictions backend/scripts/eval/openmantra_v10it/greedy/translations.jsonl \
     --references backend/scripts/eval/data/openmantra/heldout.jsonl \
     --judge openai:llama-3.3-70b-versatile \
     --limit 3 \
     --dry-run \
     --out /tmp/mqm_dryrun.json
   ```

   Dry-run prints the first chat-completions payload to stderr, never opens a
   socket, never reads `OPENAI_API_KEY`. Stub response yields `no-error` so
   the rest of the pipeline shape-checks end-to-end.

5. Run for real on a small subset first (e.g. `--limit 50`) to sanity check
   the parsed major/minor counts before scaling up.

## Recommended Groq models for GEMBA-MQM

As of 2026:

| Model | Context | Notes |
|-------|---------|-------|
| `llama-3.3-70b-versatile` | 128k | Best general-purpose, recommended default |
| `llama-3.1-70b-versatile` | 128k | Older, comparable for MQM |
| `mixtral-8x7b-32768` | 32k | Faster but weaker on Japanese |

Avoid speculative-decoding-only Groq endpoints — they sometimes drop the
"Major errors:" / "Minor errors:" structured headers GEMBA-MQM relies on.

## xCOMET-XL setup

Already pulls automatically from HF on first run via `comet.download_model`.
No setup beyond having `/home/danny/.venvs/comet/` populated (which it is).

```bash
/home/danny/.venvs/comet/bin/python backend/scripts/eval/score_jsonl_metrics.py \
  --gold-jsonl backend/scripts/eval/data/openmantra/heldout.jsonl \
  --pred-jsonl backend/scripts/eval/openmantra_v10it/greedy/translations.jsonl \
  --gold-ref-key en --pred-key en \
  --label v10it_om_xcomet \
  --metrics xcomet \
  --out-dir backend/scripts/eval/scorecards/v10it_phase0
```

Per-bubble JSON includes `xcomet_xl` (segment score in [0,1]) and
`xcomet_error_spans` (list of `{start, end, severity, confidence, text}`).
Summary JSON includes `xcomet_xl_mean`, `xcomet_error_span_counts`, and
`xcomet_total_error_spans`.

## VRAM budget

If another job is currently using the GPU (~22GB), there is usually
enough headroom for xCOMET-XL (~7GB) on a 32GB card — but at 28GB used you
will OOM. Either:

- wait for the other job's step to finish and free its activations, or
- set `CUDA_VISIBLE_DEVICES=` to force CPU (slow; only viable on small
  holdouts), or
- temporarily pause the other job with `kill -STOP <pid>` / `kill -CONT <pid>`.

## Running the full Phase 0 4-metric stack

```bash
/home/danny/.venvs/comet/bin/python backend/scripts/eval/score_jsonl_metrics.py \
  --gold-jsonl <ref.jsonl> --pred-jsonl <pred.jsonl> \
  --gold-ref-key en --pred-key en \
  --label <system>_<holdout> \
  --metrics chrf,bleu,kiwi,metricx,xcomet \
  --out-dir backend/scripts/eval/scorecards/v10it_phase0
```

Then the paired bootstrap across all metrics:

```bash
/home/danny/.venvs/comet/bin/python backend/scripts/eval/paired_bs_metric.py \
  --sys-a-per-bubble <per_bubble_v10it.json> \
  --sys-b-per-bubble <per_bubble_v9c.json> \
  --label-a v10it --label-b v9c \
  --metric-keys chrf_pp,bleu,cometkiwi_xl,metricx_24_xl,xcomet_xl \
  --lower-is-better metricx_24_xl \
  --out backend/scripts/eval/scorecards/v10it_phase0/paired_bs_<holdout>.json
```
