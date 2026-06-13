# JP→EN Manga Translator — Training Plan (v5)

Updated 2026-04-23. v6 supersedes v5. Key changes vs v4/v5:
- **SFT only**, no CPO preference stage (deferred to v2).
- **Single arm**: Qwen3-1.7B-Base primary; Qwen3-4B-Base as escalation path.
- **Deployment**: Chrome/Firefox extension + Python backend serving GGUF via `llama-cpp-python`. No mobile / `llama.rn` in v1.
- **No human eval** for v1 — automated metrics only.
- **No teacher distillation** (preserved from v4).
- **v6 data mix** (based on actual on-disk parse counts): VNTL-v3.1 is 148k pairs not 12k → gold weight 35%, no oversample. Open Mantra oversample capped at 8.4× (was 15× in v5). NSFW register coverage 50%, gold 45%. Target 200k pairs.

Authoritative implementation plan: `/home/danny/.claude/plans/read-the-plan-in-temporal-starlight.md`. This doc is kept in sync with that plan.

---

## 1. Goal

Fine-tune a ≤2B LLM (escalating to ≤4B if needed) to be the best open JP→EN manga translator we can ship, deployed as GGUF via `llama-cpp-python` behind the existing backend translation endpoint. Directional reference: match or beat `lmg-anon/vntl-llama3-8b-v2`'s VNTL-128 cosine (0.6952) despite being ~5× smaller.

## 2. Success criteria

**Directional (aim to beat, but not a ship gate):**
- VNTL-128 cosine-accuracy > 0.6986 — no ≤2B model has published this yet. Closest is `vntl-gemma2-2b` at 0.6572.

**Soft floors (probe suite + automated metrics):**
- FLORES-200 JA-EN XCOMET-XXL regression ≤ 2pp vs base-model zero-shot.
- All 8 probes in L3 pass their thresholds (see §6).
- No release-candidate metric regresses >5pp vs best prior checkpoint.

**Ship gate**: release candidate passes L3 probes green + L4 metrics acceptable; reviewed manually against ~50 bubbles from a held-out manga volume. If v1 lands with cosine 0.64–0.70, ship it — revisit with CPO in v2.

## 3. Base model — single arm

| Model | License | Size Q4_K_M | Why |
|---|---|---|---|
| `Qwen/Qwen3-1.7B-Base` (primary) | Apache 2.0 | 1.0 GB | 36T pretrain, 119 langs, Shisa-proven JA tunability. Escalate to `Qwen3-4B-Base` only if v1 underperforms. |

Escalation rule: if post-SFT VNTL-128 cosine <0.66 (i.e. worse than `vntl-gemma2-2b`), rerun the exact same data + recipe on `Qwen/Qwen3-4B-Base` (~8 h vs ~3 h).

## 4. Data — parallel corpora only (no teacher)

### Already downloaded (`backend/training/datasets/translation/`, 4.4 GB)

```
Total raw JP↔EN pairs available: ~3.3M

  JESC                                  2,797,388 pairs   CC-BY-SA 4.0    anime/TV/film dialogue
  Tatoeba JA-EN                         ~300,000          CC-BY 2.0 FR    clean short sentences
  NilanE/ParallelFiction-100k              106,000        Apache-2.0      web novel + fan-TL
  lmg-anon/VNTL raw pool                    ~10,000        Apache-2.0      VN dialogue
  lmg-anon/VNTL-v3.1-1k                     14,542        Apache-2.0      eroge VN pro-TL (GOLD)
  lmg-anon/VNTL-v3.1-1k-q                   ~10,000       Apache-2.0      preference pairs (for CPO)
  Aratako/Synthetic-JP-EN-20k                20,000       CC-BY-4.0       Claude/Nemotron synth
  NilanE/SmallParallelDocs-6k                 6,000       Apache-2.0      clean prose
  Open Mantra (5 series)                      1,593       Research        pro-translated manga (GOLD)
  SFX composite (yuiseki + nanoskript)       ~15,000      Permissive      JP→EN SFX glossaries
```

### Training-set construction

Filter + weight, target **~150k SFT pairs**:

```
Stage 1 — normalize: every source → parquet columns
         {jp, en, src, register_tag, gold_flag}
         register_tag ∈ {manga, vn_eroge, vn, anime_sub, novel, sfx, anchor, synthetic}

Stage 2 — quality filter (skip gold; apply to JESC / NilanE / Aratako / Tatoeba):
  JP: fasttext langid ≥0.70 · (hira+kata+kanji)/len ≥0.60 · 3≤chars≤400 · MeCab unk <0.4
  EN: KenLM PPL <1000 · no refusal regex · no 4-gram loop ≥3× · length ratio 0.3-4.0
  Pair: CometKiwi-23-XL ≥0.78 (fallback -22) · LaBSE cosine ≥0.70

Stage 3 — dedup within train: MinHash 5-gram, 14×8 bands, Jaccard ≥0.75
Stage 3b — cross-dedup train ∩ held-out: same MinHash LSH; any train row near-matching
           any held-out row is dropped from train before Stage 5 locks.

Stage 4 — weight + sample (v6 — rebalanced on actual on-disk counts; VNTL-v3.1
                            parses to 148k pairs, not 12k, so it carries most of the gold
                            NSFW signal without oversampling; Open Mantra oversample
                            capped at ~8×; target bumped 150k → 200k):
  register_tag   gold?   weight%   count      source                     notes
  vn_eroge       yes      35%      70,000    VNTL-v3.1-1k train         148,475 avail; use 47%, no oversample
  vn             no       15%      30,000    VNTL raw pool (filtered)   from 812k raw, filter to top 30k
  manga          yes       4%       8,000    Open Mantra (3 vols train) 955 avail; 8.4× oversample
  sfx            yes       6%      12,000    SFX merged                 16k avail, no oversample
  anime_sub      no       20%      40,000    JESC filtered              from 2.8M, filter to top 40k
  novel          no       12%      24,000    NilanE ParallelFiction     dialogue-heavy filter; cut from 18%
  synthetic      no        4%       8,000    Aratako/Synthetic-JP-EN-20k
  novel          no        2%       4,000    NilanE SmallParallelDocs   cut to minimum — formal-prose drag
  anchor         no        2%       4,000    Tatoeba jpn-eng            cut to minimum
  — total                         200,000    (gold 45% · NSFW-register 50% · SFW-dialogue 24%)

Stage 5 — hold-outs: never touch these in training
  VNTL-128 public (from vntl-leaderboard)
  Open Mantra 2 volumes (volume-level split)
  FLORES-200 JA-EN devtest (1012 pairs)
  Probe suite: ~300 hand-built adversarial (seed stub → expand manually)
  Regression canary: 500 general JA-EN (news_commentary or WMT23)
  Custom 1,500 manga OCR bubbles: ONLY IF they exist — verify before locking;
    if absent, 4,500 → 3,000 held-out total, note it.
```

### What we are NOT doing

- **No teacher distillation**. Parallel corpora only. Benefits: not capped by teacher quality, higher-fidelity human-TL data, no propagation of teacher errors.
- **No synthetic pairs from monolingual JP corpora** (alpindale VN/LN, winglian VN JSON) — using them would require a teacher. Excluded from v1; revisit in v2 if needed.
- **No CPO preference stage in v1** — SFT only. Revisit in v2 if VNTL-128 cosine motivates it.

## 5. Training — Qwen3-1.7B-Base with Unsloth

Full hyperparams live in `backend/training/configs/qwen3_sft.yaml` (canonical). Key values:

```yaml
model: Qwen/Qwen3-1.7B-Base · dtype: bf16 · max_seq_length: 2048
lora:  r=32 · alpha=64 · dropout=0.05 · targets=[q,k,v,o,gate,up,down]  (reduced from v4's r=64)
train: 2 epochs · per_device_bs=2 · grad_accum=8 (eff bs=16)
       lr=1e-4 cosine · warmup 3% · save/eval every 500 steps · keep top-3 by CometKiwi-22
output: backend/training/runs/manga-bubbles/qwen3_1p7b_sft/
```

Escalation config `qwen3_4b_sft.yaml` — same recipe, `per_device_bs=1 / grad_accum=16`.

Thinking-mode disable for Qwen3: render the chat template once with `enable_thinking=False` and **save the rendered string** as the new `chat_template`. Do NOT `.replace("{% if enable_thinking %}", "{% if false %}")` — that silently no-ops on template whitespace drift.

Prompt format (matches VNTL leaderboard so cosine eval is apples-to-apples):
```
Translate the following Japanese to English. Output only the translation.

Japanese: {jp}
English:
```

Inference sampler per Shisa's CLTL mitigation: `min_p=0.1, top_p=0.9, temperature=0.2`.

### Budget (RTX 5090, LoRA bf16)

| Stage | Wall time |
|---|---|
| Qwen3-1.7B-Base SFT (150k pairs × 2 ep) | ~3 h |
| (Escalation) Qwen3-4B-Base SFT | ~7–9 h |
| GGUF export + quantize (Q4_K_M + Q8_0) | ~15 min |

GPU contention: user pauses the running `comic-text-detector` training (27 GB on the RTX 5090) before SFT. No cloud fallback needed.

## 6. Evaluation — 4 layers

### Layer 1: Fast (every 500 training steps, 200-sample dev)

- **chrF++** (sacrebleu, char-level, CJK-friendly, free) — segment-aggregated
- **CometKiwi-22** (`Unbabel/wmt22-cometkiwi-da`, 560M, ~3 GB VRAM, 50/sec)
- Length-ratio sanity

Cost: ~2 min per eval. Plotted on wandb.

### Layer 2: Slow neural (every epoch, 2.5k samples)

- **MetricX-24-Hybrid-Large** (1.2B, ~5 GB, 20/sec) — WMT24 top-tier, handles missing refs
- **XCOMET-XL** (3.5B, ~10 GB bf16 or ~6 GB 8-bit quant) — error-span output

Run on: FLORES-1012 + VNTL-128 public + Open Mantra test ~400 + custom-500.

### Layer 3: Failure-mode probes (every checkpoint, instant)

| Probe | Rule | Target |
|---|---|---|
| Name preservation | 20-name gazetteer → exact romaji in EN | 100% |
| Honorific leak | `\b\w+-(san\|kun\|chan\|sama\|senpai\|sensei)\b` | <2% |
| Curly punctuation | count of `‘’“”…` | 0 |
| Repetition loop | VNTL's `detect_high_repetition` (substring ≥25 reps) | <1% |
| Refusal rate (adult slice) | `I can't\|I cannot\|inappropriate\|as an AI` | <1% |
| Length sanity | `len(en)/len(jp) ∈ [0.3, 4.0]` | 99% |
| SFX hit rate | 80-SFX curated dict (どきどき→thump thump etc.) | >70% |
| Idiom set | 50 hand-picked idioms, LLM-judged binary | >80% |

Fail checkpoint if any probe regresses >5pp vs best prior.

### Layer 4: Release candidate only

- **XCOMET-XXL** (10.7B, 3-bit quant ~5 GB, zero quality loss per xCOMET-lite) on full 4,500 (or 3,000 if custom-manga slice absent) held-out.
- **GEMBA-MQM** with local Qwen2.5-72B-Instruct (4-bit, ~40 GB) judge — MQM error-category output. If local judge is too slow, fall back to GPT-4o via `OPENAI_API_KEY`.
- No human eval in v1 (see §7).

Calibration sanity: if we ever do enable human eval, run GEMBA-MQM via GPT-4o on a 100-sample control first to detect local-judge drift.

### Held-out eval budget — up to 4,500 segments

| Slice | N | Notes |
|---|---|---|
| VNTL-128 (public val) | 128 | `all-mpnet-base-v2` cosine (VNTL's metric) — primary directional target |
| FLORES-200 JA-EN devtest | 1,012 | Regression canary |
| Open Mantra (2 of 5 vols, volume-level split) | ~400 | Only pro-manga pairs publicly available |
| Custom manga (from OCR, volume-level split) | 1,500 *or 0* | Include only if the corpus actually exists; otherwise drop and note |
| Probes (names, SFX, refusal, loops…) | ~300 | Seed with ~30 stub rows across 8 categories; expand manually |
| Regression canary (news/general) | 500 | `Helsinki-NLP/news_commentary` or WMT23 fallback |

Total: **4,500 if OCR slice exists, else 3,000**. Cross-dedup against train parquet at Stage 3b.

## 7. Human eval — skipped for v1

Ship from automated metrics + probe suite. Revisit with a gold tier (MQM span + pairwise + 5-scale, ~$650) for v2 once we have a baseline to compare against.

## 8. Deployment — Python backend + browser extension

```bash
# Merge LoRA, convert, quantize
python backend/scripts/train/merge_and_export_gguf.py \
  --base Qwen/Qwen3-1.7B-Base \
  --lora backend/training/runs/manga-bubbles/qwen3_1p7b_sft/final \
  --out  backend/training/weights/qwen3-mt/ \
  --quants Q4_K_M,Q8_0 \
  --llama-cpp-dir <path-to-llama.cpp>
# → backend/training/weights/qwen3-mt/{model.Q4_K_M.gguf, model.Q8_0.gguf}
```

Serve via `llama-cpp-python` inside the Python backend. Wire behind the existing `backend/app/services/local_translation_service.py` — extend it, don't duplicate. Expose the new model with a flag so we can A/B vs current translator on the same endpoint.

Default quant for production = **Q4_K_M** (lower latency, ~1.0 GB); keep Q8_0 around for quality comparison. Offload all layers to GPU when available (`n_gpu_layers=-1`). For batched bubble-translation: use `llama-cpp-python`'s `create_completion` with prompt-caching on the shared instruction prefix (save the first N tokens' KV across bubbles).

## 9. Execution sequence (~2 weeks)

**Week 1 — Data pipeline + eval harness** (runs while `comic-text-detector` training still holds the GPU)

- Loaders for 8 sources → standardized parquet schema
- Quality filters (langid, MeCab, KenLM, CometKiwi, LaBSE)
- MinHash dedup (within train + cross-dedup vs held-out)
- Weighted mix composer → `data.parquet`
- Held-out assembly (verify 4,500 target feasible)
- Eval harness L1/L2/L4 + probes.py + vntl_cosine.py runnable on any HF checkpoint (smoke against base Qwen3-1.7B for baseline)

**Week 2 — Train + export** (user pauses CTD before starting)

- Smoke-test SFT on 1k subset (~15 min sanity)
- Full SFT Qwen3-1.7B-Base (~3 h)
- L2 on best checkpoint; L3 on all; decide if escalation to Qwen3-4B is warranted
- (If escalate) SFT Qwen3-4B-Base (~8 h)
- L4 release-candidate eval
- Merge LoRA → GGUF Q4_K_M + Q8_0 → wire into `local_translation_service.py` → extension smoke test end-to-end

## 10. Open decisions

All D1–D8 from v4 are resolved. Directional summary:
- Data: parallel-only (no teacher); mix rebalanced toward manga/VN gold.
- Arms: single (Qwen3-1.7B primary, Qwen3-4B escalation).
- Preference stage: none in v1.
- Human eval: none in v1.
- Mobile: deferred.
- GPU: user pauses CTD before SFT.
- Dataset hosting: local only.
- Timeline: ~2 weeks.

---

## Files to build

```
backend/scripts/data/
  load_jesc.py load_parallelfiction.py load_vntl.py load_open_mantra.py
  load_sfx.py load_nilane_small.py load_aratako_synth.py load_tatoeba.py
  unify_schema.py filter_jp.py filter_en.py
  score_cometkiwi.py labse_filter.py dedup_minhash.py
  compose_training_mix.py build_held_out.py

backend/training/configs/
  qwen3_sft.yaml
  qwen3_4b_sft.yaml   # escalation path only

backend/scripts/train/
  sft_qwen3_unsloth.py
  merge_and_export_gguf.py

backend/scripts/eval/
  probes.py run_l1.py run_l2.py run_l4.py
  gemba_mqm_judge.py vntl_cosine.py

backend/training/eval_held_out/
  vntl128.jsonl flores_ja_en.jsonl open_mantra_test.jsonl
  probes.jsonl regression_canary.jsonl
  MANIFEST.json
```

Extend, don't create: `backend/app/services/local_translation_service.py` to route through the new GGUF.

## Kraken-agent partition (currently executing)

- Agent 1 → data pipeline (`backend/scripts/data/` + held-out construction)
- Agent 2 → eval harness (`backend/scripts/eval/`, probe set)
- Agent 3 → training wrappers + configs (`backend/scripts/train/`, `backend/training/configs/`)

Three agents, disjoint paths, running in parallel. Training execution is gated on user go-ahead.
