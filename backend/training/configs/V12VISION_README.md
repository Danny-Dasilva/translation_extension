# v12vision — Gemma-4 E4B Vision-LoRA (Phase 2: page image as context)

**Status: SCAFFOLDING ONLY. Nothing here has touched a GPU.** The text model
(v11fix7) trains first; this is ready-to-run Phase-2 scaffolding plus the
on-box verification checklist. Every API name flagged `?? VERIFY` in the
trainer must be confirmed on the box before a real run.

Files:
- `backend/training/configs/gemma4_e4b_v12vision_sft.yaml` — the config
- `backend/scripts/train/sft_gemma4_e4b_v12vision.py` — the trainer
- `backend/scripts/eval/serve_v12vision_vllm.sh` — the serve stub (untested)
- this README

---

## 1. Why a CLEAN multimodal base (not the merged text model)

Production (v11 / v11fix6 / v11fix7) is a **text-only** SFT. Its LoRA regex
targeted `model.language_model.*` **only** — the vision tower and multimodal
projector were never adapted. So the merged text checkpoint's vision tower is
**intact but unconditioned** dead weight: fine for text serving, useless as a
Phase-2 starting point.

Phase 2 = feed the **page image** as context (the #1 ROI lever for the
pronoun/speaker ceiling). That requires the original multimodal base
**`unsloth/gemma-4-E4B-it`** (a `Gemma4ForConditionalGeneration`: text + vision
+ audio towers).

> **Do NOT warm-start from a merged text checkpoint.** It bakes the text-only
> LoRA delta into the language tower while leaving the vision/projector path
> uncalibrated — you'd be adapting a vision tower against an already-shifted
> language tower.

---

## 2. The exact vision module names to LoRA

These are **verified from this repo** — `backend/scripts/eval/merge_gemma4_lora_clean.py`
(docstring + the v9c adapter's real safetensors keys) and the merged
`config.json` architecture (`text_config` 42 layers, `vision_config`
`gemma4_vision` 16 layers / hidden 768, `audio_config` 12 layers):

| Tower | Module path | Leaf type | Count |
|---|---|---|---|
| Language | `model.language_model.layers.{0..41}.{self_attn\|mlp}.{q,k,v,o\|gate,up,down}_proj` | bare `nn.Linear` | 42 × 7 = **294** |
| Vision | `model.vision_tower.encoder.layers.{0..15}.{self_attn\|mlp}.{q,k,v,o\|gate,up,down}_proj.linear` | `nn.Linear` **inside** a `Gemma4ClippableLinear` wrapper | 16 × 7 = **112** |
| Audio | `model.audio_tower.layers.{0..11}.self_attn.{...}.linear` | wrapped | 36 (**excluded**) |
| Projector | `model.multi_modal_projector.*` | see below | likely 0 LoRA-able |

**Critical nuance — the `.linear` suffix.** Vision/audio projections are
`Gemma4ClippableLinear` wrappers; the real `nn.Linear` is the child at
`...<proj>.linear`. The v9c adapter stored `...q_proj.linear.lora_A.weight` for
vision vs `...q_proj.lora_A.weight` for language. So an **explicit peft regex**
must end in `.linear` on the vision branch but NOT on the language branch.
Expected explicit-regex match count = **294 + 112 = 406** (audio excluded).
That regex is in the YAML at `lora.target_modules_regex_explicit`.

**The multimodal projector probably has nothing to LoRA.** In the Gemma family
the projector is a learned `nn.Parameter` (`mm_input_projection_weight`) + an
RMSNorm — **no `nn.Linear`** — and the clean-merge script only ever reported
sections `{language_model, vision_tower, audio_tower}`, never a projector
section. If you want to adapt the projector, set that Parameter
`requires_grad=True` directly (full-tune the one tensor) rather than expecting
LoRA to attach. **CONFIRM on the box** with the snippet below.

### Primary mechanism: unsloth boolean flags (not a hand-written regex)

The trainer uses unsloth's documented vision API and lets unsloth resolve the
wrapped Linears per-architecture:

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,    # <- the Phase-2 point
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
    random_state=42,
)
```

The trainer then **audits trainable LoRA params by tower** and **bails if
`vision_tower == 0`** (i.e. unsloth didn't recognize `gemma4_vision`'s
ClippableLinear). If it bails, fall back to the explicit regex via
`get_peft_model(target_modules=<regex>)`.

### Verify the names on the box (do this FIRST)

```bash
# Loads the base, attaches LoRA, prints vision module names + the per-tower
# LoRA section audit + one converted conversation, then EXITS without training:
cd /home/danny/Documents/personal/extension
python backend/scripts/train/sft_gemma4_e4b_v12vision.py --inspect
```

Or inspect the raw module tree directly:

```python
from transformers import Gemma4ForConditionalGeneration
m = Gemma4ForConditionalGeneration.from_pretrained(
    "unsloth/gemma-4-E4B-it", device_map="cpu", dtype="bfloat16")
import torch.nn as nn
# vision Linears (note the wrapper):
for n, mod in m.named_modules():
    if "vision_tower" in n and isinstance(mod, nn.Linear):
        print(n)                      # expect ...encoder.layers.N.self_attn.q_proj.linear
# projector — is there ANY nn.Linear? (probably not):
for n, mod in m.named_modules():
    if "multi_modal_projector" in n:
        print(type(mod).__name__, n)  # expect Parameter/RMSNorm, no Linear
```

---

## 3. The multimodal prompt format

One **per-page** dataset row → one user turn (image + OCR + context) → one
assistant turn (`en_target`). Built in `build_conversations()`:

```python
{"messages": [
  {"role": "user", "content": [
     {"type": "image", "image": <PIL.Image RGB>},          # the page image
     {"type": "text",  "text":
        "Translate this manga page from Japanese to English. Use the page "
        "image, the OCR text, and the surrounding context for speakers, "
        "pronouns, and continuity. Output only the English translation.\n\n"
        "Page OCR:\n<jp_ocr>\n\nContext:\n<page_context>"},
  ]},
  {"role": "assistant", "content": [
     {"type": "text", "text": "<en_target>"},
  ]},
]}
```

- The instruction text **mirrors the v11 page-context recipe's intent**
  (instruction + page OCR + context) but is adapted to whole-page multimodal:
  the POC schema is one row per page, so there is no per-bubble "marked line K".
- `jp_ocr` and `page_context` blocks are omitted when empty.
- Loss is **completion-only** (prompt masked); the unsloth vision collator
  handles label masking + image preprocessing.
- `?? VERIFY`: the typed-content-block schema (`{"type":"image","image":PIL}`)
  is the unsloth/HF vision-chat convention — confirm `UnslothVisionDataCollator`
  on 2026.6.7 consumes exactly this (the `--inspect` dump shows one sample).

---

## 4. Merging the vision-LoRA (for serving)

**Preferred — unsloth native merge** (handles vision + projector keys). Easiest
to do right after training in the trainer, or in a tiny standalone script:

```python
import unsloth
from unsloth import FastVisionModel
model, processor = FastVisionModel.from_pretrained(".../v12vision_poc/final")  # ?? VERIFY load-adapter API
model.save_pretrained_merged(                                                  # ?? VERIFY merge API
    ".../v12vision_poc/merged", processor, save_method="merged_16bit")
```

> **Do NOT assume the repo's `merge_gemma4_lora_clean.py` covers vision.** It
> was written for **language-only** adapters. Its key-walker *does* resolve
> `.linear` leaves (it merged v9c's vision/audio keys historically), so it may
> work — but it has **not** been validated for a vision-LoRA whose weights you
> actually intend to keep. Use the unsloth native merge unless/until the
> clean-merge path is validated for the vision tower on the box.

The merged dir must be a **full multimodal checkpoint** (vision tower +
projector intact + LoRA folded in) for the serve path below.

---

## 5. Serve path: image + text → EN

Two options; the box should be checked for which is viable.

### Option A — vLLM multimodal (preferred if supported): `serve_v12vision_vllm.sh`

Mirrors `serve_v10it_vllm.sh` but for the vision path. Key differences:
- **No MTP drafter** — the official Gemma-4 MTP drafter is text-only; pairing
  it with multimodal prompts is unverified, so it is left OFF.
- `--limit-mm-per-prompt image=1` advertises one image per prompt.
- Larger `--max-model-len` (default 4096): each image is ~**280 soft tokens**
  (`vision_soft_tokens_per_image`) on top of OCR + context + output.
- Exports `VLLM_USE_FLASHINFER_SAMPLER=0` and `TORCH_CUDA_ARCH_LIST=12.0` per
  the rig's cu130/sm120 requirements.

> **`?? VERIFY`: vLLM on this rig must support `Gemma4ForConditionalGeneration`
> MULTIMODAL (image) inputs.** vLLM supports Gemma3 vision; gemma4 vision is
> UNCONFIRMED here. If unsupported, use Option B.

Query it (OpenAI-compatible chat with an image):

```bash
# image as base64 data URL (page on local disk):
IMG_B64=$(base64 -w0 /path/to/page.png)
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "v12vision",
  "messages": [{"role":"user","content":[
     {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$IMG_B64"'"}},
     {"type":"text","text":"Translate this manga page from Japanese to English. Use the page image, the OCR text, and the surrounding context for speakers, pronouns, and continuity. Output only the English translation.\n\nPage OCR:\n<jp_ocr>\n\nContext:\n<page_context>"}
  ]}],
  "max_tokens": 256, "temperature": 0
}'
```

### Option B — unsloth native inference (guaranteed to work, single-request)

If vLLM lacks gemma4-vision support, serve via unsloth directly:

```python
import unsloth
from unsloth import FastVisionModel
from PIL import Image

model, processor = FastVisionModel.from_pretrained(".../v12vision_poc/merged")  # or base+adapter
FastVisionModel.for_inference(model)                                            # ?? VERIFY inference API

img = Image.open("/path/to/page.png").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Translate this manga page ... Output only the English translation.\n\nPage OCR:\n<jp_ocr>\n\nContext:\n<page_context>"},
]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(images=img, text=prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=256, temperature=0.0)
print(processor.decode(out[0], skip_special_tokens=True))
```

Wrap Option B in a tiny FastAPI/loop for a batch eval harness if needed; it is
slower than vLLM but does not depend on vLLM's gemma4-vision support.

---

## 6. Assumptions needing on-box verification (consolidated)

These are the `?? VERIFY` flags from the trainer/serve scaffolding:

1. **`FastVisionModel`** is the unsloth vision entrypoint and
   `FastVisionModel.from_pretrained(...)` returns `(model, processor)` on
   unsloth 2026.6.7.
2. **`FastVisionModel.get_peft_model(finetune_vision_layers=True, ...)`** boolean
   flags exist and actually adapt `gemma4_vision`'s `Gemma4ClippableLinear`
   wrappers. (The trainer's per-tower section audit catches a silent skip and
   bails — run `--inspect` first.)
3. **`from unsloth.trainer import UnslothVisionDataCollator`** import path, and
   that it consumes the typed-content-block message schema used here.
4. **`FastVisionModel.for_training(model)` / `.for_inference(model)`** mode
   toggles exist.
5. **`SFTConfig` field names on trl 0.23.1** — `max_length` (renamed from
   `max_seq_length` in trl ≥0.20), and that `dataset_kwargs={"skip_prepare_dataset":
   True}` + `remove_unused_columns=False` correctly bypass text tokenization.
6. **`SFTTrainer(processing_class=processor, ...)`** vs older `tokenizer=`.
7. **`model.save_pretrained_merged(dir, processor, save_method="merged_16bit")`**
   merge API, and whether the repo's `merge_gemma4_lora_clean.py` is safe for a
   vision-LoRA (it is not validated for it).
8. **The `multi_modal_projector` has no `nn.Linear`** (so LoRA attaches nothing
   there); adapt its Parameter directly if projector tuning is wanted.
9. **vLLM supports `Gemma4ForConditionalGeneration` multimodal (image) inputs**
   on this rig; if not, use the unsloth inference path (Option B).
10. **`--limit-mm-per-prompt`** syntax (`image=N` key=val vs `'{"image":N}'`
    JSON) for the installed vLLM build.
11. **`max_seq_length=2048`** is enough for image soft-tokens (~280) + page OCR +
    context + target — tune to the POC page-token p99.
12. **VRAM / batch**: per-device batch 2 × grad-accum 8 (eff 16) and `lr=1e-4`
    are conservative starting points for vision-LoRA on the RTX 5090; verify no
    OOM in the dry-run and revisit `r`/`lr` per dry-run loss.
