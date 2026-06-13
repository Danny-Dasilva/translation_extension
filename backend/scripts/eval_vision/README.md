# Gemma 4 31B — text vs full-page-vision vs per-bubble-vision A/B/C

Runs three translation modes on the same 45 pages and scores them:

| Mode | Inputs |
|------|--------|
| A — text-only    | `[N]JP` tagged strings from `08_translate_prompt.txt` |
| B — full page    | tagged JP + the whole `01_original.png` attached |
| C — bubble crops | tagged JP + one cropped bubble image per `[N]` tag |

## 1. Launch the model server (once, on the RTX 5090 box)

Install `llama-cpp-python` with server extras, or use the llama.cpp `llama-server` binary. For the GGUF weights on the NAS:

```bash
# Option A — llama-cpp-python server (matches existing translation stack)
uv pip install "llama-cpp-python[server]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
python -m llama_cpp.server \
  --model /mnt/nas/drive_2/ml-models/gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf \
  --clip_model_path /mnt/nas/drive_2/ml-models/gemma-4-31B-it-GGUF/mmproj-F16.gguf \
  --chat_format gemma \
  --n_gpu_layers -1 \
  --n_ctx 16384 \
  --host 127.0.0.1 --port 8080

# Option B — llama.cpp llama-server binary (best multimodal support, usually)
llama-server \
  -m /mnt/nas/drive_2/ml-models/gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf \
  --mmproj /mnt/nas/drive_2/ml-models/gemma-4-31B-it-GGUF/mmproj-F16.gguf \
  -ngl 99 -c 16384 --host 127.0.0.1 --port 8080
```

## 2. Run the A/B/C eval

```bash
cd backend
uv run python scripts/eval_vision/translate_ab.py \
  --gallery ~/manga-output/644289 \
  --out ~/manga-output/644289-abc-gemma4 \
  --pages 001 002 005 010 015 \
  --modes A B C \
  --server http://127.0.0.1:8080
```

Omit `--pages` to run the full 45. Modes default to `A B C`.

## 3. Score + view

```bash
uv run python scripts/eval_vision/score_ab.py \
  --run ~/manga-output/644289-abc-gemma4 \
  --html ~/manga-output/644289-abc-gemma4/report.html
```

Open `report.html` in a browser.
