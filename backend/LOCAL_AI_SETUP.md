# Local AI Pipeline Setup

This guide covers setting up the local AI manga translation pipeline using:
- **YOLOv10n** - Speech bubble detection (NMS-free, ~2ms)
- **PaddleOCR-VL-For-Manga** - Vision-Language OCR for Japanese text
- **HY-MT1.5-1.8B-GGUF** - Quantized translation model

## Requirements

- Python 3.11+
- CUDA-capable GPU with 6GB+ VRAM (RTX 3060/4060 or better)
- ~6GB disk space for models

## Setup Steps

### 1. Install Dependencies

```bash
cd backend
uv sync
```

This installs:
- `ultralytics` - YOLOv10 inference
- `transformers` - PaddleOCR-VL model loading
- `llama-cpp-python` - GGUF inference for translation
- `torch` - PyTorch backend

### 2. Download Models (~4GB)

```bash
# Download all models
uv run python scripts/download_models.py --all

# Or download individually:
uv run python scripts/download_models.py --ocr          # PaddleOCR-VL (~2GB)
uv run python scripts/download_models.py --translation  # HY-MT1.5 (~2GB)
```

Models are saved to `backend/app/weights/`:
```
backend/app/weights/
├── paddleocr-vl/              # OCR model
└── HY-MT1.5-1.8B-Q8_0.gguf   # Translation model
```

The YOLOv10n model is already at `backend/app/models/yolov10n_manga.pt`.

### 3. Run the Server

```bash
uv run uvicorn app.main:app --reload --port 8000
```

The server will load all three models at startup (~30 seconds).

## Performance

### Expected Latency (per image)

| Stage | Time |
|-------|------|
| Detection (YOLOv10n) | ~2ms |
| OCR per bubble | ~100ms |
| Translation per text | ~50ms |
| **Total (10 bubbles)** | **~1.5s** |

Compared to cloud pipeline (Gemini): 3-5 seconds

### VRAM Usage

| Model | VRAM |
|-------|------|
| YOLOv10n | ~100MB |
| PaddleOCR-VL | ~2GB |
| HY-MT1.5 (Q8) | ~2.5GB |
| **Total** | **~5GB** |

## Configuration

Environment variables (in `.env`) or settings in `app/config.py`:

```bash
# Model paths (defaults shown)
YOLO_MODEL_PATH=app/models/yolov10n_manga.pt
OCR_MODEL_ID=jzhang533/PaddleOCR-VL-For-Manga
TRANSLATION_MODEL_PATH=app/weights/HY-MT1.5-1.8B-Q8_0.gguf

# Performance tuning
DETECTION_CONFIDENCE=0.25
DETECTION_IMAGE_SIZE=640
TRANSLATION_BATCH_MODE=concatenated  # or "individual"
```

## Troubleshooting

### CUDA out of memory
- Reduce `DETECTION_IMAGE_SIZE` to 480
- Use CPU for translation: set `n_gpu_layers=0` in `local_translation_service.py`

### Slow first request
- Models are loaded eagerly at startup
- First inference may be slower due to CUDA kernel compilation

### Missing models
```bash
# Check what's downloaded
ls -la app/weights/

# Re-download if needed
uv run python scripts/download_models.py --all
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐
│   YOLOv10n      │───▶│  PaddleOCR-VL    │───▶│   HY-MT1.5        │
│   (Detection)   │    │  (OCR)           │    │   (Translation)   │
├─────────────────┤    ├──────────────────┤    ├───────────────────┤
│ ~2ms inference  │    │ ~100ms per crop  │    │ ~50ms per text    │
│ NMS-free        │    │ Flash Attention  │    │ GGUF Q8_0         │
│ ~100MB VRAM     │    │ ~2GB VRAM        │    │ ~2.5GB VRAM       │
└─────────────────┘    └──────────────────┘    └───────────────────┘
```

## Files

### Services
- `app/services/detector_service.py` - YOLOv10n bubble detection
- `app/services/manga_ocr_service.py` - PaddleOCR-VL text recognition
- `app/services/local_translation_service.py` - HY-MT1.5 translation

### Configuration
- `app/config.py` - Model paths and settings
- `scripts/download_models.py` - Model download utility

### Pipeline
- `app/routers/translate.py` - Orchestrates detect → crop → OCR → translate
