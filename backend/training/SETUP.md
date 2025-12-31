# YOLOv10n Training Setup Guide

Complete instructions for setting up and running the manga speech bubble detection training pipeline on any machine.

## Prerequisites

- **Python 3.11+**
- **NVIDIA GPU** with CUDA (6+ GB VRAM recommended)
- **uv** package manager ([install guide](https://github.com/astral-sh/uv))
- **Roboflow account** (free tier works)

## Step 1: Install uv (if not installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

## Step 2: Clone/Copy the Training Directory

Ensure you have the `backend/training/` directory with all files:

```
backend/training/
├── pyproject.toml
├── .gitignore
├── .python-version
├── README.md
├── configs/
│   └── default.yaml
├── scripts/
│   ├── download_dataset.py
│   ├── download_model.py
│   └── validate_dataset.py
├── train.py
├── validate.py
└── export.py
```

## Step 3: Setup Python Environment

```bash
cd backend/training

# Create virtual environment and install dependencies
uv sync
```

This installs:
- ultralytics (YOLOv10)
- torch + torchvision
- roboflow
- opencv-python
- albumentations
- and other training dependencies

## Step 4: Get Roboflow API Key (Free)

1. Go to https://app.roboflow.com/
2. Sign up or log in (free tier is sufficient)
3. Click your profile icon → **Settings**
4. Go to **API Keys** tab
5. Copy your **Private API Key**
`
## Step 5: Download Dataset

```bash
# Set your API key
export ROBOFLOW_API_KEY="44YDd0Z9w0JIbhFaJcYa"

# Download dataset (interactive selection)
uv run python scripts/download_dataset.py

# Or specify directly:
uv run python scripts/download_dataset.py --dataset manga-bubble-detect
```

Available datasets:
| Dataset | Images | Description |
|---------|--------|-------------|
| manga-bubble-detect | 4,492 | Largest, recommended |
| speech-bubble-detector | ~2,000 | Speech bubbles |
| comic-text-segmentation | ~1,500 | Comic text/bubbles |

## Step 6: Validate Dataset

```bash
uv run python scripts/validate_dataset.py
```

This checks:
- Total instance count (warns if < 10,000)
- Class distribution
- Recommends augmentation settings

## Step 7: Download YOLOv10n Model

```bash
uv run python scripts/download_model.py --model yolov10n
```

Downloads pretrained weights to `weights/yolov10n.pt` (~11 MB).

## Step 8: Start Training

```bash
# Default training (50 epochs, batch=16)
uv run python train.py

# Custom options
uv run python train.py --epochs 100 --batch 8
```

### Training Output

- Progress displayed in terminal
- Checkpoints saved to `runs/manga-bubbles/weights/`
- Best model copied to `../app/models/yolov10n_manga.pt`

### GPU Memory Issues

If you get "CUDA out of memory" errors:
```bash
uv run python train.py --batch 8   # Reduce batch size
uv run python train.py --batch 4   # Even smaller
```

## Step 9: Validate Results

```bash
uv run python validate.py
```

Reports:
- mAP@50, mAP@50-95
- Precision, Recall
- Confusion matrix

## Step 10: Export for Production

```bash
# Export to ONNX
uv run python export.py

# With options
uv run python export.py --format onnx --half  # FP16 for faster inference
```

Output saved to `../app/models/yolov10n_manga.onnx`

---

## Quick Start (All Commands)

```bash
# 1. Setup
cd backend/training
uv sync

# 2. Set API key
export ROBOFLOW_API_KEY="your_key_here"

# 3. Download everything
uv run python scripts/download_dataset.py --dataset manga-bubble-detect
uv run python scripts/download_model.py --model yolov10n

# 4. Validate dataset
uv run python scripts/validate_dataset.py

# 5. Train
uv run python train.py

# 6. Validate and export
uv run python validate.py
uv run python export.py
```

---

## Troubleshooting

### "CUDA not available"
- Install NVIDIA drivers
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### "roboflow not found"
```bash
uv sync  # Re-run dependency installation
```

### Training is slow
- Check GPU is being used: watch `nvidia-smi`
- Reduce image size: edit `configs/default.yaml`, set `imgsz: 416`

### Overfitting (val loss increases)
- Dataset may be too small
- Training will auto-stop (patience=10)
- Try: `uv run python train.py --epochs 30`

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8+ GB |
| RAM | 8 GB | 16+ GB |
| Storage | 5 GB | 20+ GB |
| GPU | GTX 1060 | RTX 3060+ |

Training time: ~30-60 min for 50 epochs on RTX 3060.

---

## Why YOLOv10?

YOLOv10 is **NMS-free** (No Non-Maximum Suppression), providing:
- Deterministic inference time regardless of detection count
- Same latency for 1 bubble or 50 bubbles
- Ideal for manga pages with many speech bubbles
