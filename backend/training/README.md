# YOLOv10n Fine-Tuning for Manga Speech Bubble Detection

Fine-tune YOLOv10-Nano on manga speech bubble datasets for the translation extension.

## Why YOLOv10?

YOLOv10 uses **NMS-free architecture** (Consistent Dual Assignments), providing:

- **Deterministic inference time** regardless of detection count
- Same latency whether the page has 1 or 50 speech bubbles
- Ideal for real-time manga translation

| Model | NMS | Inference Time | Notes |
|-------|-----|----------------|-------|
| YOLOv10n | None | ~1.5ms fixed | Deterministic |
| YOLOv11n | Required | 1.3ms + 2-10ms NMS | Variable |

For manga pages with 30+ bubbles, YOLOv10's consistent timing provides better UX.

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA (6+ GB VRAM recommended)
- [uv](https://github.com/astral-sh/uv) package manager
- Roboflow API key (free tier available)

## Quick Start

```bash
# 1. Setup environment
cd backend/training
uv sync

# 2. Set Roboflow API key
export ROBOFLOW_API_KEY="your_key_here"

# 3. Download dataset (interactive selection)
uv run python scripts/download_dataset.py

# 4. Validate dataset quality
uv run python scripts/validate_dataset.py

# 5. Download YOLOv10n pretrained weights
uv run python scripts/download_model.py

# 6. Start training
uv run python train.py

# 7. Validate results
uv run python validate.py

# 8. Export for production
uv run python export.py
```

## Getting a Roboflow API Key

1. Go to [Roboflow](https://app.roboflow.com/)
2. Sign up or log in
3. Go to Settings > API Keys
4. Copy your **Private API Key**

## Available Datasets

| Dataset | Images | Description |
|---------|--------|-------------|
| manga-bubble-detect | 4,492 | Largest available, recommended |
| speech-bubble-detector | ~2,000 | Speech bubbles for translation |
| comic-text-segmentation | ~1,500 | Comic text and bubbles |

## Training Configuration

Edit `configs/default.yaml` to customize training:

```yaml
# Key parameters
epochs: 50          # Training epochs
batch: 16           # Batch size (reduce if OOM)
imgsz: 640          # Image size

# Augmentation (aggressive for small datasets)
mosaic: 0.8         # Mosaic augmentation
mixup: 0.1          # MixUp augmentation
copy_paste: 0.1     # Copy-paste augmentation
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
uv run python train.py --batch 8
# Or even smaller:
uv run python train.py --batch 4
```

### Overfitting (val loss increases)

The default config uses aggressive augmentation. If still overfitting:
- Reduce epochs
- Increase dropout
- Use early stopping (already enabled with patience=10)

### Poor Detection Quality

1. Check dataset quality: `uv run python scripts/validate_dataset.py`
2. Ensure enough instances (>5,000 recommended)
3. Try training longer: `uv run python train.py --epochs 100`

## Project Structure

```
backend/training/
├── pyproject.toml          # Dependencies
├── configs/
│   └── default.yaml        # Training config
├── scripts/
│   ├── download_dataset.py # Dataset download
│   ├── download_model.py   # Model weights download
│   └── validate_dataset.py # Dataset quality check
├── datasets/               # Downloaded datasets (gitignored)
├── weights/                # Pretrained weights (gitignored)
├── runs/                   # Training outputs (gitignored)
├── train.py                # Main training script
├── validate.py             # Model validation
└── export.py               # ONNX export
```

## Output

After training, the best model is automatically copied to:
```
backend/app/models/yolov10n_manga.pt
```

After export:
```
backend/app/models/yolov10n_manga.onnx
```

## Integration

Use the trained model in the backend:

```python
from ultralytics import YOLO

model = YOLO("app/models/yolov10n_manga.pt")
results = model.predict(image, conf=0.25)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
```

## References

- [YOLOv10 Paper](https://arxiv.org/abs/2405.14458) - NMS-free architecture
- [Ultralytics Docs](https://docs.ultralytics.com/) - Training API
- [Roboflow](https://roboflow.com/) - Dataset hosting
