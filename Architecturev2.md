Manga Translation Extension - System Architecture (Revised v2.0)Table of Contents1.(#system-overview)2.(#technology-stack)3.(#backend-architecture-fastapi-local-ai)4.(#yolov10-fine-tuning-guide)5.(#data-flow-api-specification)System OverviewThis revised architecture replaces cloud APIs (Gemini) and generic OCR (RapidOCR) with a fully local, high-performance pipeline optimized for manga. It utilizes YOLOv10 for instant bubble detection, PaddleOCR-VL-For-Manga for specialized Japanese text recognition, and HY-MT1.5 for context-aware translation.High-Level ArchitectureCode snippetgraph LR
    A -- Base64 Image --> B(FastAPI Server)
    B -- 1. Detect Bubbles --> C
    C -- Bounding Boxes --> B
    B -- 2. Crop & OCR --> D
    D -- Japanese Text --> B
    B -- 3. Translate --> E
    E -- English Text --> B
    B -- JSON Response --> A
Technology StackBackend (Local Inference Server):Framework: FastAPI (Python 3.10+)Detection: YOLOv10-Nano (Fine-tuned on Manga Bubbles)Why: Removes NMS (Non-Maximum Suppression) overhead for lowest latency.OCR: PaddleOCR-VL-For-Manga (0.9B Parameter VLM)Why: Specialized Vision-Language model that outperforms standard OCR on vertical/stylized manga text.Translation: HY-MT1.5-1.8B-GPTQ-Int4Why: 4-bit quantization allows it to run on <2GB VRAM with extreme speed while retaining high translation quality.Server: Uvicorn (ASGI)Frontend (Browser Extension):Core: TypeScript, ViteLogic: main_extension.js (rewritten) for DOM observation.Rendering: HTML5 Canvas (overlays translated text on original image).Backend Architecture (FastAPI)Project Structurebackend/├── app/│   ├── main.py              # FastAPI entry point│   ├── services/│   │   ├── detector.py      # YOLOv10 inference│   │   ├── ocr.py           # PaddleOCR-VL inference│   │   └── translator.py    # HY-MT1.5 inference│   ├── models/│   │   └── yolov10n_manga.pt # Your fine-tuned model│   └── routers/│       └── pipeline.py      # Orchestration logic└── requirements.txt1. Detection Service (app/services/detector.py)Uses the official YOLOv10 bindings.Pythonfrom ultralytics import YOLOv10
import numpy as np

class BubbleDetector:
    def __init__(self, model_path="app/models/yolov10n_manga.pt"):
        # Load the fine-tuned YOLOv10n model
        self.model = YOLOv10(model_path)

    def detect(self, image_np):
        """
        Input: Numpy Image (OpenCV format)
        Output: List of [x1, y1, x2, y2, confidence]
        """
        # YOLOv10 is NMS-free, making this step extremely fast
        results = self.model.predict(image_np, imgsz=640, conf=0.25, verbose=False)
        
        boxes =
        for result in results:
            for box in result.boxes:
                coords = box.xyxy.cpu().numpy().astype(int)
                boxes.append(coords.tolist())
        
        # Sort boxes by reading order (Right-to-Left, Top-to-Bottom)
        # Simple heuristic: Sort by X (descending) primarily, Y (ascending) secondarily
        boxes.sort(key=lambda b: (-b, b[1]))
        return boxes
2. OCR Service (app/services/ocr.py)Integrates the specialized 0.9B parameter PaddleOCR-VL model.Pythonfrom transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image

class MangaOCRService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "jzhang533/PaddleOCR-VL-For-Manga"
        
        # Load model in bfloat16 for speed
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def recognize_text(self, image_crops):
        """
        Batch process image crops to extract Japanese text.
        """
        texts =
        for crop in image_crops:
            # Prepare inputs
            messages =}
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(images=[crop], text=[text], return_tensors="pt").to(self.device)
            
            # Generate text (set max_new_tokens low for speed, manga bubbles are short)
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Post-processing to remove prompt artifacts
            clean_text = output_text.split("OCR:")[-1].strip()
            texts.append(clean_text)
            
        return texts
3. Translation Service (app/services/translator.py)Uses the lightweight, quantized HY-MT1.5 model.Pythonfrom transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TranslatorService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "tencent/HY-MT1.5-1.8B-GPTQ-Int4"
        
        # Load quantized model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True
        )

    def translate(self, text_list):
        translated_texts =
        for text in text_list:
            # HY-MT1.5 Prompt Template for Translation
            prompt = f"Translate the following segment into English, without additional explanation. {text}"
            
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=False, 
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model.generate(inputs, max_new_tokens=256)
            decoded = self.tokenizer.decode(outputs, skip_special_tokens=True)
            
            # Extract just the translation (HY-MT usually outputs clean text with this prompt)
            translated_texts.append(decoded)
            
        return translated_texts
YOLOv10 Fine-Tuning GuideTo get YOLOv10n (Nano) to detect manga speech bubbles accurately, you must fine-tune it. The generic pre-trained model detects "persons" and "cars," not bubbles.Step 1: Prepare the DatasetGo to Roboflow Universe.Search for a dataset like "Manga Speech Bubble Dataset" (e.g., the magi-v2 or Manga109s variants).Click Download Dataset.Select format: YOLOv8 (Compatible with YOLOv10).Download the .zip and extract it. You should have a structure like:dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
Step 2: Environment SetupYOLOv10 uses a fork of Ultralytics or specific bindings. For stability, use the official implementation.Bash# Clone the official YOLOv10 repository
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10

# Install requirements
pip install -r requirements.txt
pip install -e.
Step 3: Run TrainingWe use the Nano model (yolov10n) for maximum speed.Bashyolo task=detect \
     mode=train \
     epochs=50 \
     batch=16 \
     plots=True \
     model=weights/yolov10n.pt \
     data=/path/to/your/dataset/data.yaml \
     imgsz=640
epochs: 50 is usually sufficient for fine-tuning.imgsz: 640 is standard. 1024 might improve small text detection but slows down inference.Result: After training, your new model will be saved at runs/detect/train/weights/best.pt. Move this file to your backend app/models/ folder.Step 4: Export for Inference Speed (Optional but Recommended)For production, export the PyTorch model to ONNX.Bashyolo export model=runs/detect/train/weights/best.pt format=onnx opset=13
Note: If you use ONNX, update the detector.py to use onnxruntime instead of YOLOv10 class.Data Flow & API SpecificationEndpoint: POST /translate_pageWorkflow Logic:Receive: Base64 encoded manga page.Decode: Convert Base64 to NumPy array.Detect: detector.detect(image) -> Returns list of bounding boxes [x1, y1, x2, y2].Crop: Python list comprehension to crop detections from the main image.OCR: ocr_service.recognize_text(crops) -> Returns list of Japanese strings.Translate: translator.translate(japanese_strings) -> Returns list of English strings.Format: Construct JSON response mapping Box Coordinates to Translated Text.Response JSON:JSON{
  "status": "success",
  "predictions": [
    {
      "box_2d": ,
      "original": "何だと！？",
      "translated": "What did you say!?",
      "confidence": 0.95
    },
    {
      "box_2d": ,
      "original": "逃げろ！",
      "translated": "Run away!",
      "confidence": 0.88
    }
  ]
}
Implementation NotesBatching: Both PaddleOCR-VL and HY-MT supports batching. Ensure ocr_service and translator_service process lists of strings/images, not one by one, to maximize GPU throughput.VRAM Usage:YOLOv10n: ~100MBPaddleOCR-VL (BF16): ~2GBHY-MT1.5 (Int4): ~1.5GBTotal: ~3.6GB VRAM. This fits comfortably on an RTX 3060 or 4060.