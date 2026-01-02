# Performance Optimization Reference: Image Translation Pipeline

> **Target:** <500ms total latency | **Constraint:** Maintain current OCR accuracy | **TensorRT:** Available to install

---

## Parallelization Issues & Solutions (2025-01-01)

### Issue 1: Progressive Translation Slowdown ✅ FIXED

**Symptom:** Translations slowed from 31ms → 315ms (10x degradation over 6 calls)

**Root Cause:** `asyncio.to_thread()` + `threading.Lock()` caused:
1. **Thread pool contention:** Default pool has limited workers
2. **Lock thrashing:** Multiple threads competing for single lock
3. **GIL overhead:** Python GIL + context switching accumulated
4. **Progressive degradation:** Each subsequent call waited longer for lock

**Diagnostic Evidence:**
- Test script with SAME text 10x: Consistent 62ms ✓
- Test script with DIFFERENT texts: Consistent 38ms ✓ (no slowdown!)
- API with DIFFERENT texts: 31ms → 315ms ✗ (progressive slowdown)

This proved the issue was in the async/threading layer, NOT llama-cpp.

**Temporary Fix:** Removed `asyncio.to_thread` - runs synchronously (~31ms each)

---

### Issue 2: OCR Semaphore Serialization (Known Limitation)

**Current State:** `asyncio.Semaphore(1)` serializes all OCR to single GPU thread
- Tasks spawn in parallel but execute sequentially
- Useful only for CPU preprocessing overlap

**Why:** PaddleOCR-VL's `model.generate()` is not thread-safe for concurrent calls

---

### Issue 3: Translation Parallelization Disabled

**Current State:** `_translate_parallel()` exists but uses shared instance
- With single Llama instance, parallel calls serialize anyway
- Threading approach caused progressive slowdown (Issue 1)

---

## ✅ IMPLEMENTED: Full Parallel Pipeline with Multi-Instance Pools

### Status: COMPLETE (2025-01-01)

Both OCR and Translation now use multi-instance pools for true parallel processing.
No shared locks = no contention = no progressive slowdown.

### VRAM Budget (RTX 5090 - 32GB)
```
Detection (YOLOv10n):           ~0.5GB
OCR Pool (2 × PaddleOCR-VL):    ~12-16GB (2 × 6-8GB)
Translation Pool (3 × HY-MT):   ~4.5GB   (3 × 1.5GB)
─────────────────────────────────────────────
Total:                          ~17-21GB
Available:                      32GB
Headroom:                       ~11-15GB ✓
```

### New Architecture (Fully Parallel)
```
                    ┌─────────────┐
                    │  Detection  │ 16ms
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Crops     │ (6 crops ready)
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                                   ▼
   ┌───────────┐                       ┌───────────┐
   │  OCR #1   │ ~400ms/crop           │  OCR #2   │ ~400ms/crop
   └─────┬─────┘                       └─────┬─────┘
         │                                   │
         └──────────────┬────────────────────┘
                        │
              [asyncio.Queue - as OCR completes]
                        │
         ┌──────────────┼──────────────┬──────────────┐
         ▼              ▼              ▼
   ┌───────────┐  ┌───────────┐  ┌───────────┐
   │ Trans #1  │  │ Trans #2  │  │ Trans #3  │  ~170ms each
   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
                 ┌──────▼──────┐
                 │  Response   │
                 └─────────────┘
```

### Timeline (6 bubbles, 2 OCR + 3 Translation)
```
Time:    0    200   400   600   800   1000  1200  1400
         |     |     |     |     |     |     |     |
OCR#1:   [====]     [====]     [====]           (crops 0,2,4)
OCR#2:   [====]     [====]     [====]           (crops 1,3,5)
         |     |     |     |     |     |     |
Trans#1:      [==]       [==]
Trans#2:      [==]       [==]
Trans#3:           [==]       [==]

Total: ~1200ms OCR + ~170ms final translation = ~1370ms
```

### Configuration (app/config.py)
```python
# Phase 2 Optimizations (multi-instance parallelization)
ocr_num_instances: int = 2           # Number of OCR model instances (each ~6-8GB VRAM)
translation_num_instances: int = 3    # Number of translation model instances (each ~1.5GB VRAM)
```

### Old Architecture (Single Instance - kept for reference)
```
                    ┌─────────────┐
                    │  Detection  │ 16ms
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐   ┌────────┐   ┌────────┐
         │ Crop 1 │   │ Crop 2 │   │ Crop N │
         └───┬────┘   └───┬────┘   └───┬────┘
             │            │            │
    [OCR Queue - Semaphore(1) serializes GPU]
             │
             ▼
      ┌─────────────┐
      │  OCR VLM    │ ~400ms/crop (sequential)
      └──────┬──────┘
             │
    [asyncio.Queue - as OCR completes]
             │
    ┌────────┼────────┬────────┐
    ▼        ▼        ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│Trans 1│ │Trans 2│ │Trans 3│  ~170ms each (parallel!)
└───┬───┘ └───┬───┘ └───┬───┘
    │        │        │
    └────────┴────────┘
             │
      ┌──────▼──────┐
      │  Response   │
      └─────────────┘
```

### Implementation

**File:** `app/services/manga_ocr_service.py` - MangaOCRPool class
```python
class MangaOCRPool:
    """Pool of OCR model instances for true parallel processing."""

    def __init__(self, num_instances: int = 2):
        self.instances: List[Tuple[AutoModelForCausalLM, AutoProcessor]] = []
        self.semaphores: List[asyncio.Semaphore] = []

        for i in range(num_instances):
            model = AutoModelForCausalLM.from_pretrained(model_id, ...)
            processor = AutoProcessor.from_pretrained(model_id, ...)
            self.instances.append((model, processor))
            self.semaphores.append(asyncio.Semaphore(1))

    async def recognize_parallel(self, crops: List[np.ndarray]) -> List[str]:
        """Process all crops in parallel across instances."""
        async def recognize_one(idx: int, crop: np.ndarray) -> Tuple[int, str]:
            instance_id = idx % self.num_instances
            async with self.semaphores[instance_id]:
                return (idx, self._recognize_sync(model, processor, crop))

        tasks = [recognize_one(i, crop) for i, crop in enumerate(crops)]
        results = await asyncio.gather(*tasks)
        return [r[1] for r in sorted(results, key=lambda x: x[0])]

    async def recognize_streaming(self, crops, result_queue):
        """Stream results to queue for pipeline overlap with translation."""
```

**File:** `app/services/local_translation_service.py` - LocalTranslationPool class
```python
class LocalTranslationPool:
    """Pool of Llama instances for parallel translation."""

    def __init__(self, num_instances: int = 3):
        self.instances: List[Llama] = []
        self.semaphores: List[asyncio.Semaphore] = []

        for i in range(num_instances):
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=-1,
                n_threads=2,  # Reduced per instance
            )
            self.instances.append(llm)
            self.semaphores.append(asyncio.Semaphore(1))

    async def translate_parallel(self, texts: List[str], target_lang: str) -> List[str]:
        async def translate_one(idx: int, text: str) -> Tuple[int, str]:
            instance_id = idx % len(self.instances)
            async with self.semaphores[instance_id]:
                return (idx, self._translate_sync(self.instances[instance_id], text))

        tasks = [translate_one(i, t) for i, t in enumerate(texts)]
        results = await asyncio.gather(*tasks)
        return [r[1] for r in sorted(results, key=lambda x: x[0])]

    async def translate_streaming(self, input_queue, output_queue, target_language):
        """Consume from OCR queue and produce translations."""
```

**File:** `app/routers/translate.py` - Pipeline orchestration
```python
# Initialize pools at startup
if settings.ocr_num_instances > 1:
    ocr_pool = MangaOCRPool()
else:
    ocr_service = MangaOCRService()

if settings.translation_num_instances > 1:
    translation_pool = LocalTranslationPool()
else:
    translation_service = LocalTranslationService()

# Fully parallel pipeline function
async def _process_with_pools(crops, target_language, image_idx):
    """OCR streams to queue, translation consumes in parallel."""
    ocr_queue = asyncio.Queue()

    async def ocr_producer():
        await ocr_pool.recognize_streaming(crops, ocr_queue)
        for _ in range(translation_pool.num_instances):
            await ocr_queue.put(None)  # Signal completion

    async def translation_consumer(worker_id):
        while True:
            item = await ocr_queue.get()
            if item is None:
                break
            idx, text = item
            translation = translation_pool._translate_sync(...)
            final_results[idx] = (text, translation)

    await asyncio.gather(
        ocr_producer(),
        *[translation_consumer(i) for i in range(translation_pool.num_instances)]
    )
```

### Why This Works

| Old Approach | New Approach |
|--------------|--------------|
| 1 OCR + Semaphore(1) | 2 OCR + 2 asyncio.Semaphore |
| 1 Llama + threading.Lock | 3 Llama + 3 asyncio.Semaphore |
| Sequential OCR (~2400ms) | Parallel OCR (~1200ms) |
| Thread pool contention | No shared resources |
| Lock thrashing | Each instance independent |
| GIL overhead | Sync calls in async wrappers |
| 31ms → 315ms slowdown | Consistent timing |

### Expected Performance (6 bubbles)

| Metric | Before (Sequential) | After (Parallel) | Improvement |
|--------|---------------------|------------------|-------------|
| OCR (6 crops) | 2400ms | 1200ms | **50% faster** |
| Translation (6 texts) | 1000ms | 340ms | **66% faster** |
| Pipeline overlap | 0ms | ~200ms saved | - |
| **Total** | **~3400ms** | **~1350ms** | **60% faster** |

---

## Current State Analysis

**Timing Breakdown (from example):**
```
Preprocess: 17.74ms
Detection: 30.23ms
Crop: 6.83ms
OCR: 2293.91ms ← MAJOR BOTTLENECK (76% of total time)
Translation: 375.93ms
Total: 3014.75ms
```

**Per-Bubble OCR Times (Sequential):**
- Ranges from 112ms to 725ms per crop
- Total OCR time = sum of all individual crops
- Currently running sequentially despite GPU availability

**Architecture:**
- YOLOv10n for detection (~2-30ms)
- PaddleOCR-VL-For-Manga for OCR (VLM-based, ~300-700ms/crop)
- HY-MT1.5-1.8B-Q8_0 for translation (llama-cpp, ~100-200ms/text)

---

## 10 Optimizations to Reduce Total Time

### 1. **Parallel OCR Processing with asyncio.gather** ⚡ HIGH IMPACT
**Current:** Sequential OCR - each crop waits for previous
**Proposed:** Process all crops concurrently using asyncio.gather

**Files:** `app/services/manga_ocr_service.py`, `app/routers/translate.py`

**Expected Impact:** OCR time from 2293ms → ~725ms (longest single crop)
**Implementation:**
```python
async def recognize_text_parallel(self, crops: List[np.ndarray]) -> List[str]:
    tasks = [self._recognize_single(crop) for crop in crops]
    return await asyncio.gather(*tasks)
```

---

### 2. **Parallel Translation Processing** ⚡ HIGH IMPACT
**Current:** Sequential translation in loop (375ms total)
**Proposed:** Concurrent translation with asyncio.gather

**Files:** `app/routers/translate.py:120-126`

**Expected Impact:** Translation time from 375ms → ~147ms (longest single)
**Implementation:**
```python
async def translate_parallel(texts, target_lang):
    tasks = [translation_service.translate_single(t, target_lang) for t in texts]
    return await asyncio.gather(*tasks)
```

---

### 3. **Pipeline Stage Overlapping** ⚡ HIGH IMPACT
**Current:** OCR completes fully, then translation starts
**Proposed:** Start translating early texts while OCR continues on later crops

**Files:** `app/routers/translate.py`

**Expected Impact:** Overlap ~200-300ms of work
**Implementation:** Use asyncio.as_completed() or producer-consumer pattern with asyncio.Queue

---

### 4. **TensorRT/ONNX Model Optimization** ⚡ MEDIUM-HIGH IMPACT
**Current:** PyTorch models with Flash Attention 2
**Proposed:** Convert OCR model to TensorRT or ONNX with TensorRT EP

**Files:** `app/services/manga_ocr_service.py`, new model export script

**Expected Impact:** 2-4x speedup on OCR inference
**Implementation:**
- Export PaddleOCR-VL to ONNX
- Use TensorRT execution provider
- Add model conversion script to `scripts/`

---

### 5. **Model Quantization (INT8/FP8)** ⚡ MEDIUM IMPACT
**Current:** bfloat16 for OCR, Q8_0 for translation
**Proposed:** INT8 quantization for OCR, FP8 for translation if supported

**Files:** `app/services/manga_ocr_service.py`, `app/services/local_translation_service.py`

**Expected Impact:** 20-40% speedup with minimal accuracy loss
**Implementation:**
- Use dynamic quantization for OCR
- Evaluate accuracy impact on test set

---

### 6. **True Batched OCR Inference** ⚡ MEDIUM IMPACT
**Current:** Batched method exists but may not utilize GPU efficiently
**Proposed:** Optimize batch processing with proper tensor batching

**Files:** `app/services/manga_ocr_service.py:162-265`

**Expected Impact:** Better GPU utilization, ~30% faster for 4+ crops
**Implementation:**
- Ensure padding is efficient
- Profile GPU utilization during batched inference
- Tune batch_size based on VRAM

---

### 7. **Lighter OCR Model Alternative** ⚡ MEDIUM IMPACT
**Current:** PaddleOCR-VL-For-Manga (~2GB)
**Alternative Options:**
- RapidOCR (much faster, good Japanese support)
- gnurt2041/MangaOCR (lightweight alternative)
- DeepSeek-OCR (efficient token generation)

**Files:** New service implementation or model swap

**Expected Impact:** 2-5x faster OCR at ~5-10% accuracy cost
**Trade-off:** Evaluate accuracy vs speed on test manga pages

---

### 8. **GPU Memory Pool & CUDA Streams** ⚡ MEDIUM IMPACT
**Current:** Default CUDA memory allocation
**Proposed:** Use CUDA memory pools and multiple streams

**Files:** `app/services/manga_ocr_service.py`, `app/config.py`

**Expected Impact:** Reduce memory allocation overhead, enable true parallel GPU ops
**Implementation:**
```python
# Enable memory pool
torch.cuda.set_per_process_memory_fraction(0.9)
# Use multiple CUDA streams for parallel inference
```

---

### 9. **Warm Model Cache Between Requests** ⚡ LOW-MEDIUM IMPACT
**Current:** Models warmed at startup
**Proposed:** Keep GPU memory warm with periodic dummy inference

**Files:** `app/main.py`, background task

**Expected Impact:** Consistent latency, avoid GPU cold cache
**Implementation:** Background keepalive task every 30s

---

### 10. **Streaming Response for Early Results** ⚡ LOW IMPACT (UX)
**Current:** Wait for full pipeline completion
**Proposed:** Stream partial results as bubbles complete

**Files:** `app/routers/translate.py`, `app/routers/test_page.py`

**Expected Impact:** Perceived latency reduction, first result in ~800ms
**Implementation:** Use SSE or WebSocket for streaming responses

---

## Theoretical Best-Case Timeline

With optimizations 1-3 fully implemented:

```
Current Sequential:
Detection: 30ms → OCR[1]: 297ms → OCR[2]: 647ms → ... → Trans[1]: 27ms → ...
Total: 3014ms

Optimized Parallel:
Detection: 30ms
  ↓
OCR (all parallel): 725ms (longest crop)
  ↓ (start translating as OCR completes)
Translation (parallel, overlapped): ~150ms additional
  ↓
Total: ~905ms (70% reduction)
```

With TensorRT (optimization 4):
```
OCR: 725ms → ~250ms with TensorRT
Total: ~430ms (86% reduction)
```

---

## Implementation Priority

| Priority | Optimization | Effort | Impact |
|----------|-------------|--------|--------|
| 1 | Parallel OCR (asyncio.gather) | Low | High |
| 2 | Parallel Translation | Low | Medium |
| 3 | Pipeline Overlapping | Medium | High |
| 4 | TensorRT Conversion | High | High |
| 5 | Model Quantization | Medium | Medium |
| 6 | Batched OCR Tuning | Low | Medium |
| 7 | Alternative OCR Model | Medium | Variable |
| 8 | CUDA Memory Optimization | Medium | Medium |
| 9 | Model Warmth Keepalive | Low | Low |
| 10 | Streaming Responses | Medium | Low (UX) |

---

## Critical Files to Modify

1. `app/routers/translate.py` - Pipeline orchestration (optimizations 1, 2, 3)
2. `app/services/manga_ocr_service.py` - OCR parallelization, batching (1, 4, 5, 6)
3. `app/services/local_translation_service.py` - Translation parallelization (2, 5)
4. `app/config.py` - New configuration options
5. `scripts/convert_to_tensorrt.py` - New script for model conversion (4)
6. `app/main.py` - Keepalive background task (9)

---

## Detailed Implementation Reference

### Optimization 1: Parallel OCR with asyncio.gather

**Theory:** Currently each crop is processed sequentially. With 6 crops at 112-725ms each, total = sum (2293ms). With parallel processing, total = max (725ms).

**Current Code** (`translate.py:97-107`):
```python
# Sequential - waits for each crop
if settings.ocr_use_batching:
    ocr_texts = await ocr_service.recognize_text_batch(crops, batch_size=settings.ocr_batch_size)
else:
    ocr_texts = await ocr_service.recognize_text(crops)
```

**Optimized Code:**
```python
# Parallel - all crops processed concurrently
async def recognize_text_parallel(self, crops: List[np.ndarray]) -> List[str]:
    """Process all crops in parallel using asyncio.gather."""
    if not crops:
        return []

    # Create tasks for all crops
    tasks = [asyncio.create_task(self._recognize_single(crop)) for crop in crops]

    # Wait for all to complete - takes time of longest crop only
    results = await asyncio.gather(*tasks)
    return list(results)
```

**GPU Consideration:** Multiple concurrent inference calls may compete for GPU. Solutions:
- Use CUDA streams for true parallelism
- Or use semaphore to limit concurrent GPU ops while still overlapping CPU work

**Expected Impact:** 2293ms → ~725ms (68% reduction)

---

### Optimization 2: Parallel Translation

**Theory:** Same principle - translations are independent and can run concurrently.

**Current Code** (`translate.py:120-126`):
```python
# Sequential translation
translations = []
for text in ocr_texts:
    translation_start = time.time()
    trans = await translation_service.translate_single(text, target_language)
    translations.append(trans)
```

**Optimized Code:**
```python
# Parallel translation
async def translate_parallel(texts: List[str], target_lang: str) -> List[str]:
    if not texts:
        return []

    tasks = [
        translation_service.translate_single(text, target_lang)
        for text in texts
    ]
    return await asyncio.gather(*tasks)

# Usage
translations = await translate_parallel(ocr_texts, target_language)
```

**Note:** Translation already uses `asyncio.to_thread()` internally, so parallel calls will use thread pool effectively.

**Expected Impact:** 375ms → ~147ms (60% reduction)

---

### Optimization 3: Pipeline Stage Overlapping (Producer-Consumer)

**Theory:** Start translating the first OCR result while OCR continues on remaining crops.

**Implementation Pattern:**
```python
import asyncio

async def process_with_overlap(crops, target_lang):
    queue = asyncio.Queue()
    results = [None] * len(crops)

    async def ocr_producer():
        for i, crop in enumerate(crops):
            text = await ocr_service._recognize_single(crop)
            await queue.put((i, text))
        await queue.put(None)  # Signal completion

    async def translation_consumer():
        while True:
            item = await queue.get()
            if item is None:
                break
            i, text = item
            translation = await translation_service.translate_single(text, target_lang)
            results[i] = (text, translation)

    # Run producer and consumer concurrently
    await asyncio.gather(ocr_producer(), translation_consumer())
    return results
```

**Alternative - asyncio.as_completed:**
```python
async def process_streaming():
    ocr_tasks = {asyncio.create_task(recognize(crop)): i for i, crop in enumerate(crops)}
    trans_tasks = []

    for completed in asyncio.as_completed(ocr_tasks.keys()):
        text = await completed
        idx = ocr_tasks[completed]
        # Start translation immediately
        trans_tasks.append((idx, asyncio.create_task(translate(text))))

    # Gather all translations
    results = [None] * len(crops)
    for idx, task in trans_tasks:
        results[idx] = await task
    return results
```

**Expected Impact:** Additional ~100-200ms overlap savings

---

### Optimization 4: TensorRT Model Conversion

**Theory:** TensorRT optimizes neural network inference for NVIDIA GPUs through layer fusion, kernel auto-tuning, and precision calibration.

**Installation:**
```bash
# Install TensorRT (requires CUDA)
pip install tensorrt

# For PyTorch models
pip install torch-tensorrt

# For ONNX models
pip install onnxruntime-gpu
```

**ONNX Export for PaddleOCR-VL:**
```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("jzhang533/PaddleOCR-VL-For-Manga")

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust for actual input shape
torch.onnx.export(
    model,
    dummy_input,
    "paddleocr_vl.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=14
)
```

**TensorRT Optimization via ONNX Runtime:**
```python
import onnxruntime as ort

# Create session with TensorRT execution provider
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,  # 2GB
        'trt_fp16_enable': True,  # Enable FP16 for 2x speedup
    }),
    'CUDAExecutionProvider',  # Fallback
]

session = ort.InferenceSession("paddleocr_vl.onnx", providers=providers)
```

**Note:** VLM models like PaddleOCR-VL may have complex architecture that doesn't fully convert. Consider:
- Export only the vision encoder to TensorRT
- Keep LLM decoder in PyTorch with torch.compile()

**Expected Impact:** 2-4x speedup on inference

---

### Optimization 5: Model Quantization

**Theory:** Reduce numerical precision (FP32 → FP16/INT8) for faster computation with minimal accuracy loss.

**Dynamic Quantization (CPU):**
```python
import torch

model = load_model()
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize linear layers
    dtype=torch.qint8
)
```

**Static Quantization (Better accuracy):**
```python
# Requires calibration dataset
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Run calibration
for batch in calibration_loader:
    model(batch)

torch.quantization.convert(model, inplace=True)
```

**FP16 Inference (Already enabled in your code):**
```python
# manga_ocr_service.py already uses bfloat16
self.model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Already optimized!
    ...
)
```

**INT8 for llama-cpp (Translation):**
```python
# Already using Q8_0 quantization
# Could try Q4_K_M for faster inference:
# translation_model_filename: str = "HY-MT1.5-1.8B-Q4_K_M.gguf"
```

**Expected Impact:** 20-40% speedup

---

### Optimization 6: Optimized Batch OCR

**Theory:** Process multiple images in single forward pass for better GPU utilization.

**Current Implementation Analysis:**
Your `recognize_text_batch` already implements batching, but could be optimized:

```python
async def _recognize_batch_optimized(self, crops: List[np.ndarray]) -> List[str]:
    pil_images = [Image.fromarray(crop) for crop in crops]

    # Efficient batching with proper padding
    inputs = self.processor(
        images=pil_images,
        text=[self.prompt] * len(pil_images),
        return_tensors="pt",
        padding="longest",  # More efficient than max_length
        truncation=True,
    ).to(self.device)

    with torch.no_grad(), torch.cuda.amp.autocast():  # AMP for speed
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,  # KV cache for faster generation
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

    return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
```

**Optimal Batch Size:** Profile with different sizes
```python
# config.py
ocr_batch_size: int = 4  # Default, but profile 2, 4, 8 based on VRAM
```

**Expected Impact:** ~30% faster for multi-crop images

---

### Optimization 7: Alternative Faster OCR Models

**Note:** User prefers maintaining accuracy, so this is for reference only.

**RapidOCR (2-5x faster):**
```python
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
result, _ = ocr(image)  # Returns text and bboxes
```

**PaddleOCR Lite:**
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=False,
    lang='japan',
    use_gpu=True,
    det_model_dir='path/to/lite/det',
    rec_model_dir='path/to/lite/rec',
)
```

**Accuracy comparison needed before adoption.**

---

### Optimization 8: CUDA Memory & Streams

**Theory:** Use CUDA memory pools and multiple streams for true parallel GPU execution.

**Memory Pool:**
```python
import torch

# Pre-allocate GPU memory pool
torch.cuda.set_per_process_memory_fraction(0.9)

# Enable memory-efficient attention
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

**CUDA Streams for Parallel Inference:**
```python
import torch

async def parallel_inference_with_streams(crops):
    streams = [torch.cuda.Stream() for _ in range(len(crops))]
    results = [None] * len(crops)

    for i, (crop, stream) in enumerate(zip(crops, streams)):
        with torch.cuda.stream(stream):
            results[i] = model(crop)

    # Synchronize all streams
    torch.cuda.synchronize()
    return results
```

**Note:** VLM models may not benefit as much due to sequential token generation.

**Expected Impact:** Reduced memory allocation overhead

---

### Optimization 9: Model Warmth Keepalive

**Theory:** Prevent GPU cache cooling between requests.

**Background Keepalive Task:**
```python
# app/main.py
import asyncio

async def model_keepalive():
    """Periodic dummy inference to keep GPU cache warm."""
    dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        try:
            # Quick inference to keep model warm
            await ocr_service._recognize_single(dummy_image)
        except Exception:
            pass  # Ignore errors

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing startup ...

    # Start keepalive task
    keepalive_task = asyncio.create_task(model_keepalive())

    yield

    keepalive_task.cancel()
```

**Expected Impact:** More consistent latency, especially for sporadic requests

---

### Optimization 10: Streaming Responses

**Theory:** Return partial results as they complete for better perceived performance.

**Server-Sent Events (SSE):**
```python
from fastapi.responses import StreamingResponse
import json

@router.post("/translate/stream")
async def translate_stream(request: TranslateRequest):
    async def generate():
        for i, crop in enumerate(crops):
            text = await ocr_service._recognize_single(crop)
            translation = await translation_service.translate_single(text)

            yield f"data: {json.dumps({'index': i, 'text': text, 'translation': translation})}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Client-side consumption:**
```javascript
const eventSource = new EventSource('/translate/stream');
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.done) {
        eventSource.close();
    } else {
        renderBubble(data.index, data.translation);
    }
};
```

**Expected Impact:** First result visible in ~800ms instead of waiting 3000ms

---

## Path to <500ms Target

**Phase 1: Parallelization Only (No Dependencies)**
| Stage | Before | After |
|-------|--------|-------|
| Detection | 30ms | 30ms |
| Crop | 7ms | 7ms |
| OCR | 2294ms | 725ms (parallel) |
| Translation | 376ms | 147ms (parallel) |
| **Total** | **3014ms** | **~909ms** |

**Phase 2: Add TensorRT**
| Stage | Before | After |
|-------|--------|-------|
| OCR | 725ms | ~250ms (TensorRT) |
| **Total** | **909ms** | **~434ms** |

**Phase 3: Fine-tuning**
- Pipeline overlap: -50ms
- Batch optimization: -30ms
- **Final Target: ~350-400ms** ✅

---

## Summary

To achieve <500ms while maintaining accuracy:

1. **Must implement:** Optimizations 1-3 (parallelization) → Gets you to ~900ms
2. **Required for target:** Optimization 4 (TensorRT) → Gets you to ~450ms
3. **Polish:** Optimizations 5, 6, 8, 9 → Gets you to ~350ms

Skip optimizations 7 (alternative models) and 10 (streaming) unless requirements change
