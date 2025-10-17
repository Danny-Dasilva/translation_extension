# Manga Translation Extension - System Architecture & Workflow

## Table of Contents
1. [System Overview](#system-overview)
2. [Current Extension Analysis](#current-extension-analysis)
3. [Backend Architecture (FastAPI)](#backend-architecture-fastapi)
4. [Frontend Architecture (TypeScript Extension)](#frontend-architecture-typescript-extension)
5. [Data Flow](#data-flow)
6. [API Specification](#api-specification)
7. [Implementation Phases](#implementation-phases)

---

## System Overview

### High-Level Architecture
```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Web Browser   │────────▶│  FastAPI Backend │────────▶│  Google Gemini  │
│   Extension     │         │  + RapidOCR      │         │  2.5 Flash-Lite │
│  (TypeScript)   │◀────────│                  │◀────────│                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
        │
        │ Detects manga images
        ▼
┌─────────────────┐
│   Manga Page    │
│   (Images)      │
└─────────────────┘
```

### Technology Stack

**Backend:**
- FastAPI (Python web framework)
- RapidOCR (OCR for Japanese text detection)
- Google Gemini 2.5 Flash-Lite (translation)
- Uvicorn (ASGI server)

**Frontend (Browser Extension):**
- TypeScript (type-safe JavaScript)
- pnpm (package manager)
- Vite or webpack (bundler)
- webextension-polyfill (cross-browser compatibility)

---

## Current Extension Analysis

### Existing Chrome Extension Structure

#### 1. Content Script (`main_extension.js` - 3,670 lines)

**Core Functionality:**
1. **Image Detection**
   - Scans DOM for images: `<img>`, `<canvas>`, elements with `background-image`
   - Filters translatable content using various checks
   - Uses MutationObserver to detect dynamically loaded images

2. **Image Processing**
   ```javascript
   // Converts images to base64
   const canvas = document.createElement('canvas');
   const context = canvas.getContext('2d');
   context.drawImage(element, 0, 0, width, height);
   const base64Data = canvas.toDataURL('image/jpeg');
   ```

3. **API Communication**
   ```javascript
   // Sends to backend
   chrome.runtime.sendMessage({
       kind: 'translateImage',
       base64Data: base64String,
       translateTo: 'English'
   });
   ```

4. **Translation Overlay**
   - Receives bounding boxes with translated text
   - Creates canvas/SVG overlays
   - Renders translated text on top of original image
   - Handles text fitting, font sizing, backgrounds

#### 2. Service Worker (`service_worker_prod_bin.js` - 9,038 lines)

**Functions:**
- Message passing between content script and background
- API request handling to `https://ichigoreader.com/translate`
- Settings management
- User authentication (if applicable)

### Current API Communication

**Request Format:**
```json
POST https://ichigoreader.com/translate
Content-Type: application/json

{
  "base64Images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
  ]
}
```

**Response Format:**
```json
{
  "images": [
    [
      {
        "ocrText": "別にいいだろ股開けば金やるっていってんだぜ?",
        "originalLanguage": "",
        "minX": 310,
        "minY": 986,
        "maxX": 477,
        "maxY": 1211,
        "background": "data:image/jpeg;base64,...",
        "fontHeightPx": 26,
        "fontColor": "#000000",
        "fontStrokeColor": "#FFFFFF",
        "zIndex": 1,
        "translatedText": "Isn't it fine? I'm saying I'll give you money if you just spread your legs.",
        "subtextBoxes": []
      }
    ]
  ]
}
```

---

## Backend Architecture (FastAPI)

### Project Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration (API keys, settings)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request.py       # Pydantic request models
│   │   └── response.py      # Pydantic response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ocr_service.py   # RapidOCR integration
│   │   └── translation_service.py  # Gemini integration
│   ├── routers/
│   │   ├── __init__.py
│   │   └── translate.py     # /translate endpoint
│   └── utils/
│       ├── __init__.py
│       └── image_processing.py
├── requirements.txt
├── .env                     # Environment variables
└── README.md
```

### Core Components

#### 1. FastAPI Endpoint (`app/routers/translate.py`)
```python
from fastapi import APIRouter, HTTPException
from app.models.request import TranslateRequest
from app.models.response import TranslateResponse
from app.services.ocr_service import OCRService
from app.services.translation_service import TranslationService

router = APIRouter()

@router.post("/translate", response_model=TranslateResponse)
async def translate_images(request: TranslateRequest):
    """
    Translate manga images:
    1. Decode base64 images
    2. Run OCR to detect Japanese text + bounding boxes
    3. Translate detected text using Gemini
    4. Return structured response
    """
    ocr_service = OCRService()
    translation_service = TranslationService()

    results = []
    for base64_image in request.base64Images:
        # Process image
        ocr_results = await ocr_service.detect_text(base64_image)
        translations = await translation_service.translate_batch(
            [box.text for box in ocr_results]
        )

        # Combine OCR + translations
        combined = combine_results(ocr_results, translations)
        results.append(combined)

    return TranslateResponse(images=results)
```

#### 2. OCR Service (`app/services/ocr_service.py`)
```python
from rapidocr_onnxruntime import RapidOCR
import base64
import numpy as np
from PIL import Image
import io

class OCRService:
    def __init__(self):
        self.ocr = RapidOCR()

    async def detect_text(self, base64_image: str):
        """
        Detect Japanese text and bounding boxes using RapidOCR

        Returns:
            List of OCRResult with:
            - text: detected text
            - bbox: [minX, minY, maxX, maxY]
            - confidence: detection confidence
        """
        # Decode base64
        image_data = base64.b64decode(base64_image.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        # Run OCR
        result = self.ocr(image_np)

        # Process results
        ocr_boxes = []
        if result and result[0]:
            for detection in result[0]:
                bbox, text, confidence = detection
                ocr_boxes.append({
                    'text': text,
                    'minX': int(min(p[0] for p in bbox)),
                    'minY': int(min(p[1] for p in bbox)),
                    'maxX': int(max(p[0] for p in bbox)),
                    'maxY': int(max(p[1] for p in bbox)),
                    'confidence': confidence
                })

        return ocr_boxes
```

#### 3. Translation Service (`app/services/translation_service.py`)
```python
import google.generativeai as genai
from typing import List

class TranslationService:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    async def translate_batch(self, texts: List[str], target_lang: str = 'English'):
        """
        Translate Japanese text to target language using Gemini

        Optimized for manga/dialogue:
        - Preserves casual speech patterns
        - Handles onomatopoeia
        - Maintains character voice
        """
        if not texts:
            return []

        prompt = f"""Translate the following Japanese manga dialogue to {target_lang}.
Preserve the tone, casualness, and emotion of the original text.
Return ONLY the translations, one per line, in the same order.

Texts to translate:
{chr(10).join(f'{i+1}. {text}' for i, text in enumerate(texts))}
"""

        response = self.model.generate_content(prompt)
        translations = response.text.strip().split('\n')

        # Clean up numbered format if present
        translations = [t.split('. ', 1)[-1] if '. ' in t else t
                       for t in translations]

        return translations[:len(texts)]  # Ensure same length
```

#### 4. Request/Response Models

**Request Model (`app/models/request.py`):**
```python
from pydantic import BaseModel, Field
from typing import List

class TranslateRequest(BaseModel):
    base64Images: List[str] = Field(
        ...,
        description="Array of base64-encoded images"
    )
    targetLanguage: str = Field(
        default="English",
        description="Target language for translation"
    )
```

**Response Model (`app/models/response.py`):**
```python
from pydantic import BaseModel
from typing import List, Optional

class TextBox(BaseModel):
    ocrText: str
    originalLanguage: str = ""
    minX: int
    minY: int
    maxX: int
    maxY: int
    background: str  # base64 of text region
    fontHeightPx: int
    fontColor: str
    fontStrokeColor: str
    zIndex: int
    translatedText: str
    subtextBoxes: List = []

class TranslateResponse(BaseModel):
    images: List[List[TextBox]]
```

---

## Frontend Architecture (TypeScript Extension)

### Project Structure
```
extension/
├── src/
│   ├── content/
│   │   ├── content.ts           # Main content script
│   │   ├── image-detector.ts    # Image detection logic
│   │   ├── overlay-renderer.ts  # Translation overlay
│   │   └── types.ts             # TypeScript types
│   ├── background/
│   │   ├── service-worker.ts    # Background service worker
│   │   ├── api-client.ts        # API communication
│   │   └── settings.ts          # Settings management
│   ├── popup/
│   │   ├── popup.ts
│   │   ├── popup.html
│   │   └── popup.css
│   └── shared/
│       ├── constants.ts
│       ├── utils.ts
│       └── types.ts
├── public/
│   ├── icons/
│   ├── manifest.chrome.json     # Chrome manifest v3
│   └── manifest.firefox.json    # Firefox manifest v2
├── tsconfig.json
├── vite.config.ts               # Vite bundler config
├── package.json
└── pnpm-lock.yaml
```

### Core Components

#### 1. Content Script (`src/content/content.ts`)
```typescript
import { ImageDetector } from './image-detector';
import { OverlayRenderer } from './overlay-renderer';
import { TranslationAPI } from '../shared/api-client';

class MangaTranslator {
  private detector: ImageDetector;
  private renderer: OverlayRenderer;
  private api: TranslationAPI;
  private translatedElements = new Set<HTMLElement>();

  constructor() {
    this.detector = new ImageDetector();
    this.renderer = new OverlayRenderer();
    this.api = new TranslationAPI();
  }

  async initialize() {
    // Detect initial images
    this.scanForImages();

    // Watch for new images (SPA support)
    this.observeDOM();
  }

  private scanForImages() {
    const images = this.detector.findTranslatableImages();
    images.forEach(img => this.translateImage(img));
  }

  private async translateImage(element: HTMLImageElement | HTMLCanvasElement) {
    if (this.translatedElements.has(element)) return;

    // Convert to base64
    const base64 = await this.imageToBase64(element);

    // Send to backend
    const result = await this.api.translate([base64]);

    // Render overlay
    this.renderer.overlayTranslations(element, result);
    this.translatedElements.add(element);
  }

  private async imageToBase64(element: HTMLElement): Promise<string> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;

    if (element instanceof HTMLImageElement) {
      canvas.width = element.naturalWidth;
      canvas.height = element.naturalHeight;
      ctx.drawImage(element, 0, 0);
    } else if (element instanceof HTMLCanvasElement) {
      canvas.width = element.width;
      canvas.height = element.height;
      ctx.drawImage(element, 0, 0);
    }

    return canvas.toDataURL('image/jpeg', 0.9);
  }

  private observeDOM() {
    const observer = new MutationObserver((mutations) => {
      this.scanForImages();
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }
}

// Initialize
const translator = new MangaTranslator();
translator.initialize();
```

#### 2. Image Detector (`src/content/image-detector.ts`)
```typescript
export class ImageDetector {
  findTranslatableImages(): HTMLElement[] {
    const images: HTMLElement[] = [];

    // Regular images
    document.querySelectorAll('img').forEach(img => {
      if (this.isTranslatable(img)) {
        images.push(img);
      }
    });

    // Canvases
    document.querySelectorAll('canvas').forEach(canvas => {
      if (this.isTranslatable(canvas)) {
        images.push(canvas);
      }
    });

    // Background images
    document.querySelectorAll('*').forEach(el => {
      const bgImage = window.getComputedStyle(el).backgroundImage;
      if (bgImage && bgImage !== 'none') {
        images.push(el as HTMLElement);
      }
    });

    return images;
  }

  private isTranslatable(element: HTMLElement): boolean {
    // Check minimum size
    const rect = element.getBoundingClientRect();
    if (rect.width < 100 || rect.height < 100) return false;

    // Check visibility
    if (rect.width === 0 || rect.height === 0) return false;

    // Check if already translated
    if (element.dataset.translated === 'true') return false;

    return true;
  }
}
```

#### 3. API Client (`src/background/api-client.ts`)
```typescript
import browser from 'webextension-polyfill';

interface TranslateRequest {
  base64Images: string[];
  targetLanguage?: string;
}

interface TextBox {
  ocrText: string;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  translatedText: string;
  fontHeightPx: number;
  fontColor: string;
  fontStrokeColor: string;
  zIndex: number;
}

interface TranslateResponse {
  images: TextBox[][];
}

export class TranslationAPI {
  private baseUrl = 'http://localhost:8000'; // Or production URL

  async translate(base64Images: string[]): Promise<TranslateResponse> {
    const response = await fetch(`${this.baseUrl}/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        base64Images,
        targetLanguage: 'English'
      } as TranslateRequest)
    });

    if (!response.ok) {
      throw new Error(`Translation failed: ${response.statusText}`);
    }

    return await response.json();
  }
}
```

#### 4. Overlay Renderer (`src/content/overlay-renderer.ts`)
```typescript
import { TextBox } from '../shared/types';

export class OverlayRenderer {
  overlayTranslations(element: HTMLElement, response: TranslateResponse) {
    const textBoxes = response.images[0]; // First image

    // Create canvas overlay
    const canvas = this.createOverlayCanvas(element);
    const ctx = canvas.getContext('2d')!;

    // Draw original image
    if (element instanceof HTMLImageElement) {
      ctx.drawImage(element, 0, 0);
    } else if (element instanceof HTMLCanvasElement) {
      ctx.drawImage(element, 0, 0);
    }

    // Draw translation boxes
    textBoxes.forEach(box => {
      this.drawTextBox(ctx, box, canvas.width, canvas.height);
    });

    // Replace original element
    element.replaceWith(canvas);
    canvas.dataset.translated = 'true';
  }

  private createOverlayCanvas(element: HTMLElement): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.width = element.offsetWidth;
    canvas.height = element.offsetHeight;
    canvas.className = element.className;
    canvas.style.cssText = element.style.cssText;
    return canvas;
  }

  private drawTextBox(
    ctx: CanvasRenderingContext2D,
    box: TextBox,
    canvasWidth: number,
    canvasHeight: number
  ) {
    const { minX, minY, maxX, maxY, translatedText, fontColor, fontStrokeColor } = box;

    // Calculate dimensions
    const width = maxX - minX;
    const height = maxY - minY;

    // Draw white background
    ctx.fillStyle = 'white';
    ctx.fillRect(minX, minY, width, height);

    // Draw text
    ctx.font = `${box.fontHeightPx}px sans-serif`;
    ctx.fillStyle = fontColor;
    ctx.strokeStyle = fontStrokeColor;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const centerX = minX + width / 2;
    const centerY = minY + height / 2;

    // Auto-fit text
    this.fitText(ctx, translatedText, centerX, centerY, width, height);
  }

  private fitText(
    ctx: CanvasRenderingContext2D,
    text: string,
    x: number,
    y: number,
    maxWidth: number,
    maxHeight: number
  ) {
    // Simple text fitting - split into lines if needed
    const words = text.split(' ');
    const lines: string[] = [];
    let currentLine = '';

    words.forEach(word => {
      const testLine = currentLine ? `${currentLine} ${word}` : word;
      const metrics = ctx.measureText(testLine);

      if (metrics.width > maxWidth * 0.9) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = testLine;
      }
    });
    if (currentLine) lines.push(currentLine);

    // Draw centered lines
    const lineHeight = parseInt(ctx.font);
    const totalHeight = lines.length * lineHeight;
    const startY = y - totalHeight / 2;

    lines.forEach((line, i) => {
      const lineY = startY + i * lineHeight + lineHeight / 2;
      ctx.strokeText(line, x, lineY);
      ctx.fillText(line, x, lineY);
    });
  }
}
```

#### 5. Build Configuration (`vite.config.ts`)
```typescript
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        content: resolve(__dirname, 'src/content/content.ts'),
        background: resolve(__dirname, 'src/background/service-worker.ts'),
        popup: resolve(__dirname, 'src/popup/popup.html'),
      },
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: '[name].js',
        assetFileNames: '[name].[ext]'
      }
    },
    outDir: 'dist',
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  }
});
```

#### 6. Cross-Browser Manifests

**Chrome Manifest v3 (`public/manifest.chrome.json`):**
```json
{
  "manifest_version": 3,
  "name": "Manga Translator",
  "version": "1.0.0",
  "description": "Translate manga images in real-time",
  "permissions": ["storage", "activeTab"],
  "host_permissions": ["<all_urls>"],
  "background": {
    "service_worker": "background.js"
  },
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content.js"],
      "run_at": "document_idle"
    }
  ],
  "action": {
    "default_popup": "popup.html",
    "default_icon": {
      "16": "icons/icon-16.png",
      "48": "icons/icon-48.png",
      "128": "icons/icon-128.png"
    }
  },
  "icons": {
    "16": "icons/icon-16.png",
    "48": "icons/icon-48.png",
    "128": "icons/icon-128.png"
  }
}
```

**Firefox Manifest v2 (`public/manifest.firefox.json`):**
```json
{
  "manifest_version": 2,
  "name": "Manga Translator",
  "version": "1.0.0",
  "description": "Translate manga images in real-time",
  "permissions": ["storage", "activeTab", "<all_urls>"],
  "background": {
    "scripts": ["background.js"]
  },
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content.js"],
      "run_at": "document_idle"
    }
  ],
  "browser_action": {
    "default_popup": "popup.html",
    "default_icon": {
      "16": "icons/icon-16.png",
      "48": "icons/icon-48.png",
      "128": "icons/icon-128.png"
    }
  },
  "icons": {
    "16": "icons/icon-16.png",
    "48": "icons/icon-48.png",
    "128": "icons/icon-128.png"
  }
}
```

---

## Data Flow

### Complete Translation Flow

```
1. User visits manga page
   └─▶ Content script detects images

2. Image Detection
   └─▶ Scan DOM for <img>, <canvas>, background-image
   └─▶ Filter by size, visibility, translated status

3. Image Processing (Content Script)
   └─▶ Create canvas element
   └─▶ Draw image to canvas
   └─▶ Convert to base64: canvas.toDataURL('image/jpeg')

4. Send to Background Script
   └─▶ chrome.runtime.sendMessage({
         kind: 'translateImage',
         base64Data: base64String
       })

5. Background Script → Backend API
   └─▶ POST /translate
       Body: { "base64Images": ["data:image/jpeg;base64,..."] }

6. Backend Processing
   ├─▶ Decode base64 image
   ├─▶ RapidOCR: Detect Japanese text + bounding boxes
   │   └─▶ Returns: [(bbox, text, confidence), ...]
   ├─▶ Google Gemini: Translate all detected text
   │   └─▶ Batch translation for efficiency
   └─▶ Combine OCR + translations into response format

7. Backend Response
   └─▶ {
         "images": [[
           {
             "ocrText": "原文",
             "minX": 100, "minY": 200,
             "maxX": 300, "maxY": 250,
             "translatedText": "Translated text",
             "fontHeightPx": 20,
             "fontColor": "#000000",
             ...
           }
         ]]
       }

8. Background Script → Content Script
   └─▶ Send translation results via message passing

9. Render Overlay (Content Script)
   ├─▶ Create canvas overlay matching image dimensions
   ├─▶ Draw original image
   ├─▶ For each text box:
   │   ├─▶ Draw white background at (minX, minY, maxX, maxY)
   │   ├─▶ Calculate font size and position
   │   ├─▶ Draw translated text (with stroke for outline)
   │   └─▶ Handle text fitting/wrapping
   └─▶ Replace original element with translated canvas

10. Display to User
    └─▶ User sees translated manga in real-time
```

---

## API Specification

### Endpoint: `POST /translate`

**Request:**
```typescript
{
  base64Images: string[];      // Array of base64-encoded images
  targetLanguage?: string;      // Default: "English"
}
```

**Response:**
```typescript
{
  images: Array<Array<{
    ocrText: string;            // Original Japanese text
    originalLanguage: string;   // Language code (e.g., "ja")
    minX: number;               // Bounding box coordinates
    minY: number;
    maxX: number;
    maxY: number;
    background: string;         // base64 of text region (for context)
    fontHeightPx: number;       // Suggested font size
    fontColor: string;          // Hex color (e.g., "#000000")
    fontStrokeColor: string;    // Hex stroke color
    zIndex: number;             // Layer order (1 = bottom)
    translatedText: string;     // Translated text
    subtextBoxes: any[];        // Future: nested text regions
  }>>
}
```

**Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid base64 or malformed request
- `500 Internal Server Error`: OCR or translation failure

---

## Implementation Phases

### Phase 1: Backend Setup (Days 1-3)

**Day 1: Project Setup**
- [ ] Initialize FastAPI project
- [ ] Setup virtual environment + requirements.txt
- [ ] Configure environment variables (.env)
- [ ] Create basic project structure

**Day 2: OCR Integration**
- [ ] Install RapidOCR dependencies
- [ ] Implement OCRService class
- [ ] Test with sample manga images
- [ ] Optimize bounding box detection

**Day 3: Translation Integration**
- [ ] Setup Google Gemini API
- [ ] Implement TranslationService
- [ ] Create batch translation logic
- [ ] Test translation quality with manga dialogue

**Deliverables:**
- Working `/translate` endpoint
- OCR detecting Japanese text
- Gemini translating text
- Proper response format

---

### Phase 2: Extension Rewrite (Days 4-7)

**Day 4: Project Setup**
- [ ] Initialize pnpm workspace
- [ ] Configure TypeScript + Vite
- [ ] Setup dual manifest (Chrome/Firefox)
- [ ] Install webextension-polyfill

**Day 5: Content Script**
- [ ] Port image detection logic to TypeScript
- [ ] Implement base64 conversion
- [ ] Create message passing to background

**Day 6: Background Script + API Client**
- [ ] Implement service worker (Chrome v3)
- [ ] Create API client for backend
- [ ] Handle CORS and errors

**Day 7: Overlay Rendering**
- [ ] Port canvas overlay logic
- [ ] Implement text fitting algorithms
- [ ] Add visual polish (fonts, colors, borders)

**Deliverables:**
- TypeScript extension working in Chrome
- Firefox compatibility
- Clean, type-safe codebase

---

### Phase 3: Integration & Testing (Days 8-10)

**Day 8: Integration**
- [ ] Connect extension to backend API
- [ ] Test with real manga websites
- [ ] Handle edge cases (large images, errors)

**Day 9: Optimization**
- [ ] Image compression before upload
- [ ] Caching translated images
- [ ] Reduce API calls (batch processing)

**Day 10: Testing & Documentation**
- [ ] Cross-browser testing (Chrome + Firefox)
- [ ] Performance testing
- [ ] Write user documentation
- [ ] Create demo video

**Deliverables:**
- Fully functional manga translation extension
- Production-ready backend
- Documentation

---

## Technical Considerations

### Performance Optimizations

1. **Image Compression**
   - Resize large images before sending to backend
   - Use JPEG compression (quality: 0.8-0.9)
   - Target: <2MB per image

2. **Caching**
   - Cache translated images using hash of base64
   - Store in browser storage (IndexedDB)
   - Reduces API calls for repeated manga pages

3. **Batch Processing**
   - Detect multiple images at once
   - Send up to 5 images per API request
   - Parallel processing on backend

4. **Lazy Loading**
   - Only translate images in viewport
   - Use IntersectionObserver for scroll detection

### Error Handling

1. **Network Errors**
   - Retry failed requests (max 3 attempts)
   - Display user-friendly error messages
   - Fallback to original image on failure

2. **OCR Failures**
   - Handle images with no text detected
   - Skip non-manga images (photos, logos)
   - Confidence threshold filtering

3. **Translation Errors**
   - Fallback to original text if translation fails
   - Handle API rate limits
   - Queue requests during high load

### Security Considerations

1. **API Keys**
   - Store Gemini API key in backend only
   - Use environment variables
   - Never expose in frontend

2. **CORS**
   - Configure proper CORS headers
   - Whitelist extension origins
   - Validate request origins

3. **Input Validation**
   - Validate base64 format
   - Limit image size (max 10MB)
   - Sanitize file types

---

## Development Tools & Commands

### Backend Development

```bash
# Setup
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Type checking
mypy app/
```

### Frontend Development

```bash
# Setup
cd extension
pnpm install

# Development build (Chrome)
pnpm dev:chrome

# Development build (Firefox)
pnpm dev:firefox

# Production build
pnpm build:chrome
pnpm build:firefox

# Type checking
pnpm type-check

# Linting
pnpm lint
```

### Package.json Scripts

```json
{
  "scripts": {
    "dev:chrome": "vite build --watch --mode development",
    "dev:firefox": "vite build --watch --mode development --config vite.firefox.config.ts",
    "build:chrome": "vite build --mode production",
    "build:firefox": "vite build --mode production --config vite.firefox.config.ts",
    "type-check": "tsc --noEmit",
    "lint": "eslint src --ext .ts",
    "format": "prettier --write 'src/**/*.ts'"
  }
}
```

---

## Testing Strategy

### Backend Testing

1. **Unit Tests**
   - OCRService: Test with sample images
   - TranslationService: Mock Gemini API
   - Image processing utilities

2. **Integration Tests**
   - Full `/translate` endpoint flow
   - Error handling scenarios
   - Performance benchmarks

3. **Sample Data**
   - Create test manga images
   - Various text densities
   - Different manga styles

### Extension Testing

1. **Manual Testing**
   - Test on popular manga websites
   - Verify translation accuracy
   - Check performance with many images

2. **Cross-Browser Testing**
   - Chrome (latest)
   - Firefox (latest)
   - Edge (Chromium-based)

3. **Edge Cases**
   - Very large images
   - Images with no text
   - Non-manga content
   - SPA navigation

---

## Future Enhancements

1. **Multiple Language Support**
   - Add more target languages
   - Auto-detect source language

2. **OCR Improvements**
   - Train custom model for manga fonts
   - Better handling of vertical text
   - Speech bubble detection

3. **Translation Quality**
   - Context-aware translation (previous panels)
   - Character name consistency
   - Honorific handling (-san, -kun, etc.)

4. **UI Enhancements**
   - Toggle translations on/off
   - Adjust font size/style
   - Show/hide original text

5. **Offline Mode**
   - Cache common translations
   - Downloadable translation packs

---

## Resources & References

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [RapidOCR GitHub](https://github.com/RapidAI/RapidOCR)
- [Google Gemini API](https://ai.google.dev/docs)
- [Chrome Extension Docs](https://developer.chrome.com/docs/extensions/)
- [Firefox Extension Docs](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions)
- [webextension-polyfill](https://github.com/mozilla/webextension-polyfill)

### Similar Projects
- [manga-image-translator](https://github.com/zyddnys/manga-image-translator)
- [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)

---

## Conclusion

This architecture provides a robust, scalable foundation for a manga translation extension. The separation of concerns (OCR, translation, rendering) allows for:

- **Easy maintenance**: Modular components
- **Scalability**: Backend can serve multiple extension instances
- **Extensibility**: Easy to add new features
- **Cross-browser support**: Firefox + Chrome compatible

The TypeScript rewrite brings type safety and modern development practices, while the FastAPI backend provides high performance and clean API design.
