# Manga Translator Extension

AI-powered browser extension for translating manga and comics in real-time using OCR and neural translation.

## Features

- 🔍 **Smart Detection**: Automatically detects images, canvas elements, and background images
- 🌐 **Neural Translation**: Powered by Google Gemini 2.0 Flash via FastAPI backend
- ⚡ **Real-time OCR**: Uses RapidOCR for fast Japanese text detection
- 🎨 **6 Comic Fonts**: Bangers, Kalam, Komika Jam/Slim, VTC Letterer Pro, CC Wild Words
- 🔒 **Shadow DOM**: CSS isolation prevents conflicts with website styles
- 📊 **Dynamic Updates**: ResizeObserver and IntersectionObserver for responsive overlays
- 🔄 **Canvas Monitoring**: Hash-based change detection for dynamic canvas content
- 🌍 **Per-Hostname Control**: Whitelist system for selective activation
- 🎯 **Smart Retry**: Exponential backoff for failed API requests
- 🖼️ **Background Images**: Detects and translates CSS background-image content

## Architecture

This project consists of two parts:

1. **Backend** (`backend/`): FastAPI server with RapidOCR + Google Gemini
2. **Extension** (`src/`): TypeScript browser extension with Vite bundler

## Quick Start

### 1. Backend Setup

```bash
cd backend

# Install dependencies with uv
uv sync

# Create .env file
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

See [backend/README.md](backend/README.md) for detailed backend documentation.

### 2. Extension Setup

```bash
# Install dependencies
pnpm install

# Download required fonts (see public/fonts/FONTS_README.md)
# Create extension icons (see public/icons/ICONS_README.md)

# Build extension
pnpm run build        # Chrome
pnpm run build:firefox  # Firefox
pnpm run build:all    # Both browsers

# Development mode with auto-reload
pnpm run dev          # Chrome
pnpm run dev:firefox  # Firefox
```

### 3. Load Extension

**Chrome:**
1. Open `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select `dist-chrome` directory

**Firefox:**
1. Open `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on"
3. Select any file in `dist-firefox` directory

## Usage

1. **Configure API Endpoint**: Click extension icon → Set API endpoint (default: `http://localhost:8000`)
2. **Activate for Website**: Navigate to manga site → Click extension icon to enable
3. **Translate**: Translations appear automatically or click "Translate Current Page"
4. **Customize**: Choose target language and font in extension popup

## Project Structure

```
extension/
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── main.py          # FastAPI app
│   │   ├── config.py        # Settings
│   │   ├── models/          # Pydantic models
│   │   ├── services/        # OCR + Translation
│   │   ├── routers/         # API endpoints
│   │   └── utils/           # Image processing
│   └── pyproject.toml       # uv dependencies
│
├── src/                      # Extension source
│   ├── content/             # Content script
│   │   ├── content-script.ts  # Main orchestrator
│   │   └── overlay.css      # Shadow DOM styles
│   ├── background/          # Service worker
│   │   └── service-worker.ts
│   ├── popup/               # Extension popup UI
│   │   ├── popup.html
│   │   └── popup.ts
│   ├── services/            # Core services
│   │   ├── api-client.ts        # API with retry
│   │   ├── settings-manager.ts  # Chrome storage
│   │   ├── image-detector.ts    # Find images
│   │   ├── canvas-monitor.ts    # Hash checking
│   │   └── overlay-renderer.ts  # Shadow DOM
│   ├── utils/               # Utilities
│   │   └── image-utils.ts   # Canvas, base64, etc.
│   ├── types/               # TypeScript types
│   ├── config/              # Constants
│   ├── manifest.chrome.json # Chrome Manifest v3
│   └── manifest.firefox.json # Firefox Manifest v2
│
├── public/                  # Static assets
│   ├── fonts/              # 6 comic fonts
│   └── icons/              # Extension icons
│
├── vite.config.ts          # Vite bundler config
├── tsconfig.json           # TypeScript config
└── package.json            # pnpm dependencies
```

## Technologies

### Backend
- **FastAPI**: Async Python web framework
- **RapidOCR**: ONNX-based OCR (Japanese, Chinese, Korean)
- **Google Gemini**: Neural translation optimized for manga
- **Pydantic**: Type-safe settings and validation
- **uv**: Ultra-fast Python package manager

### Extension
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast bundler with HMR
- **pnpm**: Efficient package manager
- **Shadow DOM**: CSS isolation
- **Web Extensions API**: Cross-browser compatibility

## Configuration

### Backend (.env)
```bash
GEMINI_API_KEY=your_key_here
HOST=0.0.0.0
PORT=8000
ALLOWED_ORIGINS=chrome-extension://*,moz-extension://*
DEFAULT_TARGET_LANGUAGE=English
DEFAULT_MODEL=gemini-2.0-flash-exp
```

### Extension Settings (Popup)
- API Endpoint
- Target Language
- Default Font
- Auto-translate on page load
- Per-hostname activation

## API Endpoints

### POST /translate
Translate manga images.

**Request:**
```json
{
  "base64Images": ["data:image/jpeg;base64,..."],
  "targetLanguage": "English"
}
```

**Response:**
```json
{
  "images": [[
    {
      "ocrText": "こんにちは",
      "translatedText": "Hello",
      "minX": 100, "minY": 200,
      "maxX": 300, "maxY": 250,
      "fontHeightPx": 20,
      "fontColor": "#000000",
      "fontStrokeColor": "#FFFFFF",
      "background": "data:image/jpeg;base64,...",
      "zIndex": 1,
      "subtextBoxes": []
    }
  ]]
}
```

### GET /health
Health check endpoint.

## Development

### Backend
```bash
cd backend

# Run tests (when implemented)
uv run pytest

# Type checking
uv run mypy app/

# Dev server with auto-reload
uv run uvicorn app.main:app --reload
```

### Extension
```bash
# Development with HMR
pnpm run dev

# Type checking
pnpm exec tsc --noEmit

# Build production
pnpm run build:all
```

## Known Features (from Original Extension)

All features from the original extension have been preserved:

✅ Per-hostname URL whitelist system  
✅ ResizeObserver & IntersectionObserver  
✅ 6 embedded comic fonts  
✅ Shadow DOM CSS isolation  
✅ Authentication & premium system (optional)  
✅ Canvas hash checking (1000ms interval)  
✅ Advanced overlay features (loading, errors)  
✅ Retry/polling system  
✅ Translation model selection  
✅ Referer header rules per hostname  
✅ Font color customization  
✅ Settings management (Chrome storage)  
✅ Element tracking (deduplication)  
✅ Background image detection  

## Troubleshooting

### Backend Issues
- **API Key Error**: Ensure `GEMINI_API_KEY` is set in `.env`
- **OCR Fails**: Check image format (JPEG/PNG) and size (<2MB)
- **CORS Errors**: Verify `ALLOWED_ORIGINS` includes your extension ID

### Extension Issues
- **No Translations**: Check console for errors, verify API endpoint
- **CORS Blocked**: Ensure backend CORS is configured correctly
- **Fonts Missing**: Download fonts to `public/fonts/` (see FONTS_README.md)
- **Icons Missing**: Create icons in `public/icons/` (see ICONS_README.md)

## License

MIT

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Credits

- **RapidOCR**: OCR engine
- **Google Gemini**: Translation model
- **Font Providers**: Bangers, Kalam (Google Fonts), Komika, VTC Letterer Pro, CC Wild Words
