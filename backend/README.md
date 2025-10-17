# Manga Translation API - Backend

FastAPI backend for manga image translation using RapidOCR and Google Gemini.

## Features

- 🔍 **OCR**: Text detection using RapidOCR (supports Japanese, Chinese, Korean)
- 🌐 **Translation**: Neural translation via Google Gemini 2.0 Flash
- ⚡ **Fast**: Async processing with batch translation
- 🎨 **Smart**: Automatic font size calculation and color detection
- 🔒 **Secure**: CORS-protected, environment-based configuration

## Setup

### 1. Install Dependencies (using uv)

```bash
cd backend
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run Server

```bash
# Development (with auto-reload)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### `POST /translate`

Translate manga images with OCR.

**Request:**
```json
{
  "base64Images": [
    "data:image/jpeg;base64,/9j/4AAQ..."
  ],
  "targetLanguage": "English"
}
```

**Response:**
```json
{
  "images": [[
    {
      "ocrText": "こんにちは",
      "originalLanguage": "",
      "minX": 100,
      "minY": 200,
      "maxX": 300,
      "maxY": 250,
      "background": "data:image/jpeg;base64,...",
      "fontHeightPx": 20,
      "fontColor": "#000000",
      "fontStrokeColor": "#FFFFFF",
      "zIndex": 1,
      "translatedText": "Hello",
      "subtextBoxes": []
    }
  ]]
}
```

### `GET /health`

Health check endpoint.

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── models/
│   │   ├── request.py       # Request models
│   │   └── response.py      # Response models
│   ├── services/
│   │   ├── ocr_service.py   # RapidOCR integration
│   │   └── translation_service.py  # Gemini integration
│   ├── routers/
│   │   └── translate.py     # /translate endpoint
│   └── utils/
│       └── image_processing.py  # Image utilities
├── pyproject.toml           # uv dependencies
└── .env                     # Environment config
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | *required* |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Debug mode | `True` |
| `ALLOWED_ORIGINS` | CORS origins | `chrome-extension://*,...` |
| `MAX_REQUESTS_PER_MINUTE` | Rate limit | `60` |
| `MAX_IMAGES_PER_REQUEST` | Max images | `5` |
| `DEFAULT_TARGET_LANGUAGE` | Default language | `English` |
| `DEFAULT_MODEL` | Gemini model | `gemini-2.0-flash-exp` |

## Development

```bash
# Run with auto-reload
uv run uvicorn app.main:app --reload

# Run tests (when added)
uv run pytest

# Type checking
uv run mypy app/
```

## License

MIT
