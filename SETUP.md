# Quick Setup Guide

Complete setup guide for the Manga Translator extension and backend.

## Prerequisites

- **Python 3.11+** (for backend)
- **uv** package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Node.js 18+** (for extension)
- **pnpm** package manager (`npm install -g pnpm`)
- **Google Gemini API Key** (get from https://makersuite.google.com/app/apikey)

## Step 1: Backend Setup (5 minutes)

```bash
# Navigate to backend directory
cd backend

# Install dependencies
uv sync

# Create environment file
cp .env.example .env

# Edit .env and add your GEMINI_API_KEY
nano .env  # or use your preferred editor

# Start server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify backend is running:
- Open http://localhost:8000 in browser
- You should see: `{"status":"ok","service":"Manga Translation API"}`

## Step 2: Extension Setup (10 minutes)

### Install Dependencies

```bash
# Navigate to extension root
cd ..  # if you're in backend/

# Install Node.js dependencies
pnpm install
```

### Add Required Assets

#### Fonts (Required)

Download and place these fonts in `public/fonts/`:

**Free fonts (recommended for testing):**
1. **Bangers-Regular.ttf** - https://fonts.google.com/specimen/Bangers
2. **Kalam-Regular.ttf** - https://fonts.google.com/specimen/Kalam

**Additional fonts (optional):**
3. KOMIKAX_.ttf (Komika Jam)
4. KOMIKAH_.ttf (Komika Slim)
5. VTC-Letterer-Pro.otf
6. CCWildWords-Italic.ttf

See `public/fonts/FONTS_README.md` for download links and licenses.

#### Icons (Required)

Create or download extension icons and place in `public/icons/`:
- `icon-16.png` (16x16 pixels)
- `icon-48.png` (48x48 pixels)
- `icon-128.png` (128x128 pixels)

**Quick placeholder icons:**
You can temporarily use any 16x16, 48x48, and 128x128 PNG images for testing.

### Build Extension

```bash
# Build for Chrome
pnpm run build

# OR build for Firefox
pnpm run build:firefox

# OR build for both
pnpm run build:all
```

## Step 3: Load Extension in Browser

### Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top-right)
3. Click "Load unpacked"
4. Select the `dist-chrome` directory
5. Extension should now appear in your extensions list

### Firefox

1. Open Firefox and navigate to `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on"
3. Navigate to `dist-firefox` directory
4. Select any file (e.g., `manifest.json`)
5. Extension should now appear in your add-ons list

## Step 4: Configure Extension

1. Click the extension icon in your browser toolbar
2. Set **API Endpoint** to `http://localhost:8000`
3. Choose **Target Language** (default: English)
4. Select **Default Font** (Bangers or Kalam if you downloaded them)
5. Click **Save Settings**

## Step 5: Test Translation

1. Navigate to a manga reading website (e.g., mangadex.org)
2. Click the extension icon to **enable translation for this site**
3. Status should show: "Enabled for [hostname]"
4. Click **Translate Current Page** button
5. Watch for loading indicator and translation overlays

## Troubleshooting

### Backend Issues

**"GEMINI_API_KEY field required"**
- Make sure `.env` file exists in `backend/` directory
- Verify `GEMINI_API_KEY=your_key_here` is set (no quotes)

**"Connection refused" or "Cannot connect"**
- Ensure backend is running: `cd backend && uv run uvicorn app.main:app --reload`
- Check that port 8000 is not blocked by firewall

### Extension Issues

**"Failed to load extension"**
- Verify you ran `pnpm run build` successfully
- Check that `dist-chrome` or `dist-firefox` directory exists
- Ensure icons exist in `public/icons/`

**"No translations appearing"**
- Open browser DevTools (F12) → Console tab
- Look for error messages
- Verify API endpoint is set correctly in extension popup
- Check that backend is running and accessible

**"CORS policy blocked"**
- Backend CORS should allow `chrome-extension://*` and `moz-extension://*`
- Check `ALLOWED_ORIGINS` in backend `.env`

**"Fonts not loading"**
- Ensure font files exist in `public/fonts/`
- Check browser console for 404 errors
- At minimum, download free fonts (Bangers, Kalam) from Google Fonts

### Development Tips

**Backend auto-reload:**
```bash
cd backend
uv run uvicorn app.main:app --reload
```

**Extension hot reload (Chrome):**
```bash
pnpm run dev
```

**Extension hot reload (Firefox):**
```bash
pnpm run dev:firefox
```

**View logs:**
- Backend: Terminal where uvicorn is running
- Extension: Browser DevTools → Console tab
- Content script: DevTools on web page → Console tab

## Next Steps

- **Customize settings**: Experiment with different fonts and languages
- **Test on various manga sites**: Each site may have different image structures
- **Add more fonts**: Download additional comic fonts for variety
- **Deploy backend**: Consider deploying to a cloud service (Railway, Fly.io, etc.)

## Common Manga Sites for Testing

- mangadex.org
- comick.io
- mangakatana.com
- mangafire.to
- manga-plus.shueisha.co.jp (official)

**Note:** Always respect copyright and terms of service of manga websites.

## Support

- Report issues: Create GitHub issue
- Backend API docs: http://localhost:8000/docs (when server is running)
- Extension logs: Browser DevTools → Console

## Quick Reference

```bash
# Start backend
cd backend && uv run uvicorn app.main:app --reload

# Build extension
pnpm run build

# Development mode
pnpm run dev

# View backend API docs
# Open: http://localhost:8000/docs
```

---

Happy translating! 🎌📖🌐
