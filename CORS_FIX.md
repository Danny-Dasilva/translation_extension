# CORS Error Fix

## Problem
```
CORS error converting image: [object DOMException]
Failed to create overlay: [object Event]
Translation batch failed: [object Event]
```

The extension was failing because manga images are typically hosted on different domains (CORS-protected), which prevents JavaScript from reading the image data directly.

## Solution

### 1. Enhanced `getImageBase64()` in overlay-renderer.ts

**Three-tier fallback strategy**:

```typescript
async getImageBase64(element) {
  // Tier 1: Check for cached base64 data
  if (element.dataset.originalSrc?.startsWith('data:')) {
    return element.dataset.originalSrc;
  }

  // Tier 2: Try direct canvas conversion (same-origin images)
  try {
    ctx.drawImage(element, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.85);
  } catch (error) {
    // CORS blocked...
  }

  // Tier 3: Fetch via background worker (bypasses CORS)
  const response = await browser.runtime.sendMessage({
    action: 'fetchImage',
    url: element.src,
  });

  if (response.success) {
    return response.base64;
  } else {
    throw new Error(`Failed to fetch image: ${response.error}`);
  }
}
```

### 2. Improved `loadImage()` Error Handling

```typescript
private loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => resolve(img);
    img.onerror = (error) => {
      console.error('Failed to load image:', src.substring(0, 100));
      reject(new Error('Failed to load image'));
    };

    // Only set crossOrigin for URLs, not data URIs
    if (!src.startsWith('data:')) {
      img.crossOrigin = 'anonymous';
    }

    img.src = src;
  });
}
```

### 3. Background Service Worker (Already Exists)

The background service worker at `src/background/service-worker.ts` already has a `fetchImage` handler (lines 120-133):

```typescript
case 'fetchImage':
  try {
    const response = await fetch(message.url);
    const blob = await response.blob();
    const base64 = await blobToBase64(blob);
    return { success: true, base64 };
  } catch (error) {
    return { success: false, error: error.message };
  }
```

## How It Works

1. **Content script** tries to read image directly
2. If **CORS blocks** access → requests background worker to fetch it
3. **Background worker** fetches image (extensions bypass CORS)
4. Background worker converts to base64 and returns
5. Content script uses base64 data to render translations

## Key Changes

### `src/services/overlay-renderer.ts`
- ✅ Added three-tier fallback for image fetching
- ✅ Cache check for already-fetched images
- ✅ Background worker integration for CORS-blocked images
- ✅ Better error messages
- ✅ Conditional `crossOrigin` setting (only for URLs, not data URIs)

## Testing

```bash
npm run build
# ✓ ../dist-chrome/content/content-script.js  27.04 kB │ gzip: 8.56 kB
```

**Build successful!**

### To Test:
1. Reload the extension in Chrome
2. Navigate to a manga reader site
3. The extension should now:
   - ✅ Try direct image access first (fast)
   - ✅ Fall back to background worker if CORS blocks (works)
   - ✅ Cache fetched images to avoid re-fetching
   - ✅ Show proper error messages if both fail

## Why This Works

**Browser extensions have special permissions** that allow the background worker to fetch cross-origin resources without CORS restrictions. By routing CORS-blocked image requests through the background worker, we can:

1. Bypass CORS restrictions
2. Convert images to base64
3. Return data to content script
4. Render translations on the image

This is a **standard pattern** for browser extensions that need to process cross-origin images.
