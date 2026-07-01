/**
 * Image utility functions for canvas, hash checking, and base64 conversion
 */

/**
 * Convert HTMLImageElement or HTMLCanvasElement to base64
 * Returns null if CORS prevents canvas access (caller should use image URL instead)
 */
export async function elementToBase64(
  element: HTMLImageElement | HTMLCanvasElement
): Promise<string | null> {
  if (element instanceof HTMLCanvasElement) {
    try {
      return element.toDataURL('image/jpeg', 0.85);
    } catch (error) {
      console.warn('Failed to convert canvas to base64 (CORS?):', error);
      return null;
    }
  }

  // For images, draw to canvas first
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    console.error('Failed to get canvas context');
    return null;
  }

  // Handle cross-origin images
  const img = element as HTMLImageElement;

  // Use natural dimensions for better quality
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;

  try {
    ctx.drawImage(img, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.85);
  } catch (error) {
    // CORS error - backend will need to fetch image directly
    console.warn('Failed to convert image to base64 (CORS), will use URL instead:', img.src);
    return null;
  }
}

/**
 * FNV-1a hash for a Uint8ClampedArray region
 */
function fnv1a(data: Uint8ClampedArray): number {
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < data.length; i += 16) { // Sample every 16th byte for speed
    hash ^= data[i];
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }
  return hash >>> 0; // unsigned
}

/**
 * Calculate hash of canvas content for change detection.
 * Samples 5 regions (corners + center) with FNV-1a for collision resistance.
 */
export function hashCanvas(canvas: HTMLCanvasElement): string {
  try {
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    const w = canvas.width;
    const h = canvas.height;
    const size = 50; // Sample 50x50 regions

    // 5 sample regions: top-left, top-right, bottom-left, bottom-right, center
    const regions = [
      [0, 0],
      [Math.max(0, w - size), 0],
      [0, Math.max(0, h - size)],
      [Math.max(0, w - size), Math.max(0, h - size)],
      [Math.max(0, Math.floor(w / 2 - size / 2)), Math.max(0, Math.floor(h / 2 - size / 2))],
    ];

    let combined = '';
    for (const [x, y] of regions) {
      const sw = Math.min(size, w - x);
      const sh = Math.min(size, h - y);
      if (sw <= 0 || sh <= 0) continue;
      const imageData = ctx.getImageData(x, y, sw, sh);
      combined += fnv1a(imageData.data).toString(36) + '-';
    }

    return combined;
  } catch (error) {
    console.error('Failed to hash canvas:', error);
    return '';
  }
}

/**
 * Get background image URL from element
 */
export function getBackgroundImageUrl(element: HTMLElement): string | null {
  const style = window.getComputedStyle(element);
  const bgImage = style.backgroundImage;

  if (!bgImage || bgImage === 'none') {
    return null;
  }

  // Extract URL from "url(...)"
  const match = bgImage.match(/url\(['"]?([^'"]+)['"]?\)/);
  return match ? match[1] : null;
}

/**
 * Load image from URL and convert to base64
 * Uses background service worker to bypass CORS restrictions
 * Returns null if fetch fails
 */
export async function imageUrlToBase64(url: string): Promise<string | null> {
  try {
    // Import browser API dynamically to avoid issues in non-extension contexts
    const browser = (await import('webextension-polyfill')).default;

    // Send message to background worker to fetch the image
    const response = await browser.runtime.sendMessage({
      action: 'fetchImage',
      url: url,
    });

    if (response.success) {
      return response.base64;
    } else {
      console.warn('Failed to fetch image via background worker:', response.error);
      return null;
    }
  } catch (error) {
    console.error('Failed to send message to background worker:', error);
    return null;
  }
}

/**
 * Check if element is visible in viewport
 */
export function isElementVisible(element: HTMLElement): boolean {
  const rect = element.getBoundingClientRect();
  
  return (
    rect.width > 0 &&
    rect.height > 0 &&
    rect.top < window.innerHeight &&
    rect.bottom > 0 &&
    rect.left < window.innerWidth &&
    rect.right > 0
  );
}

/**
 * Compress base64 image if it exceeds max size
 * Returns input unchanged if it's a URL (not base64)
 */
export async function compressBase64Image(
  base64OrUrl: string,
  maxSizeMB: number = 2
): Promise<string> {
  // If it's a URL (not base64), return as-is for backend to handle
  if (!base64OrUrl.startsWith('data:')) {
    return base64OrUrl;
  }

  const base64Data = base64OrUrl.split(',')[1] || base64OrUrl;

  // OPT 7: exact decoded-byte size. Base64 packs 3 bytes per 4 chars, but the
  // trailing '=' padding chars carry NO data — the old `length * 3/4` ignored
  // padding and so OVER-estimated by up to 2 bytes, which on a borderline image
  // could trip a needless lossy re-encode. Subtract the padding to get the true
  // decoded byte count, then compare against the byte limit directly (avoids
  // an extra /1024/1024 round-trip and its float drift).
  const padding = base64Data.endsWith('==') ? 2 : base64Data.endsWith('=') ? 1 : 0;
  const sizeInBytes = (base64Data.length * 3) / 4 - padding;
  const maxSizeBytes = maxSizeMB * 1024 * 1024;

  if (sizeInBytes <= maxSizeBytes) {
    return base64OrUrl;
  }

  // Load image
  const img = await loadImageFromBase64(base64OrUrl);

  // Calculate scale factor. Ratio is unit-independent: maxBytes/curBytes ==
  // maxMB/curMB, so the chosen scale is identical to the prior MB-based math
  // (only the over-limit branch reaches here, so the divisor is non-zero).
  const scale = Math.sqrt(maxSizeBytes / sizeInBytes);
  const newWidth = Math.floor(img.width * scale);
  const newHeight = Math.floor(img.height * scale);

  // Resize
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Failed to get canvas context');

  canvas.width = newWidth;
  canvas.height = newHeight;
  ctx.drawImage(img, 0, 0, newWidth, newHeight);

  return canvas.toDataURL('image/jpeg', 0.8);
}

/**
 * Load image from base64 string
 */
function loadImageFromBase64(base64: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = base64;
  });
}
