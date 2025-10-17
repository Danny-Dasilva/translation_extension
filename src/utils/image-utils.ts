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
 * Calculate hash of canvas content for change detection
 */
export function hashCanvas(canvas: HTMLCanvasElement): string {
  try {
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    // Sample pixels from canvas (top-left corner for performance)
    const imageData = ctx.getImageData(0, 0, Math.min(100, canvas.width), Math.min(100, canvas.height));
    const data = imageData.data;

    // Simple hash: sum of pixel values
    let hash = 0;
    for (let i = 0; i < data.length; i += 4) {
      hash += data[i] + data[i + 1] + data[i + 2];
    }

    return hash.toString(36);
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
  const sizeInMB = (base64Data.length * 3) / 4 / (1024 * 1024);

  if (sizeInMB <= maxSizeMB) {
    return base64OrUrl;
  }

  // Load image
  const img = await loadImageFromBase64(base64OrUrl);

  // Calculate scale factor
  const scale = Math.sqrt(maxSizeMB / sizeInMB);
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
