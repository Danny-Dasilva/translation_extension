/**
 * Content script types
 */

export interface DetectedImage {
  element: HTMLImageElement | HTMLCanvasElement;
  src: string;
  hash: string; // For canvas change detection
  boundingRect: DOMRect;
}

// Note: OverlayData is no longer used since we overwrite images directly on canvas
// Keeping for backwards compatibility if needed
export interface OverlayData {
  imageElement: HTMLImageElement | HTMLCanvasElement;
  textBoxes: import('./api').TextBox[];
  overlayContainer?: HTMLDivElement; // Optional now
  shadowRoot?: ShadowRoot; // Optional now
}

export interface TranslationCache {
  [imageHash: string]: import('./api').TextBox[];
}

export type ImageSource = 'img' | 'canvas' | 'background';

export interface ImageDetectionResult {
  element: HTMLElement;
  source: ImageSource;
  imageData: string; // base64 or URL (if CORS blocked conversion)
  imageUrl?: string; // Original URL for background worker fallback
  boundingRect: DOMRect;
}
