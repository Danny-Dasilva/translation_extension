/**
 * Image detection service for finding img, canvas, and background-image elements
 */
import { ImageDetectionResult, ImageSource } from '@/types/content';
import {
  elementToBase64,
  getBackgroundImageUrl,
  imageUrlToBase64,
  isElementVisible,
} from '@/utils/image-utils';

export class ImageDetector {
  private processedElements: Set<HTMLElement> = new Set();

  /**
   * Find all translatable images on the page
   */
  async detectImages(): Promise<ImageDetectionResult[]> {
    const results: ImageDetectionResult[] = [];

    // Detect <img> elements
    const imgElements = this.findImageElements();
    for (const img of imgElements) {
      if (this.shouldProcessElement(img)) {
        const result = await this.processImageElement(img);
        if (result) results.push(result);
      }
    }

    // Detect <canvas> elements
    const canvasElements = this.findCanvasElements();
    for (const canvas of canvasElements) {
      if (this.shouldProcessElement(canvas)) {
        const result = await this.processCanvasElement(canvas);
        if (result) results.push(result);
      }
    }

    // Detect elements with background-image
    const bgElements = this.findBackgroundImageElements();
    for (const element of bgElements) {
      if (this.shouldProcessElement(element)) {
        const result = await this.processBackgroundElement(element);
        if (result) results.push(result);
      }
    }

    return results;
  }

  /**
   * Find all <img> elements
   */
  private findImageElements(): HTMLImageElement[] {
    const images = Array.from(document.querySelectorAll('img'));
    return images.filter(img => {
      // Filter out tiny images and icons
      return (
        img.complete &&
        img.naturalWidth > 100 &&
        img.naturalHeight > 100 &&
        isElementVisible(img)
      );
    });
  }

  /**
   * Find all <canvas> elements
   */
  private findCanvasElements(): HTMLCanvasElement[] {
    const canvases = Array.from(document.querySelectorAll('canvas'));
    return canvases.filter(canvas => {
      return (
        canvas.width > 100 &&
        canvas.height > 100 &&
        isElementVisible(canvas)
      );
    });
  }

  /**
   * Find elements with background-image
   */
  private findBackgroundImageElements(): HTMLElement[] {
    const elements: HTMLElement[] = [];
    
    // Check all elements with potential background images
    const candidates = document.querySelectorAll('div, section, article, main');
    
    for (const element of Array.from(candidates)) {
      const bgUrl = getBackgroundImageUrl(element as HTMLElement);
      if (bgUrl && isElementVisible(element as HTMLElement)) {
        const rect = element.getBoundingClientRect();
        if (rect.width > 100 && rect.height > 100) {
          elements.push(element as HTMLElement);
        }
      }
    }

    return elements;
  }

  /**
   * Check if element should be processed
   */
  private shouldProcessElement(element: HTMLElement): boolean {
    // Skip if already processed
    if (this.processedElements.has(element)) {
      return false;
    }

    // Skip if part of our overlay system
    if (element.closest('.manga-translator-overlay')) {
      return false;
    }

    return true;
  }

  /**
   * Process <img> element
   */
  private async processImageElement(
    img: HTMLImageElement
  ): Promise<ImageDetectionResult | null> {
    try {
      const base64 = await elementToBase64(img);

      // If CORS prevented base64 conversion, we'll use the URL
      // Background worker will fetch and convert it
      const imageData = base64 || img.src;

      this.processedElements.add(img);

      return {
        element: img,
        source: 'img',
        imageData,
        imageUrl: img.src, // Always include URL for background worker fallback
        boundingRect: img.getBoundingClientRect(),
      };
    } catch (error) {
      console.warn('Failed to process image element:', error);
      return null;
    }
  }

  /**
   * Process <canvas> element
   */
  private async processCanvasElement(
    canvas: HTMLCanvasElement
  ): Promise<ImageDetectionResult | null> {
    try {
      const base64 = await elementToBase64(canvas);

      // Canvas CORS errors cannot be recovered (no URL fallback)
      if (!base64) {
        console.warn('Canvas has CORS protection, skipping');
        return null;
      }

      this.processedElements.add(canvas);

      return {
        element: canvas,
        source: 'canvas',
        imageData: base64,
        boundingRect: canvas.getBoundingClientRect(),
      };
    } catch (error) {
      console.warn('Failed to process canvas element:', error);
      return null;
    }
  }

  /**
   * Process element with background-image
   */
  private async processBackgroundElement(
    element: HTMLElement
  ): Promise<ImageDetectionResult | null> {
    try {
      const bgUrl = getBackgroundImageUrl(element);
      if (!bgUrl) return null;

      // Try to convert to base64 via background worker
      const base64 = await imageUrlToBase64(bgUrl);

      // If conversion failed, we'll use the URL
      // Background worker will fetch and convert it
      const imageData = base64 || bgUrl;

      this.processedElements.add(element);

      return {
        element,
        source: 'background',
        imageData,
        imageUrl: bgUrl, // Always include URL for background worker fallback
        boundingRect: element.getBoundingClientRect(),
      };
    } catch (error) {
      console.warn('Failed to process background element:', error);
      return null;
    }
  }

  /**
   * Mark element as processed
   */
  markAsProcessed(element: HTMLElement): void {
    this.processedElements.add(element);
  }

  /**
   * Check if element is processed
   */
  isProcessed(element: HTMLElement): boolean {
    return this.processedElements.has(element);
  }

  /**
   * Clear processed elements cache
   */
  clearProcessed(): void {
    this.processedElements.clear();
  }

  /**
   * Reset detector (for cleanup)
   */
  reset(): void {
    this.clearProcessed();
  }
}
