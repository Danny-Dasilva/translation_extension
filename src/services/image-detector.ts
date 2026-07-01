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
import { CONFIG } from '@/config/constants';

export class ImageDetector {
  private processedElements: Set<HTMLElement> = new Set();

  // Short-lived memo for getMainImageElement() — avoids layout thrash when
  // IntersectionObserver fires many entries in quick succession.
  private mainCacheElement: HTMLElement | null = null;
  private mainCacheExpiry: number = 0;
  private static readonly MAIN_CACHE_TTL_MS = 200;

  /**
   * Find all translatable images on the page
   */
  async detectImages(): Promise<ImageDetectionResult[]> {
    // OPT 5: encode all detected elements concurrently. The per-element
    // process* calls each do an independent drawImage+toDataURL (or a worker
    // fetch for background images); they share no state and previously ran
    // strictly one-at-a-time. Build the task list synchronously (preserving the
    // original img -> canvas -> background discovery order) then await them all
    // together. Output order is preserved because Promise.all keeps positional
    // order, and shouldProcessElement / markAsProcessed are evaluated
    // synchronously up front so a single element is never queued twice.
    const tasks: Array<Promise<ImageDetectionResult | null>> = [];

    // Detect <img> elements
    for (const img of this.findImageElements()) {
      if (this.shouldProcessElement(img)) {
        tasks.push(this.processImageElement(img));
      }
    }

    // Detect <canvas> elements
    for (const canvas of this.findCanvasElements()) {
      if (this.shouldProcessElement(canvas)) {
        tasks.push(this.processCanvasElement(canvas));
      }
    }

    // Detect elements with background-image
    for (const element of this.findBackgroundImageElements()) {
      if (this.shouldProcessElement(element)) {
        tasks.push(this.processBackgroundElement(element));
      }
    }

    const settled = await Promise.all(tasks);
    return settled.filter((r): r is ImageDetectionResult => r !== null);
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
      const el = element as HTMLElement;

      // OPT 5: cheap inline pre-filter before the expensive getComputedStyle
      // inside getBackgroundImageUrl. Most page <div>s have NO background image;
      // reading the inline style attribute (el.style.backgroundImage) is free
      // and skips the forced style recalc for them. Elements whose background is
      // set ONLY via a stylesheet have an empty inline value, so we cannot prune
      // those here — we still fall through to getBackgroundImageUrl for any
      // element without a definitive 'none' inline value, keeping detection
      // results identical. We only skip when the inline value is explicitly
      // 'none' (author overrode a sheet) — same as getBackgroundImageUrl would.
      if (el.style.backgroundImage === 'none') {
        continue;
      }

      const bgUrl = getBackgroundImageUrl(el);
      if (bgUrl && isElementVisible(el)) {
        const rect = el.getBoundingClientRect();
        if (rect.width > 100 && rect.height > 100) {
          elements.push(el);
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
   * Invalidate a single element so it can be re-processed (e.g. a manga reader
   * swapped this <img>'s `src`/`srcset` to a new page). Unlike reset(), this
   * does NOT clear every other processed element on the page.
   * Also clears the main-image memo so the swapped element is re-ranked.
   */
  invalidate(element: HTMLElement): void {
    this.processedElements.delete(element);
    this.invalidateMainCache();
  }

  /**
   * Check if element is processed
   */
  isProcessed(element: HTMLElement): boolean {
    return this.processedElements.has(element);
  }

  /**
   * Clear processed elements cache (also clears main-image memo)
   */
  clearProcessed(): void {
    this.processedElements.clear();
    this.invalidateMainCache();
  }

  /**
   * Reset detector (for cleanup)
   */
  reset(): void {
    this.clearProcessed();
    this.invalidateMainCache();
  }

  /**
   * Select the single best "main content image" on the page.
   * Candidates = <img> elements (complete, natural size >100) +
   * <canvas> elements (size >100), both visible and not inside our overlay.
   * Ranking is by rendered area (getBoundingClientRect). Applies:
   *   - min-size gate: min(w,h) >= MAIN_IMAGE_MIN_PX
   *   - dominance gate: area[0] >= MAIN_IMAGE_DOMINANCE * area[1]
   *     (auto-passes when there is only one candidate)
   */
  private selectMainElement(): HTMLElement | null {
    const candidates: Array<{ element: HTMLElement; area: number }> = [];

    for (const img of this.findImageElements()) {
      if (img.closest('.manga-translator-overlay')) continue;
      const rect = img.getBoundingClientRect();
      candidates.push({ element: img, area: rect.width * rect.height });
    }

    for (const canvas of this.findCanvasElements()) {
      if (canvas.closest('.manga-translator-overlay')) continue;
      const rect = canvas.getBoundingClientRect();
      candidates.push({ element: canvas, area: rect.width * rect.height });
    }

    if (candidates.length === 0) return null;

    // Sort descending by area
    candidates.sort((a, b) => b.area - a.area);

    const best = candidates[0];
    const bestRect = best.element.getBoundingClientRect();

    // Min-size gate
    if (
      Math.min(bestRect.width, bestRect.height) < CONFIG.MAIN_IMAGE_MIN_PX
    ) {
      return null;
    }

    // Dominance gate — auto-passes when there is only one candidate
    if (candidates.length > 1) {
      const secondArea = candidates[1].area;
      if (secondArea > 0 && best.area < CONFIG.MAIN_IMAGE_DOMINANCE * secondArea) {
        return null;
      }
    }

    return best.element;
  }

  /**
   * Return the single main content image element, with a 200 ms memo to
   * avoid repeated layout reads when the IntersectionObserver fires in a burst.
   */
  getMainImageElement(): HTMLElement | null {
    const now = performance.now();
    if (now < this.mainCacheExpiry) {
      return this.mainCacheElement;
    }
    this.mainCacheElement = this.selectMainElement();
    this.mainCacheExpiry = now + ImageDetector.MAIN_CACHE_TTL_MS;
    return this.mainCacheElement;
  }

  /**
   * Invalidate the main-image memo so the next call re-ranks candidates.
   * Call this when a src swap or page change means the winner may have changed.
   */
  invalidateMainCache(): void {
    this.mainCacheElement = null;
    this.mainCacheExpiry = 0;
  }

  /**
   * Returns true iff element is the current main content image.
   */
  isMainContentImage(element: HTMLElement): boolean {
    return this.getMainImageElement() === element;
  }
}
