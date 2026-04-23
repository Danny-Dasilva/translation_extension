/**
 * Main content script - orchestrates image detection, translation, and overlay rendering
 */
import browser from 'webextension-polyfill';
import { settingsManager } from '@/services/settings-manager';
import { apiClient } from '@/services/api-client';
import { ImageDetector } from '@/services/image-detector';
import { OverlayRenderer } from '@/services/overlay-renderer';
import { canvasMonitor } from '@/services/canvas-monitor';
import { compressBase64Image } from '@/utils/image-utils';
import { CONFIG } from '@/config/constants';

class MangaTranslatorContent {
  private imageDetector: ImageDetector;
  private overlayRenderer: OverlayRenderer;
  private resizeObserver: ResizeObserver | null = null;
  private intersectionObserver: IntersectionObserver | null = null;
  private mutationObserver: MutationObserver | null = null;
  private isEnabled: boolean = false;
  private translatingCanvases: Set<HTMLCanvasElement> = new Set();
  /** True while user is holding Alt (temporary "show original" peek). */
  private altHeld: boolean = false;
  /** True when overlay is toggled off via toolbar / hotkey tap (not Alt-held). */
  private overlayHidden: boolean = false;

  constructor() {
    this.imageDetector = new ImageDetector();
    this.overlayRenderer = new OverlayRenderer();
    this.setupAltPeekHotkey();
    this.setupEditEventListeners();
    this.initialize();
  }

  /**
   * Alt-hold "peek at original" hotkey.
   * Holding Alt sets overlay opacity to 0 (show original underneath).
   * Release restores to 0.95. Does not fire when typing in an input
   * (we only toggle when the event target is not an editable element).
   */
  private setupAltPeekHotkey(): void {
    const isEditable = (t: EventTarget | null): boolean => {
      if (!(t instanceof HTMLElement)) return false;
      const tag = t.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
      if (t.isContentEditable) return true;
      if (t.classList.contains('manga-translator-edit-textarea')) return true;
      return false;
    };

    window.addEventListener('keydown', (e) => {
      if (e.key !== 'Alt') return;
      if (this.altHeld) return;
      if (isEditable(e.target)) return;
      this.altHeld = true;
      this.overlayRenderer.setOverlayOpacity(0);
    }, true);

    window.addEventListener('keyup', (e) => {
      if (e.key !== 'Alt') return;
      if (!this.altHeld) return;
      this.altHeld = false;
      // Restore to 1.0 unless user has toggled the overlay off separately.
      this.overlayRenderer.setOverlayOpacity(this.overlayHidden ? 0 : 1);
    }, true);

    // Safety: if the tab loses focus while Alt is held, restore opacity.
    window.addEventListener('blur', () => {
      if (this.altHeld) {
        this.altHeld = false;
        this.overlayRenderer.setOverlayOpacity(this.overlayHidden ? 0 : 1);
      }
    });
  }

  /**
   * Listen for retry/edit events dispatched by per-box DOM overlays.
   * Currently we only log — full wiring (re-translate via backend) lands
   * in a future integration step.
   */
  private setupEditEventListeners(): void {
    document.addEventListener('manga-translator:retry-box', (e) => {
      const detail = (e as CustomEvent).detail;
      console.log('[Manga Translator] retry-box requested:', detail);
      // TODO: call apiClient to re-translate just this box, then re-render.
    });

    document.addEventListener('manga-translator:edit-box', (e) => {
      const detail = (e as CustomEvent).detail;
      console.log('[Manga Translator] edit-box committed:', detail);
      // TODO: patch the rendered canvas locally with the new text.
    });
  }

  /**
   * Initialize content script
   */
  private async initialize(): Promise<void> {
    // Always register message listener first (even if disabled)
    browser.runtime.onMessage.addListener((message, sender, sendResponse) => {
      this.handleMessage(message, sendResponse);
      return true; // Keep channel open for async response
    });

    // Check if extension is enabled for this hostname
    const hostname = window.location.hostname;
    this.isEnabled = await settingsManager.isEnabledForHostname(hostname);

    // Always listen for settings changes (even when disabled, so we can auto-enable)
    settingsManager.onSettingsChanged(async (settings) => {
      const enabled = await settingsManager.isEnabledForHostname(hostname);
      if (enabled !== this.isEnabled) {
        this.isEnabled = enabled;
        if (enabled) {
          this.setupResizeObserver();
          this.setupIntersectionObserver();
          this.setupCanvasMonitor();
          this.setupMutationObserver();
          this.start();
        } else {
          this.stop();
        }
      }
    });

    if (!this.isEnabled) {
      console.log(`Manga Translator: Disabled for ${hostname}`);
      return;
    }

    console.log(`Manga Translator: Enabled for ${hostname}`);

    // Setup observers
    this.setupResizeObserver();
    this.setupIntersectionObserver();
    this.setupCanvasMonitor();
    this.setupMutationObserver();

    // Start processing
    this.start();
  }

  /**
   * Start translation processing
   */
  private async start(): Promise<void> {
    console.log('Manga Translator: Starting...');
    
    // Start canvas monitor
    canvasMonitor.start();

    // Process existing images
    await this.processPage();
  }

  /**
   * Stop translation processing
   */
  private stop(): void {
    console.log('Manga Translator: Stopping...');
    
    // Stop canvas monitor
    canvasMonitor.stop();

    // Clear all overlays
    this.overlayRenderer.clearAll();

    // Reset detectors
    this.imageDetector.reset();
  }

  /**
   * Process page for translatable images
   */
  private async processPage(): Promise<void> {
    if (!this.isEnabled) return;

    try {
      // Detect all images
      const detectedImages = await this.imageDetector.detectImages();
      console.log(`Found ${detectedImages.length} translatable images`);

      // Process images in batches (up to 2 batches in parallel)
      const batchSize = CONFIG.MAX_IMAGES_PER_REQUEST;
      const batches: Array<typeof detectedImages> = [];
      for (let i = 0; i < detectedImages.length; i += batchSize) {
        batches.push(detectedImages.slice(i, i + batchSize));
      }

      const PARALLEL_BATCHES = 2;
      for (let i = 0; i < batches.length; i += PARALLEL_BATCHES) {
        const concurrent = batches.slice(i, i + PARALLEL_BATCHES);
        await Promise.all(concurrent.map((batch) => this.processBatch(batch)));
      }
    } catch (error) {
      console.error('Failed to process page:', error);
    }
  }

  /**
   * Process a batch of images
   */
  private async processBatch(
    batch: Array<{ element: HTMLElement; imageData: string; imageUrl?: string }>
  ): Promise<void> {
    const settings = await settingsManager.getSettings();

    // Show loading indicators
    const loadingElements = batch.map(({ element }) =>
      settings.showLoadingIndicator ? this.overlayRenderer.showLoading(element) : null
    );

    try {
      // Convert URLs to base64 if needed (when CORS blocked in-page conversion)
      const base64Images = await Promise.all(
        batch.map(async ({ imageData, imageUrl }) => {
          // Check if imageData is a URL (not base64)
          if (!imageData.startsWith('data:')) {
            // Fetch via background worker
            const response = await browser.runtime.sendMessage({
              action: 'fetchImage',
              url: imageUrl || imageData,
            });

            if (response.success) {
              return response.base64;
            } else {
              console.error('Failed to fetch image:', response.error);
              return imageData; // Fallback to URL (will likely fail at API)
            }
          }
          return imageData;
        })
      );

      // Compress images
      const compressedImages = await Promise.all(
        base64Images.map((imageData) => compressBase64Image(imageData, CONFIG.MAX_IMAGE_SIZE_MB))
      );

      // Translate batch
      const response = await apiClient.translate(compressedImages, settings.targetLanguage);

      // Log debug timing info if debug mode is enabled
      if (settings.showDebugInfo && response.debug?.timing) {
        const t = response.debug.timing;
        console.log(
          `[Manga Translator Debug] Timing: ` +
          `detection=${t.detection_ms?.toFixed(1)}ms, ` +
          `ocr=${t.ocr_ms?.toFixed(1)}ms, ` +
          `translation=${t.translation_ms?.toFixed(1)}ms, ` +
          `total=${t.request_total_ms?.toFixed(1)}ms`
        );
      }

      // Create overlays
      for (let i = 0; i < batch.length; i++) {
        const { element } = batch[i];
        const textBoxes = response.images[i] || [];
        // Optional: per-image inpainted plate from the LaMa service.
        // Feature-flag is implicit — renderer uses plate only if present.
        const inpaintedPlate =
          (response as any).inpainted_image_base64?.[i] ?? null;

        if (textBoxes.length > 0) {
          const isCanvas = element instanceof HTMLCanvasElement;

          // Guard: mark canvas as translating to prevent monitor from triggering re-translate
          if (isCanvas) this.translatingCanvases.add(element as HTMLCanvasElement);

          await this.overlayRenderer.createOverlay(
            element as HTMLImageElement | HTMLCanvasElement,
            textBoxes,
            settings.showDebugInfo,
            inpaintedPlate
          );

          // Update hash baseline and remove guard after writing
          if (isCanvas) {
            canvasMonitor.updateHash(element as HTMLCanvasElement);
            this.translatingCanvases.delete(element as HTMLCanvasElement);
            canvasMonitor.addCanvas(element as HTMLCanvasElement);
          }
        }
      }
    } catch (error) {
      console.error('Translation batch failed:', error);

      // Show error messages
      for (const { element } of batch) {
        this.overlayRenderer.showError(element, 'Translation failed');
      }
    } finally {
      // Remove loading indicators
      loadingElements.forEach(el => el?.remove());
    }
  }

  /**
   * Setup ResizeObserver for dynamic image resizing
   */
  private setupResizeObserver(): void {
    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const element = entry.target as HTMLElement;
        this.overlayRenderer.updateOverlayPosition(element);
      }
    });

    // Observe images as they're processed
    // Note: Images will be added to observer when overlays are created
  }

  /**
   * Setup IntersectionObserver for lazy loading
   */
  private setupIntersectionObserver(): void {
    this.intersectionObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const element = entry.target as HTMLElement;
            // Process newly visible elements
            this.processVisibleElement(element);
          }
        }
      },
      {
        rootMargin: '100px', // Pre-load images 100px before they're visible
      }
    );

    // Observe all images
    document.querySelectorAll('img, canvas').forEach((element) => {
      this.intersectionObserver?.observe(element);
    });
  }

  /**
   * Setup canvas monitor for change detection
   */
  private setupCanvasMonitor(): void {
    canvasMonitor.onChange(async (canvas) => {
      // Skip if we're currently writing translations to this canvas
      if (this.translatingCanvases.has(canvas)) return;

      // Re-process canvas when content changes
      if (this.imageDetector.isProcessed(canvas)) {
        console.log('Canvas changed, re-translating...');
        this.imageDetector.reset(); // Allow re-processing
        this.overlayRenderer.removeOverlay(canvas);

        // Re-process
        await this.processVisibleElement(canvas);
      }
    });
  }

  /**
   * Setup MutationObserver to detect dynamically added images
   */
  private setupMutationObserver(): void {
    this.mutationObserver = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const node of Array.from(mutation.addedNodes)) {
          if (!(node instanceof HTMLElement)) continue;

          // Check if the added node itself is an image/canvas
          if (node instanceof HTMLImageElement || node instanceof HTMLCanvasElement) {
            this.intersectionObserver?.observe(node);
          }

          // Check for image/canvas descendants
          const images = node.querySelectorAll?.('img, canvas');
          images?.forEach((el) => this.intersectionObserver?.observe(el));
        }
      }
    });

    this.mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  /**
   * Process a newly visible element
   */
  private async processVisibleElement(element: HTMLElement): Promise<void> {
    if (!this.isEnabled) return;
    if (this.imageDetector.isProcessed(element)) return;

    try {
      let result: { element: HTMLElement; imageData: string; imageUrl?: string } | null = null;

      if (element instanceof HTMLImageElement) {
        if (!element.complete || element.naturalWidth <= 100 || element.naturalHeight <= 100) return;
        const base64 = await import('@/utils/image-utils').then(m => m.elementToBase64(element));
        const imageData = base64 || element.src;
        this.imageDetector.markAsProcessed(element);
        result = { element, imageData, imageUrl: element.src };
      } else if (element instanceof HTMLCanvasElement) {
        if (element.width <= 100 || element.height <= 100) return;
        const base64 = await import('@/utils/image-utils').then(m => m.elementToBase64(element));
        if (!base64) return;
        this.imageDetector.markAsProcessed(element);
        result = { element, imageData: base64 };
      }

      if (result) {
        await this.processBatch([result]);
      }
    } catch (error) {
      console.error('Failed to process visible element:', error);
    }
  }

  /**
   * Handle messages from popup/background
   */
  private async handleMessage(message: any, sendResponse: (response: any) => void): Promise<void> {
    switch (message.action) {
      case 'translate':
        if (this.isEnabled) {
          await this.processPage();
          sendResponse({ success: true });
        } else {
          sendResponse({ success: false, error: 'Extension is disabled for this page' });
        }
        break;

      case 'clear':
        this.overlayRenderer.clearAll();
        this.imageDetector.reset();
        canvasMonitor.clear();
        sendResponse({ success: true });
        break;

      case 'toggle':
        this.isEnabled = !this.isEnabled;
        if (this.isEnabled) {
          await this.start();
        } else {
          this.stop();
        }
        sendResponse({ success: true, enabled: this.isEnabled });
        break;

      case 'toggleOverlay':
        // Toolbar-driven "show original" toggle (persists until toggled again).
        this.overlayHidden = !this.overlayHidden;
        this.overlayRenderer.setOverlayOpacity(this.overlayHidden ? 0 : 1);
        sendResponse({ success: true, hidden: this.overlayHidden });
        break;

      default:
        sendResponse({ success: false, error: 'Unknown action' });
    }
  }

  /**
   * Cleanup on unload
   */
  cleanup(): void {
    this.stop();
    this.resizeObserver?.disconnect();
    this.intersectionObserver?.disconnect();
    this.mutationObserver?.disconnect();
    canvasMonitor.reset();
  }
}

// Initialize content script
const translator = new MangaTranslatorContent();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  translator.cleanup();
});
