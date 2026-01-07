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
  private isEnabled: boolean = false;

  constructor() {
    this.imageDetector = new ImageDetector();
    this.overlayRenderer = new OverlayRenderer();
    this.initialize();
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

    if (!this.isEnabled) {
      console.log(`Manga Translator: Disabled for ${hostname}`);
      return;
    }

    console.log(`Manga Translator: Enabled for ${hostname}`);

    // Setup observers
    this.setupResizeObserver();
    this.setupIntersectionObserver();
    this.setupCanvasMonitor();

    // Listen for settings changes
    settingsManager.onSettingsChanged(async (settings) => {
      const enabled = await settingsManager.isEnabledForHostname(hostname);
      if (enabled !== this.isEnabled) {
        this.isEnabled = enabled;
        if (enabled) {
          this.start();
        } else {
          this.stop();
        }
      }
    });

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

      // Process images in batches
      const batchSize = CONFIG.MAX_IMAGES_PER_REQUEST;
      for (let i = 0; i < detectedImages.length; i += batchSize) {
        const batch = detectedImages.slice(i, i + batchSize);
        await this.processBatch(batch);
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

        if (textBoxes.length > 0) {
          await this.overlayRenderer.createOverlay(
            element as HTMLImageElement | HTMLCanvasElement,
            textBoxes,
            settings.showDebugInfo
          );

          // Monitor canvas for changes
          if (element instanceof HTMLCanvasElement) {
            canvasMonitor.addCanvas(element);
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
      // Re-process canvas when content changes
      if (this.imageDetector.isProcessed(canvas)) {
        console.log('Canvas changed, re-translating...');
        this.overlayRenderer.removeOverlay(canvas);
        
        // Re-process
        await this.processVisibleElement(canvas);
      }
    });
  }

  /**
   * Process a newly visible element
   */
  private async processVisibleElement(element: HTMLElement): Promise<void> {
    if (!this.isEnabled) return;
    if (this.imageDetector.isProcessed(element)) return;

    // Process single element
    // Implementation would be similar to processBatch but for one element
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
    canvasMonitor.reset();
  }
}

// Initialize content script
const translator = new MangaTranslatorContent();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  translator.cleanup();
});
