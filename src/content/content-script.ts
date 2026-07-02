/**
 * Main content script - orchestrates image detection, translation, and overlay rendering
 */
import browser from 'webextension-polyfill';
import { settingsManager } from '@/services/settings-manager';
import { apiClient } from '@/services/api-client';
import { ImageDetector } from '@/services/image-detector';
import { OverlayRenderer } from '@/services/overlay-renderer';
import { canvasMonitor } from '@/services/canvas-monitor';
import { PrefetchManager } from '@/services/prefetch-manager';
import { compressBase64Image, hashString } from '@/utils/image-utils';
import { CONFIG } from '@/config/constants';
import { logger } from '@/utils/logger';
import { FlagRequest, TranslateResponse } from '@/types/api';
import { StreamEventFrame } from '@/types/stream';
import { ExtensionSettings } from '@/types/settings';

class MangaTranslatorContent {
  private imageDetector: ImageDetector;
  private overlayRenderer: OverlayRenderer;
  private resizeObserver: ResizeObserver | null = null;
  private intersectionObserver: IntersectionObserver | null = null;
  private mutationObserver: MutationObserver | null = null;
  /** Speculative next-page prefetch (predict + warm the SW translation cache). */
  private prefetchManager: PrefetchManager | null = null;
  /** Monotonic id correlating a streaming Port request with its port messages. */
  private streamReqId = 0;
  /** Hashes we have already asked the SW to prefetch, to avoid duplicate work. */
  private prefetchedHashes: Set<string> = new Set();
  /**
   * Effective enabled state = this hostname is whitelisted AND the master
   * translation switch is ON. The content script only translates / keeps
   * overlays while this is true.
   */
  private isEnabled: boolean = false;
  /** This hostname is in the per-site activation whitelist. */
  private hostEnabled: boolean = false;
  /** Master ON/OFF translation switch (settings.translationEnabled). */
  private translationEnabled: boolean = true;
  private translatingCanvases: Set<HTMLCanvasElement> = new Set();
  /**
   * Last src we (re)processed per <img>. Used by the attribute observer to
   * detect when a reader swaps an existing <img> to a NEW page image (vs the
   * dozens of no-op/lazy-load src writes frameworks emit) so we only re-run on
   * a genuine, different, non-empty URL — never in a loop.
   */
  private lastProcessedSrc: WeakMap<HTMLElement, string> = new WeakMap();
  /** Per-element debounce timers for rapid src/srcset attribute churn. */
  private srcChangeTimers: WeakMap<HTMLElement, ReturnType<typeof setTimeout>> =
    new WeakMap();
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
   * Listen for edit/flag events dispatched by per-box DOM overlays.
   * (The per-box retry affordance was removed; only inline-edit + the
   * per-image flag button remain.)
   */
  private setupEditEventListeners(): void {
    document.addEventListener('manga-translator:edit-box', (e) => {
      const detail = (e as CustomEvent).detail;
      logger.debug('edit-box committed:', detail);
      // TODO: patch the rendered canvas locally with the new text.
    });

    // User flagged a poor translation from the per-image ⚑ button. The overlay
    // renderer has already gathered the FlagRequest payload (original image +
    // OCR/translated boxes) and provides success/error callbacks so it can flip
    // the button to ✓. We just run the SW round-trip (fire-and-forget) here.
    document.addEventListener('manga-translator:flag-image', (e) => {
      const detail = (e as CustomEvent).detail as {
        payload: FlagRequest;
        onSuccess?: () => void;
        onError?: () => void;
      };
      if (!detail?.payload) return;
      const boxCount = detail.payload.boxes?.length ?? 0;
      logger.info(`Flagging translation (${boxCount} boxes)`, detail.payload.page_url);
      apiClient
        .flagTranslation(detail.payload)
        .then((res) => {
          logger.info('Flag accepted', res?.id ?? res?.image_path ?? '');
          detail.onSuccess?.();
        })
        .catch((err) => {
          logger.error('Flag request failed', err);
          detail.onError?.();
        });
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

    // Check if extension is enabled for this hostname AND globally.
    const hostname = window.location.hostname;
    this.hostEnabled = await settingsManager.isEnabledForHostname(hostname);
    this.translationEnabled = await settingsManager.isTranslationEnabled();
    this.isEnabled = this.hostEnabled && this.translationEnabled;

    // Mirror the verbose-logging setting into the async logger.
    const initialSettings = await settingsManager.getSettings();
    logger.setVerbose(initialSettings.showDebugInfo);
    logger.setPrefix('[MT/content]');

    // Always listen for settings changes (even when disabled, so we can
    // auto-enable on either the per-host whitelist OR the master switch).
    settingsManager.onSettingsChanged(async (settings) => {
      logger.setVerbose(!!settings?.showDebugInfo);
      this.hostEnabled = await settingsManager.isEnabledForHostname(hostname);
      this.translationEnabled = await settingsManager.isTranslationEnabled();
      const enabled = this.hostEnabled && this.translationEnabled;
      if (enabled !== this.isEnabled) {
        this.isEnabled = enabled;
        if (enabled) {
          this.setupResizeObserver();
          this.setupIntersectionObserver();
          this.setupCanvasMonitor();
          this.setupMutationObserver();
          this.setupPrefetchManager();
          this.start();
        } else {
          this.stop();
        }
      }
    });

    if (!this.isEnabled) {
      logger.info(
        `Disabled for ${hostname}`,
        `(host=${this.hostEnabled}, master=${this.translationEnabled})`
      );
      return;
    }

    logger.info(`Enabled for ${hostname}`);

    // Setup observers
    this.setupResizeObserver();
    this.setupIntersectionObserver();
    this.setupCanvasMonitor();
    this.setupMutationObserver();
    this.setupPrefetchManager();

    // Start processing
    this.start();
  }

  /**
   * Start translation processing
   */
  private async start(): Promise<void> {
    logger.info('Starting...');

    // Start canvas monitor
    canvasMonitor.start();

    // Process existing images
    await this.processPage();
  }

  /**
   * Stop translation processing
   */
  private stop(): void {
    logger.info('Stopping...');

    // Stop canvas monitor
    canvasMonitor.stop();

    // Stop speculative prefetch.
    this.prefetchManager?.stop();
    this.prefetchManager = null;
    this.prefetchedHashes.clear();

    // Clear all overlays AND restore the original images underneath.
    this.overlayRenderer.clearAll();

    // Reset detectors so images get re-processed if we turn back ON.
    this.imageDetector.reset();
  }

  /**
   * Apply a freshly-computed effective enabled state. Idempotent: only starts
   * or stops when the state actually changes. Used by both the per-hostname
   * `toggle` and the master `setTranslationEnabled` paths so they converge on a
   * single, consistent enable/disable transition.
   */
  private async applyEnabledState(enabled: boolean): Promise<void> {
    if (enabled === this.isEnabled) {
      // No transition, but if turning "on" while already on do nothing; if
      // "off" while already off, also nothing. Keep idempotent.
      return;
    }
    this.isEnabled = enabled;
    if (enabled) {
      this.setupResizeObserver();
      this.setupIntersectionObserver();
      this.setupCanvasMonitor();
      this.setupMutationObserver();
      this.setupPrefetchManager();
      await this.start();
    } else {
      this.stop();
    }
  }

  /**
   * Process page for translatable images — picks the single dominant main
   * content image and routes it through the existing single-element path.
   * Eliminates the concurrent-batch flood that caused OOM on gallery pages.
   */
  private async processPage(): Promise<void> {
    if (!this.isEnabled) return;

    try {
      const main = this.imageDetector.getMainImageElement();
      if (main) {
        logger.debug('processPage: main image selected, translating single element');
        await this.processVisibleElement(main);
      } else {
        logger.debug('processPage: no dominant main image found, skipping');
      }
    } catch (error) {
      logger.error('Failed to process page:', error);
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
              logger.error('Failed to fetch image:', response.error);
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

      // Per element: (1) serve from the prefetch cache if warm, else (2) request
      // a PROGRESSIVE stream (paints detections -> bubbles -> plate as they land),
      // else (3) a single monolithic request when streaming is disabled. A cache
      // hit or a legacy/HTTP response both render via the single-pass path.
      for (let i = 0; i < batch.length; i++) {
        const element = batch[i].element as HTMLImageElement | HTMLCanvasElement;
        const compressed = compressedImages[i];
        try {
          const cached = await this.getCachedTranslation(compressed);
          if (cached) {
            logger.debug('prefetch cache hit — rendering without re-request');
            await this.renderMonolithic(element, cached, settings, compressed);
            continue;
          }

          if (settings.streamingEnabled) {
            // Holds the translatingCanvases guard for the WHOLE stream internally.
            await this.streamElement(element, compressed, settings);
          } else {
            const data = await apiClient.translate([compressed], settings.targetLanguage);
            await this.renderMonolithic(element, data, settings, compressed);
          }
        } catch (err) {
          logger.error('Translation failed for element:', err);
          this.overlayRenderer.showError(element, 'Translation failed');
        }
      }
    } catch (error) {
      logger.error('Translation batch failed:', error);

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
   * Render a monolithic {@link TranslateResponse} (prefetch cache hit, HTTP
   * fallback, or legacy WS reply) via the single-pass overlay path. Holds the
   * canvas translating-guard across the write, exactly as the old processBatch
   * did. `sourceBase64` is the compressed bytes the backend saw, so the renderer
   * lays text out in the backend's coordinate space without re-encoding.
   */
  private async renderMonolithic(
    element: HTMLImageElement | HTMLCanvasElement,
    response: TranslateResponse,
    settings: ExtensionSettings,
    sourceBase64: string
  ): Promise<void> {
    if (response.debug?.timing) {
      const t = response.debug.timing;
      logger.debug(
        `Timing: detection=${t.detection_ms?.toFixed(1)}ms, ` +
          `ocr=${t.ocr_ms?.toFixed(1)}ms, ` +
          `translation=${t.translation_ms?.toFixed(1)}ms, ` +
          `total=${t.request_total_ms?.toFixed(1)}ms`
      );
    }

    const textBoxes = response.images?.[0] || [];
    const inpaintedPlate = response.inpainted_image_base64?.[0] ?? null;
    if (textBoxes.length === 0) return;

    const isCanvas = element instanceof HTMLCanvasElement;
    if (isCanvas) this.translatingCanvases.add(element as HTMLCanvasElement);

    await this.overlayRenderer.createOverlay(
      element,
      textBoxes,
      settings.showDebugInfo,
      inpaintedPlate,
      sourceBase64
    );

    if (isCanvas) {
      canvasMonitor.updateHash(element as HTMLCanvasElement);
      this.translatingCanvases.delete(element as HTMLCanvasElement);
      canvasMonitor.addCanvas(element as HTMLCanvasElement);
    }
  }

  /**
   * Progressive translate over a long-lived runtime Port. Event frames drive the
   * incremental renderer (beginOverlay/applyTranslation/applyPlate/finish); the
   * terminal port message resolves the promise. Frames are applied through a
   * serial chain so an async beginOverlay always completes before the tl frames
   * that follow it. The canvas translating-guard is held for the entire stream.
   *
   * Backward compatible: if the backend replies legacy/monolithic (or WS fails
   * and the SW falls back to HTTP) no event frames arrive and we render the
   * single `result` via renderMonolithic.
   */
  private streamElement(
    element: HTMLImageElement | HTMLCanvasElement,
    compressed: string,
    settings: ExtensionSettings
  ): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const requestId = ++this.streamReqId;
      const isCanvas = element instanceof HTMLCanvasElement;
      if (isCanvas) this.translatingCanvases.add(element as HTMLCanvasElement);

      let settled = false;
      let chain: Promise<void> = Promise.resolve();
      const port = browser.runtime.connect({ name: 'mt-translate-stream' });

      const releaseGuard = () => {
        if (isCanvas) {
          canvasMonitor.updateHash(element as HTMLCanvasElement);
          this.translatingCanvases.delete(element as HTMLCanvasElement);
          canvasMonitor.addCanvas(element as HTMLCanvasElement);
        }
      };
      const done = (err?: Error) => {
        if (settled) return;
        settled = true;
        releaseGuard();
        try {
          port.disconnect();
        } catch {
          /* already gone */
        }
        if (err) reject(err);
        else resolve();
      };

      port.onMessage.addListener((raw) => {
        const msg = raw as {
          requestId?: number;
          type?: string;
          frame?: StreamEventFrame;
          data?: TranslateResponse;
          streamed?: boolean;
          error?: string;
        };
        if (msg?.requestId !== requestId) return;

        if (msg.type === 'event' && msg.frame) {
          const frame = msg.frame;
          chain = chain
            .then(() => this.handleStreamFrame(element, frame, settings, compressed))
            .catch((e) => logger.error('stream frame render failed:', e));
        } else if (msg.type === 'result') {
          const data = msg.data;
          const streamed = !!msg.streamed;
          chain
            .then(async () => {
              // Only render here when the backend did NOT stream (legacy/HTTP);
              // the streamed path already painted via begin/apply/finish.
              if (!streamed && data) {
                await this.renderMonolithic(element, data, settings, compressed);
              }
              done();
            })
            .catch((e) =>
              done(e instanceof Error ? e : new Error(String(e)))
            );
        } else if (msg.type === 'error') {
          const err = new Error(msg.error || 'Translation failed');
          chain.then(() => done(err)).catch(() => done(err));
        }
      });

      port.onDisconnect.addListener(() => {
        if (!settled) done(new Error('translate stream port disconnected'));
      });

      port.postMessage({
        action: 'translateStream',
        requestId,
        base64Images: [compressed],
        targetLanguage: settings.targetLanguage,
      });
    });
  }

  /**
   * Apply one streaming event frame to the incremental renderer. Awaited via the
   * per-stream serial chain so ordering (detections before tl, plate/finish
   * after) is preserved regardless of message dispatch timing.
   */
  private async handleStreamFrame(
    element: HTMLImageElement | HTMLCanvasElement,
    frame: StreamEventFrame,
    settings: ExtensionSettings,
    compressed: string
  ): Promise<void> {
    switch (frame.type) {
      case 'detections':
        await this.overlayRenderer.beginOverlay(
          element,
          frame.boxes,
          settings.showDebugInfo,
          compressed
        );
        break;
      case 'tl':
        this.overlayRenderer.applyTranslation(element, frame.index, frame.translatedText);
        break;
      case 'revise':
        this.overlayRenderer.applyTranslation(element, frame.index, frame.translatedText);
        break;
      case 'plate':
        await this.overlayRenderer.applyPlate(element, frame.data);
        break;
      case 'done':
        this.overlayRenderer.finish(element, settings.showDebugInfo);
        break;
      case 'error':
        // Leave the source image visible; the terminal port message rejects.
        this.overlayRenderer.removeOverlay(element);
        break;
    }
  }

  /**
   * Ask the service worker whether a warm prefetch result exists for these exact
   * compressed bytes. Returns the cached response or null. Only meaningful for
   * real base64 (the hash is over the bytes); URL/CORS inputs never hit.
   */
  private async getCachedTranslation(
    compressed: string
  ): Promise<TranslateResponse | null> {
    if (!compressed.startsWith('data:')) return null;
    try {
      const resp = (await browser.runtime.sendMessage({
        action: 'getCachedTranslation',
        hash: hashString(compressed),
      })) as { success?: boolean; data?: TranslateResponse } | undefined;
      if (resp?.success && resp.data) return resp.data;
    } catch (err) {
      logger.debug('getCachedTranslation failed', err);
    }
    return null;
  }

  /**
   * Create + start the speculative PrefetchManager. Idempotent: reuses the
   * existing instance if present. Prediction is gated on the effective enabled
   * state AND the prefetch setting (checked inside prefetchElement).
   */
  private setupPrefetchManager(): void {
    if (!this.prefetchManager) {
      this.prefetchManager = new PrefetchManager({
        enabled: () => this.isEnabled,
        prefetch: (el) => this.prefetchElement(el),
      });
    }
    this.prefetchManager.start();
  }

  /**
   * Speculatively translate a predicted (soon-to-be-visible) image and warm the
   * service-worker cache, keyed by the hash of its compressed bytes. Does NOT
   * render — when the image is later displayed for real, processBatch finds the
   * cache hit and renders instantly. Skips the current main image (the real path
   * owns it), already-processed elements, tiny images, and duplicates.
   */
  private async prefetchElement(element: HTMLElement): Promise<void> {
    if (!this.isEnabled) return;
    const settings = await settingsManager.getSettings();
    if (!settings.prefetchEnabled) return;

    // Don't speculatively translate the image the real path is already handling.
    if (this.imageDetector.isMainContentImage(element)) return;
    if (this.imageDetector.isProcessed(element)) return;

    let base64: string | null = null;
    if (element instanceof HTMLImageElement) {
      if (!element.complete || element.naturalWidth <= 100 || element.naturalHeight <= 100) {
        return;
      }
      base64 = await import('@/utils/image-utils').then((m) => m.elementToBase64(element));
    } else if (element instanceof HTMLCanvasElement) {
      if (element.width <= 100 || element.height <= 100) return;
      base64 = await import('@/utils/image-utils').then((m) => m.elementToBase64(element));
    } else {
      return;
    }
    if (!base64) return;

    const compressed = await compressBase64Image(base64, CONFIG.MAX_IMAGE_SIZE_MB);
    if (!compressed.startsWith('data:')) return; // need real bytes to hash

    const hash = hashString(compressed);
    if (this.prefetchedHashes.has(hash)) return;
    this.prefetchedHashes.add(hash);

    logger.debug('prefetching predicted image', hash);
    await browser.runtime.sendMessage({
      action: 'prefetchTranslate',
      base64Images: [compressed],
      targetLanguage: settings.targetLanguage,
      hash,
    });
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
        logger.debug('Canvas changed, re-translating...');
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
        // (A) src/srcset swap on an EXISTING <img> (SPA / nhentai-style readers
        // that reuse one <img> for every page). Re-detect that image.
        if (
          mutation.type === 'attributes' &&
          mutation.target instanceof HTMLImageElement
        ) {
          this.handleImageSrcChange(mutation.target);
          continue;
        }

        if (mutation.type !== 'childList') continue;

        for (const node of Array.from(mutation.addedNodes)) {
          if (!(node instanceof HTMLElement)) continue;

          // Check if the added node itself is an image/canvas
          if (node instanceof HTMLImageElement || node instanceof HTMLCanvasElement) {
            this.intersectionObserver?.observe(node);
            this.prefetchManager?.observe(node);
          }

          // Check for image/canvas descendants
          const images = node.querySelectorAll?.('img, canvas');
          images?.forEach((el) => {
            this.intersectionObserver?.observe(el);
            this.prefetchManager?.observe(el);
          });
        }
      }
    });

    this.mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
      // (B) Watch in-place src/srcset swaps so SPA readers that reuse a single
      // <img> across pages still get the new page translated.
      attributes: true,
      attributeFilter: ['src', 'srcset'],
    });
  }

  /**
   * A tracked <img>'s src/srcset changed. If it points at a genuinely NEW,
   * non-empty image we invalidate the detector entry, drop the stale overlay
   * from the previous page, and re-process the element. Loop-guarded: we only
   * act when the resolved URL actually differs from the last one we processed,
   * and we debounce rapid attribute churn.
   */
  private handleImageSrcChange(img: HTMLImageElement): void {
    if (!this.isEnabled) return;

    // Never react to src writes on our OWN overlay layers / translation images.
    if (img.dataset.mangaTranslatorTranslation || img.closest('.manga-translator-overlay')) {
      return;
    }

    // Resolve to an absolute URL; `currentSrc` reflects srcset selection once
    // loaded, falling back to the raw `src` attribute before then.
    const newSrc = img.currentSrc || img.src || '';
    if (!newSrc || newSrc.startsWith('data:')) return; // ignore blanks/inline writes

    const prev = this.lastProcessedSrc.get(img);
    if (prev === newSrc) return; // no real change → no loop

    // Debounce: readers often write src then srcset (or thrash during lazy
    // load) within the same tick. Coalesce to the last value.
    const existing = this.srcChangeTimers.get(img);
    if (existing) clearTimeout(existing);
    this.srcChangeTimers.set(
      img,
      setTimeout(() => {
        this.srcChangeTimers.delete(img);
        void this.reprocessSwappedImage(img);
      }, 150)
    );
  }

  /**
   * Re-detect a single <img> after its src settled on a new page image:
   * remove the previous page's overlay, invalidate the cache entry, and run the
   * normal visible-element path again.
   */
  private async reprocessSwappedImage(img: HTMLImageElement): Promise<void> {
    if (!this.isEnabled) return;

    const newSrc = img.currentSrc || img.src || '';
    if (!newSrc || newSrc.startsWith('data:')) return;
    if (this.lastProcessedSrc.get(img) === newSrc) return; // settled to same URL

    // Tear down the stale translation so the OLD page's text never lingers
    // over the NEW image, then allow re-detection of this element only.
    this.overlayRenderer.removeOverlay(img);
    this.imageDetector.invalidate(img);

    // Record BEFORE processing so concurrent attribute writes for the same URL
    // don't re-enqueue. processVisibleElement re-checks complete/natural size
    // and re-marks as processed.
    this.lastProcessedSrc.set(img, newSrc);

    // The freshly-swapped image may not have decoded yet; processVisibleElement
    // early-returns on !complete. Wait for it to load (no further attribute
    // mutation will arrive to retrigger us). Cap the wait so a broken/never-
    // loading src can't pin a pending callback forever.
    if (!img.complete || img.naturalWidth === 0) {
      await this.waitForImageLoad(img);
      // If the reader swapped again while we waited, defer to that newer event.
      const settled = img.currentSrc || img.src || '';
      if (settled !== newSrc) return;
    }

    await this.processVisibleElement(img);
  }

  /**
   * Resolve once an <img> finishes loading (or errors / times out). Best-effort;
   * never rejects so the re-process path stays robust.
   */
  private waitForImageLoad(img: HTMLImageElement, timeoutMs = 8000): Promise<void> {
    return new Promise<void>((resolve) => {
      if (img.complete && img.naturalWidth > 0) {
        resolve();
        return;
      }
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        img.removeEventListener('load', finish);
        img.removeEventListener('error', finish);
        clearTimeout(timer);
        resolve();
      };
      const timer = setTimeout(finish, timeoutMs);
      img.addEventListener('load', finish, { once: true });
      img.addEventListener('error', finish, { once: true });
    });
  }

  /**
   * Process a newly visible element
   */
  private async processVisibleElement(element: HTMLElement): Promise<void> {
    if (!this.isEnabled) return;
    if (this.imageDetector.isProcessed(element)) return;
    // Only translate the single dominant main-content image; skip thumbnails
    // and secondary images that don't pass the size+dominance heuristic.
    if (!this.imageDetector.isMainContentImage(element)) return;

    try {
      let result: { element: HTMLElement; imageData: string; imageUrl?: string } | null = null;

      if (element instanceof HTMLImageElement) {
        if (!element.complete || element.naturalWidth <= 100 || element.naturalHeight <= 100) return;
        const base64 = await import('@/utils/image-utils').then(m => m.elementToBase64(element));
        const imageData = base64 || element.src;
        this.imageDetector.markAsProcessed(element);
        // Record the src we processed so the attribute observer can tell a real
        // page swap apart from no-op src writes (loop guard for ISSUE 2).
        this.lastProcessedSrc.set(element, element.currentSrc || element.src || '');
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
      logger.error('Failed to process visible element:', error);
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

      case 'toggle': {
        // Per-hostname enable/disable changed in settings (from SW/popup).
        // Recompute the effective state from BOTH the host whitelist and the
        // master switch rather than blindly flipping, so the two stay in sync.
        const hostname = window.location.hostname;
        this.hostEnabled = await settingsManager.isEnabledForHostname(hostname);
        this.translationEnabled = await settingsManager.isTranslationEnabled();
        await this.applyEnabledState(this.hostEnabled && this.translationEnabled);
        sendResponse({ success: true, enabled: this.isEnabled });
        break;
      }

      case 'setTranslationEnabled': {
        // Master ON/OFF switch flipped (from popup / context-menu / hotkey).
        this.translationEnabled = !!message.enabled;
        await this.applyEnabledState(this.hostEnabled && this.translationEnabled);
        logger.info(
          `Master translation ${this.translationEnabled ? 'ON' : 'OFF'}`,
          `-> effective=${this.isEnabled}`
        );
        sendResponse({ success: true, enabled: this.isEnabled });
        break;
      }

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
