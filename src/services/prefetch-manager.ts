/**
 * PrefetchManager — speculative translation of images the reader is likely to
 * reach next, so they render instantly on display.
 *
 * Two prediction strategies, both cheap:
 *   1. VERTICAL-SCROLL readers (webtoon / long-strip): a WIDE IntersectionObserver
 *      (rootMargin ~2-3 viewport heights) fires for images well before they enter
 *      the viewport. Reuses the same observer plumbing the content-script uses for
 *      lazy display.
 *   2. PAGED readers: a `link[rel=next]` / `a[rel=next]` hint plus any already-
 *      loaded-but-hidden preloader <img> (readers commonly warm the next page into
 *      an off-screen <img>). We NEVER fetch/navigate the next document — that would
 *      be expensive and surprising; we only act on images the page already has.
 *
 * Speculative work is strictly ONE-AT-A-TIME (a queue), so it never contends with
 * the user-visible translation in flight. The actual translate+cache round-trip is
 * delegated to the `prefetch` callback the content-script provides (it owns the
 * compress + hash + service-worker request). A single `enabled()` flag gates the
 * whole thing so a setting can disable it.
 */
import { logger } from '@/utils/logger';

export interface PrefetchManagerOptions {
  /** Gate: prefetch only runs while this returns true (setting + master switch). */
  enabled: () => boolean;
  /**
   * Perform the speculative translate + cache for one element. Should resolve
   * when done (success or benign skip) and only reject on a hard error. The
   * manager awaits it to enforce one-at-a-time.
   */
  prefetch: (element: HTMLElement) => Promise<void>;
  /** Viewport-height multiplier for the look-ahead margin. Default 2.5. */
  viewportMultiplier?: number;
}

export class PrefetchManager {
  private observer: IntersectionObserver | null = null;
  private queue: HTMLElement[] = [];
  private queued: WeakSet<HTMLElement> = new WeakSet();
  private processing = false;

  constructor(private readonly opts: PrefetchManagerOptions) {}

  /**
   * Start observing images/canvases with a wide look-ahead margin and seed the
   * paged-reader heuristic. Idempotent.
   */
  start(): void {
    if (this.observer) return;
    if (typeof window === 'undefined' || typeof IntersectionObserver === 'undefined') {
      return;
    }

    const mult = this.opts.viewportMultiplier ?? 2.5;
    const margin = Math.max(200, Math.round(window.innerHeight * mult));

    this.observer = new IntersectionObserver(
      (entries) => {
        if (!this.opts.enabled()) return;
        for (const entry of entries) {
          if (entry.isIntersecting) {
            this.enqueue(entry.target as HTMLElement);
          }
        }
      },
      // Grow the root box vertically only — scroll readers advance vertically,
      // and widening horizontally would pull in unrelated sidebar thumbs.
      { rootMargin: `${margin}px 0px ${margin}px 0px` }
    );

    document.querySelectorAll('img, canvas').forEach((el) => this.observe(el));
    this.predictPagedNext();
    logger.debug(`PrefetchManager started (look-ahead ${margin}px)`);
  }

  /** Observe a newly-added element (called from the content-script's MutationObserver). */
  observe(element: Element): void {
    if (!this.observer) return;
    if (element instanceof HTMLImageElement || element instanceof HTMLCanvasElement) {
      this.observer.observe(element);
    }
  }

  /** Stop observing and drop any queued work. */
  stop(): void {
    this.observer?.disconnect();
    this.observer = null;
    this.queue.length = 0;
    this.processing = false;
  }

  /**
   * Paged-reader heuristic. If the page advertises a next page and has an
   * already-loaded off-screen <img> (a preloader), enqueue it. Intentionally
   * conservative and free of network I/O.
   */
  private predictPagedNext(): void {
    const hasNext = document.querySelector('link[rel="next"], a[rel="next"]');
    if (!hasNext) return;

    // Preload hints the page itself declared for images.
    document
      .querySelectorAll<HTMLLinkElement>('link[rel="preload"][as="image"]')
      .forEach((link) => {
        const href = link.href;
        if (!href) return;
        // Match any <img> already carrying that URL (the reader's preloader).
        document
          .querySelectorAll<HTMLImageElement>('img')
          .forEach((img) => {
            if ((img.currentSrc || img.src) === href) this.enqueue(img);
          });
      });
  }

  private enqueue(element: HTMLElement): void {
    if (!this.opts.enabled()) return;
    if (this.queued.has(element)) return;
    this.queued.add(element);
    this.queue.push(element);
    void this.drain();
  }

  /** Serialize speculative translations — strictly one at a time. */
  private async drain(): Promise<void> {
    if (this.processing) return;
    this.processing = true;
    try {
      while (this.queue.length > 0) {
        if (!this.opts.enabled()) {
          this.queue.length = 0;
          break;
        }
        const element = this.queue.shift()!;
        try {
          await this.opts.prefetch(element);
        } catch (err) {
          // Speculative work is best-effort; never surface prefetch failures.
          logger.debug('prefetch skipped/failed', err);
        }
      }
    } finally {
      this.processing = false;
    }
  }
}
