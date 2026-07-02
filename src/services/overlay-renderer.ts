/**
 * Overlay renderer using Canvas to overwrite original images.
 *
 * Features:
 *   - Binary-search font fit (replaces single-pass scale).
 *     Ref: /tmp/koharu/koharu-renderer/src/layout.rs:133-167 (run_auto)
 *   - Polygon-centroid centering (union bbox of textRegions, not outer bubble)
 *   - Font fallback chain + Google Fonts link injection
 *   - Auto-contrast detection (sample luminance around text region)
 *   - Inpainted "plate" background support (if response provides one)
 *   - Per-bubble retry button + inline edit via DOM overlay layer
 *     Ref: /tmp/koharu/ui/components/panels/TextBlocksPanel.tsx:228, 263
 *   - "Show original" Alt-hotkey (wired from content-script)
 */
import { TextBox, FlagRequest, FlagBox } from '@/types/api';
import { StreamDetectionBox } from '@/types/stream';
import { detectionBoxToTextBox } from './stream-protocol';
import { CONFIG } from '@/config/constants';
import { settingsManager } from './settings-manager';
import { logger } from '@/utils/logger';

interface RenderedImage {
  originalElement: HTMLImageElement | HTMLCanvasElement;
  newElement: HTMLImageElement | HTMLCanvasElement;
  /** Optional DOM overlay sibling that hosts per-box retry/edit affordances. */
  domOverlay?: HTMLDivElement;
  textBoxes: TextBox[];
  /**
   * The ORIGINAL source bytes the backend received (base64 data URL). Captured
   * at render time so we can (a) restore the original image when translation is
   * toggled off and (b) send it to the /flag endpoint. For <img> this is the
   * pre-overlay src; for canvas it is a snapshot taken before we drew over it.
   */
  originalImageBase64: string;
  /**
   * For <img> elements only: the exact `src` attribute that was on the element
   * before we replaced it with the rendered data URL. Used to restore the
   * original image cleanly (re-points at the live URL, not a re-encoded copy).
   */
  originalSrc?: string;
  /**
   * UI state: true once this translation has been flagged (POST /flag accepted).
   * Used to mark the ⚑ button as done and prevent a double-send.
   */
  flagged?: boolean;
}

/**
 * Live state for an in-flight PROGRESSIVE (streaming) render of one element.
 * beginOverlay creates it (source + white boxes drawn, element swapped, overlay
 * registered early); applyTranslation paints one bubble at a time onto `canvas`
 * and re-blits; applyPlate/finish recomposite via the shared paintComposite.
 */
interface StreamingRender {
  element: HTMLImageElement | HTMLCanvasElement;
  /** Offscreen render target whose pixels drive the on-page translation layer. */
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  /** The original source image (kept for recompositing on plate/revise/finish). */
  image: HTMLImageElement;
  /** Inpaint plate once an applyPlate frame lands; null until then. */
  plateImage: HTMLImageElement | null;
  fontFamily: string;
  showDebug: boolean;
  /** Full detection-order box set (mutated in place as tl frames fill text). */
  textBoxes: TextBox[];
  /** Placement-order view over the SAME box references (collision siblings). */
  sortedTextBoxes: TextBox[];
  /** detection `index` -> the TextBox it hydrated. */
  indexToBox: Map<number, TextBox>;
  /** Ink rects recorded so far (progressive collision avoidance). */
  placedRects: PlacedRect[];
  /** Background luma snapshot for auto-contrast (recaptured on recompose). */
  lumaSnapshot: LumaSnapshot | null;
  /** index -> last text drawn, for idempotent tl + revise-triggered recompose. */
  drawn: Map<number, string>;
  /** DOM overlay <img> whose src mirrors `canvas` (updated per incremental paint). */
  translationLayer: HTMLImageElement | null;
}

interface FitResult {
  fontSize: number;
  lines: string[];
  lineHeight: number;
  totalHeight: number;
  maxLineWidth: number;
  /** True if wrapping had to break inside a word (char-level) at this size. */
  brokeWord: boolean;
}

/**
 * OPT 2: a single full-canvas pixel snapshot of the composited BACKGROUND
 * (original + inpainted plate + pass-1 white rects) taken ONCE before any text
 * is drawn. Per-box luminance sampling slices into this buffer instead of doing
 * a separate getImageData GPU readback per text box. This also matches the
 * backend, which samples sample_bg_luminance() from the clean text-free plate
 * (refit_final_composites.py: text is drawn onto a separate PIL image, never
 * back into the sampled ndarray).
 */
interface LumaSnapshot {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

interface RegionBBox {
  x: number;
  y: number;
  width: number;
  height: number;
  /**
   * Which source produced this region. 'bubble' = matched speech-bubble
   * interior; 'bubble-widened' = a tall-narrow bubble interior grown
   * horizontally so horizontal EN words fit on fewer, wider lines (see
   * widenHighAspectRegion); 'regions'/'bbox' = tight text-pixel sources;
   * 'bbox-widened' = a tall-narrow CLAMPED (no-bubble) caption column grown
   * horizontally (+ bounded vertical overflow) onto the clean plate so narration
   * stops cramming to the floor (see widenNarrowCaption); 'expanded' = the
   * null-bubble fallback grew a tight source outward to give long EN text room.
   */
  source?: 'bubble' | 'bubble-widened' | 'regions' | 'bbox' | 'bbox-widened' | 'expanded';
}

/**
 * A rendered text ink-rect recorded for inter-block overlap avoidance, mirroring
 * the backend compose_final `placed_rects` entries (x0,y0,x1,y1,is_dialogue).
 */
interface PlacedRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** True when this block matched a speech bubble (dialogue). */
  isDialogue: boolean;
}

// Minimum and maximum font sizes probed by the binary search.
// Matches backend refit_final_composites.find_best_fit (min_floor=14, max_cap=96).
// The backend's clamped (no-bubble) blocks drop to a hard_floor of 9 on overflow;
// we mirror that with FONT_SIZE_HARD_FLOOR below.
const FONT_SIZE_MIN = 14;
const FONT_SIZE_MAX = 96;
// Backend's hard floor for clamped/SFX blocks that still overflow at the soft floor.
const FONT_SIZE_HARD_FLOOR = 9;
// Extra padding inside the text region (reserved for stroke + breathing room).
const TEXT_PADDING_PX = 8;
// Line spacing leading factor: backend line_height = max(ascent+descent+leading,
// font_size) with leading = 0.10 * em (refit_final_composites.line_height_px).
const LINE_LEADING_FACTOR = 0.1;
// Backend uppercases all display text before fitting (refit compose_final).
// Short-word-only char-break guard: backend wrap_greedy only hard-breaks a word
// when len(word) >= 13; shorter words overflow on one line.
const LONG_WORD_MIN_LEN = 13;
// Contrast ratio threshold below which we override API-supplied colors.
const MIN_CONTRAST_RATIO = 3.0;

// FIX (reconcile): the backend compose_final deliberately does NOT widen
// clamped/non-bubble blocks, and uses the matched bubble interior AS-IS for
// dialogue (no high-aspect widening) — blind widening overlaps neighbouring
// columns. Gate the extension's aggressive widening/expansion OFF to match this
// conservative policy. Flip to true only to restore the old (divergent) widen.
const WIDEN_REGIONS = false;

// NARROW-NARRATION widen (CLAMPED / non-bubble caption columns only), mirroring
// the backend refit_final_composites NARROW_WIDEN_* policy. Tall-narrow vertical-
// JP caption columns otherwise cram horizontal EN one-word-per-line down to the
// hard floor. For non-SFX captions whose aspect exceeds the trigger we grant a
// BOUNDED horizontal widen (+ bounded vertical overflow) onto the clean inpaint
// plate, bounded by image edges and sibling boxes (the extension analogue of the
// backend's already-placed rects). Dialogue bubbles are still NOT widened
// (WIDEN_REGIONS stays false) to keep parity with the backend.
const NARROW_WIDEN_TRIGGER = 2.5; // widen only columns with aspect h/w above this
const NARROW_WIDEN_TARGET_ASPECT = 2.0; // widen until aspect reaches this (conservative)
const NARROW_WIDEN_MAX_GROWTH = 2.5; // hard cap: never wider than this x original width

// Comic font families. Backend pick_font picks Comic Neue Bold for long/dialogue
// text (reads like human comic lettering, vs the old tall-condensed Anton) and
// Bangers for short SFX-like outbursts; we mirror that. Bangers is bundled
// locally (public/fonts); Comic Neue is served from the Google Fonts CDN because
// its TTF is not (yet) bundled — see ensureFontsInjected / tryRegisterLocalFonts.
const PRIMARY_FONT_FAMILY = 'Comic Neue';
const SFX_FONT_FAMILY = 'Bangers';
// Font sizes (px) we eagerly prime via document.fonts.load so the binary
// search measures with real glyph metrics instead of the Arial fallback.
const FONT_PRIME_SIZES = [12, 24, 48, 72];

export class OverlayRenderer {
  private renderedImages: Map<HTMLElement, RenderedImage> = new Map();
  private fontsInjected = false;
  /** Memoized promise that resolves once comic fonts are usable on canvas. */
  private fontsReadyPromise: Promise<void> | null = null;
  /**
   * Per-render memoization of layoutAtSize results, keyed by
   * `fontSize|fontFamily|maxWidth|text`. The font-fit binary search is run up to
   * twice per box (soft floor, then hard-floor retry) and identical text recurs
   * across boxes, so the same (size, family, width, text) layout is recomputed
   * many times. The cached FitResult is byte-identical to a fresh measure — only
   * the ctx.font side-effect is skipped, which downstream painters re-set
   * themselves — so this is purely a measurement-count reduction with no visual
   * change. Reset at the start of every renderTranslationsOnCanvas.
   */
  private layoutCache: Map<string, FitResult> = new Map();

  /** In-flight progressive renders, keyed by the page element being translated. */
  private streaming: Map<HTMLElement, StreamingRender> = new Map();

  constructor() {
    this.ensureFontsInjected();
  }

  /**
   * Inject font sources into the host page so our canvas font-family fallback
   * chain has comic fonts to use.
   *
   * Strategy (preferring local, falling back to CDN):
   *   1. LOCAL @font-face via the extension's web_accessible_resources
   *      (`fonts/*`, declared in manifest.chrome.json). We register these with
   *      the FontFace API using browser.runtime.getURL so they resolve in the
   *      content-script context (the `chrome-extension://__MSG_@@extension_id__`
   *      placeholder in overlay.css only resolves for manifest-referenced CSS,
   *      NOT for canvas painting). Local fonts eliminate the CDN 404/offline +
   *      latency path.
   *   2. CDN <link> (Google Fonts) as a robust fallback. Comic Neue + Bangers
   *      are OFL-licensed and safe to link.
   *
   * Bangers-Regular.ttf is bundled in public/fonts/ and declared web-accessible
   * in both manifests, so its local path is the default. Comic Neue (the backend
   * PRIMARY, refit_final_composites FONT_STACK[0]) is NOT yet bundled there, so it
   * resolves via the CDN until ComicNeue-Bold.ttf is added to public/fonts/ +
   * web_accessible_resources. The CDN also remains an offline-safety fallback.
   */
  private ensureFontsInjected(): void {
    if (this.fontsInjected) return;
    if (typeof document === 'undefined') return;
    try {
      const id = 'manga-translator-google-fonts';
      if (document.getElementById(id)) {
        this.fontsInjected = true;
        return;
      }
      const link = document.createElement('link');
      link.id = id;
      link.rel = 'stylesheet';
      link.href =
        'https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&family=Bangers&display=swap';
      (document.head || document.documentElement).appendChild(link);
      this.fontsInjected = true;
    } catch (err) {
      // Non-fatal: canvas drawing will fall back to Arial/sans-serif.
      logger.warn('font injection failed', err);
    }
  }

  /**
   * Resolve once the comic fonts are actually available for canvas drawing.
   *
   * This closes the measure-in-Arial / paint-in-Bangers race: the renderer
   * previously measured & painted BEFORE the CDN font loaded, so the binary
   * search sized against Arial metrics and the final paint used a different
   * (wider) glyph set — causing overspill. We now:
   *   1. Attempt to register a LOCAL @font-face (via runtime.getURL) and add it
   *      to document.fonts — preferred over the CDN.
   *   2. Explicitly document.fonts.load() the comic families at several sizes.
   *   3. Await document.fonts.ready as a backstop.
   *
   * Memoized so we only pay the cost once per renderer instance. All failures
   * are swallowed (we degrade to the Arial fallback rather than blocking).
   */
  private async ensureFontsReady(): Promise<void> {
    if (this.fontsReadyPromise) return this.fontsReadyPromise;

    this.fontsReadyPromise = (async () => {
      if (typeof document === 'undefined' || !('fonts' in document)) return;
      const fontSet = (document as Document).fonts;

      // 1. Best-effort LOCAL @font-face registration (preferred over CDN).
      await this.tryRegisterLocalFonts(fontSet);

      // 2. Explicitly prime the comic families at the sizes the layout probes.
      const loadJobs: Promise<unknown>[] = [];
      for (const family of [PRIMARY_FONT_FAMILY, SFX_FONT_FAMILY]) {
        for (const size of FONT_PRIME_SIZES) {
          try {
            // "AaGg" exercises ascenders/descenders so metrics are realistic.
            loadJobs.push(fontSet.load(`bold ${size}px "${family}"`, 'AaGg'));
          } catch {
            // Some browsers throw on unknown families — ignore.
          }
        }
      }
      try {
        await Promise.allSettled(loadJobs);
      } catch {
        /* ignore */
      }

      // 3. Backstop: wait for the document's overall font readiness.
      try {
        await fontSet.ready;
      } catch {
        /* ignore */
      }
    })();

    return this.fontsReadyPromise;
  }

  /**
   * Try to register the bundled comic font(s) via the FontFace API using the
   * extension's own URL. Returns silently if the asset is missing or the
   * runtime API is unavailable (e.g. unit/test context).
   */
  private async tryRegisterLocalFonts(fontSet: FontFaceSet): Promise<void> {
    try {
      // Already registered? (family present in the set) — skip.
      const browserMod = await import('webextension-polyfill');
      const runtime = browserMod.default?.runtime;
      const getURL = runtime?.getURL?.bind(runtime);
      if (!getURL) return;

      // Map of family -> bundled asset path under web_accessible_resources.
      // Comic Neue (Bold) is the backend PRIMARY (long/dialogue); Bangers is for
      // short SFX. NOTE: ComicNeue-Bold.ttf is NOT yet bundled under public/fonts
      // (only Anton/Bangers are), so the HEAD-check below fails for it and we fall
      // back to the Google Fonts CDN registered in ensureFontsInjected. Add the
      // TTF to public/fonts + web_accessible_resources to take the local path.
      const localFonts: Array<{ family: string; path: string; weight: string }> = [
        { family: PRIMARY_FONT_FAMILY, path: 'fonts/ComicNeue-Bold.ttf', weight: 'bold' },
        { family: SFX_FONT_FAMILY, path: 'fonts/Bangers-Regular.ttf', weight: 'normal' },
      ];

      for (const { family, path, weight } of localFonts) {
        try {
          const url = getURL(path);
          // HEAD-check so a missing asset doesn't spam FontFace errors.
          const probe = await fetch(url, { method: 'HEAD' });
          if (!probe.ok) continue;
          // ComicNeue-Bold.ttf is already a bold-weight face, so register it as
          // 'bold' to avoid the canvas synthesising a second (faux) bold over it.
          const face = new FontFace(family, `url(${url}) format('truetype')`, {
            weight,
          });
          const loaded = await face.load();
          fontSet.add(loaded);
        } catch {
          // Missing/forbidden asset — fall back to CDN for this family.
        }
      }
    } catch {
      // No runtime (non-extension context) — CDN fallback handles it.
    }
  }

  /**
   * Create overlay by drawing translations onto the image itself
   * This replaces the original image/canvas element
   *
   * @param imageElement target image/canvas
   * @param textBoxes   boxes (in original-image coordinates)
   * @param showDebug   draw bbox/region overlays
   * @param inpaintedBase64 optional pre-inpainted "plate" to use as background
   * @param sourceBase64 optional already-encoded source image (the EXACT bytes
   *   the backend received, i.e. the compressed base64 from content-script).
   *   Passing it avoids a redundant drawImage+toDataURL re-encode of the live
   *   <img>, and is also the correct coordinate space for the returned text
   *   boxes. Must be a `data:` base64 URL; URL/CORS-blocked sources fall back
   *   to reading the element directly.
   */
  async createOverlay(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    textBoxes: TextBox[],
    showDebug: boolean = false,
    inpaintedBase64?: string | null,
    sourceBase64?: string | null
  ): Promise<void> {
    // Remove existing overlay if any
    this.removeOverlay(imageElement);

    try {
      // Get settings
      const settings = await settingsManager.getSettings();

      // Prefer the already-encoded source the backend saw (eliminates a second
      // re-encode of the live <img>). Only usable when it's real base64 data;
      // a URL (CORS-blocked path) still needs the element read / worker fetch.
      const base64Image =
        sourceBase64 && sourceBase64.startsWith('data:')
          ? sourceBase64
          : await this.getImageBase64(imageElement);

      // Feature-flag: use inpainted plate if provided (default true when present).
      const useInpaintedPlate: boolean =
        !!inpaintedBase64 && inpaintedBase64.length > 0;

      // Create canvas with translations
      const canvas = await this.renderTranslationsOnCanvas(
        base64Image,
        textBoxes,
        imageElement,
        settings.defaultFont,
        showDebug,
        useInpaintedPlate ? inpaintedBase64! : null
      );

      // Capture the ORIGINAL <img> src BEFORE we overwrite it, so we can
      // restore the live image cleanly when translation is toggled off.
      const originalSrc =
        imageElement instanceof HTMLImageElement ? imageElement.src : undefined;

      // Replace original element with rendered canvas/image
      await this.replaceElement(imageElement, canvas);

      // Build DOM overlay with retry/edit + flag affordances.
      const domOverlay = this.buildDomOverlay(imageElement, canvas, textBoxes);

      this.renderedImages.set(imageElement, {
        originalElement: imageElement,
        newElement: canvas,
        domOverlay,
        textBoxes,
        // Source bytes the backend received (base64 data URL). Used to (a)
        // restore the original on toggle-off and (b) POST to /flag.
        originalImageBase64: base64Image,
        originalSrc,
      });
    } catch (error) {
      logger.error('Failed to create overlay:', error);
      throw error;
    }
  }

  /* ===================== PROGRESSIVE (STREAMING) API =====================
   *
   * Incremental counterpart to createOverlay, split along the existing pass
   * seams. Used by the content-script when the backend emits v:1 event frames:
   *
   *   beginOverlay(detections)  -> swap element, draw source + white boxes,
   *                                register the overlay early (source visible
   *                                immediately, before any text arrives).
   *   applyTranslation(i, text) -> typeset ONE bubble incrementally, reusing the
   *                                placement/collision computed in beginOverlay.
   *   applyPlate(b64)           -> recomposite background + redraw all text,
   *                                recapture luma.
   *   finish(debug)             -> authoritative final recompose (+ debug).
   *
   * All four are thin orchestration over the SAME internals the single-pass
   * path uses (paintComposite / drawTextBoxText / captureLumaSnapshot), so there
   * is no forked renderer.
   */

  /**
   * Begin a progressive render: draw the source image + white masks from the
   * detection geometry (no text yet), swap the element, and register the overlay
   * so the original is visible instantly while translations stream in.
   */
  async beginOverlay(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    detectionBoxes: StreamDetectionBox[],
    showDebug: boolean,
    sourceBase64?: string | null
  ): Promise<void> {
    // Tear down anything previously rendered / streaming for this element.
    this.removeOverlay(imageElement);

    const settings = await settingsManager.getSettings();

    const base64Image =
      sourceBase64 && sourceBase64.startsWith('data:')
        ? sourceBase64
        : await this.getImageBase64(imageElement);

    // Fonts must be ready before the first measure/paint (same as legacy path).
    await this.ensureFontsReady();
    this.layoutCache.clear();

    const image = await this.loadImage(base64Image);
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get canvas context');

    // Hydrate geometry-only detection boxes into TextBoxes (empty text) and
    // index them for later tl/revise frames.
    const textBoxes: TextBox[] = [];
    const indexToBox = new Map<number, TextBox>();
    for (const det of detectionBoxes) {
      const tb = detectionBoxToTextBox(det);
      textBoxes.push(tb);
      indexToBox.set(det.index, tb);
    }

    // Paint source + white boxes + luma. With empty translatedText, the text
    // pass draws nothing (drawTextBoxText early-returns), so this is exactly the
    // "background only" frame. placedRects comes back empty.
    const { sortedTextBoxes, placedRects, lumaSnapshot } = this.paintComposite(
      ctx,
      image,
      null,
      textBoxes,
      settings.defaultFont,
      false
    );

    const originalSrc =
      imageElement instanceof HTMLImageElement ? imageElement.src : undefined;

    await this.replaceElement(imageElement, canvas);
    const domOverlay = this.buildDomOverlay(imageElement, canvas, textBoxes);
    const translationLayer =
      (domOverlay?.querySelector(
        '.manga-translator-translation-layer'
      ) as HTMLImageElement | null) ?? null;

    // Register the overlay early so removeOverlay/clearAll/setOverlayOpacity and
    // the flag/edit affordances all work mid-stream.
    this.renderedImages.set(imageElement, {
      originalElement: imageElement,
      newElement: canvas,
      domOverlay,
      textBoxes,
      originalImageBase64: base64Image,
      originalSrc,
    });

    this.streaming.set(imageElement, {
      element: imageElement,
      canvas,
      ctx,
      image,
      plateImage: null,
      fontFamily: settings.defaultFont,
      showDebug,
      textBoxes,
      sortedTextBoxes,
      indexToBox,
      placedRects,
      lumaSnapshot,
      drawn: new Map(),
      translationLayer,
    });
  }

  /**
   * Apply one translated bubble to an in-flight streaming render. Idempotent by
   * `index` (a repeated identical tl is a no-op); a CHANGED text for an already-
   * drawn index (revise) triggers an authoritative recompose so the stale ink is
   * cleanly replaced.
   */
  applyTranslation(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    index: number,
    text: string
  ): void {
    const st = this.streaming.get(imageElement);
    if (!st) return;
    const box = st.indexToBox.get(index);
    if (!box) return;

    const prev = st.drawn.get(index);
    if (prev === text) return; // idempotent — nothing changed

    box.translatedText = text;

    if (prev !== undefined) {
      // Text changed for an already-painted box (revise/correction). We can't
      // erase a single box from the composited canvas, so recompose the frame.
      this.recomposeStreaming(st);
      return;
    }

    // First time this box gets text — draw just it, appending its ink rect to
    // the running collision list (progressive; reuses beginOverlay placement).
    st.drawn.set(index, text);
    this.drawTextBoxText(
      st.ctx,
      box,
      st.fontFamily,
      st.sortedTextBoxes,
      st.canvas.width,
      st.canvas.height,
      st.placedRects,
      st.lumaSnapshot
    );
    this.blitStreaming(st);
  }

  /**
   * Apply the inpaint plate to an in-flight streaming render: adopt it as the
   * background and recomposite (redraw plate + all text so far, recapture luma).
   */
  async applyPlate(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    plateBase64: string
  ): Promise<void> {
    const st = this.streaming.get(imageElement);
    if (!st) return;
    try {
      const plateSrc = plateBase64.startsWith('data:')
        ? plateBase64
        : `data:image/png;base64,${plateBase64}`;
      st.plateImage = await this.loadImage(plateSrc);
    } catch (err) {
      logger.warn('Streaming plate failed to load, keeping white-box mask:', err);
      return;
    }
    this.recomposeStreaming(st);
  }

  /**
   * Finalize an in-flight streaming render: authoritative recompose (guarantees
   * the final frame is byte-identical to the single-pass path regardless of tl
   * arrival order), optionally with debug overlays, then drop the streaming
   * state. The registered overlay (with final textBoxes) remains.
   */
  finish(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    showDebug?: boolean
  ): void {
    const st = this.streaming.get(imageElement);
    if (!st) return;
    if (showDebug !== undefined) st.showDebug = showDebug;

    this.recomposeStreaming(st);

    const rendered = this.renderedImages.get(imageElement);
    if (rendered) rendered.textBoxes = st.textBoxes;

    this.streaming.delete(imageElement);
  }

  /**
   * Full authoritative recompose of a streaming frame (background + plate + ALL
   * text in placement order + optional debug), refreshing the state's collision
   * rects, luma snapshot and drawn-index map. Used by applyPlate, revise and
   * finish. Deterministic and independent of tl arrival order.
   */
  private recomposeStreaming(st: StreamingRender): void {
    // Match the legacy per-render reset so a stale measure can't be reused.
    this.layoutCache.clear();

    const { placedRects, lumaSnapshot } = this.paintComposite(
      st.ctx,
      st.image,
      st.plateImage,
      st.textBoxes,
      st.fontFamily,
      st.showDebug
    );
    st.placedRects = placedRects;
    st.lumaSnapshot = lumaSnapshot;

    // Rebuild the drawn-index map from whatever text has arrived so far.
    st.drawn.clear();
    for (const [idx, box] of st.indexToBox) {
      if (box.translatedText && !box.skipped) st.drawn.set(idx, box.translatedText);
    }

    this.blitStreaming(st);
  }

  /**
   * Push the offscreen streaming canvas to the page: update the DOM overlay's
   * translation-layer <img> and, for a <canvas> element, redraw the live canvas.
   */
  private blitStreaming(st: StreamingRender): void {
    try {
      if (st.element instanceof HTMLCanvasElement) {
        const lctx = st.element.getContext('2d');
        if (lctx) {
          lctx.clearRect(0, 0, st.element.width, st.element.height);
          lctx.drawImage(st.canvas, 0, 0);
        }
      }
      if (st.translationLayer) {
        st.translationLayer.src = st.canvas.toDataURL('image/jpeg', 0.9);
      }
    } catch (err) {
      logger.warn('Streaming blit failed:', err);
    }
  }

  /**
   * Get base64 image data from element
   */
  private async getImageBase64(element: HTMLImageElement | HTMLCanvasElement): Promise<string> {
    if (element instanceof HTMLCanvasElement) {
      try {
        return element.toDataURL('image/jpeg', 0.85);
      } catch (error) {
        throw new Error('Cannot read canvas due to CORS protection');
      }
    }

    if (element.dataset.originalSrc && element.dataset.originalSrc.startsWith('data:')) {
      return element.dataset.originalSrc;
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get canvas context');

    canvas.width = element.naturalWidth || element.width;
    canvas.height = element.naturalHeight || element.height;

    try {
      ctx.drawImage(element, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.85);
    } catch (error) {
      logger.warn('CORS blocked image conversion, fetching via background worker:', element.src);

      const browser = (await import('webextension-polyfill')).default;
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
  }

  /**
   * Render translations onto a canvas.
   * If `inpaintedBase64` is provided, the inpainted plate is drawn as the
   * background instead of (or layered on top of) the original image.
   */
  private async renderTranslationsOnCanvas(
    base64Image: string,
    textBoxes: TextBox[],
    originalElement: HTMLImageElement | HTMLCanvasElement,
    fontFamily: string,
    showDebug: boolean = false,
    inpaintedBase64: string | null = null
  ): Promise<HTMLCanvasElement> {
    // Close the measure/paint font race: make sure the comic fonts are loaded
    // BEFORE the first layoutAtSize/measureText call below. Otherwise the
    // binary search measures Arial metrics and the paint uses Bangers ->
    // systematic overspill. Best-effort; degrades to Arial on failure.
    await this.ensureFontsReady();

    // Reset the per-render layout memo (OPT 4). Cleared each render so a stale
    // measure from a prior image / font-readiness state can never be reused.
    this.layoutCache.clear();

    // Always load original first — we may still need it for luminance sampling
    // under boxes that lack inpainted coverage (e.g. partial plates).
    const image = await this.loadImage(base64Image);

    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;

    // OPT 2: luminance now does a SINGLE full-canvas getImageData (see
    // captureLumaSnapshot) instead of one readback per box, so the
    // willReadFrequently CPU-backing hint is no longer worth the slower
    // drawImage/draw path. Dropping it does not affect pixels — it is purely a
    // performance hint, and the one remaining readback works on any canvas.
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get canvas context');

    // Load the inpaint plate (if any) up front so the shared painter can layer
    // it. Failure degrades to the white-box mask path (plateImage stays null).
    let plateImage: HTMLImageElement | null = null;
    if (inpaintedBase64) {
      try {
        const plateSrc = inpaintedBase64.startsWith('data:')
          ? inpaintedBase64
          : `data:image/png;base64,${inpaintedBase64}`;
        plateImage = await this.loadImage(plateSrc);
      } catch (err) {
        logger.warn('Inpainted plate failed to load, falling back to mask:', err);
      }
    }

    // Single-pass composite (original + plate + backgrounds + text + debug).
    // The same painter backs the incremental streaming path (beginOverlay /
    // applyPlate / finish) so both routes produce byte-identical frames.
    this.paintComposite(ctx, image, plateImage, textBoxes, fontFamily, showDebug);

    // Suppress unused-var warning for originalElement (kept for API compat)
    void originalElement;

    return canvas;
  }

  /**
   * Shared full-frame painter. Draws, in order:
   *   1. the original image,
   *   2. the inpaint plate (if `plateImage` provided) over it,
   *   3. Pass 1 backgrounds — white masks when there is NO plate, or just the
   *      WIDENED regions when there is (the plate already covers tight bubbles),
   *   4. a single luminance snapshot of the text-free background,
   *   5. Pass 2 text in placement order, threading a collision-rect list,
   *   6. Pass 3 debug overlays (optional).
   *
   * Extracted verbatim from the old renderTranslationsOnCanvas body so the
   * legacy/HTTP path is unchanged, and reused by the streaming recompose so the
   * incremental path is a thin orchestration rather than a second renderer.
   * Returns the derived placement order, the collision rects it recorded, and
   * the luma snapshot so the streaming state can keep drawing incrementally.
   */
  private paintComposite(
    ctx: CanvasRenderingContext2D,
    image: HTMLImageElement,
    plateImage: HTMLImageElement | null,
    textBoxes: TextBox[],
    fontFamily: string,
    showDebug: boolean
  ): {
    sortedTextBoxes: TextBox[];
    placedRects: PlacedRect[];
    lumaSnapshot: LumaSnapshot | null;
  } {
    const canvas = ctx.canvas;

    // Clear first so RECOMPOSITE calls (plate arrival, revise, finish) start
    // from a blank frame. On the legacy first paint the canvas is already blank,
    // so this is a no-op there.
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw original image first.
    ctx.drawImage(image, 0, 0, image.width, image.height);

    // If a plate was supplied, overlay it on top of the original at image dims.
    const hasPlate = !!plateImage;
    if (plateImage) {
      ctx.drawImage(plateImage, 0, 0, image.width, image.height);
    }

    // Placement order mirrors the backend compose_final `order`:
    //   1. DIALOGUE (bubble-matched) blocks first so they are never covered.
    //   2. then clamped caption/SFX blocks SMALLEST-area-first, so small
    //      specific narration columns claim their space and an over-large
    //      MERGED caption shrinks/clips around them instead of overlapping.
    const sortedTextBoxes = this.computePlacementOrder(textBoxes).filter(
      (tb) => !tb.skipped
    );

    // When we have an inpainted plate, skip drawing white backgrounds — the
    // plate *is* the background. Otherwise fall back to the original masked
    // rounded rects.
    if (!hasPlate) {
      for (const textBox of sortedTextBoxes) {
        this.drawTextBoxBackground(
          ctx,
          textBox,
          sortedTextBoxes,
          canvas.width,
          canvas.height
        );
      }
    } else {
      // With an inpainted plate, the plate already covers each bubble interior.
      // But WIDENED regions (high-aspect bubbles, and tall-narrow caption columns
      // grown by widenNarrowCaption) extend past the tight plate onto un-erased
      // art to fit horizontal EN — so plate the full widened region to keep that
      // text off the original art. Normal bubbles/boxes are untouched.
      for (const textBox of sortedTextBoxes) {
        const region = this.computeTextRegionBBox(
          textBox,
          sortedTextBoxes,
          canvas.width,
          canvas.height
        );
        if (region.source === 'bubble-widened' || region.source === 'bbox-widened') {
          this.drawRoundedRect(
            ctx,
            region.x,
            region.y,
            region.width,
            region.height,
            'white',
            8
          );
        }
      }
    }

    // OPT 2: snapshot the composited BACKGROUND once, here — after the plate +
    // pass-1 white rects but BEFORE any text. Auto-contrast then slices per-box
    // luminance out of this single buffer instead of issuing one getImageData
    // GPU readback per text box. This matches the backend, which samples
    // luminance from the clean text-free plate (text goes onto a separate PIL
    // image, never back into the sampled ndarray). null on a tainted canvas →
    // per-box sampling falls back and yields the same "light bg" default.
    const lumaSnapshot = this.captureLumaSnapshot(ctx);

    // Pass 2: Draw text in placement order, threading a collision-rect list so
    // later blocks avoid burying earlier ones (mirrors compose_final's
    // `placed_rects`). Boxes with empty translatedText (streaming: not yet
    // arrived) are skipped by drawTextBoxText and simply painted later.
    const placedRects: PlacedRect[] = [];
    for (const textBox of sortedTextBoxes) {
      this.drawTextBoxText(
        ctx,
        textBox,
        fontFamily,
        sortedTextBoxes,
        canvas.width,
        canvas.height,
        placedRects,
        lumaSnapshot
      );
    }

    // Pass 3 (optional): Draw debug overlays
    if (showDebug) {
      this.drawDebugOverlay(ctx, sortedTextBoxes);
    }

    return { sortedTextBoxes, placedRects, lumaSnapshot };
  }

  /**
   * Draw ONLY the white background for a text box (Pass 1 of two-pass rendering).
   *
   * Mask/plate coupling policy (reconciled with the null-bubble expansion):
   *   - IN-BUBBLE case (bubbleRect matched): the speech bubble already provides
   *     a clean background, so we keep the mask TIGHT to the text pixels
   *     (textRegions) — painting the whole bubble interior would cover bubble
   *     art / tails. Text never sits on un-erased pixels because the bubble
   *     interior is uniform.
   *   - NULL-BUBBLE / EXPANDED case (SFX over art, no bubble): the text now
   *     lays out inside an EXPANDED region (computeTextRegionBBox), which can
   *     extend beyond the tight text pixels onto un-erased art. So here we
   *     paint the plate to cover the actual LAYOUT region, guaranteeing text
   *     never sits on busy art. This is the same region findBestFit uses, so
   *     plate and text stay coupled.
   */
  private drawTextBoxBackground(
    ctx: CanvasRenderingContext2D,
    textBox: TextBox,
    allBoxes?: TextBox[],
    canvasW?: number,
    canvasH?: number
  ): void {
    const region = this.computeTextRegionBBox(textBox, allBoxes, canvasW, canvasH);

    // In-bubble: keep the mask tight to text pixels (preserve bubble art).
    if (region.source === 'bubble') {
      if (textBox.textRegions && textBox.textRegions.length > 0) {
        for (const r of textBox.textRegions) {
          this.drawRoundedRect(
            ctx,
            r.minX,
            r.minY,
            r.maxX - r.minX,
            r.maxY - r.minY,
            'white',
            4
          );
        }
      } else {
        this.drawRoundedRect(
          ctx,
          textBox.minX,
          textBox.minY,
          textBox.maxX - textBox.minX,
          textBox.maxY - textBox.minY,
          'white',
          8
        );
      }
      return;
    }

    // bubble-widened / null-bubble / expanded / tight: the layout region now
    // extends beyond the tight text pixels (and, for bubble-widened, beyond the
    // original bubble interior onto art), so the plate MUST cover the full
    // layout region — otherwise widened text would land on un-erased pixels.
    this.drawRoundedRect(
      ctx,
      region.x,
      region.y,
      region.width,
      region.height,
      'white',
      8
    );
  }

  /**
   * Compute the box we wrap and center text inside. Fallback chain:
   *   1. bubbleRect  — the matched speech-bubble interior (larger than the
   *      tight OCR bbox), so text fills the bubble and overspill resolves.
   *   2. textRegions/bbox — tight text-pixel sources. When NO qualifying
   *      bubble was matched (bubbleRect null/zero-area — e.g. SFX over art,
   *      the ~11% overspill + 27 miss boxes in the perf evidence), we do NOT
   *      fall straight to the tight box. Instead we EXPAND the tight union
   *      outward (see expandRegion) so long EN text has room to lay out at a
   *      readable size instead of overflowing an 8px-floored tight box.
   *
   * bubbleRect is optional and may be null, so older responses without it
   * still render via the (expanded) textRegions/bbox fallback.
   *
   * @param allBoxes optional sibling boxes; used to bound expansion so we
   *   don't grow a region over a neighbor's text.
   * @param canvasW/canvasH image bounds; expansion is clamped to these.
   */
  private computeTextRegionBBox(
    textBox: TextBox,
    allBoxes?: TextBox[],
    canvasW?: number,
    canvasH?: number
  ): RegionBBox {
    const b = textBox.bubbleRect;
    if (b && b.maxX > b.minX && b.maxY > b.minY) {
      const bubble: RegionBBox = {
        x: b.minX,
        y: b.minY,
        width: Math.max(1, b.maxX - b.minX),
        height: Math.max(1, b.maxY - b.minY),
        source: 'bubble',
      };
      // Backend uses the matched bubble interior AS-IS (no widening). The old
      // high-aspect widening is gated off (WIDEN_REGIONS) to match.
      if (!WIDEN_REGIONS) return bubble;
      // Tall-narrow JP bubbles (read vertically) are far too thin for
      // horizontal EN, forcing mid-word breaks. Widen them horizontally,
      // centered on the bubble, bounded by image edges + neighbors.
      return this.widenHighAspectRegion(bubble, textBox, allBoxes, canvasW, canvasH);
    }

    // CLAMPED (no bubble). Backend compose_final fits clamped blocks to the
    // BLOCK bbox (rect == block: minX..maxX). SFX/orphan boxes stay put, but a
    // tall-narrow *caption* column gets a BOUNDED horizontal widen (+ vertical
    // overflow) onto the clean plate instead of cramming to the floor — mirroring
    // the backend narrow-narration widen.
    if (!WIDEN_REGIONS) {
      const bbox: RegionBBox = {
        x: textBox.minX,
        y: textBox.minY,
        width: Math.max(1, textBox.maxX - textBox.minX),
        height: Math.max(1, textBox.maxY - textBox.minY),
        source: 'bbox',
      };
      // SFX-sized boxes are excluded (truncated to onomatopoeia, must stay put).
      if (this.isSfxSized(textBox)) return bbox;
      return this.widenNarrowCaption(bbox, textBox, allBoxes, canvasW, canvasH);
    }

    // Build the tight union (textRegions if present, else outer bbox).
    let tight: RegionBBox;
    if (textBox.textRegions && textBox.textRegions.length > 0) {
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      for (const r of textBox.textRegions) {
        if (r.minX < minX) minX = r.minX;
        if (r.minY < minY) minY = r.minY;
        if (r.maxX > maxX) maxX = r.maxX;
        if (r.maxY > maxY) maxY = r.maxY;
      }
      tight = {
        x: minX,
        y: minY,
        width: Math.max(1, maxX - minX),
        height: Math.max(1, maxY - minY),
        source: 'regions',
      };
    } else {
      tight = {
        x: textBox.minX,
        y: textBox.minY,
        width: Math.max(1, textBox.maxX - textBox.minX),
        height: Math.max(1, textBox.maxY - textBox.minY),
        source: 'bbox',
      };
    }

    // Null/zero-area bubble => expand the tight box to give text room.
    return this.expandRegion(tight, textBox, allBoxes, canvasW, canvasH);
  }

  /**
   * Grow a tight text region outward when no speech bubble was matched.
   *
   * Heuristic (documented):
   *   - Target ~2x area (matches the observed median bubbleRect/bbox expansion
   *     ratio of 2.0x in the perf evidence), via a margin of ~50% of each
   *     dimension capped at 48px per side so huge SFX boxes don't balloon.
   *   - Clamp to image bounds [0, canvasW] x [0, canvasH].
   *   - Bound by the nearest neighbor box edge on each side (when sibling boxes
   *     are provided) minus a small gutter, so we never grow over a neighbor's
   *     text. Cheap O(n) per box; n is tiny (<= ~10 blocks/page).
   */
  private expandRegion(
    tight: RegionBBox,
    self: TextBox,
    allBoxes: TextBox[] | undefined,
    canvasW: number | undefined,
    canvasH: number | undefined
  ): RegionBBox {
    const MAX_MARGIN_PX = 48;
    const NEIGHBOR_GUTTER_PX = 6;
    const marginX = Math.min(MAX_MARGIN_PX, Math.round(tight.width * 0.5));
    const marginY = Math.min(MAX_MARGIN_PX, Math.round(tight.height * 0.5));

    let left = tight.x - marginX;
    let top = tight.y - marginY;
    let right = tight.x + tight.width + marginX;
    let bottom = tight.y + tight.height + marginY;

    // Clamp to image bounds.
    const imgW = canvasW ?? Infinity;
    const imgH = canvasH ?? Infinity;
    left = Math.max(0, left);
    top = Math.max(0, top);
    right = Math.min(imgW, right);
    bottom = Math.min(imgH, bottom);

    // Bound by neighbors so we don't overlap their text (cheap rejection:
    // only neighbors that vertically/horizontally straddle us can collide).
    if (allBoxes && allBoxes.length > 1) {
      const cx = tight.x + tight.width / 2;
      const cy = tight.y + tight.height / 2;
      for (const other of allBoxes) {
        if (other === self) continue;
        const oL = other.minX;
        const oT = other.minY;
        const oR = other.maxX;
        const oB = other.maxY;
        // Vertical overlap with the tight band => can clip our horizontal grow.
        const vOverlap = oB > tight.y && oT < tight.y + tight.height;
        if (vOverlap) {
          if (oR <= cx) left = Math.max(left, oR + NEIGHBOR_GUTTER_PX);
          if (oL >= cx) right = Math.min(right, oL - NEIGHBOR_GUTTER_PX);
        }
        const hOverlap = oR > tight.x && oL < tight.x + tight.width;
        if (hOverlap) {
          if (oB <= cy) top = Math.max(top, oB + NEIGHBOR_GUTTER_PX);
          if (oT >= cy) bottom = Math.min(bottom, oT - NEIGHBOR_GUTTER_PX);
        }
      }
    }

    // Never shrink below the tight box (neighbor clamps could over-constrain).
    left = Math.min(left, tight.x);
    top = Math.min(top, tight.y);
    right = Math.max(right, tight.x + tight.width);
    bottom = Math.max(bottom, tight.y + tight.height);

    return {
      x: left,
      y: top,
      width: Math.max(1, right - left),
      height: Math.max(1, bottom - top),
      source: 'expanded',
    };
  }

  /**
   * Widen a tall-narrow speech-bubble interior horizontally so horizontal
   * English words fit on fewer, wider lines.
   *
   * WHY: JP is set in vertical columns, so its bubbles are tall and thin
   * (h/w up to ~3-5x in the perf evidence). Laying horizontal EN into that
   * literal interior forces tiny per-line widths and mid-word breaks
   * ("MOMMY" -> "MOM"/"MY"). Trading a little of the (over-abundant) height
   * for width lets whole words sit on one line.
   *
   * Heuristic (documented):
   *   - Only engage for ASPECT_TRIGGER = height/width > 1.6 (normal/wide
   *     bubbles are left untouched -> backward compatible).
   *   - Target a final aspect of TARGET_ASPECT = 1.2 (slightly taller than
   *     wide reads naturally for short stacked phrases) by growing width,
   *     capped at MAX_WIDTH_GROWTH = 2.6x the bubble width so we never balloon.
   *   - Keep the box centered on the ORIGINAL bubble center.
   *   - Clamp to image bounds and to neighbor box edges (minus a gutter) so we
   *     don't overrun an adjacent bubble/panel. Cheap O(n); n is tiny.
   *   - Height is left untouched (the bubble already has ample height); only
   *     width grows. If neighbors clamp width hard, we keep whatever room we
   *     could win — never narrower than the original bubble.
   */
  private widenHighAspectRegion(
    bubble: RegionBBox,
    self: TextBox,
    allBoxes: TextBox[] | undefined,
    canvasW: number | undefined,
    canvasH: number | undefined
  ): RegionBBox {
    const ASPECT_TRIGGER = 1.6;
    const TARGET_ASPECT = 1.2; // desired height/width after widening
    const MAX_WIDTH_GROWTH = 2.6;
    const NEIGHBOR_GUTTER_PX = 6;

    const aspect = bubble.height / bubble.width;
    if (aspect <= ASPECT_TRIGGER) return bubble; // normal/wide -> unchanged.

    // Desired width to reach TARGET_ASPECT, capped by MAX_WIDTH_GROWTH.
    const desiredWidth = Math.min(
      bubble.height / TARGET_ASPECT,
      bubble.width * MAX_WIDTH_GROWTH
    );
    if (desiredWidth <= bubble.width) return bubble;

    const cx = bubble.x + bubble.width / 2;
    let left = cx - desiredWidth / 2;
    let right = cx + desiredWidth / 2;
    const top = bubble.y;
    const bottom = bubble.y + bubble.height;

    // Clamp to image bounds.
    const imgW = canvasW ?? Infinity;
    left = Math.max(0, left);
    right = Math.min(imgW, right);

    // Bound horizontal growth by neighbors that vertically overlap us.
    if (allBoxes && allBoxes.length > 1) {
      for (const other of allBoxes) {
        if (other === self) continue;
        const ob = other.bubbleRect;
        const oL = ob ? ob.minX : other.minX;
        const oT = ob ? ob.minY : other.minY;
        const oR = ob ? ob.maxX : other.maxX;
        const oB = ob ? ob.maxY : other.maxY;
        const vOverlap = oB > top && oT < bottom;
        if (!vOverlap) continue;
        if (oR <= cx) left = Math.max(left, oR + NEIGHBOR_GUTTER_PX);
        if (oL >= cx) right = Math.min(right, oL - NEIGHBOR_GUTTER_PX);
      }
    }

    // Never shrink below the original bubble width.
    left = Math.min(left, bubble.x);
    right = Math.max(right, bubble.x + bubble.width);

    void canvasH;
    return {
      x: left,
      y: top,
      width: Math.max(1, right - left),
      height: Math.max(1, bottom - top),
      source: 'bubble-widened',
    };
  }

  /**
   * Bounded widen for a tall-narrow CLAMPED (no-bubble) caption column, mirroring
   * the backend refit_final_composites narrow-narration widen.
   *
   *   - Only engages above NARROW_WIDEN_TRIGGER aspect (h/w); normal/wide columns
   *     are returned unchanged.
   *   - Widens horizontally toward NARROW_WIDEN_TARGET_ASPECT, capped at
   *     NARROW_WIDEN_MAX_GROWTH x the original width; centered on the column.
   *   - Grants a modest BOUNDED vertical overflow (~18%, matching the backend
   *     overflow_frac) onto the clean plate so the readable size isn't clipped.
   *   - Clamps to image bounds and to sibling box edges (minus a gutter) — the
   *     extension analogue of the backend's already-placed-rect bound — so it
   *     never grows over a neighbour's text. Never smaller than the original box.
   */
  private widenNarrowCaption(
    bbox: RegionBBox,
    self: TextBox,
    allBoxes: TextBox[] | undefined,
    canvasW: number | undefined,
    canvasH: number | undefined
  ): RegionBBox {
    const aspect = bbox.height / bbox.width;
    if (aspect <= NARROW_WIDEN_TRIGGER) return bbox; // normal/wide -> unchanged.

    const desiredWidth = Math.min(
      bbox.height / NARROW_WIDEN_TARGET_ASPECT,
      bbox.width * NARROW_WIDEN_MAX_GROWTH
    );
    if (desiredWidth <= bbox.width) return bbox;

    const NEIGHBOR_GUTTER_PX = 6;
    const imgW = canvasW ?? Infinity;
    const imgH = canvasH ?? Infinity;
    const cx = bbox.x + bbox.width / 2;
    const cy = bbox.y + bbox.height / 2;
    const top0 = bbox.y;
    const bottom0 = bbox.y + bbox.height;

    // Horizontal widen, bounded by image edges + vertically-overlapping siblings.
    let left = Math.max(0, cx - desiredWidth / 2);
    let right = Math.min(imgW, cx + desiredWidth / 2);
    if (allBoxes && allBoxes.length > 1) {
      for (const other of allBoxes) {
        if (other === self) continue;
        const ob = other.bubbleRect;
        const oL = ob ? ob.minX : other.minX;
        const oT = ob ? ob.minY : other.minY;
        const oR = ob ? ob.maxX : other.maxX;
        const oB = ob ? ob.maxY : other.maxY;
        if (!(oB > top0 && oT < bottom0)) continue; // no vertical overlap
        if (oR <= cx) left = Math.max(left, oR + NEIGHBOR_GUTTER_PX);
        if (oL >= cx) right = Math.min(right, oL - NEIGHBOR_GUTTER_PX);
      }
    }
    left = Math.min(left, bbox.x); // never narrower than the original box
    right = Math.max(right, bbox.x + bbox.width);

    // Bounded vertical overflow onto the clean plate, bounded by image edges +
    // horizontally-overlapping siblings of the (now widened) column.
    const vpad = Math.round(bbox.height * 0.18);
    let top = Math.max(0, top0 - Math.floor(vpad / 2));
    let bottom = Math.min(imgH, bottom0 + Math.floor(vpad / 2));
    if (allBoxes && allBoxes.length > 1) {
      for (const other of allBoxes) {
        if (other === self) continue;
        const ob = other.bubbleRect;
        const oL = ob ? ob.minX : other.minX;
        const oT = ob ? ob.minY : other.minY;
        const oR = ob ? ob.maxX : other.maxX;
        const oB = ob ? ob.maxY : other.maxY;
        if (!(oR > left && oL < right)) continue; // no horizontal overlap
        if (oB <= cy) top = Math.max(top, oB + NEIGHBOR_GUTTER_PX);
        if (oT >= cy) bottom = Math.min(bottom, oT - NEIGHBOR_GUTTER_PX);
      }
    }
    top = Math.min(top, bbox.y);
    bottom = Math.max(bottom, bbox.y + bbox.height);

    return {
      x: left,
      y: top,
      width: Math.max(1, right - left),
      height: Math.max(1, bottom - top),
      source: 'bbox-widened',
    };
  }

  /**
   * Compute the placement order for the page, mirroring the backend
   * compose_final `order`: dialogue (bubble-matched) blocks first, then clamped
   * caption/SFX blocks smallest-bbox-area first.
   */
  private computePlacementOrder(textBoxes: TextBox[]): TextBox[] {
    return [...textBoxes].sort((a, b) => {
      const aDialogue = this.isDialogueBlock(a) ? 0 : 1;
      const bDialogue = this.isDialogueBlock(b) ? 0 : 1;
      if (aDialogue !== bDialogue) return aDialogue - bDialogue;
      const aArea = (a.maxX - a.minX) * (a.maxY - a.minY);
      const bArea = (b.maxX - b.minX) * (b.maxY - b.minY);
      return aArea - bArea; // smallest-area-first among same dialogue class
    });
  }

  /**
   * A block is DIALOGUE when it matched a (non-degenerate) speech bubble — the
   * same test computeTextRegionBBox uses to take the 'bubble' branch. Mirrors
   * the backend `is_dialogue = fit_rect is not None`.
   */
  private isDialogueBlock(textBox: TextBox): boolean {
    const b = textBox.bubbleRect;
    return !!(b && b.maxX > b.minX && b.maxY > b.minY);
  }

  /**
   * Backend _is_sfx_sized: small/orphan boxes (the verbose-gloss offenders).
   * The extension TextBox has no `orphan` flag, so we use only the size
   * heuristic (short side <= 48 px OR area <= 9000 px).
   */
  private isSfxSized(textBox: TextBox): boolean {
    const w = Math.abs(textBox.maxX - textBox.minX);
    const h = Math.abs(textBox.maxY - textBox.minY);
    const shortSide = Math.min(w, h);
    const area = w * h;
    return shortSide <= 48 || area <= 9000;
  }

  /**
   * Draw the text for a text box (Pass 2). Replicates the per-block portion of
   * the backend compose_final: normalize + uppercase, SFX truncation for small
   * clamped boxes, font selection (Anton vs Bangers), clamped-floor retry,
   * luminance auto-contrast, and inter-block collision suppression that records
   * each rendered ink rect into `placedRects`.
   */
  private drawTextBoxText(
    ctx: CanvasRenderingContext2D,
    textBox: TextBox,
    fontFamily: string,
    allBoxes: TextBox[] | undefined,
    canvasW: number | undefined,
    canvasH: number | undefined,
    placedRects: PlacedRect[],
    lumaSnapshot?: LumaSnapshot | null
  ): void {
    const raw = textBox.translatedText;
    // A skipped region (e.g. already-English) keeps its original pixels — never
    // mask or overlay text on it. Defensive: filtered upstream too.
    if (!raw || textBox.skipped) return;

    const isDialogue = this.isDialogueBlock(textBox);
    const isClamped = !isDialogue; // caption / orphan / SFX over art
    const sfxSized = this.isSfxSized(textBox);

    // FIX A.3: cap verbose SFX glosses in small/orphan clamped boxes to
    // onomatopoeia length so they don't overflow onto neighbours.
    let text = raw;
    if (isClamped) {
      text = truncateSfxText(text, sfxSized);
    }
    // FIX #2: normalize to the ASCII subset our display fonts cover, then
    // UPPERCASE (English manga dialogue is conventionally all-caps).
    text = normalizeForDisplay(text).trim().toUpperCase();
    if (!text) return;

    // Center inside the text-region bbox (union of textRegions if available).
    // Pass siblings + image bounds so the region (incl. high-aspect bubble
    // widening) is identical to the one used to paint the background.
    const region = this.computeTextRegionBBox(textBox, allBoxes, canvasW, canvasH);

    // FIX #1: choose Anton (long/dialogue) vs Bangers (short SFX) per the
    // backend pick_font, evaluated on the uppercased display text. We still
    // honour an explicit non-default settings.defaultFont if one was supplied.
    const usePicked = !fontFamily || fontFamily.trim().length === 0;
    const family = usePicked ? this.pickFontFamily(text) : fontFamily;

    // Available area after padding.
    const availWidth = Math.max(1, region.width - TEXT_PADDING_PX * 2);
    const availHeight = Math.max(1, region.height - TEXT_PADDING_PX * 2);

    // Binary-search for the largest font size that fits (soft floor).
    let fit = this.findBestFit(ctx, text, availWidth, availHeight, family, FONT_SIZE_MIN);

    // FIX #3: clamped (no-bubble) blocks still overflowing at the soft floor
    // retry once down to the hard floor before accepting overflow.
    if (isClamped && (fit.maxLineWidth > availWidth || fit.totalHeight > availHeight)) {
      fit = this.findBestFit(ctx, text, availWidth, availHeight, family, FONT_SIZE_HARD_FLOOR);
    }

    // Predict the rendered ink rect (centered, clamped to the region) for
    // collision tests, mirroring compose_final's rendered_rect.
    const renderedW = Math.min(fit.maxLineWidth, region.width);
    const renderedH = Math.min(fit.totalHeight, region.height);
    const cx = region.x + region.width / 2;
    const cy = region.y + region.height / 2;
    const rx0 = Math.round(cx - renderedW / 2);
    const ry0 = Math.round(cy - renderedH / 2);
    const renderedRect: PlacedRect = {
      x0: rx0,
      y0: ry0,
      x1: rx0 + Math.round(renderedW),
      y1: ry0 + Math.round(renderedH),
      isDialogue,
    };

    // FIX A.2 / collision avoidance (compose_final).
    if (isClamped) {
      // Orphan/SFX over DIALOGUE -> always suppress; SFX-sized over ANY placed
      // block -> suppress; a larger caption is kept (drawn shrunk/clipped).
      for (const pr of placedRects) {
        if (!rectsOverlap(renderedRect, pr)) continue;
        if (pr.isDialogue) return; // never cover dialogue
        if (sfxSized) return; // stray SFX over another caption
      }
    } else {
      // Dialogue: skip only when clearly (>=60% of own area) buried under an
      // already-placed rect (detection duplicates / heavy overlap).
      const rrArea = Math.max(
        1,
        (renderedRect.x1 - renderedRect.x0) * (renderedRect.y1 - renderedRect.y0)
      );
      for (const pr of placedRects) {
        const ix0 = Math.max(renderedRect.x0, pr.x0);
        const iy0 = Math.max(renderedRect.y0, pr.y0);
        const ix1 = Math.min(renderedRect.x1, pr.x1);
        const iy1 = Math.min(renderedRect.y1, pr.y1);
        if (ix1 <= ix0 || iy1 <= iy0) continue;
        const inter = (ix1 - ix0) * (iy1 - iy0);
        if (inter / rrArea >= 0.6) return; // clearly buried
      }
    }

    // Auto-contrast. Sample the rendered background luminance, then pick colors.
    const { fontColor, strokeColor } = this.resolveColors(
      ctx,
      textBox,
      region,
      lumaSnapshot
    );

    this.drawWrappedText(
      ctx,
      fit.lines,
      region.x,
      region.y,
      region.width,
      region.height,
      family,
      fit.fontSize,
      fit.lineHeight,
      fontColor,
      strokeColor
    );

    // Record the rendered ink rect so later blocks avoid burying this one.
    placedRects.push(renderedRect);
  }

  /**
   * Binary search for the largest font size within [FONT_SIZE_MIN, FONT_SIZE_MAX]
   * whose wrapped lines fit inside (availWidth x availHeight) WITHOUT breaking
   * inside a word.
   *
   * Mirrors koharu run_auto (low=min, high=max, widen while it fits) but adds a
   * hard "no mid-word break" constraint, which is the dominant readability bug
   * in tall-narrow bubbles ("MOMMY" -> "MOM"/"MY"). Because a word only breaks
   * when it is wider than availWidth, and word width shrinks monotonically with
   * font size, "the longest word fits the width" is itself monotonic in size —
   * so we can binary-search it. Priority of the accepted layout:
   *
   *   1. Largest size that fits H + W AND does not break a word  (preferred)
   *   2. else: largest size that fits H + W (may break a word — only happens
   *      when a single word cannot fit the width even near FONT_SIZE_MIN)
   *   3. else: `minSize`, accepting overflow (koharu's tiny-box behavior)
   *
   * `minSize` defaults to the soft floor FONT_SIZE_MIN (14, matching the backend
   * min_floor); clamped/SFX blocks retry with FONT_SIZE_HARD_FLOOR (9) on
   * overflow, mirroring compose_final's two-stage fit.
   */
  private findBestFit(
    ctx: CanvasRenderingContext2D,
    text: string,
    availWidth: number,
    availHeight: number,
    fontFamily: string,
    minSize: number = FONT_SIZE_MIN
  ): FitResult {
    let low = minSize;
    let high = FONT_SIZE_MAX;
    // best = largest size fitting H+W with NO mid-word break (priority 1).
    let best: FitResult | null = null;
    // fallback = largest size fitting H+W even if it broke a word (priority 2).
    let fallback: FitResult | null = null;

    while (low <= high) {
      const mid = (low + high) >> 1;
      const attempt = this.layoutAtSize(ctx, text, availWidth, mid, fontFamily);
      const fitsH = attempt.totalHeight <= availHeight;
      const fitsW = attempt.maxLineWidth <= availWidth;

      if (fitsH && fitsW && !attempt.brokeWord) {
        // Clean fit — try to grow.
        best = attempt;
        if (!fallback || attempt.fontSize > fallback.fontSize) fallback = attempt;
        low = mid + 1;
      } else if (fitsH && fitsW && attempt.brokeWord) {
        // Fits the box but only by char-breaking the longest word. The
        // longest word is too wide at this size, so a CLEAN fit can only exist
        // at a SMALLER size — search down. Remember it as a fallback.
        if (!fallback || attempt.fontSize > fallback.fontSize) fallback = attempt;
        high = mid - 1;
      } else {
        // Overflows the box — shrink.
        high = mid - 1;
      }
    }

    // Priority 1: largest clean (no mid-word break) fit.
    if (best) return best;
    // Priority 2: largest box-fitting layout even if it broke a word.
    if (fallback) return fallback;
    // Priority 3: nothing fit — minSize floor, accept overflow.
    return this.layoutAtSize(ctx, text, availWidth, minSize, fontFamily);
  }

  /**
   * Wrap `text` at `fontSize` and measure block width/height.
   *
   * FIX #4: line height mirrors the backend line_height_px:
   *   line_height = max(ascent + descent + leading, font_size),  leading = 0.10*em
   * using FONT-level metrics (fontBoundingBoxAscent/Descent, the canvas analogue
   * of PIL's font.getmetrics()) — NOT the per-line ink bbox — so the height is
   * constant per size and matches the backend. The max(...,font_size) guard
   * keeps short-glyph lines from collapsing tighter than the backend.
   */
  private layoutAtSize(
    ctx: CanvasRenderingContext2D,
    text: string,
    maxWidth: number,
    fontSize: number,
    fontFamily: string
  ): FitResult {
    // OPT 4 (safe variant): memoize identical layouts. The binary search probes
    // and the soft-floor/hard-floor retry frequently re-request the same
    // (size, family, width, text); returning the cached FitResult is identical
    // to recomputing it (only the transient ctx.font set is skipped, which
    // paint paths re-apply). No effect on the chosen font size or pixels.
    const key = `${fontSize}|${fontFamily}|${maxWidth}|${text}`;
    const cached = this.layoutCache.get(key);
    if (cached) return cached;
    const result = this.layoutAtSizeUncached(ctx, text, maxWidth, fontSize, fontFamily);
    this.layoutCache.set(key, result);
    return result;
  }

  private layoutAtSizeUncached(
    ctx: CanvasRenderingContext2D,
    text: string,
    maxWidth: number,
    fontSize: number,
    fontFamily: string
  ): FitResult {
    ctx.font = this.buildFontString(fontSize, fontFamily, 'bold');
    const { lines, brokeWord } = this.wrapTextAtFont(ctx, text, maxWidth);

    // Font-level metrics: prefer fontBoundingBox* (font ascent/descent). Fall
    // back to the 0.8/0.2 em split when a browser doesn't expose them.
    const probe = ctx.measureText('AaGg');
    const fontAscent =
      (probe as any).fontBoundingBoxAscent ?? fontSize * 0.8;
    const fontDescent =
      (probe as any).fontBoundingBoxDescent ?? fontSize * 0.2;
    const leading = Math.max(0, Math.round(fontSize * LINE_LEADING_FACTOR));
    const lineHeight = Math.max(
      fontAscent + fontDescent + leading,
      Math.round(fontSize)
    );

    let maxLineWidth = 0;
    for (const ln of lines) {
      const w = ctx.measureText(ln).width;
      if (w > maxLineWidth) maxLineWidth = w;
    }

    const totalHeight = lineHeight * lines.length;

    return {
      fontSize,
      lines: lines.length > 0 ? lines : [text],
      lineHeight,
      totalHeight: totalHeight || lineHeight,
      maxLineWidth,
      brokeWord,
    };
  }

  /**
   * Greedy word-wrap that mirrors the backend wrap_greedy (refit_final_composites).
   *
   * FIX #5: a word wider than the box is only HARD-BROKEN by character when it
   * is the start of a fresh line AND len(word) >= LONG_WORD_MIN_LEN (13). A
   * shorter word that overflows ("MOMMY") is left to overflow on one line — a
   * slight overhang reads far better than "MO/MM/Y". This matches the backend's
   * `len(w) >= 13` guard exactly (the prior code char-broke ANY over-wide word).
   *
   * Returns `{ lines, brokeWord }` where `brokeWord` is true iff a word had to
   * be split mid-word; the font-fit search uses it to prefer a smaller clean
   * size over a mid-word break (see findBestFit).
   */
  private wrapTextAtFont(
    ctx: CanvasRenderingContext2D,
    text: string,
    maxWidth: number
  ): { lines: string[]; brokeWord: boolean } {
    const words = text.split(/\s+/).filter(Boolean);
    const lines: string[] = [];
    let cur = '';
    let brokeWord = false;

    const measure = (s: string) => ctx.measureText(s).width;

    for (const w of words) {
      const trial = (cur ? `${cur} ${w}` : w).trim();
      if (measure(trial) <= maxWidth || !cur) {
        // A word wider than the box: only hard-break LONG words (>=13 chars)
        // when starting a fresh line; shorter words overflow on one line.
        if (measure(w) > maxWidth && !cur && w.length >= LONG_WORD_MIN_LEN) {
          brokeWord = true;
          let frag = '';
          for (const ch of w) {
            if (measure(frag + ch) > maxWidth && frag) {
              lines.push(frag);
              frag = ch;
            } else {
              frag += ch;
            }
          }
          cur = frag;
        } else {
          cur = trial;
        }
      } else {
        lines.push(cur);
        cur = w;
      }
    }
    if (cur) lines.push(cur);
    return {
      lines: lines.length > 0 ? lines : [text],
      brokeWord,
    };
  }

  /**
   * Build a canvas font string with a safe fallback chain.
   */
  private buildFontString(size: number, family: string, weight = 'bold'): string {
    const safeFamily = family && family.trim().length > 0 ? family : PRIMARY_FONT_FAMILY;
    // Fallback chain mirrors the backend FONT_STACK ordering: Comic Neue (primary)
    // then Bangers/Anton comic faces, then a wide-coverage system fallback for any
    // leaked glyph.
    return `${weight} ${size}px "${safeFamily}", "Comic Neue", "Bangers", "Anton", "Noto Sans", "Arial", sans-serif`;
  }

  /**
   * Choose the display font family for a block, mirroring the backend
   * refit_final_composites.pick_font: short exclamatory or short all-caps text
   * (SFX-like outbursts) -> Bangers (comic punch); everything else -> Comic Neue.
   *
   * NOTE: the backend evaluates this on the UPPERCASED display text (compose_final
   * uppercases before pick_font is called via _pick_renderable_font(pick_font(text))),
   * so we pass already-uppercased text here too.
   */
  private pickFontFamily(displayText: string): string {
    const cleaned = displayText.trim();
    if (!cleaned) return PRIMARY_FONT_FAMILY;
    const exclam =
      (cleaned.match(/!/g)?.length ?? 0) + (cleaned.match(/\?/g)?.length ?? 0);
    const short = cleaned.length <= 8;
    // After uppercasing, isupper() is true iff there is at least one letter and
    // no lowercase letters. Mirror that: has a letter and equals its upper-case.
    const hasAlpha = /[a-z]/i.test(cleaned);
    const allcaps = hasAlpha && cleaned === cleaned.toUpperCase();
    if ((short && exclam >= 1) || (allcaps && cleaned.length <= 16)) {
      return SFX_FONT_FAMILY;
    }
    return PRIMARY_FONT_FAMILY;
  }

  /**
   * Pick text/stroke colors. If API-supplied colors have acceptable contrast
   * (>= MIN_CONTRAST_RATIO), keep them. Otherwise override via luminance sampling.
   */
  private resolveColors(
    ctx: CanvasRenderingContext2D,
    textBox: TextBox,
    region: RegionBBox,
    lumaSnapshot?: LumaSnapshot | null
  ): { fontColor: string; strokeColor: string } {
    const apiFont = textBox.fontColor || '';
    const apiStroke = textBox.fontStrokeColor || '';

    // Use API-supplied colors only when both parse AND have a sane contrast.
    const parsedFont = parseColor(apiFont);
    const parsedStroke = parseColor(apiStroke);
    if (parsedFont && parsedStroke) {
      const ratio = contrastRatio(parsedFont, parsedStroke);
      if (ratio >= MIN_CONTRAST_RATIO) {
        return { fontColor: apiFont, strokeColor: apiStroke };
      }
    }

    // FIX #6: auto-contrast over the FULL region rect using BT.601 luma,
    // MEDIAN + dark_fraction (share of pixels with luma<96). Go WHITE text when
    // the median is dark (<140) OR a meaningful share is dark (>0.35) — mirrors
    // the backend sample_bg_luminance + decision in compose_final. The prior
    // 8-point MEAN+>128 flipped to black text on mostly-dark art with a few
    // bright specks.
    const { median, darkFraction } = this.sampleRegionLuminance(
      ctx,
      region,
      lumaSnapshot
    );
    if (median < 140 || darkFraction > 0.35) {
      // Dark background → white text on black stroke.
      return { fontColor: '#FFFFFF', strokeColor: '#000000' };
    }
    // Bright background → dark text on light stroke.
    return { fontColor: '#000000', strokeColor: '#FFFFFF' };
  }

  /**
   * OPT 2: capture ONE full-canvas pixel buffer up front so per-box luminance
   * sampling can slice into it instead of doing a getImageData GPU readback per
   * box. Returns null on a tainted (cross-origin) canvas; callers then fall
   * back to per-box getImageData, which throws and yields the same light-bg
   * default — identical behavior, just without the up-front read.
   */
  private captureLumaSnapshot(
    ctx: CanvasRenderingContext2D
  ): LumaSnapshot | null {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    if (width <= 0 || height <= 0) return null;
    try {
      const data = ctx.getImageData(0, 0, width, height).data;
      return { data, width, height };
    } catch {
      return null;
    }
  }

  /**
   * Sample the FULL region rect from the (already-composited) canvas and return
   * `{ median, darkFraction }` of BT.601 luminance in [0, 255], mirroring the
   * backend sample_bg_luminance. To bound cost on large rects, we subsample on
   * a fixed grid stride rather than reading every pixel.
   *
   * When a `lumaSnapshot` (OPT 2) is supplied we read pixel values from that
   * single pre-captured full-canvas buffer; the region bounds, stride, sample
   * grid and luma formula are IDENTICAL to the per-box getImageData path, so
   * the returned (median, darkFraction) are byte-for-byte the same — only the
   * source of the bytes differs (one shared readback vs. N readbacks).
   */
  private sampleRegionLuminance(
    ctx: CanvasRenderingContext2D,
    region: RegionBBox,
    lumaSnapshot?: LumaSnapshot | null
  ): { median: number; darkFraction: number } {
    const cw = ctx.canvas.width;
    const ch = ctx.canvas.height;
    const x0 = Math.max(0, Math.floor(region.x));
    const y0 = Math.max(0, Math.floor(region.y));
    const x1 = Math.min(cw, Math.ceil(region.x + region.width));
    const y1 = Math.min(ch, Math.ceil(region.y + region.height));
    if (x1 <= x0 || y1 <= y0) return { median: 255, darkFraction: 0 };

    const w = x1 - x0;
    const h = y1 - y0;

    // OPT 2: prefer the shared full-canvas snapshot. `snapStride` is the row
    // width to index into: for the snapshot it's the full canvas width and the
    // region pixel (xx,yy) lives at ((y0+yy)*cw + (x0+xx)); for the fallback
    // per-region buffer it's the region width w and the pixel is at (yy*w+xx).
    let data: Uint8ClampedArray;
    let snapStride: number;
    let baseX: number;
    let baseY: number;
    if (lumaSnapshot && lumaSnapshot.width === cw && lumaSnapshot.height === ch) {
      data = lumaSnapshot.data;
      snapStride = cw;
      baseX = x0;
      baseY = y0;
    } else {
      try {
        data = ctx.getImageData(x0, y0, w, h).data;
      } catch {
        // Tainted canvas (cross-origin) — assume light background → dark text.
        return { median: 255, darkFraction: 0 };
      }
      snapStride = w;
      baseX = 0;
      baseY = 0;
    }

    // Cap samples (~4096) by striding so the median sort stays cheap.
    const totalPixels = w * h;
    const maxSamples = 4096;
    const stride = Math.max(1, Math.floor(Math.sqrt(totalPixels / maxSamples)));
    const lums: number[] = [];
    let dark = 0;
    for (let yy = 0; yy < h; yy += stride) {
      for (let xx = 0; xx < w; xx += stride) {
        const idx = ((baseY + yy) * snapStride + (baseX + xx)) * 4;
        const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
        lums.push(lum);
        if (lum < 96) dark += 1;
      }
    }
    if (lums.length === 0) return { median: 255, darkFraction: 0 };
    lums.sort((a, b) => a - b);
    const mid = lums.length >> 1;
    const median =
      lums.length % 2 === 1 ? lums[mid] : (lums[mid - 1] + lums[mid]) / 2;
    return { median, darkFraction: dark / lums.length };
  }

  /**
   * Draw debug overlay showing detection boxes and text regions
   */
  private drawDebugOverlay(ctx: CanvasRenderingContext2D, textBoxes: TextBox[]): void {
    for (const textBox of textBoxes) {
      const x = textBox.minX;
      const y = textBox.minY;
      const width = textBox.maxX - textBox.minX;
      const height = textBox.maxY - textBox.minY;

      ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      if (textBox.textRegions && textBox.textRegions.length > 0) {
        ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.lineWidth = 1;

        for (const region of textBox.textRegions) {
          const rw = region.maxX - region.minX;
          const rh = region.maxY - region.minY;
          ctx.fillRect(region.minX, region.minY, rw, rh);
          ctx.strokeRect(region.minX, region.minY, rw, rh);
        }
      }

      const label = `z:${textBox.zIndex || 1}${textBox.confidence ? ` c:${(textBox.confidence * 100).toFixed(0)}%` : ''}`;
      ctx.font = '10px monospace';
      ctx.fillStyle = 'rgba(255, 0, 0, 0.9)';
      ctx.fillText(label, x + 2, y - 3);
    }
  }

  /**
   * Draw a rounded rectangle
   */
  private drawRoundedRect(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    width: number,
    height: number,
    fillColor: string,
    radius: number
  ): void {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();
  }

  /**
   * Draw wrapped text centered in a bounding box.
   * Uses an explicit lineHeight from the fit result so lines don't walk.
   */
  private drawWrappedText(
    ctx: CanvasRenderingContext2D,
    lines: string[],
    boxX: number,
    boxY: number,
    boxWidth: number,
    boxHeight: number,
    fontFamily: string,
    fontSize: number,
    lineHeight: number,
    fontColor: string,
    strokeColor: string
  ): void {
    ctx.font = this.buildFontString(fontSize, fontFamily, 'bold');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const totalTextHeight = lines.length * lineHeight;

    // Center vertically in the region (union of textRegions if available).
    let startY = boxY + (boxHeight - totalTextHeight) / 2 + lineHeight / 2;

    if (startY < boxY + lineHeight / 2) {
      startY = boxY + lineHeight / 2;
    }

    const centerX = boxX + boxWidth / 2;
    // FIX #7: stroke width mirrors the backend stroke_w = max(3, min(8,
    // round(font.size * 0.14))). PIL's stroke_width is the full OUTER thickness,
    // but canvas strokeText centres the stroke on the glyph path (half inside),
    // so we double it to land the same visible outer thickness as the backend.
    const backendStroke = Math.max(3, Math.min(8, Math.round(fontSize * 0.14)));
    const strokeWidth = backendStroke * 2;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const yPos = startY + i * lineHeight;

      // Don't draw if we've walked past the box.
      if (yPos > boxY + boxHeight - lineHeight / 2 + 1) {
        break;
      }

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = strokeWidth;
      ctx.lineJoin = 'round';
      ctx.strokeText(line, centerX, yPos);

      ctx.fillStyle = fontColor;
      ctx.fillText(line, centerX, yPos);
    }
  }

  /**
   * Load an image from base64 or URL
   */
  private loadImage(src: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();

      img.onload = () => resolve(img);
      img.onerror = () => {
        logger.error('Failed to load image:', src.substring(0, 100));
        reject(new Error('Failed to load image'));
      };

      if (!src.startsWith('data:')) {
        img.crossOrigin = 'anonymous';
      }

      img.src = src;
    });
  }

  /**
   * Apply the rendered translation to the page.
   *
   * PROGRESSIVE ENHANCEMENT: the live page <img> is NEVER mutated. The
   * translation is shown by the on-top translation layer built in
   * buildDomOverlay (a child <img> inside the absolutely-positioned overlay),
   * which sits above the page image. This guarantees the original image keeps
   * loading/displaying immediately and is never blanked while we translate —
   * and avoids the CORS/SPA-revert hazards of writing `img.src` directly.
   *
   * For <canvas> we still draw onto the live canvas (its own pixels are the
   * only way to back the on-top layer for tainted/contextless canvases, and
   * restoreOriginal can repaint it on toggle-off).
   */
  private async replaceElement(
    originalElement: HTMLImageElement | HTMLCanvasElement,
    canvas: HTMLCanvasElement
  ): Promise<void> {
    if (originalElement instanceof HTMLCanvasElement) {
      const ctx = originalElement.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, originalElement.width, originalElement.height);
        ctx.drawImage(canvas, 0, 0);
      }
    }
    // <img>: intentionally no `src` mutation — the on-top translation layer
    // (buildDomOverlay) renders the translated frame above the untouched image.
  }

  /**
   * Build a per-image DOM overlay hosting retry/edit affordances for each
   * text box. The overlay is absolutely positioned on top of the image and
   * kept in sync with its client rect on resize/scroll.
   *
   * Reference: koharu's TextBlocksPanel.tsx:228,263 — per-box edit + retry.
   */
  private buildDomOverlay(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    canvas: HTMLCanvasElement,
    textBoxes: TextBox[]
  ): HTMLDivElement | undefined {
    if (typeof document === 'undefined') return undefined;

    const overlay = document.createElement('div');
    overlay.className = CONFIG.CSS_CLASSES.OVERLAY_CONTAINER + ' manga-translator-box-layer';
    overlay.dataset.mangaTranslatorOverlay = '1';

    // Position the overlay over the image. We use position:absolute and
    // match the image's offset within its offset parent.
    overlay.style.position = 'absolute';
    overlay.style.pointerEvents = 'none';
    overlay.style.zIndex = '999999';
    overlay.style.left = '0';
    overlay.style.top = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';

    const host = (imageElement.parentElement || document.body) as HTMLElement;

    // Ensure host is a positioning context — but don't stomp an explicit
    // `position` already set by the site.
    const hostPos = window.getComputedStyle(host).position;
    if (hostPos === 'static') {
      host.style.position = 'relative';
    }

    // Position overlay to exactly cover the image within the host.
    overlay.style.left = `${imageElement.offsetLeft}px`;
    overlay.style.top = `${imageElement.offsetTop}px`;
    overlay.style.width = `${imageElement.clientWidth || canvas.width}px`;
    overlay.style.height = `${imageElement.clientHeight || canvas.height}px`;

    // Figure out the scale from canvas (natural image) coordinates to
    // rendered pixels.
    const scaleX =
      (imageElement.clientWidth || canvas.width) / Math.max(1, canvas.width);
    const scaleY =
      (imageElement.clientHeight || canvas.height) / Math.max(1, canvas.height);

    // FRAMEWORK-PROOF TRANSLATION LAYER.
    //
    // Manga readers are usually SPAs (React/Vue/etc.) that re-render and REVERT
    // any `img.src` we set back to their own page URL, wiping the translated
    // image. To survive that, we paint the rendered translation as a visible
    // layer INSIDE this overlay (which the framework doesn't touch) instead of
    // depending on `img.src`. It is the FIRST child of the overlay so it sits
    // BELOW the per-box boxDivs / flag button (later children, painted on top)
    // but ABOVE the page image (the overlay itself has a high z-index). It is
    // sized to exactly cover the image and is pointer-events:none so it never
    // intercepts clicks meant for the boxDivs/flag button.
    //
    // The PIXELS are byte-identical to the canvas render (parity-matched to the
    // backend) — we only change WHERE the canvas is shown (overlay child) vs
    // mutating `img.src`. We render to the same data URL the <img> branch of
    // replaceElement would have used.
    const translationLayer = document.createElement('img');
    translationLayer.className = 'manga-translator-translation-layer';
    translationLayer.dataset.mangaTranslatorTranslation = '1';
    translationLayer.src = canvas.toDataURL('image/jpeg', 0.9);
    translationLayer.style.position = 'absolute';
    translationLayer.style.left = '0';
    translationLayer.style.top = '0';
    translationLayer.style.width = '100%';
    translationLayer.style.height = '100%';
    // Match the canvas aspect exactly (the canvas already encodes the full
    // original-image frame, so 'fill' reproduces what `img.src = dataURL` did:
    // the rendered frame stretched to the image box). pointer-events:none keeps
    // clicks flowing through to the boxDivs / flag button above it.
    translationLayer.style.objectFit = 'fill';
    translationLayer.style.pointerEvents = 'none';
    translationLayer.style.zIndex = '0';
    translationLayer.style.userSelect = 'none';
    translationLayer.draggable = false;
    overlay.appendChild(translationLayer);

    textBoxes.forEach((box, boxIndex) => {
      const boxDiv = document.createElement('div');
      boxDiv.className = CONFIG.CSS_CLASSES.TEXT_BOX + ' manga-translator-box';
      boxDiv.dataset.boxIndex = String(boxIndex);
      boxDiv.style.position = 'absolute';
      boxDiv.style.left = `${box.minX * scaleX}px`;
      boxDiv.style.top = `${box.minY * scaleY}px`;
      boxDiv.style.width = `${(box.maxX - box.minX) * scaleX}px`;
      boxDiv.style.height = `${(box.maxY - box.minY) * scaleY}px`;
      boxDiv.style.pointerEvents = 'auto';
      boxDiv.style.cursor = 'text';
      // Stack above the translation layer (z-index:0) so the edit hit-area and
      // any edit UI sit on top of the rendered translation.
      boxDiv.style.zIndex = '1';

      // Double-click to edit inline. (The per-box visible retry button has been
      // removed; the dblclick-to-edit affordance is kept.)
      boxDiv.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        this.enterEditMode(boxDiv, boxIndex, box, imageElement, canvas);
      });

      overlay.appendChild(boxDiv);
    });

    // Per-image "flag bad translation" affordance. A tiny ⚑ button pinned to the
    // top-right corner of the overlay, shown on hover (see overlay.css). Clicking
    // it dispatches a `manga-translator:flag-image` event the content-script
    // handles (it has the apiClient + does the SW round-trip). The button lives
    // on the DOM overlay, NOT the canvas, so it never alters the rendered pixels
    // the backend-parity test checks.
    const flagBtn = document.createElement('button');
    flagBtn.type = 'button';
    flagBtn.className = 'manga-translator-flag-btn';
    flagBtn.title = 'Flag this translation as poor (saves the image for fine-tuning)';
    flagBtn.textContent = '⚑'; // ⚑
    // Fully inline-styled so it renders identically without depending on
    // overlay.css being injected into the host page. Subtle by default, more
    // opaque on hover. Pinned to the overlay's top-right corner.
    flagBtn.style.cssText = [
      'position:absolute',
      'top:4px',
      'right:4px',
      'width:22px',
      'height:22px',
      'padding:0',
      'margin:0',
      'border:none',
      'border-radius:4px',
      'background:rgba(17,24,39,0.55)',
      'color:#fff',
      'font-size:13px',
      'line-height:22px',
      'text-align:center',
      'cursor:pointer',
      'opacity:0.25',
      'transition:opacity 0.15s ease, background 0.15s ease',
      'pointer-events:auto',
      'z-index:1000001',
    ].join(';');
    flagBtn.addEventListener('mouseenter', () => {
      if (!flagBtn.classList.contains('flagged')) flagBtn.style.opacity = '1';
    });
    flagBtn.addEventListener('mouseleave', () => {
      if (!flagBtn.classList.contains('flagged')) flagBtn.style.opacity = '0.25';
    });
    flagBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      e.preventDefault();
      this.requestFlag(imageElement, flagBtn);
    });
    overlay.appendChild(flagBtn);

    host.appendChild(overlay);
    return overlay;
  }

  /**
   * Build the POST /flag payload for a rendered image from the TextBox[] that
   * were drawn for it, plus the captured ORIGINAL source bytes. Coordinates are
   * passed through in the backend's (original-image pixel) space. Returns null
   * if the image is no longer tracked.
   */
  private buildFlagPayload(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    targetLanguage: string
  ): FlagRequest | null {
    const rendered = this.renderedImages.get(imageElement);
    if (!rendered) return null;

    const boxes: FlagBox[] = rendered.textBoxes.map((b) => ({
      ocr_text: b.ocrText,
      translated_text: b.translatedText,
      minX: b.minX,
      minY: b.minY,
      maxX: b.maxX,
      maxY: b.maxY,
    }));

    return {
      image_base64: rendered.originalImageBase64,
      page_url: typeof location !== 'undefined' ? location.href : '',
      target_language: targetLanguage,
      boxes,
    };
  }

  /**
   * Fire the flag request for an image. Idempotent per image — once flagged (or
   * while in-flight) the button is disabled. Gathers the payload, dispatches a
   * `manga-translator:flag-image` event with it + a pair of callbacks so the
   * content-script can run the actual SW round-trip and report back; we then
   * flip the button to a ✓ (success) or restore it (failure).
   */
  private requestFlag(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    flagBtn: HTMLButtonElement
  ): void {
    const rendered = this.renderedImages.get(imageElement);
    if (!rendered || rendered.flagged) return;

    // Optimistic disable to avoid double-send while the request is in flight.
    flagBtn.disabled = true;
    flagBtn.classList.add('flagging');

    settingsManager
      .getSettings()
      .then((settings) => {
        const payload = this.buildFlagPayload(imageElement, settings.targetLanguage);
        if (!payload) {
          flagBtn.disabled = false;
          flagBtn.classList.remove('flagging');
          return;
        }
        document.dispatchEvent(
          new CustomEvent('manga-translator:flag-image', {
            detail: {
              payload,
              onSuccess: () => {
                const r = this.renderedImages.get(imageElement);
                if (r) r.flagged = true;
                flagBtn.classList.remove('flagging');
                flagBtn.classList.add('flagged');
                flagBtn.textContent = '✓'; // ✓
                flagBtn.title = 'Flagged — thanks!';
                // Lock in a confirmed (green, fully opaque) state.
                flagBtn.style.background = 'rgba(16,185,129,0.9)';
                flagBtn.style.opacity = '1';
              },
              onError: () => {
                flagBtn.disabled = false;
                flagBtn.classList.remove('flagging');
                flagBtn.style.opacity = '0.25';
                flagBtn.title = 'Flag failed — click to retry';
              },
            },
          })
        );
      })
      .catch(() => {
        flagBtn.disabled = false;
        flagBtn.classList.remove('flagging');
      });
  }

  /**
   * Turn a box into an editable textarea. On blur / Enter-without-shift, emit
   * a `manga-translator:edit-box` CustomEvent with the new text.
   */
  private enterEditMode(
    boxDiv: HTMLDivElement,
    boxIndex: number,
    box: TextBox,
    imageElement: HTMLImageElement | HTMLCanvasElement,
    canvas: HTMLCanvasElement
  ): void {
    // Prevent recursive entries.
    if (boxDiv.querySelector('textarea')) return;

    const ta = document.createElement('textarea');
    ta.className = 'manga-translator-edit-textarea';
    ta.value = box.translatedText ?? '';
    ta.style.width = '100%';
    ta.style.height = '100%';
    ta.style.boxSizing = 'border-box';
    ta.style.resize = 'none';
    ta.style.border = '2px dashed #4a90e2';
    ta.style.background = 'rgba(255,255,255,0.9)';
    ta.style.font = '14px sans-serif';
    ta.style.padding = '4px';

    const commit = () => {
      const newText = ta.value;
      box.translatedText = newText;
      const detail = {
        boxIndex,
        newText,
        originalText: box.ocrText,
        imageElement,
      };
      document.dispatchEvent(
        new CustomEvent('manga-translator:edit-box', { detail })
      );
      ta.remove();
      // Hint: trigger a re-render by dispatching a generic refresh event.
      // Implementers may choose to listen to edit-box and redraw.
      void canvas;
    };

    ta.addEventListener('blur', commit);
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        ta.blur();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        ta.remove();
      }
    });

    boxDiv.appendChild(ta);
    ta.focus();
    ta.select();
  }

  /**
   * Set overlay canvas + DOM overlay opacity (0 = show original, 1 = translated).
   * Used by the "hold Alt to peek" hotkey in content-script.
   */
  setOverlayOpacity(opacity: number): void {
    for (const rendered of this.renderedImages.values()) {
      if (rendered.newElement instanceof HTMLCanvasElement) {
        (rendered.newElement as HTMLCanvasElement).style.opacity = String(opacity);
      } else if (rendered.newElement instanceof HTMLImageElement) {
        rendered.newElement.style.opacity = String(opacity);
      }
      if (rendered.domOverlay) {
        rendered.domOverlay.style.opacity = String(opacity);
      }
    }
  }

  /**
   * Remove overlay (restore original if possible)
   */
  removeOverlay(imageElement: HTMLElement): void {
    // Abandon any in-flight progressive render for this element.
    this.streaming.delete(imageElement);
    const rendered = this.renderedImages.get(imageElement);
    if (rendered) {
      this.restoreOriginal(rendered);
      rendered.domOverlay?.remove();
      this.renderedImages.delete(imageElement);
    }
  }

  /**
   * Restore the original image underneath:
   *   - <img>: nothing to restore — replaceElement no longer mutates `src`,
   *     so the live image is already the original. We only clear any opacity
   *     we may have set. (Re-pointing `src` here would fight readers that have
   *     since swapped the image to a new page.)
   *   - <canvas>: redraw the saved original base64 snapshot back onto it.
   * Best-effort; failures are swallowed so toggling OFF never throws.
   */
  private restoreOriginal(rendered: RenderedImage): void {
    try {
      const el = rendered.originalElement;
      if (el instanceof HTMLImageElement) {
        el.style.opacity = '';
      } else if (el instanceof HTMLCanvasElement) {
        const src = rendered.originalImageBase64;
        el.style.opacity = '';
        if (!src) return;
        const img = new Image();
        img.onload = () => {
          try {
            const ctx = el.getContext('2d');
            if (!ctx) return;
            ctx.clearRect(0, 0, el.width, el.height);
            ctx.drawImage(img, 0, 0, el.width, el.height);
          } catch {
            /* canvas restore is best-effort */
          }
        };
        img.src = src;
      }
    } catch {
      /* never let a restore failure surface on the toggle path */
    }
  }

  /**
   * Show loading indicator
   */
  showLoading(imageElement: HTMLElement): HTMLDivElement {
    // PROGRESSIVE ENHANCEMENT: the badge must NEVER obscure the image content
    // while we translate. It is a small spinner-only chip pinned to the image's
    // TOP-RIGHT corner (not centered over it), with no opaque backdrop covering
    // the page. The original image stays fully visible the whole time.
    const rect = imageElement.getBoundingClientRect();
    const loading = document.createElement('div');
    loading.className = CONFIG.CSS_CLASSES.LOADING;
    // Clamp into the viewport so a corner just off-screen still shows the chip.
    const left = Math.max(4, Math.min(rect.right - 28, window.innerWidth - 28));
    const top = Math.max(4, rect.top + 8);
    loading.style.cssText = `
      position: fixed;
      left: ${left}px;
      top: ${top}px;
      background: rgba(0, 0, 0, 0.45);
      padding: 4px;
      border-radius: 50%;
      z-index: 1000000;
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: none;
    `;

    const spinner = document.createElement('div');
    spinner.className = 'manga-translator-loading-spinner';
    spinner.style.cssText = `
      display: inline-block;
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255, 255, 255, 0.35);
      border-radius: 50%;
      border-top-color: white;
      animation: spin 0.6s linear infinite;
    `;
    loading.appendChild(spinner);

    document.body.appendChild(loading);
    return loading;
  }

  /**
   * Show error message
   */
  showError(imageElement: HTMLElement, message: string): HTMLDivElement {
    const rect = imageElement.getBoundingClientRect();
    const error = document.createElement('div');
    error.className = CONFIG.CSS_CLASSES.ERROR;
    error.textContent = message;
    error.style.cssText = `
      position: fixed;
      left: ${rect.left + 10}px;
      top: ${rect.top + 10}px;
      background: rgba(220, 38, 38, 0.9);
      color: white;
      padding: 8px 16px;
      border-radius: 4px;
      font-family: system-ui;
      font-size: 12px;
      z-index: 1000000;
      cursor: pointer;
    `;

    error.addEventListener('click', () => error.remove());
    document.body.appendChild(error);

    setTimeout(() => error.remove(), 5000);

    return error;
  }

  /**
   * Clear all overlays and restore the original images underneath. Called when
   * translation is toggled OFF (master switch / per-host disable).
   */
  clearAll(): void {
    this.streaming.clear();
    for (const rendered of this.renderedImages.values()) {
      this.restoreOriginal(rendered);
      rendered.domOverlay?.remove();
    }
    this.renderedImages.clear();
  }

  /**
   * Reset renderer
   */
  reset(): void {
    this.clearAll();
  }

  /**
   * Update overlay position. Repositions DOM overlay to match the image rect.
   */
  updateOverlayPosition(imageElement: HTMLElement): void {
    const rendered = this.renderedImages.get(imageElement);
    if (!rendered || !rendered.domOverlay) return;

    const img = rendered.originalElement;
    rendered.domOverlay.style.left = `${(img as HTMLElement).offsetLeft}px`;
    rendered.domOverlay.style.top = `${(img as HTMLElement).offsetTop}px`;
    rendered.domOverlay.style.width = `${
      (img as HTMLElement).clientWidth || rendered.newElement.clientWidth || 0
    }px`;
    rendered.domOverlay.style.height = `${
      (img as HTMLElement).clientHeight || rendered.newElement.clientHeight || 0
    }px`;
  }
}

/* -------------------------- utility helpers -------------------------- */

/**
 * Character substitutions mirroring the backend refit_final_composites
 * _DISPLAY_REPLACE map. The Latin display fonts (Anton/Bangers) only cover
 * ASCII + a sliver of Latin-1, so non-ASCII punctuation is remapped to an
 * ASCII equivalent to avoid tofu squares. Keep in sync with the backend.
 */
const DISPLAY_REPLACE: Record<string, string> = {
  '…': '...', // …
  '⋯': '...', // ⋯
  '—': '-', // em-dash
  '–': '-', // en-dash
  '−': '-', // minus
  'ー': '-', // ー prolonged sound mark
  '‘': "'",
  '’': "'",
  '“': '"',
  '”': '"',
  '«': '"',
  '»': '"',
  '「': '"',
  '」': '"',
  '『': '"',
  '』': '"',
  '（': '(',
  '）': ')',
  '。': '.',
  '、': ',',
  '．': '.',
  '，': ',',
  '？': '?',
  '！': '!',
  '：': ':',
  '；': ';',
  '・': '.',
  '·': '.',
  '〜': '~',
  '～': '~',
  '　': ' ', // ideographic space
  '​': '', // zero-width space
  '‌': '',
  '‍': '',
  '﻿': '',
};

/** Axis-aligned overlap test for two ink rects (mirrors _rects_overlap). */
function rectsOverlap(a: PlacedRect, b: PlacedRect, pad = 0): boolean {
  return (
    a.x0 < b.x1 + pad &&
    b.x0 < a.x1 + pad &&
    a.y0 < b.y1 + pad &&
    b.y0 < a.y1 + pad
  );
}

const SFX_MAX_CHARS = 16;

/**
 * Shorten a verbose SFX gloss to onomatopoeia length for a small clamped box,
 * mirroring the backend _truncate_sfx_text. Only applies when the box is
 * SFX-sized; already-short strings pass through unchanged.
 */
function truncateSfxText(text: string, sfxSized: boolean): string {
  if (!text) return text;
  const s = text.trim();
  if (s.length <= SFX_MAX_CHARS || !sfxSized) return s;
  // Prefer a clean word boundary within the budget.
  const words = s.replace(/,/g, ' ').split(/\s+/).filter(Boolean);
  let out = '';
  for (const w of words) {
    const trial = (out ? `${out} ${w}` : w).trim();
    if (trial.length > SFX_MAX_CHARS) break;
    out = trial;
  }
  if (!out) out = s.slice(0, SFX_MAX_CHARS); // single very long word
  return out;
}

/**
 * Replicate the backend normalize_for_display: substitute font-incompatible
 * characters with ASCII equivalents (fullwidth ASCII letters/digits FF01..FF5E
 * map to their U+0021.. counterparts), then the caller uppercases. Idempotent.
 */
function normalizeForDisplay(text: string): string {
  if (!text) return text;
  let out = '';
  for (const ch of text) {
    const direct = DISPLAY_REPLACE[ch];
    if (direct !== undefined) {
      out += direct;
      continue;
    }
    const code = ch.codePointAt(0)!;
    // Fullwidth ASCII range FF01..FF5E -> normal ASCII 0x21..0x7E.
    if (code >= 0xff01 && code <= 0xff5e) {
      out += String.fromCharCode(code - 0xff01 + 0x21);
    } else {
      out += ch;
    }
  }
  return out;
}

interface RGB {
  r: number;
  g: number;
  b: number;
}

/**
 * Parse '#RRGGBB', '#RGB', or 'rgb(r,g,b)' into RGB. Returns null on failure.
 */
function parseColor(input: string): RGB | null {
  if (!input) return null;
  const s = input.trim().toLowerCase();
  if (s.startsWith('#')) {
    let hex = s.slice(1);
    if (hex.length === 3) {
      hex = hex
        .split('')
        .map((c) => c + c)
        .join('');
    }
    if (hex.length !== 6) return null;
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    if ([r, g, b].some((v) => Number.isNaN(v))) return null;
    return { r, g, b };
  }
  const m = s.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
  if (m) {
    return { r: +m[1], g: +m[2], b: +m[3] };
  }
  return null;
}

/** Relative luminance per WCAG. */
function relLum({ r, g, b }: RGB): number {
  const toLin = (c: number) => {
    const v = c / 255;
    return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * toLin(r) + 0.7152 * toLin(g) + 0.0722 * toLin(b);
}

function contrastRatio(a: RGB, b: RGB): number {
  const la = relLum(a);
  const lb = relLum(b);
  const [hi, lo] = la > lb ? [la, lb] : [lb, la];
  return (hi + 0.05) / (lo + 0.05);
}
