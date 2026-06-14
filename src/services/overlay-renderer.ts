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
import { TextBox } from '@/types/api';
import { CONFIG } from '@/config/constants';
import { settingsManager } from './settings-manager';

interface RenderedImage {
  originalElement: HTMLImageElement | HTMLCanvasElement;
  newElement: HTMLImageElement | HTMLCanvasElement;
  /** Optional DOM overlay sibling that hosts per-box retry/edit affordances. */
  domOverlay?: HTMLDivElement;
  textBoxes: TextBox[];
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
   * 'expanded' = the null-bubble fallback grew a tight source outward to give
   * long EN text room.
   */
  source?: 'bubble' | 'bubble-widened' | 'regions' | 'bbox' | 'expanded';
}

// Minimum and maximum font sizes probed by the binary search.
const FONT_SIZE_MIN = 8;
const FONT_SIZE_MAX = 72;
// Extra padding inside the text region (reserved for stroke + breathing room).
const TEXT_PADDING_PX = 8;
// Line spacing multiplier applied on top of measured ascent+descent.
const LINE_GAP_FACTOR = 1.1;
// Contrast ratio threshold below which we override API-supplied colors.
const MIN_CONTRAST_RATIO = 3.0;

// Comic font families we try to guarantee are loaded before the first
// measure/paint pass. Order = preference (local bundled first, then CDN).
const PRIMARY_FONT_FAMILY = 'Bangers';
const SECONDARY_FONT_FAMILY = 'Fredoka';
// Font sizes (px) we eagerly prime via document.fonts.load so the binary
// search measures with real glyph metrics instead of the Arial fallback.
const FONT_PRIME_SIZES = [12, 24, 48, 72];

export class OverlayRenderer {
  private renderedImages: Map<HTMLElement, RenderedImage> = new Map();
  private fontsInjected = false;
  /** Memoized promise that resolves once comic fonts are usable on canvas. */
  private fontsReadyPromise: Promise<void> | null = null;

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
   *   2. CDN <link> (Google Fonts) as a robust fallback. Bangers + Fredoka are
   *      OFL-licensed and safe to link.
   *
   * NOTE: local TTFs are not yet shipped in public/fonts/ (only a README),
   * so the local path is best-effort and silently falls back to the CDN.
   * FOLLOW-UP: bundle Bangers-Regular.ttf (OFL) into public/fonts/ to make the
   * local path the default and drop the CDN dependency entirely.
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
        'https://fonts.googleapis.com/css2?family=Bangers&family=Fredoka:wght@400;600&display=swap';
      (document.head || document.documentElement).appendChild(link);
      this.fontsInjected = true;
    } catch (err) {
      // Non-fatal: canvas drawing will fall back to Arial/sans-serif.
      console.warn('Manga Translator: font injection failed', err);
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
      for (const family of [PRIMARY_FONT_FAMILY, SECONDARY_FONT_FAMILY]) {
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
      const localFonts: Array<{ family: string; path: string }> = [
        { family: PRIMARY_FONT_FAMILY, path: 'fonts/Bangers-Regular.ttf' },
      ];

      for (const { family, path } of localFonts) {
        try {
          const url = getURL(path);
          // HEAD-check so a missing asset doesn't spam FontFace errors.
          const probe = await fetch(url, { method: 'HEAD' });
          if (!probe.ok) continue;
          const face = new FontFace(family, `url(${url}) format('truetype')`, {
            weight: 'normal',
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
   */
  async createOverlay(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    textBoxes: TextBox[],
    showDebug: boolean = false,
    inpaintedBase64?: string | null
  ): Promise<void> {
    // Remove existing overlay if any
    this.removeOverlay(imageElement);

    try {
      // Get settings
      const settings = await settingsManager.getSettings();

      // Get base64 image data (original page image)
      const base64Image = await this.getImageBase64(imageElement);

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

      // Replace original element with rendered canvas/image
      await this.replaceElement(imageElement, canvas);

      // Build DOM overlay with retry/edit affordances.
      const domOverlay = this.buildDomOverlay(imageElement, canvas, textBoxes);

      this.renderedImages.set(imageElement, {
        originalElement: imageElement,
        newElement: canvas,
        domOverlay,
        textBoxes,
      });
    } catch (error) {
      console.error('Failed to create overlay:', error);
      throw error;
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
      console.warn('CORS blocked image conversion, fetching via background worker:', element.src);

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

    // Always load original first — we may still need it for luminance sampling
    // under boxes that lack inpainted coverage (e.g. partial plates).
    const image = await this.loadImage(base64Image);

    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Failed to get canvas context');

    // Draw original image first
    ctx.drawImage(image, 0, 0, image.width, image.height);

    // If the backend supplied an inpainted plate, overlay it on top of the
    // original. We draw the plate at image dimensions — backend is expected
    // to return a plate aligned with the source frame.
    if (inpaintedBase64) {
      try {
        const plateSrc = inpaintedBase64.startsWith('data:')
          ? inpaintedBase64
          : `data:image/png;base64,${inpaintedBase64}`;
        const plate = await this.loadImage(plateSrc);
        ctx.drawImage(plate, 0, 0, image.width, image.height);
      } catch (err) {
        console.warn('Inpainted plate failed to load, falling back to mask:', err);
      }
    }

    // Sort text boxes by z-index (lower first, so they're drawn first)
    const sortedTextBoxes = [...textBoxes].sort((a, b) => {
      const aZ = a.zIndex || 1;
      const bZ = b.zIndex || 1;
      return aZ - bZ;
    });

    // When we have an inpainted plate, skip drawing white backgrounds — the
    // plate *is* the background. Otherwise fall back to the original masked
    // rounded rects.
    if (!inpaintedBase64) {
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
      // But high-aspect bubbles are WIDENED past that interior (onto un-erased
      // art) to fit horizontal EN — so plate just the *extra* widened area to
      // keep that text off the original art. Normal bubbles are untouched.
      for (const textBox of sortedTextBoxes) {
        const region = this.computeTextRegionBBox(
          textBox,
          sortedTextBoxes,
          canvas.width,
          canvas.height
        );
        if (region.source === 'bubble-widened') {
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

    // Pass 2: Draw ALL text on top of all backgrounds
    for (const textBox of sortedTextBoxes) {
      this.drawTextBoxText(
        ctx,
        textBox,
        fontFamily,
        sortedTextBoxes,
        canvas.width,
        canvas.height
      );
    }

    // Pass 3 (optional): Draw debug overlays
    if (showDebug) {
      this.drawDebugOverlay(ctx, sortedTextBoxes);
    }

    // Suppress unused-var warning for originalElement (kept for API compat)
    void originalElement;

    return canvas;
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
      // Tall-narrow JP bubbles (read vertically) are far too thin for
      // horizontal EN, forcing mid-word breaks. Widen them horizontally,
      // centered on the bubble, bounded by image edges + neighbors.
      return this.widenHighAspectRegion(bubble, textBox, allBoxes, canvasW, canvasH);
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
   * Draw ONLY the text for a text box (Pass 2 of two-pass rendering)
   */
  private drawTextBoxText(
    ctx: CanvasRenderingContext2D,
    textBox: TextBox,
    fontFamily: string,
    allBoxes?: TextBox[],
    canvasW?: number,
    canvasH?: number
  ): void {
    const text = textBox.translatedText;
    if (!text) return;

    // Center inside the text-region bbox (union of textRegions if available).
    // Pass siblings + image bounds so the region (incl. high-aspect bubble
    // widening) is identical to the one used to paint the background.
    const region = this.computeTextRegionBBox(textBox, allBoxes, canvasW, canvasH);

    // Available area after padding.
    const availWidth = Math.max(1, region.width - TEXT_PADDING_PX * 2);
    const availHeight = Math.max(1, region.height - TEXT_PADDING_PX * 2);

    // Binary-search for the largest font size that fits.
    const fit = this.findBestFit(ctx, text, availWidth, availHeight, fontFamily);

    // Auto-contrast. Sample the rendered background at the text region's
    // center to determine luminance, then pick/validate colors.
    const { fontColor, strokeColor } = this.resolveColors(ctx, textBox, region);

    this.drawWrappedText(
      ctx,
      fit.lines,
      region.x,
      region.y,
      region.width,
      region.height,
      fontFamily,
      fit.fontSize,
      fit.lineHeight,
      fontColor,
      strokeColor
    );
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
   *   3. else: FONT_SIZE_MIN, accepting overflow (koharu's tiny-box behavior)
   */
  private findBestFit(
    ctx: CanvasRenderingContext2D,
    text: string,
    availWidth: number,
    availHeight: number,
    fontFamily: string
  ): FitResult {
    let low = FONT_SIZE_MIN;
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
    // Priority 3: nothing fit — FONT_SIZE_MIN, accept overflow.
    return this.layoutAtSize(ctx, text, availWidth, FONT_SIZE_MIN, fontFamily);
  }

  /**
   * Wrap `text` at `fontSize` and measure total height via
   * actualBoundingBoxAscent + actualBoundingBoxDescent summed across lines,
   * with LINE_GAP_FACTOR applied between lines.
   */
  private layoutAtSize(
    ctx: CanvasRenderingContext2D,
    text: string,
    maxWidth: number,
    fontSize: number,
    fontFamily: string
  ): FitResult {
    ctx.font = this.buildFontString(fontSize, fontFamily, 'bold');
    const { lines, brokeWord } = this.wrapTextAtFont(ctx, text, maxWidth);

    let totalHeight = 0;
    let maxLineWidth = 0;
    let representativeAscent = fontSize * 0.8;
    let representativeDescent = fontSize * 0.2;

    for (let i = 0; i < lines.length; i++) {
      const metrics = ctx.measureText(lines[i]);
      const ascent =
        (metrics as any).actualBoundingBoxAscent ?? fontSize * 0.8;
      const descent =
        (metrics as any).actualBoundingBoxDescent ?? fontSize * 0.2;
      const lineHeightThis = (ascent + descent) * LINE_GAP_FACTOR;
      totalHeight += lineHeightThis;
      if (metrics.width > maxLineWidth) maxLineWidth = metrics.width;
      if (i === 0) {
        representativeAscent = ascent;
        representativeDescent = descent;
      }
    }

    // Represent the (approx) constant line height we'll use in draw-time
    // using the first line's ascent+descent. This keeps baselines stable.
    const lineHeight =
      (representativeAscent + representativeDescent) * LINE_GAP_FACTOR;

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
   * Word-wrap helper that assumes the caller already set ctx.font.
   * Also splits absurdly long single words by char if needed.
   *
   * Returns `{ lines, brokeWord }` where `brokeWord` is true iff a single word
   * had to be split mid-word (character-level) because it could not fit
   * `maxWidth` on its own. The font-fit search uses `brokeWord` to AVOID such
   * sizes whenever a larger-but-non-breaking size could not be found — mid-word
   * breaks ("MOMMY" -> "MOM"/"MY") are the dominant readability bug, so we only
   * accept them as a true last resort (see findBestFit).
   */
  private wrapTextAtFont(
    ctx: CanvasRenderingContext2D,
    text: string,
    maxWidth: number
  ): { lines: string[]; brokeWord: boolean } {
    const words = text.split(/\s+/).filter(Boolean);
    const lines: string[] = [];
    let currentLine = '';
    let brokeWord = false;

    const pushLongWord = (word: string): void => {
      // Word alone doesn't fit; break by character (last-resort, flagged).
      brokeWord = true;
      let buf = '';
      for (const ch of word) {
        const test = buf + ch;
        if (ctx.measureText(test).width > maxWidth && buf) {
          lines.push(buf);
          buf = ch;
        } else {
          buf = test;
        }
      }
      if (buf) {
        currentLine = buf;
      }
    };

    for (const word of words) {
      const testLine = currentLine ? `${currentLine} ${word}` : word;
      const metrics = ctx.measureText(testLine);

      if (metrics.width > maxWidth && currentLine) {
        lines.push(currentLine);
        // Check if word itself fits.
        if (ctx.measureText(word).width > maxWidth) {
          currentLine = '';
          pushLongWord(word);
        } else {
          currentLine = word;
        }
      } else if (metrics.width > maxWidth && !currentLine) {
        // Even the first word doesn't fit — char-wrap it.
        pushLongWord(word);
      } else {
        currentLine = testLine;
      }
    }

    if (currentLine) lines.push(currentLine);
    return {
      lines: lines.length > 0 ? lines : [text],
      brokeWord,
    };
  }

  /**
   * Build a canvas font string with a safe fallback chain.
   */
  private buildFontString(size: number, family: string, weight = 'bold'): string {
    const safeFamily = family && family.trim().length > 0 ? family : 'Bangers';
    return `${weight} ${size}px "${safeFamily}", "Bangers", "Fredoka", "Noto Sans", "Arial", sans-serif`;
  }

  /**
   * Pick text/stroke colors. If API-supplied colors have acceptable contrast
   * (>= MIN_CONTRAST_RATIO), keep them. Otherwise override via luminance sampling.
   */
  private resolveColors(
    ctx: CanvasRenderingContext2D,
    textBox: TextBox,
    region: RegionBBox
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

    // Auto-contrast: sample ~8 pixels around the region and average luminance.
    const meanLum = this.sampleMeanLuminance(ctx, region);
    if (meanLum > 128) {
      // Bright background → dark text on light stroke.
      return { fontColor: '#111111', strokeColor: '#FFFFFF' };
    }
    return { fontColor: '#FFFFFF', strokeColor: '#000000' };
  }

  /**
   * Sample up to ~8 points around the region's center and return mean
   * luminance in [0, 255].
   */
  private sampleMeanLuminance(
    ctx: CanvasRenderingContext2D,
    region: RegionBBox
  ): number {
    const cx = region.x + region.width / 2;
    const cy = region.y + region.height / 2;
    const rx = Math.max(2, Math.floor(region.width / 4));
    const ry = Math.max(2, Math.floor(region.height / 4));

    // 8 sample offsets arranged in a small diamond/grid.
    const offsets: Array<[number, number]> = [
      [0, 0],
      [-rx, 0],
      [rx, 0],
      [0, -ry],
      [0, ry],
      [-rx, -ry],
      [rx, -ry],
      [-rx, ry],
    ];

    let total = 0;
    let count = 0;
    for (const [dx, dy] of offsets) {
      const sx = Math.round(clamp(cx + dx, 0, ctx.canvas.width - 1));
      const sy = Math.round(clamp(cy + dy, 0, ctx.canvas.height - 1));
      try {
        const data = ctx.getImageData(sx, sy, 1, 1).data;
        const lum = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2];
        total += lum;
        count += 1;
      } catch {
        // getImageData can throw on tainted canvases (cross-origin) —
        // fall through to default.
      }
    }
    if (count === 0) return 255; // Assume light background → dark text.
    return total / count;
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
    const strokeWidth = Math.max(2, Math.round(fontSize * 0.12));

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
        console.error('Failed to load image:', src.substring(0, 100));
        reject(new Error('Failed to load image'));
      };

      if (!src.startsWith('data:')) {
        img.crossOrigin = 'anonymous';
      }

      img.src = src;
    });
  }

  /**
   * Replace the original element with the rendered canvas
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
    } else {
      const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
      originalElement.src = dataUrl;
    }
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

      // Retry icon.
      const retryBtn = document.createElement('button');
      retryBtn.type = 'button';
      retryBtn.className = 'manga-translator-retry-btn';
      retryBtn.title = 'Retry translation';
      retryBtn.textContent = '↻'; // ↻
      retryBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        e.preventDefault();
        const detail = {
          boxIndex,
          originalText: box.ocrText,
          translatedText: box.translatedText,
          imageElement,
        };
        document.dispatchEvent(
          new CustomEvent('manga-translator:retry-box', { detail })
        );
      });
      boxDiv.appendChild(retryBtn);

      // Double-click to edit inline.
      boxDiv.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        this.enterEditMode(boxDiv, boxIndex, box, imageElement, canvas);
      });

      overlay.appendChild(boxDiv);
    });

    host.appendChild(overlay);
    return overlay;
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
    const rendered = this.renderedImages.get(imageElement);
    if (rendered) {
      rendered.domOverlay?.remove();
      this.renderedImages.delete(imageElement);
    }
  }

  /**
   * Show loading indicator
   */
  showLoading(imageElement: HTMLElement): HTMLDivElement {
    const rect = imageElement.getBoundingClientRect();
    const loading = document.createElement('div');
    loading.className = CONFIG.CSS_CLASSES.LOADING;
    loading.style.cssText = `
      position: fixed;
      left: ${rect.left + rect.width / 2 - 60}px;
      top: ${rect.top + rect.height / 2 - 20}px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 10px 20px;
      border-radius: 4px;
      font-family: system-ui;
      font-size: 14px;
      z-index: 1000000;
      display: flex;
      align-items: center;
      gap: 8px;
    `;

    const spinner = document.createElement('div');
    spinner.className = 'manga-translator-loading-spinner';
    spinner.style.cssText = `
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: white;
      animation: spin 0.6s linear infinite;
    `;
    loading.appendChild(spinner);

    const text = document.createElement('span');
    text.textContent = 'Translating...';
    loading.appendChild(text);

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
   * Clear all overlays
   */
  clearAll(): void {
    for (const rendered of this.renderedImages.values()) {
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

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
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
