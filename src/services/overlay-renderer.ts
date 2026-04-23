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
}

interface RegionBBox {
  x: number;
  y: number;
  width: number;
  height: number;
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

export class OverlayRenderer {
  private renderedImages: Map<HTMLElement, RenderedImage> = new Map();
  private fontsInjected = false;

  constructor() {
    this.ensureFontsInjected();
  }

  /**
   * Inject a `<link>` into the host page pointing at a comic-style Google Font
   * so our canvas font-family fallback chain has something to use.
   *
   * Documented license: Bangers is OFL (https://fonts.google.com/specimen/Bangers),
   * Fredoka is OFL (https://fonts.google.com/specimen/Fredoka). Both are
   * safe to link directly via Google Fonts CSS API.
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

    // Sort text boxes by z-index (lower first)
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
        this.drawTextBoxBackground(ctx, textBox);
      }
    }

    // Pass 2: Draw ALL text on top of all backgrounds
    for (const textBox of sortedTextBoxes) {
      this.drawTextBoxText(ctx, textBox, fontFamily);
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
   * Draw ONLY the white background for a text box (Pass 1 of two-pass rendering)
   */
  private drawTextBoxBackground(ctx: CanvasRenderingContext2D, textBox: TextBox): void {
    const x = textBox.minX;
    const y = textBox.minY;
    const width = textBox.maxX - textBox.minX;
    const height = textBox.maxY - textBox.minY;

    if (textBox.textRegions && textBox.textRegions.length > 0) {
      for (const region of textBox.textRegions) {
        const rw = region.maxX - region.minX;
        const rh = region.maxY - region.minY;
        this.drawRoundedRect(ctx, region.minX, region.minY, rw, rh, 'white', 4);
      }
    } else {
      this.drawRoundedRect(ctx, x, y, width, height, 'white', 8);
    }
  }

  /**
   * Compute the *text-region bbox* (union of textRegions if provided, else
   * the outer bubble bbox). This is what we wrap and center text inside —
   * not the full bubble, so we don't overrun rounded edges.
   */
  private computeTextRegionBBox(textBox: TextBox): RegionBBox {
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
      return {
        x: minX,
        y: minY,
        width: Math.max(1, maxX - minX),
        height: Math.max(1, maxY - minY),
      };
    }
    return {
      x: textBox.minX,
      y: textBox.minY,
      width: Math.max(1, textBox.maxX - textBox.minX),
      height: Math.max(1, textBox.maxY - textBox.minY),
    };
  }

  /**
   * Draw ONLY the text for a text box (Pass 2 of two-pass rendering)
   */
  private drawTextBoxText(
    ctx: CanvasRenderingContext2D,
    textBox: TextBox,
    fontFamily: string
  ): void {
    const text = textBox.translatedText;
    if (!text) return;

    // Center inside the text-region bbox (union of textRegions if available).
    const region = this.computeTextRegionBBox(textBox);

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
   * whose wrapped lines fit inside (availWidth x availHeight).
   *
   * Mirrors koharu run_auto: low=min, high=max, widen while it fits.
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
    let best: FitResult | null = null;

    while (low <= high) {
      const mid = (low + high) >> 1;
      const attempt = this.layoutAtSize(ctx, text, availWidth, mid, fontFamily);
      const fitsH = attempt.totalHeight <= availHeight;
      const fitsW = attempt.maxLineWidth <= availWidth;
      if (fitsH && fitsW) {
        best = attempt;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }

    // If nothing fit at the minimum size, fall back to FONT_SIZE_MIN and
    // accept overflow (koharu does this for tiny boxes).
    if (!best) {
      best = this.layoutAtSize(ctx, text, availWidth, FONT_SIZE_MIN, fontFamily);
    }
    return best;
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
    const lines = this.wrapTextAtFont(ctx, text, maxWidth);

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
    };
  }

  /**
   * Word-wrap helper that assumes the caller already set ctx.font.
   * Also splits absurdly long single words by char if needed.
   */
  private wrapTextAtFont(
    ctx: CanvasRenderingContext2D,
    text: string,
    maxWidth: number
  ): string[] {
    const words = text.split(/\s+/).filter(Boolean);
    const lines: string[] = [];
    let currentLine = '';

    const pushLongWord = (word: string): void => {
      // Word alone doesn't fit; break by character.
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
    return lines.length > 0 ? lines : [text];
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
