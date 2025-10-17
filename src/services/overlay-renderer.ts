/**
 * Overlay renderer using Canvas to overwrite original images
 * Matches behavior of original main_extension.js
 */
import { TextBox } from '@/types/api';
import { CONFIG } from '@/config/constants';
import { settingsManager } from './settings-manager';

interface RenderedImage {
  originalElement: HTMLImageElement | HTMLCanvasElement;
  newElement: HTMLImageElement | HTMLCanvasElement;
}

export class OverlayRenderer {
  private renderedImages: Map<HTMLElement, RenderedImage> = new Map();

  /**
   * Create overlay by drawing translations onto the image itself
   * This replaces the original image/canvas element
   */
  async createOverlay(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    textBoxes: TextBox[]
  ): Promise<void> {
    // Remove existing overlay if any
    this.removeOverlay(imageElement);

    try {
      // Get settings
      const settings = await settingsManager.getSettings();

      // Get base64 image data
      const base64Image = await this.getImageBase64(imageElement);

      // Create canvas with translations
      const canvas = await this.renderTranslationsOnCanvas(
        base64Image,
        textBoxes,
        imageElement,
        settings.defaultFont
      );

      // Replace original element with rendered canvas/image
      await this.replaceElement(imageElement, canvas);

      this.renderedImages.set(imageElement, {
        originalElement: imageElement,
        newElement: canvas,
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

    // For img elements, first check if we already have base64 data stored
    if (element.dataset.originalSrc && element.dataset.originalSrc.startsWith('data:')) {
      return element.dataset.originalSrc;
    }

    // Try to draw to canvas first (will work for same-origin images)
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get canvas context');

    // Use natural dimensions
    canvas.width = element.naturalWidth || element.width;
    canvas.height = element.naturalHeight || element.height;

    try {
      ctx.drawImage(element, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.85);
    } catch (error) {
      console.warn('CORS blocked image conversion, fetching via background worker:', element.src);

      // Fetch via background worker to bypass CORS
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
   * Render translations onto a canvas
   */
  private async renderTranslationsOnCanvas(
    base64Image: string,
    textBoxes: TextBox[],
    originalElement: HTMLImageElement | HTMLCanvasElement,
    fontFamily: string
  ): Promise<HTMLCanvasElement> {
    // Load the image
    const image = await this.loadImage(base64Image);

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Failed to get canvas context');

    // Draw original image
    ctx.drawImage(image, 0, 0, image.width, image.height);

    // Calculate scaling factors
    // The textBox coordinates are in "original image" coordinates
    // We need to scale them to match the canvas size
    const displayedRect = originalElement.getBoundingClientRect();
    const naturalWidth = image.width;
    const naturalHeight = image.height;

    // Sort text boxes by z-index (lower first, so they're drawn first)
    const sortedTextBoxes = [...textBoxes].sort((a, b) => {
      const aZ = a.zIndex || 1;
      const bZ = b.zIndex || 1;
      return aZ - bZ;
    });

    // Draw each text box
    for (const textBox of sortedTextBoxes) {
      await this.drawTextBox(ctx, textBox, fontFamily, naturalWidth, naturalHeight);
    }

    return canvas;
  }

  /**
   * Draw a single text box on the canvas
   */
  private async drawTextBox(
    ctx: CanvasRenderingContext2D,
    textBox: TextBox,
    fontFamily: string,
    imageWidth: number,
    imageHeight: number
  ): Promise<void> {
    // Text box coordinates (already in original image coordinates from API)
    const x = textBox.minX;
    const y = textBox.minY;
    const width = textBox.maxX - textBox.minX;
    const height = textBox.maxY - textBox.minY;

    // Draw white rounded rectangle background
    this.drawRoundedRect(ctx, x, y, width, height, 'white', 8);

    // Prepare text rendering
    const text = textBox.translatedText;
    const fontSize = textBox.fontHeightPx || 20;
    const fontColor = textBox.fontColor || '#000000';
    const strokeColor = textBox.fontStrokeColor || '#FFFFFF';

    // Word wrap and fit text
    const lines = this.wrapText(ctx, text, width - 20, fontFamily, fontSize); // 20px padding

    // Calculate actual font size that fits
    let actualFontSize = fontSize;
    const lineHeight = fontSize * 1.25;
    const totalHeight = lines.length * lineHeight;

    // Scale down font if text doesn't fit vertically
    if (totalHeight > height - 20) {
      const scale = (height - 20) / totalHeight;
      actualFontSize = Math.max(12, fontSize * scale);
    }

    // Draw the text
    this.drawWrappedText(
      ctx,
      lines,
      x,
      y,
      width,
      height,
      fontFamily,
      actualFontSize,
      fontColor,
      strokeColor
    );
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
   * Wrap text to fit within a given width
   */
  private wrapText(
    ctx: CanvasRenderingContext2D,
    text: string,
    maxWidth: number,
    fontFamily: string,
    fontSize: number
  ): string[] {
    ctx.font = `${fontSize}px ${fontFamily}`;

    const words = text.split(' ');
    const lines: string[] = [];
    let currentLine = '';

    for (const word of words) {
      const testLine = currentLine ? `${currentLine} ${word}` : word;
      const metrics = ctx.measureText(testLine);

      if (metrics.width > maxWidth && currentLine) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = testLine;
      }
    }

    if (currentLine) {
      lines.push(currentLine);
    }

    return lines.length > 0 ? lines : [text];
  }

  /**
   * Draw wrapped text centered in a bounding box
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
    fontColor: string,
    strokeColor: string
  ): void {
    ctx.font = `bold ${fontSize}px ${fontFamily}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const lineHeight = fontSize * 1.25;
    const totalTextHeight = lines.length * lineHeight;

    // Center vertically
    let startY = boxY + (boxHeight - totalTextHeight) / 2 + lineHeight / 2;

    // Clamp to box bounds
    if (startY < boxY + lineHeight / 2) {
      startY = boxY + lineHeight / 2;
    }

    const centerX = boxX + boxWidth / 2;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const yPos = startY + i * lineHeight;

      // Don't draw if we're outside the box
      if (yPos > boxY + boxHeight - lineHeight / 2) {
        break;
      }

      // Draw stroke (outline)
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 3;
      ctx.strokeText(line, centerX, yPos);

      // Draw fill (main text)
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
      img.onerror = (error) => {
        console.error('Failed to load image:', src.substring(0, 100));
        reject(new Error('Failed to load image'));
      };

      // Only set crossOrigin for URLs, not for data URIs
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
      // For canvas elements, just draw onto them
      const ctx = originalElement.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, originalElement.width, originalElement.height);
        ctx.drawImage(canvas, 0, 0);
      }
    } else {
      // For img elements, convert canvas to data URL and replace src
      const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
      originalElement.src = dataUrl;
    }
  }

  /**
   * Remove overlay (restore original if possible)
   */
  removeOverlay(imageElement: HTMLElement): void {
    const rendered = this.renderedImages.get(imageElement);
    if (rendered) {
      // Note: We can't easily restore the original image since we've overwritten it
      // This is the same limitation as the original extension
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

    // Auto-remove after 5 seconds
    setTimeout(() => error.remove(), 5000);

    return error;
  }

  /**
   * Clear all overlays
   */
  clearAll(): void {
    // Note: Can't restore originals since we've overwritten them
    this.renderedImages.clear();
  }

  /**
   * Reset renderer
   */
  reset(): void {
    this.clearAll();
  }

  /**
   * Update overlay position (not needed for canvas approach)
   */
  updateOverlayPosition(_imageElement: HTMLElement): void {
    // No-op: Canvas rendering doesn't require position updates
  }
}
