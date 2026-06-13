/**
 * Dev visual-test harness entry.
 *
 * Loads a real manga page image + a real /translate response JSON and renders
 * them through the PRODUCTION OverlayRenderer (imported unchanged from
 * src/services/overlay-renderer.ts). The composite is what the screenshot
 * script captures.
 *
 * Page selection via URL query: ?page=637653_010
 *
 * Data is served by the harness vite server (see vite.config.ts):
 *   /api/image?page=<id>     -> the source webp (image/webp)
 *   /api/response?page=<id>  -> the TranslateResponse JSON
 *
 * Both `responses` dir and `images` dir are configured by env vars read by the
 * vite plugin at server start (HARNESS_RESPONSES_DIR / HARNESS_IMAGES_DIR).
 *
 * The harness sets `window.__renderState` to one of:
 *   'rendering' | 'done' | 'error:<msg>'
 * so the Playwright screenshot script can poll for completion deterministically.
 */
import type { TextBox, TranslateResponse } from '@/types/api';
import { OverlayRenderer } from '@/services/overlay-renderer';

declare global {
  interface Window {
    __renderState?: string;
    __renderError?: string;
  }
}

const statusEl = document.getElementById('status') as HTMLDivElement;

function setState(state: string, err?: string): void {
  window.__renderState = state;
  if (err) window.__renderError = err;
  if (statusEl) {
    statusEl.textContent = err ? `${state} ${err}` : state;
    statusEl.classList.toggle('error', state.startsWith('error'));
  }
}

function getPageId(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get('page') || '637653_010';
}

async function loadImageElement(url: string): Promise<HTMLImageElement> {
  // Fetch as a blob and use an object URL so the canvas is NOT tainted
  // (same-origin blob:) — this lets the renderer's getImageData() auto-contrast
  // sampling run for real, exactly as it would on a CORS-clean page.
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`image fetch ${resp.status}`);
  const blob = await resp.blob();
  const objUrl = URL.createObjectURL(blob);
  const img = document.getElementById('page-img') as HTMLImageElement;
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('image decode failed'));
    img.src = objUrl;
  });
  // Render at natural size so the screenshot == the canvas the renderer makes.
  img.width = img.naturalWidth;
  img.height = img.naturalHeight;
  return img;
}

async function main(): Promise<void> {
  setState('rendering');
  try {
    const page = getPageId();

    // 1. Load the real response JSON + the real source image in parallel.
    const [respJson, img] = await Promise.all([
      fetch(`/api/response?page=${encodeURIComponent(page)}`).then((r) => {
        if (!r.ok) throw new Error(`response fetch ${r.status}`);
        return r.json() as Promise<TranslateResponse>;
      }),
      loadImageElement(`/api/image?page=${encodeURIComponent(page)}`),
    ]);

    const textBoxes: TextBox[] = (respJson.images && respJson.images[0]) || [];
    const inpainted: string | null =
      (respJson.inpainted_image_base64 && respJson.inpainted_image_base64[0]) ||
      null;

    // Size the host to the natural image so buildDomOverlay positions boxes
    // 1:1 with canvas coordinates (scaleX/scaleY == 1).
    const host = document.getElementById('page-host') as HTMLDivElement;
    host.style.position = 'relative';
    host.style.width = `${img.naturalWidth}px`;
    host.style.height = `${img.naturalHeight}px`;

    // 2. Make sure comic fonts are actually usable before we render, so the
    // binary-search font metrics match the painted glyphs (Bangers, not Arial).
    // The renderer awaits this internally too, but priming here removes any
    // first-paint race in the harness.
    try {
      await (document as Document).fonts.load('bold 48px "Bangers"', 'AaGg');
      await (document as Document).fonts.load('bold 48px "Fredoka"', 'AaGg');
      await (document as Document).fonts.ready;
    } catch {
      /* degrade to renderer's own font handling */
    }

    // 3. Run the REAL production renderer.
    const renderer = new OverlayRenderer();
    await renderer.createOverlay(img, textBoxes, false, inpainted);

    // 4. Belt-and-suspenders: wait for fonts again (CDN may have resolved
    // during render) and one rAF so the replaced <img> src has painted.
    try {
      await (document as Document).fonts.ready;
    } catch {
      /* ignore */
    }
    await new Promise<void>((r) => requestAnimationFrame(() => r()));

    setState('done');
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    // eslint-disable-next-line no-console
    console.error('[harness] render failed:', err);
    setState('error', msg);
  }
}

void main();
