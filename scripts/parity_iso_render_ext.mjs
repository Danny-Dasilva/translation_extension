#!/usr/bin/env node
/**
 * parity_iso_render_ext.mjs — Path A of the ISOLATED renderer-parity test.
 *
 * Drives the REAL built extension (production OverlayRenderer in dist-chrome)
 * but FREEZES the translation data: a Playwright route fulfills POST /translate
 * with the captured iso_<page>/response.json so the content script renders the
 * EXACT same TextBox[] + inpaint plate that path B (backend PIL) renders. This
 * isolates the renderer from inference variance.
 *
 * The SW tries WebSocket first; we abort the WS upgrade so it falls back to the
 * HTTP path (which is the one that carries inpainted_image_base64 AND is the one
 * our route intercepts). Net: identical input to both renderers.
 *
 * Output: backend/.bench/_parity/iso_<page>/ext_render.png (+ .jpg renderer-exact)
 *
 * Usage: node scripts/parity_iso_render_ext.mjs <page>   (044 | 030)
 */
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const PAGE = process.argv[2] || '044';
const EXT_DIR = path.join(ROOT, 'dist-chrome');
const CHROME_BIN = '/home/danny/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome';
const ISO_DIR = path.join(ROOT, `backend/.bench/_parity/iso_${PAGE}`);
const RESP_JSON = path.join(ISO_DIR, 'response.json');
const SRC_WEBP = path.join(ROOT, `backend/.bench/full_pipeline_v2/588828_mesu2_insp/${PAGE}/01_source.webp`);
const OUT_PNG = path.join(ISO_DIR, 'ext_render.png');
const USER_DATA = path.join('/tmp', `parity_iso_profile_${PAGE}`);

fs.mkdirSync(ISO_DIR, { recursive: true });
fs.rmSync(USER_DATA, { recursive: true, force: true });

for (const [p, what] of [[CHROME_BIN, 'chrome'], [SRC_WEBP, 'source'], [RESP_JSON, 'captured response']]) {
  if (!fs.existsSync(p)) { console.error(`FATAL: ${what} not found at ${p}`); process.exit(2); }
}

const captured = JSON.parse(fs.readFileSync(RESP_JSON, 'utf-8'));
// The renderer keys per-image: response.images[i] + inpainted_image_base64[i].
// Our test page has exactly one <img>, so the captured single-image response is
// already in the right shape. Ensure the HTTP-success envelope the SW expects.
const fulfilledBody = JSON.stringify({
  success: true,
  images: captured.images,
  inpainted_image_base64: captured.inpainted_image_base64 || [],
  debug: captured.debug,
});

const imgBytes = fs.readFileSync(SRC_WEBP);
const html = `<!DOCTYPE html><html><head><meta charset="utf-8">
<title>iso parity ${PAGE}</title>
<style>html,body{margin:0;padding:0;background:#222}</style></head>
<body><img id="manga" src="/manga.webp" style="display:block"></body></html>`;

const server = http.createServer((req, res) => {
  if (req.url === '/manga.webp') {
    res.writeHead(200, { 'Content-Type': 'image/webp', 'Cache-Control': 'no-store' });
    res.end(imgBytes);
  } else {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(html);
  }
});

const log = (...a) => console.log('[iso-A]', ...a);
const result = { page: PAGE, loads: false, routedTranslate: false, rendered: false, boxesRendered: null, outPng: OUT_PNG };

async function main() {
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  const PAGE_URL = `http://localhost:${port}/`;
  log('serving test page at', PAGE_URL);

  const context = await chromium.launchPersistentContext(USER_DATA, {
    executablePath: CHROME_BIN,
    headless: false,
    args: [
      '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
      `--disable-extensions-except=${EXT_DIR}`, `--load-extension=${EXT_DIR}`,
    ],
    viewport: { width: 1400, height: 2000 },
    ignoreHTTPSErrors: true,
  });

  // ---- ROUTE: fulfill POST /translate with the captured (frozen) response ----
  // Matches the SW's HTTP fallback fetch(`${endpoint}/translate`). We fulfill
  // ONLY the POST; everything else passes through.
  await context.route('**/translate', async (route) => {
    const req = route.request();
    if (req.method() === 'POST') {
      result.routedTranslate = true;
      log('intercepted POST /translate -> fulfilling with captured response');
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: { 'Access-Control-Allow-Origin': '*' },
        body: fulfilledBody,
      });
    } else {
      await route.continue();
    }
  });

  // ---- find the MV3 service worker ----
  let sw = context.serviceWorkers()[0];
  if (!sw) sw = await context.waitForEvent('serviceworker', { timeout: 20000 }).catch(() => null);
  if (!sw) { throw new Error('extension service worker never started (MV3 load failed)'); }
  const extId = new URL(sw.url()).host;
  result.loads = true;
  log('SW up, ext id =', extId);

  // Force the HTTP fallback: neuter WebSocket inside the SW so translateViaWebSocket
  // throws immediately and the code falls back to translateViaHttp (which both
  // carries the plate AND is the request our route intercepts).
  try {
    await sw.evaluate(() => {
      self.WebSocket = class {
        constructor() { throw new Error('WS disabled for iso-parity (force HTTP path)'); }
      };
    });
    log('disabled WebSocket in SW (forcing HTTP /translate path)');
  } catch (e) { log('WS disable failed (non-fatal):', e.message); }

  // ---- seed settings via popup (same as parity_e2e.mjs) ----
  const popup = await context.newPage();
  await popup.goto(`chrome-extension://${extId}/popup/popup.html`, { waitUntil: 'domcontentloaded' });
  await popup.evaluate(async () => {
    await chrome.storage.sync.set({ manga_translator_settings: {
      apiEndpoint: 'http://localhost:8001', targetLanguage: 'English', autoTranslate: true,
      activeUrls: ['localhost'], defaultFont: 'Anton', useCache: false,
      showLoadingIndicator: false, showDebugInfo: false, isPremium: false,
    }});
  });
  await popup.close();
  log('seeded settings');

  const page = await context.newPage();
  page.on('console', (m) => {
    const t = m.text();
    if (m.type() === 'error' || /Manga Translator|Translation|render|box|overlay/i.test(t)) {
      log(`page[${m.type()}]:`, t.slice(0, 300));
    }
  });
  page.on('pageerror', (e) => log('page EXCEPTION:', String(e.message || e).slice(0, 300)));
  await page.goto(PAGE_URL, { waitUntil: 'load' });
  log('test page loaded; nudging settings to trigger auto-translate...');

  const p2 = await context.newPage();
  await p2.goto(`chrome-extension://${extId}/popup/popup.html`, { waitUntil: 'domcontentloaded' });
  await p2.evaluate(async () => {
    const cur = (await chrome.storage.sync.get('manga_translator_settings')).manga_translator_settings;
    await chrome.storage.sync.set({ manga_translator_settings: { ...cur, _nudge: Date.now() } });
  });
  await p2.close();

  // ---- wait for renderer to flip <img>.src to a data: URL ----
  const imgHandle = await page.$('#manga');
  let rendered = false;
  const deadline = Date.now() + Number(process.env.PARITY_WAIT_MS || 60000);
  while (Date.now() < deadline) {
    const src = await imgHandle.evaluate((el) => el.src).catch(() => '');
    if (src.startsWith('data:image')) { rendered = true; break; }
    await page.waitForTimeout(400);
  }
  if (!rendered) { throw new Error('renderer never replaced image (no data: URL)'); }
  result.rendered = true;
  log('renderer produced data:image output');

  // capture the renderer-exact bytes (the data URL) + a PNG screenshot
  const dataUrl = await imgHandle.evaluate((el) => el.src);
  const b64 = dataUrl.split(',')[1];
  fs.writeFileSync(OUT_PNG.replace(/\.png$/, '.jpg'), Buffer.from(b64, 'base64'));
  await imgHandle.screenshot({ path: OUT_PNG });
  result.boxesRendered = await page.evaluate(() =>
    document.querySelectorAll('.manga-translator-box').length || null);
  log('saved ext render (jpg renderer-exact + png screenshot)');

  await context.close();
}

main()
  .then(() => {
    console.log('\n===ISO_A_RESULT===');
    console.log(JSON.stringify(result, null, 2));
    server.close();
    process.exit(0);
  })
  .catch((e) => {
    result.error = String(e.message || e);
    console.error('[iso-A] ERROR:', result.error);
    console.log('\n===ISO_A_RESULT===');
    console.log(JSON.stringify(result, null, 2));
    server.close();
    process.exit(1);
  });
