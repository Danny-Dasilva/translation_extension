#!/usr/bin/env node
/**
 * parity_e2e.mjs — Full extension E2E "ready to install + same visual as backend" gate.
 *
 * Flow:
 *   1. Launch full Chrome-for-Testing (148) with the built MV3 extension loaded
 *      (persistent context, --headless=new).
 *   2. Serve a local HTML page (over http://localhost) containing ONE manga
 *      <img> = a bench source image. Hostname = "localhost" so we can whitelist it.
 *   3. Seed chrome.storage.sync via the extension's own popup page so the content
 *      script's onSettingsChanged listener auto-enables for localhost and
 *      auto-translates (no manual button needed).
 *   4. Wait for the <img> src to flip to a data:image/jpeg URL (renderer output).
 *   5. Save the rendered output PNG. Confirm a POST /translate (or WS) hit the
 *      backend (Playwright network capture on the SW + backend log diff).
 *
 * Pixel comparison vs backend 11_final.webp is done by a separate Python script.
 *
 * Usage: node scripts/parity_e2e.mjs <page>   (page = 044 or 030; default 044)
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
const PARITY_DIR = path.join(ROOT, 'backend/.bench/_parity');
const SRC_WEBP = path.join(ROOT, `backend/.bench/full_pipeline_v2/588828_mesu2_insp/${PAGE}/01_source.webp`);
const OUT_PNG = path.join(PARITY_DIR, `ext_render_${PAGE}.png`);
const USER_DATA = path.join('/tmp', `parity_profile_${PAGE}`);

fs.mkdirSync(PARITY_DIR, { recursive: true });
// Fresh profile each run so storage isn't stale.
fs.rmSync(USER_DATA, { recursive: true, force: true });

if (!fs.existsSync(CHROME_BIN)) {
  console.error(`FATAL: chrome binary not found at ${CHROME_BIN}`);
  process.exit(2);
}
if (!fs.existsSync(SRC_WEBP)) {
  console.error(`FATAL: source image not found at ${SRC_WEBP}`);
  process.exit(2);
}

// ---- tiny static server: serves the manga image + a test HTML page ----
const imgBytes = fs.readFileSync(SRC_WEBP);
const html = `<!DOCTYPE html><html><head><meta charset="utf-8">
<title>parity test ${PAGE}</title>
<style>html,body{margin:0;padding:0;background:#222}</style></head>
<body>
<img id="manga" src="/manga.webp" style="display:block">
</body></html>`;

const server = http.createServer((req, res) => {
  if (req.url === '/manga.webp') {
    res.writeHead(200, { 'Content-Type': 'image/webp', 'Cache-Control': 'no-store' });
    res.end(imgBytes);
  } else {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(html);
  }
});

const log = (...a) => console.log('[parity]', ...a);
const result = {
  page: PAGE,
  loads: false,
  connects: false,
  translates: false,
  connectionPath: null, // 'WS' | 'HTTP'
  outPng: OUT_PNG,
  translateRequests: [],
  wsConnections: [],
  swErrors: [],
  pageErrors: [],
  boxesRendered: null,
  blocker: null,
};

async function main() {
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  const PAGE_URL = `http://localhost:${port}/`;
  log('serving test page at', PAGE_URL);

  const context = await chromium.launchPersistentContext(USER_DATA, {
    executablePath: CHROME_BIN,
    headless: false, // we pass --headless=new ourselves (MV3 SW needs new-headless)
    args: [
      '--headless=new',
      '--no-sandbox',
      '--disable-gpu',
      '--disable-dev-shm-usage',
      `--disable-extensions-except=${EXT_DIR}`,
      `--load-extension=${EXT_DIR}`,
    ],
    viewport: { width: 1400, height: 2000 },
    ignoreHTTPSErrors: true,
  });

  // ---- capture network at the context level (catches SW fetches) ----
  context.on('request', (req) => {
    const url = req.url();
    if (url.includes('/translate') && req.method() === 'POST') {
      result.translateRequests.push(url);
      result.connectionPath = result.connectionPath || 'HTTP';
      log('captured POST', url);
    }
  });
  context.on('response', (res) => {
    const url = res.url();
    if (url.includes('/translate')) log('response', res.status(), url);
  });
  context.on('weberror', (e) => result.pageErrors.push(String(e.error())));

  // ---- find the extension's MV3 service worker to get its id ----
  let sw = context.serviceWorkers()[0];
  if (!sw) {
    log('waiting for service worker...');
    sw = await context.waitForEvent('serviceworker', { timeout: 20000 }).catch(() => null);
  }
  if (!sw) {
    result.blocker = 'extension service worker never started (MV3 load failed)';
    throw new Error(result.blocker);
  }
  const extId = new URL(sw.url()).host;
  result.loads = true;
  log('extension service worker up, id =', extId);

  // The WS client lives in the SW. Hook it from the SW context so we can tell
  // whether the WS path or the HTTP path actually carried the translation.
  try {
    await sw.evaluate(() => {
      const _WS = self.WebSocket;
      self.__wsUrls = [];
      self.__wsClosed = [];
      self.WebSocket = class extends _WS {
        constructor(url, proto) {
          super(url, proto);
          self.__wsUrls.push(String(url));
          this.addEventListener('close', () => self.__wsClosed.push(String(url)));
        }
      };
    });
  } catch (e) {
    log('WS hook failed (non-fatal):', e.message);
  }

  // ---- seed settings via the extension popup page (real chrome.storage.sync) ----
  const popup = await context.newPage();
  await popup.goto(`chrome-extension://${extId}/popup/popup.html`, { waitUntil: 'domcontentloaded' });
  await popup.evaluate(async () => {
    const SET = {
      apiEndpoint: 'http://localhost:8001',
      targetLanguage: 'English',
      autoTranslate: true,
      activeUrls: ['localhost'],
      defaultFont: 'Anton',
      useCache: false,
      showLoadingIndicator: false,
      showDebugInfo: false,
      isPremium: false,
    };
    // chrome.storage.sync is the store both SW + content script read.
    await chrome.storage.sync.set({ manga_translator_settings: SET });
  });
  log('seeded chrome.storage.sync (activeUrls=[localhost], autoTranslate=true)');
  await popup.close();

  // ---- open the test page; content script will auto-enable via storage change ----
  const page = await context.newPage();
  page.on('console', (m) => {
    const t = m.text();
    // log everything except the noisiest; surface errors/warnings always
    if (m.type() === 'error' || m.type() === 'warning' || /Manga Translator|WebSocket|Translation|Found .* translatable|Timing|overlay|render|box/i.test(t)) {
      log(`page[${m.type()}]:`, t.slice(0, 400));
    }
  });
  page.on('pageerror', (e) => log('page EXCEPTION:', String(e.message || e).slice(0, 400)));
  await page.goto(PAGE_URL, { waitUntil: 'load' });
  log('test page loaded; waiting for content script to enable + translate...');

  // The content script reads storage on init. If the page loaded a hair before
  // storage was set, the storage.onChanged path enables it. If it loaded after,
  // initialize() already saw it enabled. To be safe, nudge a storage write so
  // onSettingsChanged fires regardless of ordering.
  const p2 = await context.newPage();
  await p2.goto(`chrome-extension://${extId}/popup/popup.html`, { waitUntil: 'domcontentloaded' });
  await p2.evaluate(async () => {
    const cur = (await chrome.storage.sync.get('manga_translator_settings')).manga_translator_settings;
    await chrome.storage.sync.set({ manga_translator_settings: { ...cur, _nudge: Date.now() } });
  });
  await p2.close();

  // ---- wait for the renderer to flip <img>.src to a data: URL ----
  const imgHandle = await page.$('#manga');
  let rendered = false;
  const deadline = Date.now() + Number(process.env.PARITY_WAIT_MS || 90000); // backend OCR can take ~10-15s
  while (Date.now() < deadline) {
    const src = await imgHandle.evaluate((el) => el.src).catch(() => '');
    if (src.startsWith('data:image')) { rendered = true; break; }
    await page.waitForTimeout(500);
  }

  if (!rendered) {
    // Maybe it errored or produced 0 boxes — capture diagnostics.
    const swDiag = await sw.evaluate(() => ({ wsUrls: self.__wsUrls || [], wsClosed: self.__wsClosed || [] })).catch(() => ({}));
    result.wsConnections = swDiag.wsUrls || [];
    result.blocker = result.blocker || 'renderer never replaced the image (no data: URL after 90s)';
    log('NO RENDER. SW WS urls:', JSON.stringify(swDiag));
    throw new Error(result.blocker);
  }

  result.translates = true;
  log('renderer replaced image with a data:image URL — translation rendered.');

  // ---- determine connection path from the SW WS hook ----
  const swDiag = await sw.evaluate(() => ({ wsUrls: self.__wsUrls || [], wsClosed: self.__wsClosed || [] }));
  result.wsConnections = swDiag.wsUrls || [];
  if (result.wsConnections.some((u) => u.includes('/ws/translate'))) {
    // WS was opened. Was it actually used (vs. opened-then-fell-back)? If a
    // POST /translate also fired, HTTP carried it; otherwise WS did.
    result.connectionPath = result.translateRequests.length > 0 ? 'HTTP (WS attempted, fell back)' : 'WS';
  } else if (result.translateRequests.length > 0) {
    result.connectionPath = 'HTTP';
  }
  result.connects = result.wsConnections.length > 0 || result.translateRequests.length > 0;

  // ---- capture the rendered output at full natural resolution ----
  // Read the data URL bytes directly (this IS the renderer's exact output —
  // no browser re-scaling), and also a viewport screenshot for sanity.
  const dataUrl = await imgHandle.evaluate((el) => el.src);
  const b64 = dataUrl.split(',')[1];
  fs.writeFileSync(OUT_PNG.replace(/\.png$/, '.jpg'), Buffer.from(b64, 'base64'));
  // Also screenshot the element to PNG for convenient viewing.
  await imgHandle.screenshot({ path: OUT_PNG });
  log('saved render JPEG (renderer-exact) and PNG screenshot');

  // count rendered boxes from the DOM overlay
  result.boxesRendered = await page.evaluate(() =>
    document.querySelectorAll('.manga-translator-box').length || null
  );

  await context.close();
}

main()
  .then(() => {
    console.log('\n===PARITY_RESULT_JSON===');
    console.log(JSON.stringify(result, null, 2));
    server.close();
    process.exit(0);
  })
  .catch((e) => {
    result.error = String(e.message || e);
    console.error('[parity] ERROR:', result.error);
    console.log('\n===PARITY_RESULT_JSON===');
    console.log(JSON.stringify(result, null, 2));
    server.close();
    process.exit(1);
  });
