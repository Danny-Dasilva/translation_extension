#!/usr/bin/env node
/**
 * Screenshot driver for the dev visual-test harness.
 *
 * Boots the harness vite dev server, navigates Playwright/chromium to each page,
 * waits for fonts + the real OverlayRenderer to finish (window.__renderState),
 * and writes a full-page PNG per page.
 *
 * USAGE
 *   node dev/visual-test/screenshot.mjs                  # baseline defaults
 *   node dev/visual-test/screenshot.mjs --pages 637653_045,637653_090
 *   node dev/visual-test/screenshot.mjs \
 *        --responses /tmp/transperf_evidence_after \
 *        --out dev/visual-test/out/after
 *
 * FLAGS / ENV (flag wins over env)
 *   --responses <dir>   HARNESS_RESPONSES_DIR  (default /tmp/transperf_evidence)
 *   --images <dir>      HARNESS_IMAGES_DIR     (default /tmp/transperf_imgcache)
 *   --out <dir>         output dir             (default dev/visual-test/out/baseline)
 *   --pages a,b,c       comma list of page ids (default: discovered from responses)
 *   --port <n>          HARNESS_PORT
 *
 * Auto-installs chromium if missing (npx playwright install chromium).
 */
import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';
import { execSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const val =
        argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true';
      out[key] = val;
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));

const RESPONSES_DIR =
  args.responses || process.env.HARNESS_RESPONSES_DIR || '/tmp/transperf_evidence';
const IMAGES_DIR =
  args.images || process.env.HARNESS_IMAGES_DIR || '/tmp/transperf_imgcache';
const OUT_DIR = path.resolve(
  args.out ? args.out : path.join(__dirname, 'out/baseline')
);
const PORT = Number(args.port || process.env.HARNESS_PORT || 5199);

// Page list: explicit --pages, else every <id>.json in the responses dir.
function discoverPages() {
  if (args.pages) return args.pages.split(',').map((s) => s.trim()).filter(Boolean);
  return fs
    .readdirSync(RESPONSES_DIR)
    .filter((f) => f.endsWith('.json'))
    .map((f) => f.replace(/\.json$/, ''))
    .sort();
}

async function ensureChromium() {
  try {
    const { chromium } = await import('playwright');
    // Probe executable; throws if not installed.
    chromium.executablePath();
    return chromium;
  } catch {
    console.log('[screenshot] installing chromium for Playwright…');
    execSync('npx playwright install chromium', { stdio: 'inherit' });
    const { chromium } = await import('playwright');
    return chromium;
  }
}

async function main() {
  const pages = discoverPages();
  if (pages.length === 0) {
    console.error(`[screenshot] no pages found in ${RESPONSES_DIR}`);
    process.exit(1);
  }
  fs.mkdirSync(OUT_DIR, { recursive: true });

  console.log('[screenshot] config:');
  console.log('  responses :', RESPONSES_DIR);
  console.log('  images    :', IMAGES_DIR);
  console.log('  out       :', OUT_DIR);
  console.log('  pages     :', pages.join(', '));

  // Expose dirs to the vite data-middleware.
  process.env.HARNESS_RESPONSES_DIR = RESPONSES_DIR;
  process.env.HARNESS_IMAGES_DIR = IMAGES_DIR;
  process.env.HARNESS_PORT = String(PORT);

  const server = await createServer({
    configFile: path.join(__dirname, 'vite.config.ts'),
    root: __dirname,
  });
  await server.listen();
  const addr = server.httpServer.address();
  const base = `http://127.0.0.1:${addr.port}`;
  console.log('[screenshot] harness server at', base);

  const chromium = await ensureChromium();
  const browser = await chromium.launch();

  const results = [];
  try {
    for (const page of pages) {
      const ctx = await browser.newContext({ deviceScaleFactor: 1 });
      const pg = await ctx.newPage();
      const consoleErrors = [];
      pg.on('console', (m) => {
        if (m.type() === 'error') consoleErrors.push(m.text());
      });

      const target = `${base}/?page=${encodeURIComponent(page)}`;
      await pg.goto(target, { waitUntil: 'load' });

      let state = 'timeout';
      try {
        await pg.waitForFunction(
          () =>
            window.__renderState === 'done' ||
            (window.__renderState || '').startsWith('error'),
          { timeout: 30000 }
        );
        state = await pg.evaluate(() => window.__renderState);
      } catch {
        state = 'timeout';
      }

      // Settle a beat for the CDN font + final paint.
      await pg.waitForTimeout(400);

      const outFile = path.join(OUT_DIR, `${page}.png`);
      // Screenshot the page-host element (exact composite, no chrome).
      const el = await pg.$('#page-host');
      if (el) {
        await el.screenshot({ path: outFile });
      } else {
        await pg.screenshot({ path: outFile, fullPage: true });
      }

      const ok = state === 'done';
      results.push({ page, state, ok, outFile, consoleErrors });
      console.log(
        `[screenshot] ${ok ? 'OK ' : 'FAIL'} ${page} -> ${outFile} (state=${state})${
          consoleErrors.length ? ` errors=${consoleErrors.length}` : ''
        }`
      );
      if (consoleErrors.length) {
        consoleErrors.slice(0, 3).forEach((e) => console.log('      console:', e));
      }
      await ctx.close();
    }
  } finally {
    await browser.close();
    await server.close();
  }

  const fails = results.filter((r) => !r.ok);
  console.log(
    `[screenshot] done: ${results.length - fails.length}/${results.length} ok`
  );
  fs.writeFileSync(
    path.join(OUT_DIR, '_results.json'),
    JSON.stringify(results, null, 2)
  );
  process.exit(fails.length ? 2 : 0);
}

main().catch((err) => {
  console.error('[screenshot] fatal:', err);
  process.exit(1);
});
