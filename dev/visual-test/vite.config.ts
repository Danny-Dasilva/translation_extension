/**
 * Vite config for the standalone dev visual-test harness.
 *
 * - root = this folder (serves index.html / main.ts)
 * - `@` alias -> ../../src so we import the REAL OverlayRenderer + types
 * - `webextension-polyfill` alias -> our stub (documented in mocks/)
 * - dev-server middleware exposes:
 *       /api/response?page=<id>  -> <RESPONSES_DIR>/<id>.json
 *       /api/image?page=<id>     -> mapped source webp (NAS or local cache)
 *
 * Configurable via env vars (set by screenshot.mjs, overridable by hand):
 *   HARNESS_RESPONSES_DIR  default /tmp/transperf_evidence
 *   HARNESS_IMAGES_DIR     default /tmp/transperf_imgcache  (local webp cache)
 *   HARNESS_NAS_BASE       default the nhentai gallery root (fallback source)
 */
import { defineConfig, type Plugin } from 'vite';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.resolve(__dirname, '../../src');

const RESPONSES_DIR =
  process.env.HARNESS_RESPONSES_DIR || '/tmp/transperf_evidence';
const IMAGES_DIR = process.env.HARNESS_IMAGES_DIR || '/tmp/transperf_imgcache';
const NAS_BASE =
  process.env.HARNESS_NAS_BASE ||
  '/mnt/nas/drive_2/onlyfans/external_content/nhentai';

// Map a gallery prefix -> NAS folder name. Add new galleries here as needed.
const GALLERY_DIRS: Record<string, string> = {
  '637653': '637653_Haha to Ochite Iku Part 12',
  '653631': '653631_Haha to Ochite Iku Part 13',
};

/** Resolve the source webp for a page id like "637653_010". */
function resolveImagePath(pageId: string): string | null {
  // 1. Prefer a local cache copy (fast, avoids NAS latency).
  const cached = path.join(IMAGES_DIR, `${pageId}.webp`);
  if (fs.existsSync(cached)) return cached;

  // 2. Fall back to the NAS gallery: <prefix>_*/<NNN>.webp
  const [prefix, num] = pageId.split('_');
  const dir = GALLERY_DIRS[prefix];
  if (dir && num) {
    const nas = path.join(NAS_BASE, dir, `${num}.webp`);
    if (fs.existsSync(nas)) return nas;
  }
  return null;
}

function dataMiddleware(): Plugin {
  return {
    name: 'harness-data-middleware',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = new URL(req.url || '', 'http://localhost');
        const page = (url.searchParams.get('page') || '').replace(
          /[^a-zA-Z0-9_]/g,
          ''
        );

        if (url.pathname === '/api/response') {
          const file = path.join(RESPONSES_DIR, `${page}.json`);
          if (!page || !fs.existsSync(file)) {
            res.statusCode = 404;
            res.end(`response not found: ${file}`);
            return;
          }
          res.setHeader('Content-Type', 'application/json');
          fs.createReadStream(file).pipe(res);
          return;
        }

        if (url.pathname === '/api/image') {
          const file = page ? resolveImagePath(page) : null;
          if (!file) {
            res.statusCode = 404;
            res.end(`image not found for page: ${page}`);
            return;
          }
          res.setHeader('Content-Type', 'image/webp');
          fs.createReadStream(file).pipe(res);
          return;
        }

        next();
      });
    },
  };
}

export default defineConfig({
  root: __dirname,
  resolve: {
    alias: {
      '@': SRC,
      // Documented stub so the real renderer/settingsManager run without a
      // browser-extension runtime. See mocks/webextension-polyfill.ts.
      'webextension-polyfill': path.resolve(
        __dirname,
        'mocks/webextension-polyfill.ts'
      ),
    },
  },
  // The renderer's constants.ts reads import.meta.env.VITE_API_* — harmless
  // defaults are fine here.
  server: {
    host: '127.0.0.1',
    port: Number(process.env.HARNESS_PORT) || 5199,
    strictPort: false,
  },
  plugins: [dataMiddleware()],
});
