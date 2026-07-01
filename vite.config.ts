import { defineConfig } from 'vite';
import webExtension from 'vite-plugin-web-extension';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const browser = process.env.BROWSER || 'chrome';

export default defineConfig({
  root: 'src',
  envDir: __dirname, // Load .env from project root
  plugins: [
    webExtension({
      manifest: browser === 'firefox' ? 'manifest.firefox.json' : 'manifest.chrome.json',
      watchFilePaths: ['**/*'],
      browser: browser as 'chrome' | 'firefox',
      // Emit assets referenced by the manifest/content-script but not picked up
      // as JS entry points. Paths are relative to Vite's `root` (= src/), so this
      // emits src/content/overlay.css -> dist/content/overlay.css, matching the
      // web_accessible_resource and the runtime load in content-script.ts.
      additionalInputs: ['content/overlay.css'],
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  build: {
    outDir: path.resolve(__dirname, `dist-${browser}`),
    emptyOutDir: true,
    sourcemap: process.env.NODE_ENV === 'development',
  },
  publicDir: path.resolve(__dirname, 'public'),
});
