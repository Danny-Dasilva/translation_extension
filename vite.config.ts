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
