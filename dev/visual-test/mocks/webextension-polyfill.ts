/**
 * Minimal stub for `webextension-polyfill` used ONLY by the dev visual-test
 * harness (aliased in dev/visual-test/vite.config.ts).
 *
 * WHY: the production OverlayRenderer + settingsManager import the real
 * `webextension-polyfill` (`browser.*`). In a plain browser page (no extension
 * runtime) that module's APIs are undefined and throw. We provide just enough
 * surface for the code paths the harness exercises:
 *
 *   - settingsManager.getSettings() -> browser.storage.sync.get(KEY)
 *       We return {} so it falls back to DEFAULT_SETTINGS (defaultFont:'Bangers').
 *       => The renderer runs its REAL settings path; the only fiction is "no
 *          stored user overrides", which is exactly the default render.
 *
 *   - OverlayRenderer.tryRegisterLocalFonts() -> runtime.getURL(path)
 *       We deliberately return `runtime` WITHOUT getURL so the local-font
 *       registration short-circuits (the `if (!getURL) return;` guard) and the
 *       renderer falls back to the CDN Bangers/Fredoka — the same fallback path
 *       that runs in production when public/fonts/*.ttf aren't bundled (they
 *       aren't, per the source comment at overlay-renderer.ts:90-94).
 *
 *   - getImageBase64() CORS fallback -> runtime.sendMessage({action:'fetchImage'})
 *       Not reached in the harness: we feed same-origin (blob:) images so the
 *       canvas read succeeds directly. Provided as a no-op for safety.
 *
 * Everything the screenshot shows is therefore produced by the REAL renderer
 * code; this stub only replaces the extension-runtime plumbing.
 */

const browser = {
  storage: {
    sync: {
      // Return empty -> settingsManager merges over DEFAULT_SETTINGS.
      get: async (_keys?: unknown): Promise<Record<string, unknown>> => ({}),
      set: async (_items?: unknown): Promise<void> => {},
    },
    local: {
      get: async (_keys?: unknown): Promise<Record<string, unknown>> => ({}),
      set: async (_items?: unknown): Promise<void> => {},
    },
  },
  runtime: {
    // NOTE: intentionally NO getURL -> forces CDN font fallback (documented).
    sendMessage: async (_msg?: unknown): Promise<{ success: boolean }> => ({
      success: false,
    }),
  },
};

export default browser;
