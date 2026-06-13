# Manga Overlay Visual-Test Harness

Renders the **production** frontend typesetter
(`src/services/overlay-renderer.ts`) against real `/translate` response JSONs +
the real manga page images in a **real Chromium**, and screenshots the
composite for visual QA (text size / placement / overspill / font / contrast).

This replaces the old PIL approximation (`/tmp/transperf_evidence/*_composite.png`),
which diverged from the browser because PIL FreeType metrics != browser
`measureText` and it omitted the frontend auto-contrast color logic. Here the
**actual `OverlayRenderer` canvas output** is captured.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Page shell; hosts `#page-host > #page-img`, preloads comic fonts. |
| `main.ts` | Imports the **real** `OverlayRenderer`, loads image + response, calls `createOverlay`. Sets `window.__renderState`. |
| `vite.config.ts` | Dev server: `@`→`src`, `webextension-polyfill`→stub, `/api/response` + `/api/image` middleware. |
| `mocks/webextension-polyfill.ts` | Documented stub for the extension runtime (see Mocks below). |
| `screenshot.mjs` | Playwright driver: boots vite, navigates per page, waits for render, saves PNG. |
| `out/` | Screenshots (`baseline/`, `after/`, …) + `_results.json`. |

## Run

Baseline (defaults: responses `/tmp/transperf_evidence`, images
`/tmp/transperf_imgcache`, out `out/baseline`):

```bash
node dev/visual-test/screenshot.mjs
```

Specific pages:

```bash
node dev/visual-test/screenshot.mjs --pages 637653_045,637653_090
```

Re-run against NEW data later (the whole point):

```bash
node dev/visual-test/screenshot.mjs \
  --responses /tmp/transperf_evidence_after \
  --out dev/visual-test/out/after
```

Flags (flag > env > default): `--responses` (`HARNESS_RESPONSES_DIR`),
`--images` (`HARNESS_IMAGES_DIR`), `--out`, `--pages a,b,c`, `--port`
(`HARNESS_PORT`). With no `--pages`, every `<id>.json` in the responses dir is
rendered.

Interactive debugging (no screenshot): `npx vite --config dev/visual-test/vite.config.ts`
then open `http://127.0.0.1:5199/?page=637653_010`.

## Data mapping

- Response JSON `<gallery>_<NNN>.json` (a raw `TranslateResponse`) →
  page image `<gallery dir>/<NNN>.webp`.
- Galleries are mapped in `vite.config.ts` `GALLERY_DIRS`. Add new prefixes there.
- Images resolve from the **local cache** `HARNESS_IMAGES_DIR` first (fast),
  then fall back to the NAS gallery root `HARNESS_NAS_BASE`. To pre-warm the
  cache, copy the needed `<NNN>.webp` to `<HARNESS_IMAGES_DIR>/<gallery>_<NNN>.webp`.
- Boxes are in natural-image pixel coords; webps are full-res (1280×1807), so
  the host is sized 1:1 and the screenshot is pixel-faithful to the canvas.

## Mocks / caveats (fidelity)

Everything the screenshot shows is produced by the **real** renderer. The only
fiction is the extension-runtime plumbing:

- **`webextension-polyfill` → `mocks/webextension-polyfill.ts`**
  - `storage.sync.get` returns `{}` → `settingsManager` falls back to
    `DEFAULT_SETTINGS` (`defaultFont: 'Bangers'`). Faithful to a default-config
    user (no per-user font override applied).
  - `runtime` has **no `getURL`** → `tryRegisterLocalFonts()` short-circuits and
    the renderer uses the **CDN Bangers/Fredoka** fallback. This matches
    production today, where `public/fonts/*.ttf` are not bundled
    (see `overlay-renderer.ts:90-94`). If local TTFs are shipped later, glyph
    metrics could differ slightly from this harness.
  - `runtime.sendMessage` is a no-op (`{success:false}`). **Not reached**: images
    are loaded as same-origin `blob:` URLs, so the canvas is untainted, the
    direct `toDataURL`/`getImageData` path succeeds, and the CORS background-fetch
    fallback never fires. Auto-contrast luminance sampling therefore runs for
    real.

- **Fonts:** the harness `await document.fonts.ready` (and explicitly primes
  Bangers/Fredoka) before screenshotting, so the binary-search font metrics and
  the painted glyphs are both Bangers — no measure-in-Arial/paint-in-Bangers
  race. Requires network access to Google Fonts; offline runs would degrade to
  Arial fallback (and the screenshot would honestly reflect that).

- **No CSS scaling:** rendered at natural size, `deviceScaleFactor: 1`. The
  per-box DOM overlay (retry/edit affordances) is built but `pointer-events`
  are invisible in a static screenshot — display QA targets the canvas text.

`out/<set>/_results.json` records per-page `state` (`done`/`error`/`timeout`)
and any browser console errors.
