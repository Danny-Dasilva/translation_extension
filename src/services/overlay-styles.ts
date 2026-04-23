/**
 * Inline CSS styles for overlay (Shadow DOM).
 *
 * Also contains a lightweight `@import` for two open-license comic-style
 * Google Fonts (Bangers, Fredoka) so the canvas font-family fallback chain
 * has real families to resolve. Fonts are also link-injected at runtime
 * from OverlayRenderer for environments where this stylesheet isn't
 * reached (e.g., canvas rendering runs before shadow DOM attaches).
 *
 *   Bangers:  https://fonts.google.com/specimen/Bangers   (OFL)
 *   Fredoka:  https://fonts.google.com/specimen/Fredoka   (OFL)
 */

export const OVERLAY_STYLES = `
/* Pull in open-license comic/display fonts for the canvas fallback chain. */
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=Fredoka:wght@400;600&display=swap');

/* Local fallback font-face — lets Chrome resolve "Bangers" even if the
   external stylesheet is blocked. Uses the same Google-hosted font file. */
@font-face {
  font-family: 'Bangers';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: local('Bangers'),
       url('https://fonts.gstatic.com/s/bangers/v24/FeVQS0BTqb0h60ACL5la2bxii28.woff2') format('woff2');
}

/* Shadow DOM styles for manga translation overlay */
:host {
  all: initial;
  display: block;
}

.manga-translator-container,
.manga-translator-overlay {
  position: absolute;
  pointer-events: none;
  z-index: 999999;
}

.manga-translator-text-box {
  position: absolute;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  word-wrap: break-word;
  pointer-events: none;
  user-select: none;
  line-height: 1.2;
  padding: 2px;
}

/* The DOM overlay layer sits over the canvas and hosts per-box affordances. */
.manga-translator-box-layer {
  position: absolute;
  pointer-events: none;
}

.manga-translator-box {
  position: absolute;
  pointer-events: auto;
  border: 1px dashed transparent;
  transition: border-color 120ms ease-out, background-color 120ms ease-out;
  box-sizing: border-box;
}

.manga-translator-box:hover,
.manga-translator-box:focus-within {
  border-color: rgba(74, 144, 226, 0.9);
  background-color: rgba(74, 144, 226, 0.07);
}

.manga-translator-box .manga-translator-retry-btn {
  position: absolute;
  top: 2px;
  right: 2px;
  width: 22px;
  height: 22px;
  padding: 0;
  border: none;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  font-size: 14px;
  line-height: 22px;
  text-align: center;
  cursor: pointer;
  opacity: 0;
  pointer-events: auto;
  transition: opacity 120ms ease-out, background-color 120ms ease-out;
}

.manga-translator-box:hover .manga-translator-retry-btn,
.manga-translator-box:focus-within .manga-translator-retry-btn {
  opacity: 1;
}

.manga-translator-box .manga-translator-retry-btn:hover {
  background: rgba(74, 144, 226, 0.95);
}

.manga-translator-edit-textarea {
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  resize: none;
  border: 2px dashed #4a90e2;
  background: rgba(255, 255, 255, 0.92);
  color: #111;
  padding: 4px;
  font-family: 'Fredoka', 'Noto Sans', Arial, sans-serif;
  font-size: 14px;
}

.manga-translator-text-box-text {
  display: block;
  width: 100%;
  text-shadow:
    -1px -1px 0 var(--stroke-color),
    1px -1px 0 var(--stroke-color),
    -1px 1px 0 var(--stroke-color),
    1px 1px 0 var(--stroke-color);
  font-weight: bold;
  font-family: 'Bangers', 'Fredoka', 'Noto Sans', Arial, sans-serif;
}

.manga-translator-loading {
  position: absolute;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 10px 20px;
  border-radius: 4px;
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 14px;
  z-index: 1000000;
  pointer-events: none;
}

.manga-translator-loading-spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: spin 0.6s linear infinite;
  margin-right: 8px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.manga-translator-error {
  position: absolute;
  background: rgba(220, 38, 38, 0.9);
  color: white;
  padding: 8px 16px;
  border-radius: 4px;
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 12px;
  z-index: 1000000;
  pointer-events: auto;
  cursor: pointer;
}
`;
