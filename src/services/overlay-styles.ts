/**
 * Inline CSS styles for overlay (Shadow DOM)
 * This avoids issues with loading external CSS files
 */

export const OVERLAY_STYLES = `
/* Shadow DOM styles for manga translation overlay */
:host {
  all: initial;
  display: block;
}

.manga-translator-container {
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

.manga-translator-text-box-text {
  display: block;
  width: 100%;
  text-shadow: 
    -1px -1px 0 var(--stroke-color),
    1px -1px 0 var(--stroke-color),
    -1px 1px 0 var(--stroke-color),
    1px 1px 0 var(--stroke-color);
  font-weight: bold;
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
