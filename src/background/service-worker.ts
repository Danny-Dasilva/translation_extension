/**
 * Background service worker for Chrome extension
 */
import browser from 'webextension-polyfill';
import { settingsManager } from '@/services/settings-manager';
import { webSocketClient } from '@/services/websocket-client';
import { CONFIG } from '@/config/constants';
import { TranslateResponse, FlagRequest, FlagResponse } from '@/types/api';
import { StreamEventFrame } from '@/types/stream';
import { logger } from '@/utils/logger';

logger.setPrefix('[MT/sw]');

/**
 * PREFETCH CACHE (LRU, bounded).
 *
 * The content-script's PrefetchManager speculatively translates images it
 * predicts the reader will reach next, keyed by a fast hash of the COMPRESSED
 * image bytes (the exact bytes the backend sees). When the page actually
 * displays that image the content-script asks for the cached result instead of
 * re-requesting, so a predicted page renders instantly.
 *
 * A plain Map preserves insertion order, which we use for O(1) LRU eviction:
 * `get` re-inserts to mark recency; `put` drops the oldest key past the bound.
 */
const PREFETCH_CACHE_MAX = 5;
const prefetchCache = new Map<string, TranslateResponse>();

function prefetchCacheGet(hash: string): TranslateResponse | undefined {
  const hit = prefetchCache.get(hash);
  if (hit) {
    // Bump recency.
    prefetchCache.delete(hash);
    prefetchCache.set(hash, hit);
  }
  return hit;
}

function prefetchCachePut(hash: string, data: TranslateResponse): void {
  if (prefetchCache.has(hash)) prefetchCache.delete(hash);
  prefetchCache.set(hash, data);
  while (prefetchCache.size > PREFETCH_CACHE_MAX) {
    const oldest = prefetchCache.keys().next().value;
    if (oldest === undefined) break;
    prefetchCache.delete(oldest);
  }
}

/**
 * Translate a single image the "monolithic" way (no streaming): WebSocket
 * first, HTTP fallback. Shared by the prefetch path and any non-streaming
 * caller. Streaming is intentionally NOT used for prefetch — a speculative
 * request has no visible overlay to paint into, so we only want the final
 * assembled result to cache.
 */
async function translateMonolithic(
  base64Images: string[],
  targetLanguage: string
): Promise<TranslateResponse> {
  try {
    const ws = await translateViaWebSocket(base64Images, targetLanguage);
    if (ws.success && ws.data) return ws.data;
  } catch (err) {
    logger.warn('Prefetch WS failed, falling back to HTTP:', err);
  }
  return await translateViaHttp(base64Images, targetLanguage);
}

// Create context menu.
// NOTE: Firefox for Android does NOT implement browser.contextMenus — accessing
// it throws and would kill the whole background script. Feature-detect and no-op
// there (the popup toggle + content-script paths still work without menus).
async function createContextMenu() {
  if (!browser.contextMenus) return;
  // Per-site activation toggle.
  await browser.contextMenus.create({
    id: 'toggle-manga-translator',
    title: 'Enable Manga Translator for this site',
    contexts: ['page', 'image'],
  });
  // Master ON/OFF switch (global, independent of the per-site whitelist).
  await browser.contextMenus.create({
    id: 'toggle-translation-enabled',
    title: 'Turn Translation OFF',
    contexts: ['page', 'image'],
  });
}

// Update context menu titles based on current state
async function updateContextMenu(hostname: string) {
  if (!browser.contextMenus) return; // Firefox for Android: no contextMenus API
  const isEnabled = await settingsManager.isEnabledForHostname(hostname);
  await browser.contextMenus.update('toggle-manga-translator', {
    title: isEnabled
      ? `Disable Manga Translator for ${hostname}`
      : `Enable Manga Translator for ${hostname}`,
  });

  const translationOn = await settingsManager.isTranslationEnabled();
  await browser.contextMenus.update('toggle-translation-enabled', {
    title: translationOn ? 'Turn Translation OFF' : 'Turn Translation ON',
  });
}

// Initialize extension on install
browser.runtime.onInstalled.addListener(async (details) => {
  if (details.reason === 'install') {
    logger.info('Manga Translator installed');

    // Initialize default settings + mirror verbose flag into the logger.
    const settings = await settingsManager.loadSettings();
    logger.setVerbose(settings.showDebugInfo);

    // Create context menu
    await createContextMenu();

    // Open welcome page (optional)
    // browser.tabs.create({ url: 'popup/popup.html' });
  } else if (details.reason === 'update') {
    logger.info('Manga Translator updated');

    // Ensure context menu exists
    await createContextMenu();
  }
});

/**
 * Flip the master translation ON/OFF switch, persist it, update the context
 * menu, and notify the content script so it clears/restores overlays. Shared by
 * the context-menu item, the popup, and any hotkey path.
 */
async function applyTranslationToggle(tabId?: number, hostname?: string): Promise<boolean> {
  const current = await settingsManager.isTranslationEnabled();
  const next = !current;
  await settingsManager.setTranslationEnabled(next);
  logger.info(`Master translation switched ${next ? 'ON' : 'OFF'}`);

  if (hostname) await updateContextMenu(hostname);

  if (tabId !== undefined) {
    try {
      await browser.tabs.sendMessage(tabId, {
        action: 'setTranslationEnabled',
        enabled: next,
      });
    } catch (err) {
      // Content script may not be injected on this page — non-fatal.
      logger.warn('Could not notify content script of translation toggle', err);
    }
  }
  return next;
}

// Handle context menu clicks.
// Optional-chain the top-level access: Firefox for Android has no contextMenus
// API, and an unguarded `browser.contextMenus.onClicked` here throws at module
// load and kills the ENTIRE background script (→ all translate messages go
// unanswered → "translation failed"). With `?.` the listener is simply not
// registered on Android; the popup toggle drives the same logic.
browser.contextMenus?.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'toggle-manga-translator' && tab?.url) {
    const url = new URL(tab.url);
    const hostname = url.hostname;
    const isEnabled = await settingsManager.isEnabledForHostname(hostname);

    if (isEnabled) {
      await settingsManager.removeActiveUrl(hostname);
      logger.info(`Disabled for ${hostname}`);
    } else {
      await settingsManager.addActiveUrl(hostname);
      logger.info(`Enabled for ${hostname}`);
    }

    // Update context menu title
    await updateContextMenu(hostname);

    // Notify content script
    if (tab.id) {
      await browser.tabs.sendMessage(tab.id, { action: 'toggle' });
    }
  } else if (info.menuItemId === 'toggle-translation-enabled') {
    const hostname = tab?.url ? new URL(tab.url).hostname : undefined;
    await applyTranslationToggle(tab?.id, hostname);
  }
});

// Update context menu when tab changes
browser.tabs.onActivated.addListener(async (activeInfo) => {
  const tab = await browser.tabs.get(activeInfo.tabId);
  if (tab.url) {
    const url = new URL(tab.url);
    await updateContextMenu(url.hostname);
  }
});

// Update context menu when URL changes
browser.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.url) {
    const url = new URL(changeInfo.url);
    await updateContextMenu(url.hostname);
  }
});

// Handle messages from content scripts and popup
browser.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender).then(sendResponse);
  return true; // Keep channel open for async response
});

async function handleMessage(message: any, sender: browser.Runtime.MessageSender): Promise<any> {
  switch (message.action) {
    case 'getSettings':
      return await settingsManager.getSettings();

    case 'saveSettings':
      await settingsManager.saveSettings(message.settings);
      return { success: true };

    case 'addActiveUrl':
      await settingsManager.addActiveUrl(message.hostname);
      return { success: true };

    case 'removeActiveUrl':
      await settingsManager.removeActiveUrl(message.hostname);
      return { success: true };

    case 'getActiveUrls':
      const urls = await settingsManager.getActiveUrls();
      return { urls };

    case 'isEnabled':
      const enabled = await settingsManager.isEnabledForHostname(message.hostname);
      return { enabled };

    case 'getTranslationEnabled':
      return { enabled: await settingsManager.isTranslationEnabled() };

    case 'setTranslationEnabled': {
      // Master ON/OFF switch toggle from the popup. Persist + notify content.
      const result = await applyTranslationToggle(message.tabId, message.hostname);
      return { success: true, enabled: result };
    }

    case 'flagTranslation': {
      // Forward a poor-translation report to the backend /flag endpoint.
      // Done here (in the SW, which has host_permissions) so the cross-origin
      // POST is not mixed-content/CORS-blocked from the page.
      try {
        const data = await postFlag(message.payload as FlagRequest);
        return { success: true, data };
      } catch (error) {
        logger.error('Flag request failed', error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Flag request failed',
        };
      }
    }

    case 'fetchImage':
      // Fetch cross-origin image and convert to base64
      try {
        const response = await fetch(message.url);
        if (!response.ok) {
          return { success: false, error: `Failed to fetch image: ${response.status}` };
        }
        const blob = await response.blob();
        const base64 = await blobToBase64(blob);
        return { success: true, base64 };
      } catch (error) {
        logger.error('Failed to fetch image:', error);
        return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
      }

    case 'translateImages': {
      // Try WebSocket first, fall back to HTTP
      const t0 = Date.now();
      try {
        const wsResult = await translateViaWebSocket(
          message.base64Images,
          message.targetLanguage
        );
        if (wsResult.success) {
          logger.info(
            `translate done (ws) in ${Date.now() - t0}ms`,
            `n=${message.base64Images?.length ?? 0}`
          );
          return wsResult;
        }
        // If WebSocket failed, fall back to HTTP
        logger.warn('WebSocket translation failed, falling back to HTTP:', wsResult.error);
      } catch (wsError) {
        logger.warn('WebSocket translation error, falling back to HTTP:', wsError);
      }

      // HTTP fallback
      try {
        const data = await translateViaHttp(
          message.base64Images,
          message.targetLanguage
        );
        logger.info(`translate done (http) in ${Date.now() - t0}ms`);
        return { success: true, data };
      } catch (error) {
        logger.error('Translation API call failed:', error);
        return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
      }
    }

    case 'getCachedTranslation': {
      // Content-script asks whether a predicted image was already translated by
      // the prefetch path. Keyed by the fast hash of the compressed bytes.
      const data = prefetchCacheGet(message.hash);
      return data ? { success: true, data } : { success: false };
    }

    case 'prefetchTranslate': {
      // Speculative translation of a predicted next-page image. Result is cached
      // (not returned for rendering). Deduped by hash so we never translate the
      // same predicted image twice.
      const hash: string = message.hash;
      if (prefetchCache.has(hash)) {
        return { success: true, cached: true };
      }
      try {
        const data = await translateMonolithic(
          message.base64Images,
          message.targetLanguage
        );
        prefetchCachePut(hash, data);
        logger.info(`prefetch cached (hash=${hash}, size=${prefetchCache.size})`);
        return { success: true, cached: false };
      } catch (error) {
        logger.warn('Prefetch translate failed:', error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Prefetch failed',
        };
      }
    }

    case 'translate':
      // Forward translate message to content script
      if (message.tabId) {
        await browser.tabs.sendMessage(message.tabId, { action: 'translate' });
        return { success: true };
      }
      return { success: false, error: 'No tab ID provided' };

    case 'clear':
      // Forward clear message to content script
      if (message.tabId) {
        await browser.tabs.sendMessage(message.tabId, { action: 'clear' });
        return { success: true };
      }
      return { success: false, error: 'No tab ID provided' };

    default:
      return { success: false, error: 'Unknown action' };
  }
}

/**
 * Convert blob to base64 data URL
 */
async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
      } else {
        reject(new Error('Failed to convert blob to base64'));
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// Handle browser action clicks
browser.action?.onClicked.addListener(async (tab) => {
  if (!tab.id) return;

  // Toggle extension for current hostname
  const url = new URL(tab.url || '');
  const hostname = url.hostname;
  
  const isEnabled = await settingsManager.isEnabledForHostname(hostname);
  
  if (isEnabled) {
    await settingsManager.removeActiveUrl(hostname);
    logger.info(`Disabled for ${hostname}`);
  } else {
    await settingsManager.addActiveUrl(hostname);
    logger.info(`Enabled for ${hostname}`);
  }

  // Notify content script
  await browser.tabs.sendMessage(tab.id, {
    action: 'toggle',
  });
});

/**
 * Convert base64 data URL to ArrayBuffer
 */
function base64ToArrayBuffer(base64: string): ArrayBuffer {
  // Remove data URL prefix if present
  const base64Data = base64.includes(',') ? base64.split(',')[1] : base64;
  const binaryString = atob(base64Data);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

/**
 * Translate images via WebSocket (binary upload).
 *
 * `onFrame`, when supplied, receives each v:1 streaming event frame as it
 * arrives (with the 0-based image index it belongs to) so a caller holding a
 * long-lived Port can forward progressive updates to the tab. The underlying
 * websocket-client transparently supports BOTH the legacy monolithic response
 * (onFrame simply never fires) and the streaming path, so this function's
 * assembled return value is identical in either case.
 */
async function translateViaWebSocket(
  base64Images: string[],
  targetLanguage: string,
  onFrame?: (imageIndex: number, frame: StreamEventFrame) => void
): Promise<{ success: boolean; data?: TranslateResponse; error?: string }> {
  try {
    // Connect to WebSocket
    await webSocketClient.connect(targetLanguage);

    // Process images one at a time (WebSocket protocol is request-response)
    const allTextBoxes: any[][] = [];
    const allPlates: (string | null)[] = [];
    let debugInfo: TranslateResponse['debug'] | undefined;

    for (let i = 0; i < base64Images.length; i++) {
      // Convert base64 to binary
      const imageBuffer = base64ToArrayBuffer(base64Images[i]);

      // Send via WebSocket, forwarding stream frames (if any) tagged with the
      // image index so the content-script can route them to the right overlay.
      const response = await webSocketClient.send(
        imageBuffer,
        onFrame ? { onEvent: (frame) => onFrame(i, frame) } : undefined
      );

      if (response.success === false) {
        return { success: false, error: (response as any).error || 'Translation failed' };
      }

      // Collect results
      if (response.images && response.images.length > 0) {
        allTextBoxes.push(...response.images);
      }
      // Collect per-image plate (streaming path assembles it; legacy may too).
      if (response.inpainted_image_base64 && response.inpainted_image_base64.length > 0) {
        allPlates.push(...response.inpainted_image_base64);
      } else {
        allPlates.push(null);
      }

      // Keep the last debug info
      if (response.debug) {
        debugInfo = response.debug;
      }
    }

    const result: TranslateResponse = {
      success: true,
      images: allTextBoxes,
      inpainted_image_base64: allPlates,
      debug: debugInfo,
    };

    return { success: true, data: result };
  } catch (error) {
    logger.error('WebSocket translation error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'WebSocket translation failed',
    };
  }
}

/**
 * Translate images via HTTP (base64 upload - fallback)
 */
async function translateViaHttp(
  base64Images: string[],
  targetLanguage: string
): Promise<TranslateResponse> {
  const settings = await settingsManager.getSettings();
  const endpoint = CONFIG.DEFAULT_API_ENDPOINT;

  const response = await fetch(`${endpoint}/translate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(settings.apiKey && { Authorization: `Bearer ${settings.apiKey}` }),
    },
    body: JSON.stringify({
      base64Images,
      targetLanguage,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `API request failed: ${response.status}`);
  }

  return await response.json();
}

/**
 * POST a poor-translation report to the backend /flag endpoint.
 *
 * The body field names are the EXACT snake_case contract the backend expects
 * (image_base64, page_url, target_language, boxes[].{ocr_text, translated_text,
 * minX, minY, maxX, maxY}). The payload is built on the content side and passed
 * through verbatim — we only add auth headers here.
 */
async function postFlag(payload: FlagRequest): Promise<FlagResponse> {
  const settings = await settingsManager.getSettings();
  const endpoint = CONFIG.DEFAULT_API_ENDPOINT;

  const response = await fetch(`${endpoint}/flag`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(settings.apiKey && { Authorization: `Bearer ${settings.apiKey}` }),
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let detail = `Flag request failed: ${response.status}`;
    try {
      const err = await response.json();
      detail = (err as { detail?: string }).detail || detail;
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }

  return (await response.json()) as FlagResponse;
}

/**
 * STREAMING TRANSLATE PORT (MV3-safe progressive delivery).
 *
 * The content-script opens a long-lived `browser.runtime` Port named
 * `mt-translate-stream` and posts a single `translateStream` request. We run the
 * websocket translate with an onFrame forwarder so each v:1 event frame is
 * pushed to the tab as `{ type:'event', ... }`, then post the final assembled
 * response as `{ type:'result', ... }`. If the backend replies legacy/monolithic
 * (or WS fails and we fall back to HTTP) no event frames fire and the caller
 * simply renders from the single `result` — preserving the existing path.
 *
 * A Port (not sendMessage) is required because MV3 service workers can suspend;
 * an open Port keeps the worker alive for the duration of the stream and gives
 * us a channel to push N messages for one request.
 */
browser.runtime.onConnect.addListener((port) => {
  if (port.name !== 'mt-translate-stream') return;

  port.onMessage.addListener(async (raw) => {
    const msg = raw as {
      action?: string;
      requestId?: number;
      base64Images?: string[];
      targetLanguage?: string;
    };
    if (msg?.action !== 'translateStream') return;

    const requestId = msg.requestId;
    const base64Images = msg.base64Images ?? [];
    const targetLanguage = msg.targetLanguage ?? 'English';
    const t0 = Date.now();
    let streamed = false;

    const safePost = (payload: Record<string, unknown>) => {
      try {
        port.postMessage({ requestId, ...payload });
      } catch (err) {
        // Port may have disconnected (tab navigated away) — non-fatal.
        logger.warn('stream port post failed', err);
      }
    };

    try {
      const wsResult = await translateViaWebSocket(
        base64Images,
        targetLanguage,
        (imageIndex, frame) => {
          streamed = true;
          safePost({ type: 'event', imageIndex, frame });
        }
      );

      if (wsResult.success && wsResult.data) {
        logger.info(
          `translateStream done (ws${streamed ? '/streamed' : ''}) in ${Date.now() - t0}ms`
        );
        safePost({ type: 'result', data: wsResult.data, streamed });
        return;
      }
      logger.warn('translateStream WS failed, HTTP fallback:', wsResult.error);
    } catch (wsErr) {
      logger.warn('translateStream WS error, HTTP fallback:', wsErr);
    }

    // HTTP fallback — single monolithic result, no incremental events.
    try {
      const data = await translateViaHttp(base64Images, targetLanguage);
      logger.info(`translateStream done (http) in ${Date.now() - t0}ms`);
      safePost({ type: 'result', data, streamed: false });
    } catch (error) {
      logger.error('translateStream failed:', error);
      safePost({
        type: 'error',
        error: error instanceof Error ? error.message : 'Translation failed',
      });
    }
  });
});

logger.info('Background service worker loaded');
