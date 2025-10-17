/**
 * Background service worker for Chrome extension
 */
import browser from 'webextension-polyfill';
import { settingsManager } from '@/services/settings-manager';

// Create context menu
async function createContextMenu() {
  await browser.contextMenus.create({
    id: 'toggle-manga-translator',
    title: 'Enable Manga Translator for this site',
    contexts: ['page', 'image'],
  });
}

// Update context menu title based on current state
async function updateContextMenu(hostname: string) {
  const isEnabled = await settingsManager.isEnabledForHostname(hostname);
  await browser.contextMenus.update('toggle-manga-translator', {
    title: isEnabled
      ? `Disable Manga Translator for ${hostname}`
      : `Enable Manga Translator for ${hostname}`,
  });
}

// Initialize extension on install
browser.runtime.onInstalled.addListener(async (details) => {
  if (details.reason === 'install') {
    console.log('Manga Translator installed');

    // Initialize default settings
    await settingsManager.loadSettings();

    // Create context menu
    await createContextMenu();

    // Open welcome page (optional)
    // browser.tabs.create({ url: 'popup/popup.html' });
  } else if (details.reason === 'update') {
    console.log('Manga Translator updated');

    // Ensure context menu exists
    await createContextMenu();
  }
});

// Handle context menu clicks
browser.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'toggle-manga-translator' && tab?.url) {
    const url = new URL(tab.url);
    const hostname = url.hostname;
    const isEnabled = await settingsManager.isEnabledForHostname(hostname);

    if (isEnabled) {
      await settingsManager.removeActiveUrl(hostname);
      console.log(`Disabled for ${hostname}`);
    } else {
      await settingsManager.addActiveUrl(hostname);
      console.log(`Enabled for ${hostname}`);
    }

    // Update context menu title
    await updateContextMenu(hostname);

    // Notify content script
    if (tab.id) {
      await browser.tabs.sendMessage(tab.id, { action: 'toggle' });
    }
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
        console.error('Failed to fetch image:', error);
        return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
      }

    case 'translateImages':
      // Make API call to translation service
      try {
        const settings = await settingsManager.getSettings();
        const endpoint = settings.apiEndpoint || 'http://localhost:8000';

        const response = await fetch(`${endpoint}/translate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(settings.apiKey && { 'Authorization': `Bearer ${settings.apiKey}` }),
          },
          body: JSON.stringify({
            base64Images: message.base64Images,
            targetLanguage: message.targetLanguage,
          }),
        });

        if (!response.ok) {
          const error = await response.json();
          return { success: false, error: error.detail || `API request failed: ${response.status}` };
        }

        const data = await response.json();
        return { success: true, data };
      } catch (error) {
        console.error('Translation API call failed:', error);
        return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
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
    console.log(`Disabled for ${hostname}`);
  } else {
    await settingsManager.addActiveUrl(hostname);
    console.log(`Enabled for ${hostname}`);
  }

  // Notify content script
  await browser.tabs.sendMessage(tab.id, {
    action: 'toggle',
  });
});

console.log('Manga Translator: Background service worker loaded');
