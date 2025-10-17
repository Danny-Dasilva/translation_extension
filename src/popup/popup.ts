/**
 * Popup script for extension settings
 */
import browser from 'webextension-polyfill';
import { settingsManager } from '@/services/settings-manager';

// Get current tab hostname
async function getCurrentHostname(): Promise<string | null> {
  const tabs = await browser.tabs.query({ active: true, currentWindow: true });
  if (tabs[0]?.url) {
    const url = new URL(tabs[0].url);
    return url.hostname;
  }
  return null;
}

// Update toggle button state
async function updateToggleButton() {
  const hostname = await getCurrentHostname();
  if (!hostname) return;

  const toggleButton = document.getElementById('toggle-site') as HTMLButtonElement;
  const isEnabled = await settingsManager.isEnabledForHostname(hostname);

  if (toggleButton) {
    if (isEnabled) {
      toggleButton.textContent = `Disable for ${hostname}`;
      toggleButton.className = 'toggle-disable';
    } else {
      toggleButton.textContent = `Enable for ${hostname}`;
      toggleButton.className = 'toggle-enable';
    }
  }
}

// Toggle extension for current site
async function toggleSite() {
  const hostname = await getCurrentHostname();
  if (!hostname) return;

  const isEnabled = await settingsManager.isEnabledForHostname(hostname);

  if (isEnabled) {
    await settingsManager.removeActiveUrl(hostname);
    showStatus('disabled', `Disabled for ${hostname}`);
  } else {
    await settingsManager.addActiveUrl(hostname);
    showStatus('enabled', `Enabled for ${hostname}`);
  }

  // Update button
  await updateToggleButton();

  // Notify content script
  const tabs = await browser.tabs.query({ active: true, currentWindow: true });
  if (tabs[0]?.id) {
    await browser.tabs.sendMessage(tabs[0].id, { action: 'toggle' });
  }

  setTimeout(() => hideStatus(), 2000);
}

// Load current settings
async function loadSettings() {
  const settings = await settingsManager.getSettings();

  const apiEndpointInput = document.getElementById('api-endpoint') as HTMLInputElement;
  const targetLanguageSelect = document.getElementById('target-language') as HTMLSelectElement;
  const defaultFontSelect = document.getElementById('default-font') as HTMLSelectElement;
  const autoTranslateCheckbox = document.getElementById('auto-translate') as HTMLInputElement;

  if (apiEndpointInput) apiEndpointInput.value = settings.apiEndpoint;
  if (targetLanguageSelect) targetLanguageSelect.value = settings.targetLanguage;
  if (defaultFontSelect) defaultFontSelect.value = settings.defaultFont;
  if (autoTranslateCheckbox) autoTranslateCheckbox.checked = settings.autoTranslate;

  // Update toggle button
  await updateToggleButton();
}

// Save settings
async function saveSettings() {
  const apiEndpointInput = document.getElementById('api-endpoint') as HTMLInputElement;
  const targetLanguageSelect = document.getElementById('target-language') as HTMLSelectElement;
  const defaultFontSelect = document.getElementById('default-font') as HTMLSelectElement;
  const autoTranslateCheckbox = document.getElementById('auto-translate') as HTMLInputElement;

  await settingsManager.saveSettings({
    apiEndpoint: apiEndpointInput.value,
    targetLanguage: targetLanguageSelect.value,
    defaultFont: defaultFontSelect.value as any,
    autoTranslate: autoTranslateCheckbox.checked,
  });

  showStatus('saved', 'Settings saved!');
  setTimeout(() => hideStatus(), 2000);
}

// Translate current page
async function translateCurrentPage() {
  const hostname = await getCurrentHostname();
  if (!hostname) {
    showStatus('error', 'Unable to get current page');
    setTimeout(() => hideStatus(), 3000);
    return;
  }

  // Check if enabled, auto-enable if not
  const isEnabled = await settingsManager.isEnabledForHostname(hostname);
  if (!isEnabled) {
    await settingsManager.addActiveUrl(hostname);
    await updateToggleButton();
    showStatus('enabled', `Enabled and translating ${hostname}...`);
  } else {
    showStatus('enabled', 'Translating...');
  }

  const tabs = await browser.tabs.query({ active: true, currentWindow: true });
  if (tabs[0]?.id) {
    try {
      const response = await browser.runtime.sendMessage({
        action: 'translate',
        tabId: tabs[0].id,
      });

      if (response && response.success) {
        showStatus('enabled', 'Translation started!');
      } else {
        showStatus('error', response?.error || 'Translation failed');
      }
    } catch (error) {
      console.error('Translation error:', error);
      showStatus('error', 'Translation failed - check console');
    }

    setTimeout(() => hideStatus(), 3000);
  }
}

// Clear overlays
async function clearOverlays() {
  const tabs = await browser.tabs.query({ active: true, currentWindow: true });
  if (tabs[0]?.id) {
    await browser.runtime.sendMessage({ action: 'clear', tabId: tabs[0].id });
    showStatus('disabled', 'Overlays cleared!');
    setTimeout(() => hideStatus(), 2000);
  }
}

// Show status message
function showStatus(type: 'enabled' | 'disabled' | 'saved' | 'error', message: string) {
  const statusDiv = document.getElementById('status');
  if (statusDiv) {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    statusDiv.style.display = 'block';
  }
}

// Hide status message
function hideStatus() {
  const statusDiv = document.getElementById('status');
  if (statusDiv) {
    statusDiv.style.display = 'none';
  }
}

// Initialize popup
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();

  const toggleButton = document.getElementById('toggle-site');
  const saveButton = document.getElementById('save-settings');
  const translateButton = document.getElementById('translate-now');
  const clearButton = document.getElementById('clear-overlays');

  if (toggleButton) {
    toggleButton.addEventListener('click', toggleSite);
  }

  if (saveButton) {
    saveButton.addEventListener('click', saveSettings);
  }

  if (translateButton) {
    translateButton.addEventListener('click', translateCurrentPage);
  }

  if (clearButton) {
    clearButton.addEventListener('click', clearOverlays);
  }
});
