/**
 * Settings management using Chrome storage API
 */
import browser from 'webextension-polyfill';
import { ExtensionSettings, DEFAULT_SETTINGS, HostnameConfig } from '@/types/settings';
import { CONFIG } from '@/config/constants';

export class SettingsManager {
  private static instance: SettingsManager;
  private settings: ExtensionSettings | null = null;

  private constructor() {}

  static getInstance(): SettingsManager {
    if (!SettingsManager.instance) {
      SettingsManager.instance = new SettingsManager();
    }
    return SettingsManager.instance;
  }

  /**
   * Load settings from Chrome storage
   */
  async loadSettings(): Promise<ExtensionSettings> {
    try {
      const result = await browser.storage.sync.get(CONFIG.STORAGE_KEYS.SETTINGS);
      this.settings = result[CONFIG.STORAGE_KEYS.SETTINGS] || DEFAULT_SETTINGS;
      return this.settings;
    } catch (error) {
      console.error('Failed to load settings:', error);
      this.settings = DEFAULT_SETTINGS;
      return DEFAULT_SETTINGS;
    }
  }

  /**
   * Save settings to Chrome storage
   */
  async saveSettings(settings: Partial<ExtensionSettings>): Promise<void> {
    try {
      const currentSettings = await this.getSettings();
      const updatedSettings = { ...currentSettings, ...settings };
      await browser.storage.sync.set({
        [CONFIG.STORAGE_KEYS.SETTINGS]: updatedSettings,
      });
      this.settings = updatedSettings;
    } catch (error) {
      console.error('Failed to save settings:', error);
      throw error;
    }
  }

  /**
   * Get current settings (from cache or storage)
   */
  async getSettings(): Promise<ExtensionSettings> {
    if (!this.settings) {
      return await this.loadSettings();
    }
    return this.settings;
  }

  /**
   * Check if extension is enabled for current hostname
   */
  async isEnabledForHostname(hostname: string): Promise<boolean> {
    const settings = await this.getSettings();
    
    // If no activeUrls configured, extension is disabled everywhere
    if (!settings.activeUrls || settings.activeUrls.length === 0) {
      return false;
    }
    
    // Check if hostname matches any pattern in activeUrls
    return settings.activeUrls.some(pattern => {
      // Support wildcards like "*.example.com"
      const regex = new RegExp(
        '^' + pattern.replace(/\*/g, '.*').replace(/\./g, '\\.') + '$'
      );
      return regex.test(hostname);
    });
  }

  /**
   * Add hostname to activation whitelist
   */
  async addActiveUrl(hostname: string): Promise<void> {
    const settings = await this.getSettings();
    const activeUrls = settings.activeUrls || [];
    
    if (!activeUrls.includes(hostname)) {
      activeUrls.push(hostname);
      await this.saveSettings({ activeUrls });
    }
  }

  /**
   * Remove hostname from activation whitelist
   */
  async removeActiveUrl(hostname: string): Promise<void> {
    const settings = await this.getSettings();
    const activeUrls = (settings.activeUrls || []).filter(url => url !== hostname);
    await this.saveSettings({ activeUrls });
  }

  /**
   * Get active URLs list
   */
  async getActiveUrls(): Promise<string[]> {
    const settings = await this.getSettings();
    return settings.activeUrls || [];
  }

  /**
   * Reset settings to defaults
   */
  async resetSettings(): Promise<void> {
    await browser.storage.sync.set({
      [CONFIG.STORAGE_KEYS.SETTINGS]: DEFAULT_SETTINGS,
    });
    this.settings = DEFAULT_SETTINGS;
  }

  /**
   * Listen for settings changes
   */
  onSettingsChanged(callback: (settings: ExtensionSettings) => void): void {
    browser.storage.onChanged.addListener((changes, areaName) => {
      if (areaName === 'sync' && changes[CONFIG.STORAGE_KEYS.SETTINGS]) {
        const newSettings = changes[CONFIG.STORAGE_KEYS.SETTINGS].newValue;
        this.settings = newSettings;
        callback(newSettings);
      }
    });
  }
}

// Export singleton instance
export const settingsManager = SettingsManager.getInstance();
