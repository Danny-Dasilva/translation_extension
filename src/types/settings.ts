/**
 * Settings and configuration types
 */
import { CONFIG } from '@/config/constants';

export interface ExtensionSettings {
  // API Configuration
  apiEndpoint: string;
  apiKey?: string;
  
  // Translation Settings
  targetLanguage: string;
  autoTranslate: boolean;

  // Master ON/OFF switch. When false, the content script does not auto-translate
  // and clears/restores any existing overlays (independent of per-hostname
  // activation and the Alt-hold "peek original" behavior).
  translationEnabled: boolean;

  // Per-hostname activation whitelist
  activeUrls: string[]; // List of hostnames where extension is enabled
  
  // Font Settings
  defaultFont: FontName;
  customFontColor?: string;
  customStrokeColor?: string;
  
  // Feature Flags
  useCache: boolean;
  showLoadingIndicator: boolean;
  showDebugInfo: boolean; // Show detection boxes and timing info

  /**
   * Progressive (streaming) rendering. When true (default) the content script
   * requests translations over a long-lived Port and paints detections /
   * bubbles / plate incrementally as the backend streams v:1 event frames. When
   * false, or when the backend replies with a legacy monolithic response, it
   * falls back to a single-pass render. Backend support is auto-detected per
   * response, so disabling this only skips the Port path.
   */
  streamingEnabled: boolean;

  /**
   * Speculative prefetch of predicted next-page images. When true (default) the
   * PrefetchManager translates images approaching the viewport ahead of time and
   * caches them in the service worker so they render instantly on display.
   */
  prefetchEnabled: boolean;

  // Premium/Auth (optional)
  isPremium: boolean;
  authToken?: string;
}

export type FontName = 
  | 'Bangers'
  | 'Kalam'
  | 'Komika Jam'
  | 'Komika Slim'
  | 'VTC Letterer Pro'
  | 'CC Wild Words';

export const DEFAULT_SETTINGS: ExtensionSettings = {
  apiEndpoint: CONFIG.DEFAULT_API_ENDPOINT,
  targetLanguage: 'English',
  autoTranslate: false,
  translationEnabled: true,
  activeUrls: [],
  defaultFont: 'Bangers',
  useCache: true,
  showLoadingIndicator: true,
  showDebugInfo: false,
  streamingEnabled: true,
  prefetchEnabled: true,
  isPremium: false,
};

export interface HostnameConfig {
  hostname: string;
  enabled: boolean;
  refererRule?: string; // Optional custom referer header
}
