/**
 * Settings and configuration types
 */

export interface ExtensionSettings {
  // API Configuration
  apiEndpoint: string;
  apiKey?: string;
  
  // Translation Settings
  targetLanguage: string;
  autoTranslate: boolean;
  
  // Per-hostname activation whitelist
  activeUrls: string[]; // List of hostnames where extension is enabled
  
  // Font Settings
  defaultFont: FontName;
  customFontColor?: string;
  customStrokeColor?: string;
  
  // Feature Flags
  useCache: boolean;
  showLoadingIndicator: boolean;
  
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
  apiEndpoint: 'http://localhost:8000',
  targetLanguage: 'English',
  autoTranslate: false,
  activeUrls: [],
  defaultFont: 'Bangers',
  useCache: true,
  showLoadingIndicator: true,
  isPremium: false,
};

export interface HostnameConfig {
  hostname: string;
  enabled: boolean;
  refererRule?: string; // Optional custom referer header
}
