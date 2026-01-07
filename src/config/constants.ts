/**
 * Global constants and configuration
 */

export const CONFIG = {
  // API
  DEFAULT_API_ENDPOINT: 'http://100.97.223.100:8000',
  WS_ENDPOINT: 'ws://100.97.223.100:8000',
  API_TIMEOUT: 30000, // 30 seconds
  MAX_RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second
  
  // Image Processing
  MAX_IMAGES_PER_REQUEST: 5,
  MAX_IMAGE_SIZE_MB: 2,
  
  // Polling and Observers
  CANVAS_CHECK_INTERVAL: 1000, // 1 second (matching existing extension)
  RESIZE_DEBOUNCE: 300, // milliseconds
  
  // Storage Keys
  STORAGE_KEYS: {
    SETTINGS: 'manga_translator_settings',
    ACTIVE_URLS: 'manga_translator_active_urls',
    CACHE: 'manga_translator_cache',
    AUTH_TOKEN: 'manga_translator_auth_token',
  },
  
  // CSS Classes
  CSS_CLASSES: {
    OVERLAY_CONTAINER: 'manga-translator-overlay',
    TEXT_BOX: 'manga-translator-text-box',
    LOADING: 'manga-translator-loading',
    ERROR: 'manga-translator-error',
  },
  
  // Font Settings
  FONTS: [
    'Bangers',
    'Kalam',
    'Komika Jam',
    'Komika Slim',
    'VTC Letterer Pro',
    'CC Wild Words',
  ] as const,
  
  DEFAULT_FONT_SIZE_RANGE: {
    MIN: 12,
    MAX: 50,
  },
} as const;

export const SUPPORTED_LANGUAGES = [
  'English',
  'Spanish',
  'French',
  'German',
  'Italian',
  'Portuguese',
  'Russian',
  'Chinese',
  'Korean',
] as const;

export type SupportedLanguage = typeof SUPPORTED_LANGUAGES[number];
