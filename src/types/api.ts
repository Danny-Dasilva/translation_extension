/**
 * API request and response types matching backend models
 */

export interface TranslateRequest {
  // Can be either base64 data URLs or regular image URLs (for CORS-blocked images)
  base64Images: string[];
  targetLanguage: string;
}

export interface TextRegion {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface TextBox {
  ocrText: string;
  originalLanguage: string;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  background: string; // base64
  /**
   * Backend-estimated glyph height in pixels for this block. Used by the
   * renderer as the initial seed / soft cap for the font-fit binary search
   * (see findBestFit). Optional + may be 0 on older responses, in which case
   * the renderer falls back to a pure 8..72 search.
   */
  fontHeightPx: number;
  fontColor: string;
  fontStrokeColor: string;
  zIndex: number;
  translatedText: string;
  subtextBoxes: TextBox[];
  textRegions?: TextRegion[]; // Precise text regions for targeted masking
  bubbleRect?: TextRegion | null; // Speech-bubble interior rect this block matched (null = no qualifying bubble, e.g. SFX over art)
  confidence?: number; // Detection confidence
  ocrTimeMs?: number; // OCR timing
  translateTimeMs?: number; // Translation timing
}

export interface DebugTiming {
  preprocess_ms?: number;
  detection_ms?: number;
  crop_ms?: number;
  ocr_ms?: number;
  translation_ms?: number;
  text_extract_ms?: number;
  request_total_ms?: number;
  ws_frame_bytes?: number;
}

export interface TranslateResponse {
  success?: boolean;
  session_id?: string;
  images: TextBox[][];
  /**
   * Optional per-image inpainted "plate" image (base64 data URL or raw base64 PNG).
   * When present, the frontend may use it as the background behind translated text
   * instead of masking the original image with white rectangles.
   * Index aligns with `images[]`.
   */
  inpainted_image_base64?: (string | null)[];
  debug?: {
    timing?: DebugTiming;
    total_ms?: number;
  };
}

export interface APIError {
  detail: string;
}
