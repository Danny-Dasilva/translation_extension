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
  fontHeightPx: number;
  fontColor: string;
  fontStrokeColor: string;
  zIndex: number;
  translatedText: string;
  subtextBoxes: TextBox[];
  textRegions?: TextRegion[]; // Precise text regions for targeted masking
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
  debug?: {
    timing?: DebugTiming;
    total_ms?: number;
  };
}

export interface APIError {
  detail: string;
}
