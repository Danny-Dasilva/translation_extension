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
}

export interface TranslateResponse {
  images: TextBox[][];
}

export interface APIError {
  detail: string;
}
