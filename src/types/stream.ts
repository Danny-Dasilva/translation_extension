/**
 * Versioned streaming event-frame protocol (server -> client) for progressive
 * translation delivery.
 *
 * The binary image UPLOAD (client -> server) is unchanged. The server may reply
 * in one of two shapes on the SAME websocket:
 *
 *   1. LEGACY (monolithic): a single {@link TranslateResponse} JSON object, as
 *      it does today. The client detects this (no `v` field) and resolves the
 *      request exactly as before.
 *
 *   2. STREAM (this module): a sequence of `{ v: 1, type, session_id, ... }`
 *      event frames the client applies incrementally, resolving the overall
 *      request when the terminal `done` frame arrives.
 *
 * Ordering / idempotency contract the backend must honour:
 *   - `detections` arrives FIRST (geometry only, no translated text).
 *   - `tl` frames arrive in ANY order, each addressing a box by `index`.
 *     Re-sending the same `index` is idempotent (last write wins).
 *   - `revise` optionally supersedes a prior `tl` for the same `index`.
 *   - `plate` (inpaint background) arrives AT MOST ONCE per image.
 *   - exactly one terminal frame: `done` (success) or `error`.
 *
 * A single websocket message carries ONE image's stream. `image_index` is
 * reserved for a future multi-image batch on one socket; it defaults to 0 and
 * the current one-request-one-image flow never sets it.
 */
import type { TextRegion, DebugTiming } from './api';

/** Wire version for the event-frame protocol. Bump on breaking changes. */
export const STREAM_PROTOCOL_VERSION = 1;

/**
 * Geometry-only subset of a TextBox emitted in the `detections` frame. Text
 * content (translatedText / ocrText) is intentionally absent here — it arrives
 * later via `tl` frames keyed by the same `index`. All fields except `index`
 * and the bbox are optional so a minimal backend can emit just coordinates.
 */
export interface StreamDetectionBox {
  /** Stable per-image box id used to correlate `tl`/`revise` frames. */
  index: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  originalLanguage?: string;
  /** Backend glyph-height seed for the font-fit search (see TextBox.fontHeightPx). */
  fontHeightPx?: number;
  fontColor?: string;
  fontStrokeColor?: string;
  zIndex?: number;
  /** Precise text-pixel regions for targeted masking. */
  textRegions?: TextRegion[];
  /** Matched speech-bubble interior rect (null = SFX over art, no bubble). */
  bubbleRect?: TextRegion | null;
  confidence?: number;
  /** Forward-compat: a deliberately skipped region (already-English, etc.). */
  skipped?: boolean;
}

interface StreamFrameBase {
  v: 1;
  session_id: string;
  /** Which image in a (future) multi-image batch. Defaults to 0. */
  image_index?: number;
}

/** Geometry for every detected box. Emitted first. */
export interface StreamDetectionsFrame extends StreamFrameBase {
  type: 'detections';
  boxes: StreamDetectionBox[];
}

/** A translated (and OCR'd) box. Idempotent by `index`; any order. */
export interface StreamTlFrame extends StreamFrameBase {
  type: 'tl';
  index: number;
  translatedText: string;
  ocrText?: string;
}

/** Optional correction that supersedes a prior `tl` for the same `index`. */
export interface StreamReviseFrame extends StreamFrameBase {
  type: 'revise';
  index: number;
  translatedText: string;
}

/** The inpainted background plate (base64, with or without data: prefix). ≤1 per image. */
export interface StreamPlateFrame extends StreamFrameBase {
  type: 'plate';
  data: string;
}

/** Terminal success frame. */
export interface StreamDoneFrame extends StreamFrameBase {
  type: 'done';
  debug?: {
    timing?: DebugTiming;
    total_ms?: number;
  };
}

/** Terminal failure frame. */
export interface StreamErrorFrame extends StreamFrameBase {
  type: 'error';
  error: string;
}

/** Discriminated union of all server->client event frames. */
export type StreamEventFrame =
  | StreamDetectionsFrame
  | StreamTlFrame
  | StreamReviseFrame
  | StreamPlateFrame
  | StreamDoneFrame
  | StreamErrorFrame;

/** The frame `type` values, useful for exhaustive checks. */
export type StreamEventType = StreamEventFrame['type'];
