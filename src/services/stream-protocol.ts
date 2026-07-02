/**
 * Pure, dependency-free helpers for the versioned streaming event protocol.
 *
 * This module deliberately imports ONLY types (erased at build/run time) so it
 * can be unit-tested in isolation under Node's native TS type-stripping without
 * pulling in the extension/browser runtime. See scripts/test_stream_protocol.mjs.
 *
 *   - Mode detection: tell a legacy monolithic {@link TranslateResponse} apart
 *     from a v:1 {@link StreamEventFrame}.
 *   - Detection-box -> TextBox hydration (geometry now, text filled later).
 *   - {@link StreamAssembler}: an order-independent, idempotent state machine
 *     that folds a sequence of event frames into a legacy-shaped response so the
 *     overall promise can resolve with the same object callers already handle.
 */
import type { TextBox, TranslateResponse } from '@/types/api';
import type {
  StreamDetectionBox,
  StreamEventFrame,
  StreamDoneFrame,
  StreamErrorFrame,
} from '@/types/stream';

/**
 * True when `msg` is a v:1 streaming event frame (as opposed to a legacy
 * monolithic response). We key off the numeric `v` discriminator plus a string
 * `type`; a legacy TranslateResponse has neither.
 */
export function isStreamEventFrame(msg: unknown): msg is StreamEventFrame {
  if (!msg || typeof msg !== 'object') return false;
  const m = msg as Record<string, unknown>;
  return m.v === 1 && typeof m.type === 'string';
}

/** True for the terminal frames that close a stream. */
export function isTerminalFrame(
  frame: StreamEventFrame
): frame is StreamDoneFrame | StreamErrorFrame {
  return frame.type === 'done' || frame.type === 'error';
}

/**
 * Hydrate a geometry-only detection box into a full TextBox with empty text.
 * Mirrors the fields the renderer reads; unknowns default to renderer-safe
 * values (empty strings, 0) so the existing single-pass painters work unchanged
 * once translatedText is later filled in.
 */
export function detectionBoxToTextBox(box: StreamDetectionBox): TextBox {
  return {
    ocrText: '',
    originalLanguage: box.originalLanguage ?? '',
    minX: box.minX,
    minY: box.minY,
    maxX: box.maxX,
    maxY: box.maxY,
    background: '',
    fontHeightPx: box.fontHeightPx ?? 0,
    fontColor: box.fontColor ?? '',
    fontStrokeColor: box.fontStrokeColor ?? '',
    zIndex: box.zIndex ?? 0,
    translatedText: '',
    subtextBoxes: [],
    textRegions: box.textRegions,
    bubbleRect: box.bubbleRect ?? null,
    confidence: box.confidence,
    skipped: box.skipped,
  };
}

/**
 * Folds a single image's event stream into TextBox[] + plate + debug, producing
 * a legacy-shaped {@link TranslateResponse} on demand. Idempotent and
 * order-independent: `tl`/`revise` frames are applied by box `index`, later
 * writes win, and a `tl` that arrives before its `detections` box is buffered
 * and reconciled once geometry appears.
 */
export class StreamAssembler {
  private boxesByIndex = new Map<number, TextBox>();
  /** Insertion order of detection boxes, so images[] preserves detect order. */
  private order: number[] = [];
  /** tl/revise text that arrived before the matching detection box. */
  private pendingText = new Map<number, { translatedText: string; ocrText?: string }>();
  private plate: string | null = null;
  private debug: TranslateResponse['debug'] | undefined;
  private sessionId: string | undefined;
  private done = false;
  private errored: string | null = null;

  /** Session id from the first frame that carried one. */
  get session(): string | undefined {
    return this.sessionId;
  }

  /** True once a terminal `done` frame has been applied. */
  get isDone(): boolean {
    return this.done;
  }

  /** The error message if a terminal `error` frame was applied, else null. */
  get error(): string | null {
    return this.errored;
  }

  /**
   * Apply one event frame to the accumulated state. Safe to call repeatedly
   * with the same frame (idempotent). Returns the frame for call-site chaining.
   */
  apply(frame: StreamEventFrame): StreamEventFrame {
    if (frame.session_id && !this.sessionId) this.sessionId = frame.session_id;

    switch (frame.type) {
      case 'detections': {
        for (const b of frame.boxes) {
          if (!this.boxesByIndex.has(b.index)) this.order.push(b.index);
          const tb = detectionBoxToTextBox(b);
          // Reconcile any text that raced ahead of its geometry.
          const pending = this.pendingText.get(b.index);
          if (pending) {
            tb.translatedText = pending.translatedText;
            if (pending.ocrText !== undefined) tb.ocrText = pending.ocrText;
            this.pendingText.delete(b.index);
          }
          this.boxesByIndex.set(b.index, tb);
        }
        break;
      }
      case 'tl':
      case 'revise': {
        const existing = this.boxesByIndex.get(frame.index);
        if (existing) {
          existing.translatedText = frame.translatedText;
          if (frame.type === 'tl' && frame.ocrText !== undefined) {
            existing.ocrText = frame.ocrText;
          }
        } else {
          // Buffer until the detection box arrives.
          const prior = this.pendingText.get(frame.index);
          this.pendingText.set(frame.index, {
            translatedText: frame.translatedText,
            ocrText:
              frame.type === 'tl' && frame.ocrText !== undefined
                ? frame.ocrText
                : prior?.ocrText,
          });
        }
        break;
      }
      case 'plate':
        this.plate = frame.data;
        break;
      case 'done':
        this.debug = frame.debug;
        this.done = true;
        break;
      case 'error':
        this.errored = frame.error;
        this.done = true;
        break;
    }
    return frame;
  }

  /** TextBoxes in detection order (geometry + whatever text has arrived). */
  textBoxes(): TextBox[] {
    return this.order.map((i) => this.boxesByIndex.get(i)!).filter(Boolean);
  }

  /** The inpaint plate base64 if a `plate` frame was seen, else null. */
  plateBase64(): string | null {
    return this.plate;
  }

  /**
   * Assemble a legacy-shaped response so `WebSocketClient.send` can resolve with
   * the exact object the non-streaming path returns. `images` is a single-image
   * array-of-arrays to match the batch shape callers index with `images[i]`.
   */
  toResponse(): TranslateResponse {
    return {
      success: this.errored ? false : true,
      session_id: this.sessionId,
      images: [this.textBoxes()],
      inpainted_image_base64: [this.plate],
      debug: this.debug,
    };
  }
}
