/**
 * Lightweight async logger.
 *
 * Goal: logging must NEVER block the translate/render hot path. All log calls
 * are O(1) (push to an in-memory buffer) and the actual `console.*` work is
 * deferred to a microtask / idle callback / setTimeout(0), so a burst of logs
 * inside a tight loop (e.g. per-text-box rendering) costs only array pushes on
 * the critical path.
 *
 * Levels: debug | info | warn | error. Verbose levels (debug/info) are gated by
 * the `showDebugInfo` setting — when verbose logging is OFF they are dropped at
 * the call site (no buffering, no flush) so there is zero overhead. warn/error
 * always pass through.
 *
 * Works in both the content-script / overlay (DOM) context and the
 * service-worker context — `requestIdleCallback` is feature-detected and we
 * fall back to `setTimeout(0)`.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  level: LogLevel;
  args: unknown[];
  ts: number;
}

const LEVEL_RANK: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};

// warn/error are always emitted; debug/info require verbose mode.
const VERBOSE_THRESHOLD = LEVEL_RANK.warn;

/**
 * Schedule `cb` to run off the critical path. Prefers requestIdleCallback
 * (when the page is idle), falls back to queueMicrotask, then setTimeout(0).
 * All three guarantee the current synchronous frame finishes first.
 */
function scheduleIdle(cb: () => void): void {
  const ric = (globalThis as any).requestIdleCallback as
    | ((cb: () => void, opts?: { timeout: number }) => number)
    | undefined;
  if (typeof ric === 'function') {
    ric(cb, { timeout: 1000 });
    return;
  }
  if (typeof queueMicrotask === 'function') {
    queueMicrotask(cb);
    return;
  }
  setTimeout(cb, 0);
}

export class AsyncLogger {
  private static instance: AsyncLogger;

  /** Pending entries waiting to be flushed to the console. */
  private buffer: LogEntry[] = [];
  /** True once a flush has been scheduled but not yet run. */
  private flushScheduled = false;
  /** When true, debug/info are emitted; otherwise dropped at the call site. */
  private verbose = false;
  /** Prefix prepended to every console line. */
  private prefix = '[Manga Translator]';

  /**
   * Optional sink for warn/error entries that should be forwarded somewhere
   * (e.g. the backend). Fire-and-forget — never awaited on the hot path.
   */
  private remoteSink: ((entries: LogEntry[]) => void) | null = null;
  /** Buffer of warn/error entries pending remote forward. */
  private remoteBuffer: LogEntry[] = [];
  private remoteFlushScheduled = false;

  private constructor() {}

  static getInstance(): AsyncLogger {
    if (!AsyncLogger.instance) {
      AsyncLogger.instance = new AsyncLogger();
    }
    return AsyncLogger.instance;
  }

  /**
   * Enable/disable verbose (debug/info) logging. Wire this to the
   * `showDebugInfo` setting. warn/error are unaffected.
   */
  setVerbose(verbose: boolean): void {
    this.verbose = verbose;
  }

  /** Override the per-line prefix (e.g. "[MT/sw]" vs "[MT/content]"). */
  setPrefix(prefix: string): void {
    this.prefix = prefix;
  }

  /**
   * Register a fire-and-forget sink for warn/error entries. The sink is invoked
   * off the critical path with a batch of pending entries. Pass `null` to
   * disable forwarding.
   */
  setRemoteSink(sink: ((entries: LogEntry[]) => void) | null): void {
    this.remoteSink = sink;
  }

  debug(...args: unknown[]): void {
    this.enqueue('debug', args);
  }

  info(...args: unknown[]): void {
    this.enqueue('info', args);
  }

  warn(...args: unknown[]): void {
    this.enqueue('warn', args);
  }

  error(...args: unknown[]): void {
    this.enqueue('error', args);
  }

  /**
   * O(1) on the hot path: gate verbose levels, push to the buffer, and ensure a
   * single deferred flush is scheduled. The actual console work happens later.
   */
  private enqueue(level: LogLevel, args: unknown[]): void {
    const rank = LEVEL_RANK[level];
    // Drop verbose logs entirely when not in verbose mode — no buffering cost.
    if (rank < VERBOSE_THRESHOLD && !this.verbose) return;

    this.buffer.push({ level, args, ts: Date.now() });

    // Queue warn/error for optional remote forwarding (separate buffer so the
    // console flush and the remote flush are independent).
    if (rank >= VERBOSE_THRESHOLD && this.remoteSink) {
      this.remoteBuffer.push({ level, args, ts: Date.now() });
      this.scheduleRemoteFlush();
    }

    if (!this.flushScheduled) {
      this.flushScheduled = true;
      scheduleIdle(() => this.flush());
    }
  }

  /** Drain the console buffer. Runs off the critical path. */
  private flush(): void {
    this.flushScheduled = false;
    const entries = this.buffer;
    this.buffer = [];

    for (const entry of entries) {
      const fn =
        entry.level === 'error'
          ? console.error
          : entry.level === 'warn'
            ? console.warn
            : entry.level === 'info'
              ? console.info
              : console.debug;
      try {
        fn(this.prefix, ...entry.args);
      } catch {
        // Never let a logging failure surface on the app path.
      }
    }
  }

  private scheduleRemoteFlush(): void {
    if (this.remoteFlushScheduled) return;
    this.remoteFlushScheduled = true;
    scheduleIdle(() => this.flushRemote());
  }

  /** Fire-and-forget the queued warn/error entries to the remote sink. */
  private flushRemote(): void {
    this.remoteFlushScheduled = false;
    if (!this.remoteSink || this.remoteBuffer.length === 0) return;
    const entries = this.remoteBuffer;
    this.remoteBuffer = [];
    try {
      this.remoteSink(entries);
    } catch {
      // Swallow — remote logging is best-effort.
    }
  }
}

/** Shared singleton logger. */
export const logger = AsyncLogger.getInstance();
