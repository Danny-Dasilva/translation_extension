/**
 * Canvas monitoring service for detecting changes via hash checking
 */
import { hashCanvas } from '@/utils/image-utils';
import { CONFIG } from '@/config/constants';

interface CanvasState {
  canvas: HTMLCanvasElement;
  hash: string;
  lastChecked: number;
}

export class CanvasMonitor {
  private canvasStates: Map<HTMLCanvasElement, CanvasState> = new Map();
  private intervalId: number | null = null;
  private onChangeCallbacks: Set<(canvas: HTMLCanvasElement) => void> = new Set();

  /**
   * Start monitoring canvases
   */
  start(): void {
    if (this.intervalId !== null) {
      return; // Already running
    }

    this.intervalId = window.setInterval(() => {
      this.checkAllCanvases();
    }, CONFIG.CANVAS_CHECK_INTERVAL);

    console.log('Canvas monitor started');
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    console.log('Canvas monitor stopped');
  }

  /**
   * Add canvas to monitor
   */
  addCanvas(canvas: HTMLCanvasElement): void {
    if (this.canvasStates.has(canvas)) {
      return;
    }

    const hash = hashCanvas(canvas);
    this.canvasStates.set(canvas, {
      canvas,
      hash,
      lastChecked: Date.now(),
    });
  }

  /**
   * Remove canvas from monitoring
   */
  removeCanvas(canvas: HTMLCanvasElement): void {
    this.canvasStates.delete(canvas);
  }

  /**
   * Check all monitored canvases for changes
   */
  private checkAllCanvases(): void {
    const now = Date.now();

    for (const [canvas, state] of this.canvasStates.entries()) {
      // Skip if not in DOM anymore
      if (!document.contains(canvas)) {
        this.canvasStates.delete(canvas);
        continue;
      }

      try {
        const currentHash = hashCanvas(canvas);
        
        if (currentHash !== state.hash) {
          console.log('Canvas content changed, triggering update');
          state.hash = currentHash;
          state.lastChecked = now;

          // Notify callbacks
          this.onChangeCallbacks.forEach(callback => callback(canvas));
        } else {
          state.lastChecked = now;
        }
      } catch (error) {
        console.error('Error checking canvas:', error);
      }
    }
  }

  /**
   * Register callback for canvas changes
   */
  onChange(callback: (canvas: HTMLCanvasElement) => void): void {
    this.onChangeCallbacks.add(callback);
  }

  /**
   * Unregister callback
   */
  offChange(callback: (canvas: HTMLCanvasElement) => void): void {
    this.onChangeCallbacks.delete(callback);
  }

  /**
   * Get monitored canvas count
   */
  getMonitoredCount(): number {
    return this.canvasStates.size;
  }

  /**
   * Clear all monitored canvases
   */
  clear(): void {
    this.canvasStates.clear();
  }

  /**
   * Reset monitor
   */
  reset(): void {
    this.stop();
    this.clear();
    this.onChangeCallbacks.clear();
  }
}

// Export singleton instance
export const canvasMonitor = new CanvasMonitor();
