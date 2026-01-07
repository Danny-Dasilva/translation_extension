/**
 * WebSocket client for manga translation service
 * Provides binary image upload for faster translation compared to HTTP/base64
 */
import { TranslateResponse } from '@/types/api';
import { CONFIG } from '@/config/constants';

interface PendingRequest {
  resolve: (response: TranslateResponse) => void;
  reject: (error: Error) => void;
  timeout: ReturnType<typeof setTimeout>;
}

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private pendingRequest: PendingRequest | null = null;
  private currentLanguage: string = 'English';
  private connectionPromise: Promise<void> | null = null;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 3;

  /**
   * Check if WebSocket is connected and ready
   */
  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Connect to WebSocket server
   */
  async connect(targetLanguage: string): Promise<void> {
    // If already connected with same language, reuse connection
    if (this.isConnected && this.currentLanguage === targetLanguage) {
      return;
    }

    // If connecting, wait for existing connection
    if (this.connectionPromise) {
      await this.connectionPromise;
      // Check if language matches after connection
      if (this.currentLanguage === targetLanguage) {
        return;
      }
      // Otherwise disconnect and reconnect with new language
      await this.disconnect();
    }

    this.currentLanguage = targetLanguage;
    this.connectionPromise = this.createConnection(targetLanguage);

    try {
      await this.connectionPromise;
    } finally {
      this.connectionPromise = null;
    }
  }

  /**
   * Create WebSocket connection
   */
  private createConnection(targetLanguage: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = `${CONFIG.WS_ENDPOINT}/ws/translate/${encodeURIComponent(targetLanguage)}`;
      console.log(`[WebSocket] Connecting to ${wsUrl}`);

      this.ws = new WebSocket(wsUrl);

      const connectionTimeout = setTimeout(() => {
        if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
          this.ws.close();
          reject(new Error('WebSocket connection timeout'));
        }
      }, 10000); // 10 second timeout

      this.ws.onopen = () => {
        clearTimeout(connectionTimeout);
        console.log('[WebSocket] Connected');
        this.reconnectAttempts = 0;
        resolve();
      };

      this.ws.onmessage = (event: MessageEvent) => {
        this.handleMessage(event);
      };

      this.ws.onerror = (event: Event) => {
        clearTimeout(connectionTimeout);
        console.error('[WebSocket] Error:', event);
        this.handleError(event);
      };

      this.ws.onclose = (event: CloseEvent) => {
        clearTimeout(connectionTimeout);
        console.log(`[WebSocket] Closed: code=${event.code}, reason=${event.reason}`);
        this.handleClose(event);
        if (!this.isConnected) {
          reject(new Error(`WebSocket closed: ${event.reason || 'Unknown reason'}`));
        }
      };
    });
  }

  /**
   * Send binary image data and receive translation response
   */
  async send(imageData: ArrayBuffer): Promise<TranslateResponse> {
    if (!this.isConnected) {
      throw new Error('WebSocket not connected');
    }

    // Only one request at a time (WebSocket returns responses in order)
    if (this.pendingRequest) {
      throw new Error('Request already in progress');
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        if (this.pendingRequest) {
          this.pendingRequest = null;
          reject(new Error('WebSocket request timeout'));
        }
      }, CONFIG.API_TIMEOUT);

      this.pendingRequest = { resolve, reject, timeout };

      try {
        this.ws!.send(imageData);
        console.log(`[WebSocket] Sent ${imageData.byteLength} bytes`);
      } catch (error) {
        clearTimeout(timeout);
        this.pendingRequest = null;
        reject(error);
      }
    });
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleMessage(event: MessageEvent): void {
    if (!this.pendingRequest) {
      console.warn('[WebSocket] Received message but no pending request');
      return;
    }

    const { resolve, reject, timeout } = this.pendingRequest;
    this.pendingRequest = null;
    clearTimeout(timeout);

    try {
      const response = JSON.parse(event.data) as TranslateResponse;

      if (response.success === false) {
        reject(new Error((response as any).error || 'Translation failed'));
        return;
      }

      console.log(`[WebSocket] Received response with ${response.images?.[0]?.length || 0} text boxes`);
      resolve(response);
    } catch (error) {
      reject(new Error(`Failed to parse WebSocket response: ${error}`));
    }
  }

  /**
   * Handle WebSocket error
   */
  private handleError(event: Event): void {
    console.error('[WebSocket] Connection error');

    // Reject pending request if any
    if (this.pendingRequest) {
      const { reject, timeout } = this.pendingRequest;
      this.pendingRequest = null;
      clearTimeout(timeout);
      reject(new Error('WebSocket connection error'));
    }
  }

  /**
   * Handle WebSocket close
   */
  private handleClose(event: CloseEvent): void {
    this.ws = null;

    // Reject pending request if any
    if (this.pendingRequest) {
      const { reject, timeout } = this.pendingRequest;
      this.pendingRequest = null;
      clearTimeout(timeout);
      reject(new Error(`WebSocket closed unexpectedly: ${event.reason || 'Unknown'}`));
    }
  }

  /**
   * Disconnect WebSocket
   */
  async disconnect(): Promise<void> {
    if (this.ws) {
      // Clear pending request
      if (this.pendingRequest) {
        const { reject, timeout } = this.pendingRequest;
        this.pendingRequest = null;
        clearTimeout(timeout);
        reject(new Error('WebSocket disconnected'));
      }

      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  /**
   * Check if should attempt reconnection
   */
  shouldReconnect(): boolean {
    return this.reconnectAttempts < this.maxReconnectAttempts;
  }

  /**
   * Increment reconnect counter
   */
  incrementReconnectAttempts(): void {
    this.reconnectAttempts++;
  }
}

// Export singleton instance
export const webSocketClient = new WebSocketClient();
