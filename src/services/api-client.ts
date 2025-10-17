/**
 * API client for manga translation service with retry logic
 */
import browser from 'webextension-polyfill';
import { TranslateRequest, TranslateResponse, APIError } from '@/types/api';
import { CONFIG } from '@/config/constants';
import { settingsManager } from './settings-manager';

export class APIClient {
  private static instance: APIClient;

  private constructor() {}

  static getInstance(): APIClient {
    if (!APIClient.instance) {
      APIClient.instance = new APIClient();
    }
    return APIClient.instance;
  }

  /**
   * Translate images with automatic retry logic
   */
  async translate(
    base64Images: string[],
    targetLanguage: string
  ): Promise<TranslateResponse> {
    const request: TranslateRequest = {
      base64Images,
      targetLanguage,
    };

    return await this.executeWithRetry(async () => {
      return await this.makeTranslateRequest(request);
    });
  }

  /**
   * Make translation API request via background service worker
   * This bypasses CORS restrictions
   */
  private async makeTranslateRequest(
    request: TranslateRequest
  ): Promise<TranslateResponse> {
    try {
      // Send message to background worker to make the API call
      const response = await browser.runtime.sendMessage({
        action: 'translateImages',
        base64Images: request.base64Images,
        targetLanguage: request.targetLanguage,
      });

      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || 'API request failed');
      }
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Unknown error occurred');
    }
  }

  /**
   * Execute function with exponential backoff retry
   */
  private async executeWithRetry<T>(
    fn: () => Promise<T>,
    attempt: number = 1
  ): Promise<T> {
    try {
      return await fn();
    } catch (error) {
      if (attempt >= CONFIG.MAX_RETRY_ATTEMPTS) {
        console.error(`Failed after ${attempt} attempts:`, error);
        throw error;
      }

      // Exponential backoff: 1s, 2s, 4s, etc.
      const delay = CONFIG.RETRY_DELAY * Math.pow(2, attempt - 1);
      console.warn(`Attempt ${attempt} failed, retrying in ${delay}ms...`, error);

      await this.sleep(delay);
      return await this.executeWithRetry(fn, attempt + 1);
    }
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      const settings = await settingsManager.getSettings();
      const endpoint = settings.apiEndpoint || CONFIG.DEFAULT_API_ENDPOINT;

      const response = await fetch(`${endpoint}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });

      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
}

// Export singleton instance
export const apiClient = APIClient.getInstance();
