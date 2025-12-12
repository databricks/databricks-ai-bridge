/**
 * HTTP client for Databricks Model Serving with SSE streaming support
 */

import { Config, ConfigOptions } from "@databricks/sdk-experimental";
import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  DatabricksApiError,
} from "../types.js";

export interface DatabricksHttpClientOptions {
  timeout?: number;
  maxRetries?: number;
}

export interface DatabricksHttpClient {
  /**
   * Make a non-streaming chat completion request
   */
  chatCompletion(
    endpoint: string,
    request: ChatCompletionRequest
  ): Promise<ChatCompletionResponse>;

  /**
   * Make a streaming chat completion request
   * Returns an async generator of SSE chunks
   */
  chatCompletionStream(
    endpoint: string,
    request: ChatCompletionRequest,
    signal?: AbortSignal
  ): AsyncGenerator<ChatCompletionChunk>;

  /**
   * Get the underlying Config object
   */
  getConfig(): Config;
}

/**
 * Error thrown when a Databricks API request fails
 */
export class DatabricksRequestError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly errorCode?: string,
    public readonly response?: DatabricksApiError
  ) {
    super(message);
    this.name = "DatabricksRequestError";
  }
}

/**
 * Parse SSE (Server-Sent Events) stream from response body
 */
async function* parseSSEStream(
  body: ReadableStream<Uint8Array>
): AsyncGenerator<ChatCompletionChunk> {
  const decoder = new TextDecoder();
  const reader = body.getReader();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        // Process any remaining data in buffer
        if (buffer.trim()) {
          const lines = buffer.split("\n");
          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("data: ")) {
              const data = trimmed.slice(6);
              if (data !== "[DONE]") {
                try {
                  yield JSON.parse(data) as ChatCompletionChunk;
                } catch {
                  // Ignore malformed JSON in final chunk
                }
              }
            }
          }
        }
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      // Process complete events (separated by double newline)
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        const lines = part.split("\n");
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            const data = trimmed.slice(6);
            if (data === "[DONE]") {
              return;
            }
            try {
              yield JSON.parse(data) as ChatCompletionChunk;
            } catch (e) {
              // Skip malformed JSON
              console.warn("Failed to parse SSE data:", data, e);
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Create a Databricks HTTP client for chat completions
 */
export function createDatabricksHttpClient(
  configOrOptions: Config | ConfigOptions,
  options: DatabricksHttpClientOptions = {}
): DatabricksHttpClient {
  const config =
    configOrOptions instanceof Config ? configOrOptions : new Config(configOrOptions);

  const { timeout = 60000, maxRetries = 2 } = options;

  async function makeRequest(
    endpoint: string,
    request: ChatCompletionRequest,
    stream: boolean
  ): Promise<Response> {
    await config.ensureResolved();
    const host = await config.getHost();

    const url = new URL(`/serving-endpoints/${endpoint}/invocations`, host);

    const headers = new Headers({
      "Content-Type": "application/json",
      Accept: stream ? "text/event-stream" : "application/json",
    });

    // Apply authentication
    await config.authenticate(headers);

    const body = JSON.stringify({
      ...request,
      stream,
    });

    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
          const response = await fetch(url.toString(), {
            method: "POST",
            headers,
            body,
            signal: controller.signal,
          });

          clearTimeout(timeoutId);

          if (!response.ok) {
            const errorBody = await response.text();
            let errorData: DatabricksApiError;
            try {
              errorData = JSON.parse(errorBody);
            } catch {
              errorData = { message: errorBody };
            }

            const errorMessage =
              errorData.error?.message ||
              errorData.message ||
              `Request failed with status ${response.status}`;
            const errorCode = errorData.error?.code || errorData.error_code;

            // Don't retry 4xx errors (except 429)
            if (response.status >= 400 && response.status < 500 && response.status !== 429) {
              throw new DatabricksRequestError(
                errorMessage,
                response.status,
                errorCode,
                errorData
              );
            }

            // Retry 5xx errors and 429
            lastError = new DatabricksRequestError(
              errorMessage,
              response.status,
              errorCode,
              errorData
            );
          } else {
            return response;
          }
        } catch (e) {
          clearTimeout(timeoutId);
          if (e instanceof DatabricksRequestError) {
            throw e;
          }
          lastError = e as Error;
          // Retry on network errors
        }
      } catch (e) {
        if (e instanceof DatabricksRequestError && e.status && e.status >= 400 && e.status < 500) {
          throw e;
        }
        lastError = e as Error;
      }

      // Wait before retry with exponential backoff
      if (attempt < maxRetries) {
        await new Promise((resolve) => setTimeout(resolve, Math.pow(2, attempt) * 1000));
      }
    }

    throw lastError || new Error("Request failed after retries");
  }

  return {
    async chatCompletion(
      endpoint: string,
      request: ChatCompletionRequest
    ): Promise<ChatCompletionResponse> {
      const response = await makeRequest(endpoint, request, false);
      return (await response.json()) as ChatCompletionResponse;
    },

    async *chatCompletionStream(
      endpoint: string,
      request: ChatCompletionRequest,
      signal?: AbortSignal
    ): AsyncGenerator<ChatCompletionChunk> {
      const response = await makeRequest(endpoint, request, true);

      if (!response.body) {
        throw new DatabricksRequestError("Response body is empty for streaming request");
      }

      for await (const chunk of parseSSEStream(response.body)) {
        if (signal?.aborted) {
          throw new Error("AbortError: Request was aborted");
        }
        yield chunk;
      }
    },

    getConfig(): Config {
      return config;
    },
  };
}
