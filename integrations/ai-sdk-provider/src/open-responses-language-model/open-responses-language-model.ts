import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3FinishReason,
} from '@ai-sdk/provider';
import { convertToOpenResponsesInput } from './open-responses-convert-to-input';
import { createOpenResponsesStreamTransformer } from './open-responses-stream-transformer';

interface OpenResponsesLanguageModelConfig {
  modelId: string;
  baseUrl: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
}

/**
 * Language model implementation for OpenResponses format
 * OpenResponses is used by declarative agent apps built with Databricks Agent Framework
 *
 * Specification: https://github.com/databricks/openresponses
 */
export class OpenResponsesLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = 'v3';
  readonly provider = 'databricks';
  readonly modelId: string;
  readonly defaultObjectGenerationMode = undefined;

  private readonly config: OpenResponsesLanguageModelConfig;

  constructor(config: OpenResponsesLanguageModelConfig) {
    this.modelId = config.modelId;
    this.config = config;
  }

  async doGenerate(
    options: LanguageModelV3CallOptions,
  ): Promise<{
    text?: string;
    toolCalls?: Array<{
      toolCallType: 'function';
      toolCallId: string;
      toolName: string;
      args: string;
    }>;
    finishReason: LanguageModelV3FinishReason;
    usage: {
      promptTokens: number;
      completionTokens: number;
    };
    rawCall: { rawPrompt: unknown; rawSettings: Record<string, unknown> };
    rawResponse?: { headers?: Record<string, string> };
    warnings?: Array<{ type: string; message: string }>;
    providerMetadata?: Record<string, Record<string, unknown>>;
    logprobs?: undefined;
    reasoning?: string;
  }> {
    // For non-streaming, we collect the full stream into a single response
    const { stream, rawCall } = await this.doStream(options);
    const reader = stream.getReader();
    let fullText = '';
    let finishReason: LanguageModelV3FinishReason = {
      unified: 'stop',
      raw: 'stop',
    };
    let usage = {
      promptTokens: 0,
      completionTokens: 0,
    };
    let providerMetadata: Record<string, Record<string, unknown>> | undefined;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        if (value.type === 'text-delta') {
          fullText += value.textDelta;
        } else if (value.type === 'finish') {
          finishReason = {
            unified: value.finishReason,
            raw: value.finishReason,
          };
          usage = value.usage;
          providerMetadata = value.providerMetadata;
        } else if (value.type === 'error') {
          throw value.error;
        }
      }

      return {
        text: fullText,
        finishReason,
        usage,
        rawCall,
        providerMetadata,
      };
    } finally {
      reader.releaseLock();
    }
  }

  async doStream(
    options: LanguageModelV3CallOptions,
  ): Promise<{
    stream: ReadableStream<import('@ai-sdk/provider').LanguageModelV3StreamPart>;
    rawCall: { rawPrompt: unknown; rawSettings: Record<string, unknown> };
    rawResponse?: { headers?: Record<string, string> };
    warnings?: Array<{ type: string; message: string }>;
  }> {
    const input = convertToOpenResponsesInput({
      prompt: options.prompt,
    });

    const fetchImpl = this.config.fetch ?? fetch;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.config.headers,
      ...options.headers,
    };

    const response = await fetchImpl(this.config.baseUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(input),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `OpenResponses API error (${response.status}): ${errorText}`,
      );
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    // Transform the SSE stream to AI SDK stream parts
    const stream = response.body.pipeThrough(
      createOpenResponsesStreamTransformer(),
    );

    return {
      stream,
      rawCall: {
        rawPrompt: input,
        rawSettings: {},
      },
      rawResponse: {
        headers: Object.fromEntries(response.headers.entries()),
      },
    };
  }
}
