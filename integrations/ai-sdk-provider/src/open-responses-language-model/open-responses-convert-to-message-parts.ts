import type { LanguageModelV3StreamPart } from '@ai-sdk/provider';
import type { OpenResponsesChunk } from './open-responses-schema';

/**
 * Convert OpenResponses events to AI SDK stream parts
 * Returns a single stream part or null if the event should be skipped
 */
export function convertOpenResponsesChunkToStreamPart(
  chunk: OpenResponsesChunk,
): LanguageModelV3StreamPart | null {
  switch (chunk.type) {
    case 'response.output_text.delta':
      return {
        type: 'text-delta',
        textDelta: chunk.delta,
      };

    case 'response.output_item.done':
      return {
        type: 'finish',
        finishReason: 'stop',
        usage: {
          promptTokens: 0, // OpenResponses doesn't provide token counts
          completionTokens: 0,
        },
        providerMetadata: {
          databricks: {
            format: 'openresponses',
            fullContent: chunk.item.content,
          },
        },
      };

    case 'response.error':
      return {
        type: 'error',
        error: new Error(chunk.error.message),
      };

    default:
      // Unknown event type - skip it
      return null;
  }
}
