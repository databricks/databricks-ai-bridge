import type { LanguageModelV3StreamPart } from '@ai-sdk/provider';
import { openResponsesChunkSchema } from './open-responses-schema';
import { convertOpenResponsesChunkToStreamPart } from './open-responses-convert-to-message-parts';

/**
 * Create a TransformStream that parses OpenResponses SSE format and converts to AI SDK stream parts
 *
 * OpenResponses uses Server-Sent Events (SSE) format:
 * data: {"type": "response.output_text.delta", "delta": "Hello"}\n
 * data: {"type": "response.output_item.done", "item": {...}}\n
 * data: [DONE]\n
 */
export function createOpenResponsesStreamTransformer(): TransformStream<
  Uint8Array,
  LanguageModelV3StreamPart
> {
  let buffer = '';
  const decoder = new TextDecoder();

  return new TransformStream({
    transform(chunk, controller) {
      // Decode incoming bytes and add to buffer
      buffer += decoder.decode(chunk, { stream: true });

      // Split by newlines to get individual lines
      const lines = buffer.split('\n');

      // Keep the last incomplete line in buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        // Skip empty lines or lines that don't start with "data:"
        if (!line.trim() || !line.startsWith('data:')) {
          continue;
        }

        // Extract the data payload
        const dataStr = line.slice(5).trim(); // Remove "data:" prefix

        // Skip [DONE] marker
        if (dataStr === '[DONE]') {
          continue;
        }

        try {
          // Parse JSON
          const json = JSON.parse(dataStr);

          // Validate with schema
          const parsed = openResponsesChunkSchema.safeParse(json);

          if (parsed.success) {
            // Convert to AI SDK stream part
            const streamPart = convertOpenResponsesChunkToStreamPart(parsed.data);
            if (streamPart) {
              controller.enqueue(streamPart);
            }
          } else {
            // Schema validation failed - log warning but don't break stream
            console.warn(
              '[OpenResponses] Failed to validate event:',
              parsed.error.message,
            );
          }
        } catch (error) {
          // JSON parse error - log warning but don't break stream
          console.warn('[OpenResponses] Failed to parse JSON:', error);
        }
      }
    },

    flush(controller) {
      // Process any remaining buffer content
      if (buffer.trim() && buffer.startsWith('data:')) {
        const dataStr = buffer.slice(5).trim();
        if (dataStr && dataStr !== '[DONE]') {
          try {
            const json = JSON.parse(dataStr);
            const parsed = openResponsesChunkSchema.safeParse(json);
            if (parsed.success) {
              const streamPart = convertOpenResponsesChunkToStreamPart(parsed.data);
              if (streamPart) {
                controller.enqueue(streamPart);
              }
            }
          } catch (error) {
            // Ignore errors in flush
          }
        }
      }
    },
  });
}
