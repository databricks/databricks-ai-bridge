import type { LanguageModelV3Prompt } from '@ai-sdk/provider';
import type { OpenResponsesRequest } from './open-responses-api-types';

/**
 * Convert AI SDK messages to OpenResponses format
 * OpenResponses expects a simple messages array with role and content
 */
export function convertToOpenResponsesInput({
  prompt,
}: {
  prompt: LanguageModelV3Prompt;
}): OpenResponsesRequest {
  return {
    messages: prompt.map((msg) => {
      // Convert 'developer' role to 'system' (OpenResponses doesn't have 'developer')
      const role = msg.role === 'developer' ? 'system' : msg.role;

      // Extract text content from message parts
      let content = '';
      if (Array.isArray(msg.content)) {
        content = msg.content
          .map((part) => {
            if (part.type === 'text') {
              return part.text;
            }
            // Skip non-text parts (images, etc.)
            return '';
          })
          .filter(Boolean)
          .join('\n');
      } else {
        content = msg.content;
      }

      return {
        role: role as 'user' | 'assistant' | 'system',
        content,
      };
    }),
    stream: true,
  };
}
