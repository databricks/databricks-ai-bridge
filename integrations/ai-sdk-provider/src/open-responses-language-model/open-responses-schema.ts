import { z } from 'zod';

/**
 * OpenResponses format event schemas
 * These are the events sent by declarative agent apps using the OpenResponses format
 * Specification: https://github.com/databricks/openresponses
 */

// Text delta event - contains actual text content
export const openResponsesTextDeltaSchema = z.object({
  type: z.literal('response.output_text.delta'),
  delta: z.string(),
});

// Item done event - sent when response complete
export const openResponsesItemDoneSchema = z.object({
  type: z.literal('response.output_item.done'),
  item: z.object({
    role: z.enum(['assistant', 'user', 'system']),
    content: z.string(),
  }),
});

// Error event
export const openResponsesErrorSchema = z.object({
  type: z.literal('response.error'),
  error: z.object({
    message: z.string(),
    type: z.string().optional(),
  }),
});

// Union of all OpenResponses events - using discriminatedUnion for better type inference
export const openResponsesChunkSchema = z.discriminatedUnion('type', [
  openResponsesTextDeltaSchema,
  openResponsesItemDoneSchema,
  openResponsesErrorSchema,
]);

// TypeScript types
export type OpenResponsesTextDelta = z.infer<typeof openResponsesTextDeltaSchema>;
export type OpenResponsesItemDone = z.infer<typeof openResponsesItemDoneSchema>;
export type OpenResponsesError = z.infer<typeof openResponsesErrorSchema>;
export type OpenResponsesChunk = z.infer<typeof openResponsesChunkSchema>;
