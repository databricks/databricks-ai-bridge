import { z } from 'zod/v4'

// Tool call schema for OpenAI-compatible format
export const toolCallSchema = z.object({
  id: z.string(),
  type: z.literal('function'),
  function: z.object({
    name: z.string(),
    arguments: z.string(), // JSON string of arguments
  }),
})

// Zod schemas mirroring FMAPI chat chunk types
export const reasoningSummarySchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('summary_text'),
    text: z.string(),
    signature: z.string().optional(),
  }),
  z.object({
    type: z.literal('summary_encrypted_text'),
    data: z.string(),
  }),
])

export const contentItemSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('text'),
    text: z.string(),
    citation: z.unknown().optional(),
  }),
  z.object({
    type: z.literal('image'),
    image_url: z.string(),
  }),
  z.object({
    type: z.literal('reasoning'),
    summary: z.array(reasoningSummarySchema),
  }),
])

// Tool call delta for streaming
export const toolCallDeltaSchema = z.object({
  index: z.number(),
  id: z.string().optional(),
  type: z.literal('function').optional(),
  function: z
    .object({
      name: z.string().optional(),
      arguments: z.string().optional(),
    })
    .optional(),
})

export const fmapiChunkSchema = z.object({
  id: z.string(),
  created: z.number(),
  model: z.string(),
  usage: z
    .object({
      prompt_tokens: z.number().nullable().optional(),
      completion_tokens: z.number().nullable().optional(),
      total_tokens: z.number().nullable().optional(),
    })
    .nullable()
    .optional(),
  object: z.literal('chat.completion.chunk'),
  choices: z.array(
    z.object({
      index: z.number(),
      delta: z.object({
        role: z.union([z.literal('assistant'), z.null(), z.undefined()]).optional(),
        // content can be string, array of content items, null (when tool_calls present), or omitted
        content: z.union([z.string(), z.array(contentItemSchema), z.null()]).optional(),
        tool_calls: z.array(toolCallDeltaSchema).optional(),
      }),
      finish_reason: z.union([z.literal('stop'), z.literal('tool_calls'), z.null()]).optional(),
    })
  ),
})

export const fmapiResponseSchema = z.object({
  id: z.string(),
  created: z.number(),
  model: z.string(),
  usage: z
    .object({
      prompt_tokens: z.number(),
      completion_tokens: z.number(),
      total_tokens: z.number(),
    })
    .nullable()
    .optional(),
  choices: z.array(
    z.object({
      message: z.object({
        role: z.union([z.literal('assistant'), z.literal('user'), z.literal('tool')]),
        content: z.union([z.string(), z.array(contentItemSchema), z.null()]).optional(),
        tool_calls: z.array(toolCallSchema).optional(),
      }),
      finish_reason: z.union([z.literal('stop'), z.literal('tool_calls'), z.null()]).optional(),
    })
  ),
})

// Input message schema for requests (different from response format)
// Tool messages in requests need tool_call_id at the message level
export type FmapiInputMessage =
  | {
      role: 'system' | 'user' | 'assistant'
      content: string | z.infer<typeof contentItemSchema>[] | null
      tool_calls?: z.infer<typeof toolCallSchema>[]
    }
  | {
      role: 'tool'
      tool_call_id: string
      content: string
    }

// Exported types for type-only imports in other modules
export type FmapiChunk = z.infer<typeof fmapiChunkSchema>
export type FmapiResponse = z.infer<typeof fmapiResponseSchema>
export type FmapiMessage = FmapiResponse['choices'][number]['message']
export type FmapiContentItem = z.infer<typeof contentItemSchema>
export type FmapiToolCall = z.infer<typeof toolCallSchema>
