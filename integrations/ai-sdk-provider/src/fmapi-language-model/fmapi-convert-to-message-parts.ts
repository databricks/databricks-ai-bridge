import type { LanguageModelV2Content, LanguageModelV2StreamPart } from '@ai-sdk/provider'
import type { FmapiChunk, FmapiContentItem, FmapiResponse, FmapiToolCall } from './fmapi-schema'

export const convertFmapiChunkToMessagePart = (
  chunk: FmapiChunk,
  toolCallIdsByIndex?: Map<number, string>
): LanguageModelV2StreamPart[] => {
  const parts: LanguageModelV2StreamPart[] = []
  if (chunk.choices.length === 0) return parts
  const choice = chunk.choices[0]

  // Handle streaming tool calls
  if (choice.delta.tool_calls && choice.delta.tool_calls.length > 0) {
    for (const toolCallDelta of choice.delta.tool_calls) {
      const index = toolCallDelta.index

      // If this is the start of a tool call (has id and name), track the ID
      if (toolCallDelta.id && toolCallDelta.function?.name) {
        toolCallIdsByIndex?.set(index, toolCallDelta.id)
        parts.push({
          type: 'tool-input-start',
          id: toolCallDelta.id,
          toolName: toolCallDelta.function.name,
        })
      }

      // If this has argument delta, use tracked ID or fallback
      if (toolCallDelta.function?.arguments) {
        const id = toolCallDelta.id ?? toolCallIdsByIndex?.get(index) ?? `tool-call-${index}`
        parts.push({
          type: 'tool-input-delta',
          id,
          delta: toolCallDelta.function.arguments,
        })
      }
    }
  }

  if (typeof choice.delta.content === 'string') {
    // Skip empty string content to avoid spurious text-start/text-end cycles
    if (choice.delta.content) {
      parts.push({ type: 'text-delta', id: chunk.id, delta: choice.delta.content })
    }
  } else if (Array.isArray(choice.delta.content)) {
    parts.push(...mapContentItemsToStreamParts(choice.delta.content, chunk.id))
  }

  return parts
}

export const convertFmapiResponseToMessagePart = (
  response: FmapiResponse
): LanguageModelV2Content[] => {
  const parts: LanguageModelV2Content[] = []
  if (response.choices.length === 0) return parts
  const choice = response.choices[0]

  // Handle OpenAI-format tool_calls first
  if (choice.message.tool_calls && choice.message.tool_calls.length > 0) {
    for (const toolCall of choice.message.tool_calls) {
      parts.push(convertToolCallToContent(toolCall))
    }
    // If there's also text content, include it
    if (typeof choice.message.content === 'string' && choice.message.content) {
      parts.push({ type: 'text', text: choice.message.content })
    }
    return parts
  }

  if (typeof choice.message.content === 'string') {
    parts.push({ type: 'text', text: choice.message.content })
  } else {
    parts.push(...mapContentItemsToProviderContent(choice.message.content ?? []))
  }

  return parts
}

const convertToolCallToContent = (toolCall: FmapiToolCall): LanguageModelV2Content => {
  return {
    type: 'tool-call',
    toolCallId: toolCall.id,
    toolName: toolCall.function.name,
    input: toolCall.function.arguments,
  }
}

const mapContentItemsToStreamParts = (
  items: FmapiContentItem[],
  id: string
): LanguageModelV2StreamPart[] => {
  const parts: LanguageModelV2StreamPart[] = []
  for (const item of items) {
    switch (item.type) {
      case 'text':
        parts.push({ type: 'text-delta', id, delta: item.text })
        break
      case 'image':
        // Images are currently not supported in stream parts
        break
      case 'reasoning': {
        for (const summary of item.summary.filter((s) => s.type === 'summary_text')) {
          parts.push({ type: 'reasoning-delta', id, delta: summary.text })
        }
        break
      }
    }
  }
  return parts
}

const mapContentItemsToProviderContent = (items: FmapiContentItem[]): LanguageModelV2Content[] => {
  const parts: LanguageModelV2Content[] = []
  for (const item of items) {
    switch (item.type) {
      case 'text':
        parts.push({ type: 'text', text: item.text })
        break
      case 'image':
        // Images are currently not supported in content parts
        break
      case 'reasoning': {
        for (const summary of item.summary.filter((s) => s.type === 'summary_text')) {
          parts.push({ type: 'reasoning', text: summary.text })
        }
        break
      }
    }
  }
  return parts
}
