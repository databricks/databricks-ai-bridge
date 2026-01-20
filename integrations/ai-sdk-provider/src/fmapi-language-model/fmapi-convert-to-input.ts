import type { LanguageModelV2Message, LanguageModelV2ToolResultPart } from '@ai-sdk/provider'
import type { FmapiInputMessage, FmapiContentItem } from './fmapi-schema'

type LanguageModelV2SystemMessage = Extract<LanguageModelV2Message, { role: 'system' }>
type LanguageModelV2UserMessage = Extract<LanguageModelV2Message, { role: 'user' }>
type LanguageModelV2AssistantMessage = Extract<LanguageModelV2Message, { role: 'assistant' }>
type LanguageModelV2ToolMessage = Extract<LanguageModelV2Message, { role: 'tool' }>

export const convertPromptToFmapiMessages = (
  prompt: LanguageModelV2Message[]
): { messages: Array<FmapiInputMessage> } => {
  const messages: Array<FmapiInputMessage> = []

  for (const message of prompt) {
    switch (message.role) {
      case 'system':
        messages.push({
          role: 'system',
          content: convertSystemContent(message),
        })
        break
      case 'user':
        messages.push({
          role: 'user',
          content: convertUserContent(message),
        })
        break
      case 'assistant':
        messages.push(convertAssistantMessage(message))
        break
      case 'tool':
        // Tool messages need special handling - one message per tool result
        messages.push(...convertToolMessages(message))
        break
    }
  }

  return { messages }
}

const convertSystemContent = (message: LanguageModelV2SystemMessage): FmapiContentItem[] => {
  return [{ type: 'text', text: message.content }]
}

const convertUserContent = (message: LanguageModelV2UserMessage): FmapiContentItem[] => {
  const items: FmapiContentItem[] = []

  for (const part of message.content) {
    switch (part.type) {
      case 'text':
        items.push({ type: 'text', text: part.text })
        break
      case 'file':
        if (part.mediaType.startsWith('image/')) {
          const url = toHttpUrlString(part.data)
          if (url) items.push({ type: 'image', image_url: url })
        }
        break
    }
  }

  return items
}

const convertAssistantMessage = (message: LanguageModelV2AssistantMessage): FmapiInputMessage => {
  const contentItems: FmapiContentItem[] = []
  const toolCalls: Array<{
    id: string
    type: 'function'
    function: { name: string; arguments: string }
  }> = []

  for (const part of message.content) {
    switch (part.type) {
      case 'text':
        contentItems.push({ type: 'text', text: part.text })
        break
      case 'file':
        if (part.mediaType.startsWith('image/')) {
          const url = toHttpUrlString(part.data)
          if (url) contentItems.push({ type: 'image', image_url: url })
        }
        break
      case 'reasoning':
        contentItems.push({
          type: 'reasoning',
          summary: [{ type: 'summary_text', text: part.text }],
        })
        break
      case 'tool-call':
        // Convert to OpenAI tool_calls format
        toolCalls.push({
          id: part.toolCallId,
          type: 'function',
          function: {
            name: part.toolName,
            arguments: typeof part.input === 'string' ? part.input : JSON.stringify(part.input),
          },
        })
        break
    }
  }

  return {
    role: 'assistant',
    // Use null instead of empty string when there's no text content
    // (required by Claude API when tool_calls are present)
    content: contentItems.length === 0 ? null : contentItems,
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  }
}

const convertToolMessages = (message: LanguageModelV2ToolMessage): FmapiInputMessage[] => {
  const messages: FmapiInputMessage[] = []

  for (const part of message.content) {
    if (part.type === 'tool-result') {
      // Each tool result becomes a separate tool message with tool_call_id
      const content = convertToolResultOutputToContentValue(part.output)
      messages.push({
        role: 'tool',
        tool_call_id: part.toolCallId,
        content: typeof content === 'string' ? content : JSON.stringify(content),
      })
    }
  }

  return messages
}

const toHttpUrlString = (data: URL | string | Uint8Array): string | null => {
  if (data instanceof URL) return data.toString()
  if (typeof data === 'string') {
    if (data.startsWith('http://') || data.startsWith('https://')) return data
  }
  return null
}

const convertToolResultOutputToContentValue = (
  output: LanguageModelV2ToolResultPart['output']
): unknown => {
  switch (output.type) {
    case 'text':
    case 'error-text':
      return output.value
    case 'json':
    case 'error-json':
      return output.value
    case 'content':
      return output.value
    default:
      return null
  }
}
