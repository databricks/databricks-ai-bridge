import type {
  LanguageModelV3Message,
  LanguageModelV3ToolResultPart,
  LanguageModelV3ToolCallPart,
} from '@ai-sdk/provider'
import { parseProviderOptions } from '@ai-sdk/provider-utils'
import { z } from 'zod/v4'
import type { FmapiInputMessage, FmapiContentItem } from './fmapi-schema'

type LanguageModelV3SystemMessage = Extract<LanguageModelV3Message, { role: 'system' }>
type LanguageModelV3UserMessage = Extract<LanguageModelV3Message, { role: 'user' }>
type LanguageModelV3AssistantMessage = Extract<LanguageModelV3Message, { role: 'assistant' }>
type LanguageModelV3ToolMessage = Extract<LanguageModelV3Message, { role: 'tool' }>

export const convertPromptToFmapiMessages = async (
  prompt: LanguageModelV3Message[]
): Promise<{ messages: Array<FmapiInputMessage> }> => {
  const messages: Array<FmapiInputMessage> = []

  for (const message of prompt) {
    switch (message.role) {
      case 'system':
        messages.push(convertSystemMessage(message))
        break
      case 'user':
        messages.push(convertUserMessage(message))
        break
      case 'assistant':
        messages.push(await convertAssistantMessage(message))
        break
      case 'tool':
        // Tool messages need special handling - one message per tool result
        messages.push(...convertToolMessages(message))
        break
    }
  }

  return { messages }
}

const convertSystemMessage = (message: LanguageModelV3SystemMessage): FmapiInputMessage => {
  return {
    role: 'system',
    content: [{ type: 'text', text: message.content }],
  }
}

const convertUserMessage = (message: LanguageModelV3UserMessage): FmapiInputMessage => {
  const content: FmapiContentItem[] = []

  for (const part of message.content) {
    switch (part.type) {
      case 'text':
        content.push({ type: 'text', text: part.text })
        break
      case 'file':
        if (part.mediaType.startsWith('image/')) {
          const url = toHttpUrlString(part.data)
          if (url) content.push({ type: 'image', image_url: url })
        }
        break
    }
  }

  return { role: 'user', content }
}

const convertAssistantMessage = async (
  message: LanguageModelV3AssistantMessage
): Promise<FmapiInputMessage> => {
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
      case 'tool-call': {
        // Parse provider options to get the actual tool name
        const toolName = await getToolNameFromPart(part)
        // Convert to OpenAI tool_calls format
        toolCalls.push({
          id: part.toolCallId,
          type: 'function',
          function: {
            name: toolName,
            arguments: typeof part.input === 'string' ? part.input : JSON.stringify(part.input),
          },
        })
        break
      }
    }
  }

  return {
    role: 'assistant',
    // Use null instead of empty string when there's no text content
    content: contentItems.length === 0 ? null : contentItems,
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  }
}

const convertToolMessages = (message: LanguageModelV3ToolMessage): FmapiInputMessage[] => {
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
  output: LanguageModelV3ToolResultPart['output']
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

const ProviderOptionsSchema = z.object({
  toolName: z.string().nullish(),
})

const getToolNameFromPart = async (part: LanguageModelV3ToolCallPart): Promise<string> => {
  const providerOptions = await parseProviderOptions({
    provider: 'databricks',
    providerOptions: part.providerOptions,
    schema: ProviderOptionsSchema,
  })
  // Use the actual tool name from provider metadata if available,
  // otherwise fall back to part.toolName
  return providerOptions?.toolName ?? part.toolName
}
