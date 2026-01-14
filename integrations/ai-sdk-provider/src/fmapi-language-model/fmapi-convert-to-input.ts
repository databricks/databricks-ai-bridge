import type { LanguageModelV2Message, LanguageModelV2ToolResultPart } from '@ai-sdk/provider'
import type { FmapiMessage } from './fmapi-schema'
import { serializeToolCall, serializeToolResult } from './fmapi-tags'

type FmapiContentItem = Exclude<NonNullable<FmapiMessage['content']>, string>[number]
type LanguageModelV2SystemMessage = Extract<LanguageModelV2Message, { role: 'system' }>
type LanguageModelV2UserMessage = Extract<LanguageModelV2Message, { role: 'user' }>
type LanguageModelV2AssistantMessage = Extract<LanguageModelV2Message, { role: 'assistant' }>
type LanguageModelV2ToolMessage = Extract<LanguageModelV2Message, { role: 'tool' }>

export const convertPromptToFmapiMessages = (
  prompt: LanguageModelV2Message[]
): { messages: Array<FmapiMessage> } => {
  const messages: Array<FmapiMessage> = prompt.map((message) => {
    const role = message.role === 'system' ? 'user' : message.role

    let contentItems: FmapiContentItem[] = []

    switch (message.role) {
      case 'system':
        contentItems = convertSystemContent(message)
        break
      case 'user':
        contentItems = convertUserContent(message)
        break
      case 'assistant':
        contentItems = convertAssistantContent(message)
        break
      case 'tool':
        contentItems = convertToolContent(message)
        break
    }

    const content = contentItems.length === 0 ? '' : contentItems
    return { role, content }
  })
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

const convertAssistantContent = (message: LanguageModelV2AssistantMessage): FmapiContentItem[] => {
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
      case 'reasoning':
        items.push({
          type: 'reasoning',
          summary: [{ type: 'summary_text', text: part.text }],
        })
        break
      case 'tool-call':
        items.push({
          type: 'text',
          text: serializeToolCall({
            id: part.toolCallId,
            name: part.toolName,
            arguments: part.input,
          }),
        })
        break
      case 'tool-result':
        items.push({
          type: 'text',
          text: serializeToolResult({
            id: part.toolCallId,
            content: convertToolResultOutputToContentValue(part.output),
          }),
        })
        break
    }
  }

  return items
}

const convertToolContent = (message: LanguageModelV2ToolMessage): FmapiContentItem[] => {
  const items: FmapiContentItem[] = []

  for (const part of message.content) {
    if (part.type === 'tool-result') {
      items.push({
        type: 'text',
        text: serializeToolResult({
          id: part.toolCallId,
          content: convertToolResultOutputToContentValue(part.output),
        }),
      })
    }
  }

  return items
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
