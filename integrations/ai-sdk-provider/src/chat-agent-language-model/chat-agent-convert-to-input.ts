import type { LanguageModelV2Prompt, LanguageModelV2ToolResultPart } from '@ai-sdk/provider'
import type { ChatAgentResponse } from './chat-agent-schema'

type ChatAgentMessage = ChatAgentResponse['messages'][number]
type LanguageModelV2UserMessage = Extract<LanguageModelV2Prompt[number], { role: 'user' }>
type LanguageModelV2AssistantMessage = Extract<LanguageModelV2Prompt[number], { role: 'assistant' }>
type LanguageModelV2ToolMessage = Extract<LanguageModelV2Prompt[number], { role: 'tool' }>

export const convertLanguageModelV2PromptToChatAgentResponse = (
  prompt: LanguageModelV2Prompt
): ChatAgentResponse['messages'] => {
  const messages: ChatAgentResponse['messages'] = []
  let messageIndex = 0

  for (const msg of prompt) {
    switch (msg.role) {
      case 'system':
        // System messages are prompt-only; they don't exist in ChatAgent responses.
        break

      case 'user': {
        const converted = convertUserMessage(msg, messageIndex)
        messages.push(converted)
        messageIndex++
        break
      }

      case 'assistant': {
        const converted = convertAssistantMessage(msg, messageIndex)
        messages.push(...converted)
        messageIndex += converted.length
        break
      }

      case 'tool': {
        const converted = convertToolMessage(msg, messageIndex)
        messages.push(...converted)
        messageIndex += converted.length
        break
      }
    }
  }

  return messages
}

const convertUserMessage = (
  msg: LanguageModelV2UserMessage,
  messageIndex: number
): ChatAgentMessage => {
  const text = (msg.content ?? [])
    .filter((part): part is Extract<typeof part, { type: 'text' }> => part.type === 'text')
    .map((part) => part.text)
    .join('\n')

  return {
    role: 'user',
    content: text,
    id: `user-${messageIndex}`,
  }
}

const convertAssistantMessage = (
  msg: LanguageModelV2AssistantMessage,
  startIndex: number
): ChatAgentMessage[] => {
  const messages: ChatAgentMessage[] = []
  let messageIndex = startIndex

  const textContent = (msg.content ?? [])
    .filter((part) => part.type === 'text' || part.type === 'reasoning')
    .map((part) => (part.type === 'text' ? part.text : part.text))
    .join('\n')

  const toolCalls = (msg.content ?? [])
    .filter(
      (part): part is Extract<typeof part, { type: 'tool-call' }> => part.type === 'tool-call'
    )
    .map((call) => ({
      type: 'function' as const,
      id: call.toolCallId,
      function: {
        name: call.toolName,
        arguments: typeof call.input === 'string' ? call.input : JSON.stringify(call.input ?? {}),
      },
    }))

  messages.push({
    role: 'assistant',
    content: textContent,
    id: `assistant-${messageIndex++}`,
    tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
  })

  // Convert any tool results embedded in the assistant message into separate tool messages.
  for (const part of msg.content ?? []) {
    if (part.type === 'tool-result') {
      messages.push({
        role: 'tool',
        name: part.toolName,
        content: convertToolResultOutput(part.output),
        tool_call_id: part.toolCallId,
        id: `tool-${messageIndex++}`,
      })
    }
  }

  return messages
}

const convertToolMessage = (
  msg: LanguageModelV2ToolMessage,
  startIndex: number
): ChatAgentMessage[] => {
  const messages: ChatAgentMessage[] = []
  let messageIndex = startIndex

  for (const part of msg.content ?? []) {
    if (part.type === 'tool-result') {
      messages.push({
        role: 'tool',
        name: part.toolName,
        content: convertToolResultOutput(part.output),
        tool_call_id: part.toolCallId,
        id: `tool-${messageIndex++}`,
      })
    }
  }

  return messages
}

const convertToolResultOutput = (output: LanguageModelV2ToolResultPart['output']): string => {
  switch (output.type) {
    case 'text':
    case 'error-text':
      return output.value
    case 'json':
    case 'error-json':
      return JSON.stringify(output.value)
    case 'content':
      return output.value
        .map((p) => (p.type === 'text' ? p.text : ''))
        .filter(Boolean)
        .join('\n')
    default:
      return ''
  }
}
