import type { LanguageModelV3Content, LanguageModelV3StreamPart } from '@ai-sdk/provider'
import type { ChatAgentChunk, ChatAgentResponse } from './chat-agent-schema'

export const convertChatAgentChunkToMessagePart = (
  chunk: ChatAgentChunk
): LanguageModelV3StreamPart[] => {
  const parts = [] as LanguageModelV3StreamPart[]
  if (chunk.delta.role === 'assistant') {
    if (chunk.delta.content) {
      parts.push({
        type: 'text-delta',
        id: chunk.delta.id,
        delta: chunk.delta.content,
      })
    }
    chunk.delta.tool_calls?.forEach((toolCall) => {
      parts.push({
        type: 'tool-call',
        toolCallId: toolCall.id,
        input: toolCall.function.arguments,
        toolName: toolCall.function.name,
        dynamic: true,
        providerExecuted: true,
      })
    })
  } else if (chunk.delta.role === 'tool') {
    parts.push({
      type: 'tool-result',
      toolCallId: chunk.delta.tool_call_id,
      result: chunk.delta.content,
      toolName: chunk.delta.name ?? 'unknown',
    })
  }
  return parts
}

export const convertChatAgentResponseToMessagePart = (
  response: ChatAgentResponse
): LanguageModelV3Content[] => {
  const parts: LanguageModelV3Content[] = []
  for (const message of response.messages) {
    if (message.role === 'assistant') {
      parts.push({
        type: 'text',
        text: message.content,
      })
      for (const part of message.tool_calls ?? []) {
        parts.push({
          type: 'tool-call',
          toolCallId: part.id,
          input: part.function.arguments,
          toolName: part.function.name,
          dynamic: true,
          providerExecuted: true,
        })
      }
    } else if (message.role === 'tool') {
      parts.push({
        type: 'tool-result',
        toolCallId: message.tool_call_id,
        result: message.content,
        toolName: message.name ?? 'unknown',
      })
    }
  }
  return parts
}
