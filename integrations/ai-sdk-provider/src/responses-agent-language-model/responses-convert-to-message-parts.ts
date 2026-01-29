import type { JSONValue, LanguageModelV3Content, LanguageModelV3StreamPart } from '@ai-sdk/provider'
import { randomUUID } from 'node:crypto'
import { type ResponsesAgentChunk, type ResponsesAgentResponse } from './responses-agent-schema'
import {
  MCP_APPROVAL_REQUEST_TYPE,
  MCP_APPROVAL_RESPONSE_TYPE,
  createApprovalStatusOutput,
} from '../mcp'

export type ConvertChunkOptions = {
  useRemoteToolCalling: boolean
  toolNamesByCallId?: Map<string, string>
}

export const convertResponsesAgentChunkToMessagePart = (
  chunk: ResponsesAgentChunk,
  options: ConvertChunkOptions = { useRemoteToolCalling: false }
): LanguageModelV3StreamPart[] => {
  const parts: LanguageModelV3StreamPart[] = []

  if ('error' in chunk) {
    parts.push({
      type: 'error',
      error: chunk.error,
    })
    return parts
  }

  switch (chunk.type) {
    case 'response.output_text.delta':
      parts.push({
        type: 'text-delta',
        id: chunk.item_id,
        delta: chunk.delta,
        providerMetadata: {
          databricks: {
            itemId: chunk.item_id,
          },
        },
      })
      break

    case 'response.reasoning_summary_text.delta':
      parts.push({
        type: 'reasoning-delta',
        id: chunk.item_id,
        delta: chunk.delta,
        providerMetadata: {
          databricks: {
            itemId: chunk.item_id,
          },
        },
      })
      break

    case 'function_call_output':
      parts.push({
        type: 'tool-result',
        toolCallId: chunk.call_id,
        result: chunk.output != null ? (chunk.output as NonNullable<JSONValue>) : {},
        toolName: options.toolNamesByCallId?.get(chunk.call_id) ?? 'unknown',
      })
      break

    case 'response.output_item.done':
      parts.push(...convertOutputItemDone(chunk.item, options))
      break
    case 'response.output_text.annotation.added':
      parts.push({
        type: 'source',
        url: chunk.annotation.url,
        title: chunk.annotation.title,
        id: randomUUID(),
        sourceType: 'url',
      })
      break

    case 'error':
      parts.push({
        type: 'error',
        error: chunk,
      })
      break

    default: {
      void (chunk as never)
      break
    }
  }

  return parts
}

type OutputItemDoneItem = Extract<
  ResponsesAgentChunk,
  { type: 'response.output_item.done' }
>['item']

const convertOutputItemDone = (
  item: OutputItemDoneItem,
  options: ConvertChunkOptions
): LanguageModelV3StreamPart[] => {
  switch (item.type) {
    case 'message': {
      const firstContent = item.content[0]
      if (!firstContent) return []
      return [
        {
          type: 'text-delta',
          id: item.id,
          delta: firstContent.text,
          providerMetadata: {
            databricks: {
              itemId: item.id,
              itemType: 'response.output_item.done',
            },
          },
        },
      ]
    }

    case 'function_call':
      return [
        {
          type: 'tool-call',
          toolCallId: item.call_id,
          toolName: item.name,
          input: item.arguments,
          ...(options.useRemoteToolCalling && {
            dynamic: true,
            providerExecuted: true,
          }),
          providerMetadata: {
            databricks: {
              itemId: item.id,
            },
          },
        },
      ]

    case 'function_call_output':
      return [
        {
          type: 'tool-result',
          toolCallId: item.call_id,
          result: item.output != null ? (item.output as NonNullable<JSONValue>) : {},
          toolName: options.toolNamesByCallId?.get(item.call_id) ?? 'unknown',
        },
      ]

    case 'reasoning': {
      const firstSummary = item.summary[0]
      if (!firstSummary) return []
      return [
        {
          type: 'reasoning-start',
          id: item.id,
        },
        {
          type: 'reasoning-delta',
          id: item.id,
          delta: firstSummary.text,
          providerMetadata: {
            databricks: {
              itemId: item.id,
            },
          },
        },
        {
          type: 'reasoning-end',
          id: item.id,
        },
      ]
    }

    case 'mcp_approval_request':
      return [
        {
          type: 'tool-call',
          toolCallId: item.id,
          toolName: item.name,
          input: item.arguments,
          ...(options.useRemoteToolCalling && {
            dynamic: true,
            providerExecuted: true,
          }),
          providerMetadata: {
            databricks: {
              type: MCP_APPROVAL_REQUEST_TYPE,
              itemId: item.id,
              serverLabel: item.server_label,
            },
          },
        },
      ]

    case 'mcp_approval_response':
      return [
        {
          type: 'tool-result',
          toolCallId: item.approval_request_id,
          toolName: options.toolNamesByCallId?.get(item.approval_request_id) ?? 'mcp_approval',
          result: createApprovalStatusOutput(item.approve),
          providerMetadata: {
            databricks: {
              type: MCP_APPROVAL_RESPONSE_TYPE,
              ...(item.id != null && { itemId: item.id }),
            },
          },
        },
      ]

    default:
      void (item satisfies never)
      return []
  }
}

export type ConvertResponseOptions = {
  useRemoteToolCalling: boolean
}

export const convertResponsesAgentResponseToMessagePart = (
  response: ResponsesAgentResponse,
  options: ConvertResponseOptions = { useRemoteToolCalling: false }
): LanguageModelV3Content[] => {
  const parts: LanguageModelV3Content[] = []

  // Build a map of call_id -> tool_name from function_call outputs
  const toolNamesByCallId = new Map<string, string>()
  for (const output of response.output) {
    if (output.type === 'function_call') {
      toolNamesByCallId.set(output.call_id, output.name)
    } else if (output.type === 'mcp_approval_request') {
      toolNamesByCallId.set(output.id, output.name)
    }
  }

  for (const output of response.output) {
    switch (output.type) {
      case 'message': {
        for (const content of output.content) {
          if (content.type === 'output_text') {
            parts.push({
              type: 'text',
              text: content.text,
              providerMetadata: {
                databricks: {
                  itemId: output.id,
                },
              },
            })
          }
        }
        break
      }

      case 'function_call':
        parts.push({
          type: 'tool-call',
          toolCallId: output.call_id,
          toolName: output.name,
          input: output.arguments,
          ...(options.useRemoteToolCalling && {
            dynamic: true,
            providerExecuted: true,
          }),
          providerMetadata: {
            databricks: {
              itemId: output.id,
            },
          },
        })
        break

      case 'reasoning':
        for (const summary of output.summary) {
          if (summary.type === 'summary_text') {
            parts.push({
              type: 'reasoning',
              text: summary.text,
              providerMetadata: {
                databricks: {
                  itemId: output.id,
                },
              },
            })
          }
        }
        break

      case 'function_call_output':
        parts.push({
          type: 'tool-result',
          result: output.output as NonNullable<JSONValue>,
          toolCallId: output.call_id,
          toolName: toolNamesByCallId.get(output.call_id) ?? 'unknown',
        })
        break

      case 'mcp_approval_request':
        parts.push({
          type: 'tool-call',
          toolCallId: output.id,
          toolName: output.name,
          input: output.arguments,
          providerMetadata: {
            databricks: {
              type: MCP_APPROVAL_REQUEST_TYPE,
              itemId: output.id,
              serverLabel: output.server_label,
            },
          },
        })
        break

      case 'mcp_approval_response':
        parts.push({
          type: 'tool-result',
          toolCallId: output.approval_request_id,
          toolName: toolNamesByCallId.get(output.approval_request_id) ?? 'mcp_approval',
          result: createApprovalStatusOutput(output.approve),
          providerMetadata: {
            databricks: {
              type: MCP_APPROVAL_RESPONSE_TYPE,
              ...(output.id != null && { itemId: output.id }),
            },
          },
        })
        break
      default: {
        void (output satisfies never)
        break
      }
    }
  }

  return parts
}
