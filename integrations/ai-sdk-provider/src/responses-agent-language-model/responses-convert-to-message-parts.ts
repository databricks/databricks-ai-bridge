import type { LanguageModelV2Content, LanguageModelV2StreamPart } from '@ai-sdk/provider'
import { randomUUID } from 'node:crypto'
import { type ResponsesAgentChunk, type ResponsesAgentResponse } from './responses-agent-schema'
import { DATABRICKS_TOOL_CALL_ID } from '../tools'
import {
  MCP_APPROVAL_REQUEST_TYPE,
  MCP_APPROVAL_RESPONSE_TYPE,
  createApprovalStatusOutput,
} from '../mcp'

export const convertResponsesAgentChunkToMessagePart = (
  chunk: ResponsesAgentChunk
): LanguageModelV2StreamPart[] => {
  const parts: LanguageModelV2StreamPart[] = []

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
        result: chunk.output,
        toolName: DATABRICKS_TOOL_CALL_ID,
      })
      break

    case 'response.output_item.done':
      parts.push(...convertOutputItemDone(chunk.item))
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

const convertOutputItemDone = (item: OutputItemDoneItem): LanguageModelV2StreamPart[] => {
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
          toolName: DATABRICKS_TOOL_CALL_ID,
          input: item.arguments,
          providerMetadata: {
            databricks: {
              toolName: item.name,
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
          result: item.output,
          toolName: DATABRICKS_TOOL_CALL_ID,
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
          toolName: DATABRICKS_TOOL_CALL_ID,
          input: item.arguments,
          providerMetadata: {
            databricks: {
              type: MCP_APPROVAL_REQUEST_TYPE,
              toolName: item.name,
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
          toolName: DATABRICKS_TOOL_CALL_ID,
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

export const convertResponsesAgentResponseToMessagePart = (
  response: ResponsesAgentResponse
): LanguageModelV2Content[] => {
  const parts: LanguageModelV2Content[] = []

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
          toolName: DATABRICKS_TOOL_CALL_ID,
          input: output.arguments,
          providerMetadata: {
            databricks: {
              toolName: output.name,
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
          result: output.output,
          toolCallId: output.call_id,
          toolName: DATABRICKS_TOOL_CALL_ID,
        })
        break

      case 'mcp_approval_request':
        parts.push({
          type: 'tool-call',
          toolCallId: output.id,
          toolName: DATABRICKS_TOOL_CALL_ID,
          input: output.arguments,
          providerMetadata: {
            databricks: {
              type: MCP_APPROVAL_REQUEST_TYPE,
              toolName: output.name,
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
          toolName: DATABRICKS_TOOL_CALL_ID,
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
