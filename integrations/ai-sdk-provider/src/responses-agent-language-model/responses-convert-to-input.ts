import {
  UnsupportedFunctionalityError,
  type SharedV3Warning,
  type LanguageModelV3Prompt,
  type LanguageModelV3ToolResultPart,
} from '@ai-sdk/provider'
import { parseProviderOptions } from '@ai-sdk/provider-utils'
import { z } from 'zod/v4'
import type { ResponsesInput } from './responses-api-types'

export async function convertToResponsesInput({
  prompt,
  systemMessageMode,
}: {
  prompt: LanguageModelV3Prompt
  systemMessageMode: 'system' | 'developer' | 'remove'
}): Promise<{
  input: ResponsesInput
  warnings: Array<SharedV3Warning>
}> {
  const input: ResponsesInput = []
  const warnings: Array<SharedV3Warning> = []

  // Track processed approval IDs to avoid duplicates
  const processedApprovalIds = new Set<string>()

  // Track which approval IDs have a function_call_output (tool result)
  // If we have a function_call_output for an MCP approval, we should NOT also send mcp_approval_response
  // because the function_call_output already implies approval and sending both causes
  // "each tool_use must have a single result" error
  const approvalIdsWithToolResult = new Set<string>()

  // Map tool call results to a map by tool call id so we can insert them into the input in the correct order,
  // right after the tool call that produced them.
  const toolCallResultsByToolCallId = prompt
    .filter((p) => p.role === 'tool')
    .flatMap((p) => p.content)
    .reduce(
      (reduction, toolCallResult) => {
        if (toolCallResult.type === 'tool-result') {
          reduction[toolCallResult.toolCallId] = toolCallResult
        }
        return reduction
      },
      {} as Record<string, LanguageModelV3ToolResultPart>
    )

  for (const { role, content } of prompt) {
    switch (role) {
      case 'system': {
        switch (systemMessageMode) {
          case 'system':
            input.push({ role: 'system', content })
            break

          case 'developer':
            input.push({ role: 'developer', content })
            break

          case 'remove':
            warnings.push({
              type: 'other',
              message: 'system messages are removed for this model',
            })
            break

          default: {
            const _exhaustiveCheck: never = systemMessageMode
            throw new Error(`Unsupported system message mode: ${String(_exhaustiveCheck)}`)
          }
        }
        break
      }

      case 'user':
        input.push({
          role: 'user',
          content: content.map((part) => {
            switch (part.type) {
              case 'text':
                return { type: 'input_text', text: part.text }
              default:
                throw new UnsupportedFunctionalityError({
                  functionality: `part ${JSON.stringify(part)}`,
                })
            }
          }),
        })
        break

      case 'assistant':
        for (const part of content) {
          const providerOptions = await parseProviderOptions({
            provider: 'databricks',
            providerOptions: part.providerOptions,
            schema: ProviderOptionsSchema,
          })
          const itemId = providerOptions?.itemId ?? undefined
          switch (part.type) {
            case 'text': {
              input.push({
                role: 'assistant',
                content: [{ type: 'output_text', text: part.text }],
                id: itemId,
              })
              break
            }
            case 'tool-call': {
              const toolName = providerOptions?.toolName ?? part.toolName
              const approvalRequestId = providerOptions?.approvalRequestId

              // Check if this is an MCP approval request (has approvalRequestId in metadata)
              if (approvalRequestId) {
                const serverLabel = providerOptions?.serverLabel ?? ''
                input.push({
                  type: 'mcp_approval_request',
                  id: approvalRequestId,
                  name: toolName,
                  arguments: JSON.stringify(part.input),
                  server_label: serverLabel,
                })
                // Don't add tool result here - it will be handled via tool-approval-response
                break
              }

              input.push({
                type: 'function_call',
                call_id: part.toolCallId,
                name: toolName,
                arguments: JSON.stringify(part.input),
                id: itemId,
              })
              const toolCallResult = toolCallResultsByToolCallId[part.toolCallId]
              if (toolCallResult) {
                input.push({
                  type: 'function_call_output',
                  call_id: part.toolCallId,
                  output: convertToolResultOutputToString(toolCallResult.output),
                })
              }
              break
            }

            case 'tool-result': {
              input.push({
                type: 'function_call_output',
                call_id: part.toolCallId,
                output: convertToolResultOutputToString(part.output),
              })
              // Track this tool call ID - if it matches an MCP approval request,
              // we should NOT also generate an mcp_approval_response
              approvalIdsWithToolResult.add(part.toolCallId)
              break
            }

            case 'reasoning': {
              if (!itemId) break
              input.push({
                type: 'reasoning',
                summary: [{ type: 'summary_text', text: part.text }],
                id: itemId,
              })
              break
            }
          }
        }
        break

      case 'tool':
        // Handle tool-approval-response parts (from AI SDK v6 native tool approval)
        for (const part of content) {
          if (part.type === 'tool-approval-response') {
            // Skip if already processed
            if (processedApprovalIds.has(part.approvalId)) {
              continue
            }
            processedApprovalIds.add(part.approvalId)

            // Skip if there's already a function_call_output for this approval ID
            // The function_call_output implies the tool was approved and executed,
            // so we don't need a separate mcp_approval_response (which would cause
            // "each tool_use must have a single result" error)
            if (approvalIdsWithToolResult.has(part.approvalId)) {
              continue
            }

            input.push({
              type: 'mcp_approval_response',
              id: part.approvalId,
              approval_request_id: part.approvalId,
              approve: part.approved,
              ...(part.reason && { reason: part.reason }),
            })
          }
          // Note: tool-result parts are handled when processing the corresponding
          // tool-call in the assistant message, so we skip them here.
        }
        break

      default: {
        const _exhaustiveCheck: never = role
        throw new Error(`Unsupported role: ${String(_exhaustiveCheck)}`)
      }
    }
  }

  return { input, warnings }
}

const ProviderOptionsSchema = z.object({
  itemId: z.string().nullish(),
  toolName: z.string().nullish(),
  serverLabel: z.string().nullish(),
  approvalRequestId: z.string().nullish(),
})

export type ProviderOptions = z.infer<typeof ProviderOptionsSchema>

const convertToolResultOutputToString = (
  output: LanguageModelV3ToolResultPart['output']
): string => {
  switch (output.type) {
    case 'text':
    case 'error-text':
      return output.value
    case 'execution-denied':
      return output.reason ?? 'Execution denied'
    case 'json':
    case 'error-json':
    case 'content':
      return JSON.stringify(output.value)
  }
}
