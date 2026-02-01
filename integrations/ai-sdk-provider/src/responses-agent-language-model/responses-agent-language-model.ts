import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
} from '@ai-sdk/provider'
import {
  type ParseResult,
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  postJsonToApi,
} from '@ai-sdk/provider-utils'
import { z } from 'zod/v4'
import type { DatabricksLanguageModelConfig } from '../databricks-provider'
import {
  responsesAgentResponseSchema,
  looseResponseAgentChunkSchema,
  type responsesAgentChunkSchema,
} from './responses-agent-schema'
import {
  convertResponsesAgentChunkToMessagePart,
  convertResponsesAgentResponseToMessagePart,
} from './responses-convert-to-message-parts'
import { convertToResponsesInput } from './responses-convert-to-input'
import { getDatabricksLanguageModelTransformStream } from '../stream-transformers/databricks-stream-transformer'
import { prepareResponsesTools } from './responses-prepare-tools'
import { callOptionsToResponsesArgs } from './call-options-to-responses-args'

function mapResponsesFinishReason({
  finishReason,
  hasToolCalls,
}: {
  finishReason: string | null | undefined
  hasToolCalls: boolean
}): LanguageModelV3FinishReason {
  let unified: LanguageModelV3FinishReason['unified']
  switch (finishReason) {
    case undefined:
    case null:
      unified = hasToolCalls ? 'tool-calls' : 'stop'
      break
    case 'max_output_tokens':
      unified = 'length'
      break
    case 'content_filter':
      unified = 'content-filter'
      break
    default:
      unified = hasToolCalls ? 'tool-calls' : 'other'
  }
  return { raw: finishReason ?? undefined, unified }
}

export class DatabricksResponsesAgentLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = 'v3'

  readonly modelId: string

  private readonly config: DatabricksLanguageModelConfig

  constructor(modelId: string, config: DatabricksLanguageModelConfig) {
    this.modelId = modelId
    this.config = config
  }

  get provider(): string {
    return this.config.provider
  }

  readonly supportedUrls: Record<string, RegExp[]> = {}

  async doGenerate(
    options: Parameters<LanguageModelV3['doGenerate']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV3['doGenerate']>>> {
    const { warnings, ...networkArgs } = await this.getArgs({
      config: this.config,
      options,
      stream: false,
      modelId: this.modelId,
    })

    const { value: response } = await postJsonToApi({
      ...networkArgs,
      successfulResponseHandler: createJsonResponseHandler(responsesAgentResponseSchema),
      failedResponseHandler: createJsonErrorResponseHandler({
        errorSchema: z.any(),
        errorToMessage: (error) => JSON.stringify(error),
        isRetryable: () => false,
      }),
    })

    const content = convertResponsesAgentResponseToMessagePart(response)
    const hasToolCalls = content.some((p) => p.type === 'tool-call')

    // Extract trace_id from both root level (legacy) and nested databricks_output structure
    const traceId =
      response.trace_id ?? response.databricks_output?.trace?.info?.trace_id ?? undefined
    const spanId = response.span_id ?? undefined

    // Create a normalized response body with trace_id at root level for easier access
    const responseBody = {
      ...response,
      ...(traceId && { trace_id: traceId }),
      ...(spanId && { span_id: spanId }),
    }

    return {
      content,
      finishReason: mapResponsesFinishReason({
        finishReason: response.incomplete_details?.reason,
        hasToolCalls,
      }),
      usage: {
        inputTokens: {
          total: response.usage?.input_tokens ?? 0,
          noCache: 0,
          cacheRead: 0,
          cacheWrite: 0,
        },
        outputTokens: { total: response.usage?.output_tokens ?? 0, text: 0, reasoning: 0 },
      },
      warnings,
      response: {
        body: responseBody,
      },
    }
  }

  async doStream(
    options: Parameters<LanguageModelV3['doStream']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV3['doStream']>>> {
    const { warnings, ...networkArgs } = await this.getArgs({
      config: this.config,
      options,
      stream: true,
      modelId: this.modelId,
    })

    const { responseHeaders, value: response } = await postJsonToApi({
      ...networkArgs,
      failedResponseHandler: createJsonErrorResponseHandler({
        errorSchema: z.any(),
        errorToMessage: (error) => JSON.stringify(error),
        isRetryable: () => false,
      }),
      successfulResponseHandler: createEventSourceResponseHandler(looseResponseAgentChunkSchema),
      abortSignal: options.abortSignal,
    })

    let finishReason: LanguageModelV3FinishReason = {
      raw: undefined,
      unified: 'stop',
    }
    const usage: LanguageModelV3Usage = {
      inputTokens: {
        total: 0,
        noCache: 0,
        cacheRead: 0,
        cacheWrite: 0,
      },
      outputTokens: {
        total: 0,
        text: 0,
        reasoning: 0,
      },
    }

    const allParts: LanguageModelV3StreamPart[] = []
    const useRemoteToolCalling = this.config.useRemoteToolCalling ?? false
    // Track tool call IDs to tool names for looking up tool names in function_call_output events
    const toolNamesByCallId = new Map<string, string>()

    // Create a mutable object to capture trace_id and span_id from responses.completed event
    // This object will be mutated as the stream is consumed
    const responseBody: Record<string, unknown> = {}

    return {
      stream: response
        .pipeThrough(
          new TransformStream<
            ParseResult<z.infer<typeof responsesAgentChunkSchema>>,
            LanguageModelV3StreamPart
          >({
            start(controller) {
              controller.enqueue({ type: 'stream-start', warnings })
            },

            transform(chunk, controller) {
              if (options.includeRawChunks) {
                controller.enqueue({ type: 'raw', rawValue: chunk.rawValue })
              }

              // handle failed chunk parsing / validation:
              if (!chunk.success) {
                finishReason = { raw: undefined, unified: 'error' }
                controller.enqueue({ type: 'error', error: chunk.error })
                return
              }

              if (chunk.value.type === 'responses.completed') {
                const hasToolCalls = allParts.some((p) => p.type === 'tool-call')
                finishReason = mapResponsesFinishReason({
                  finishReason: chunk.value.response.incomplete_details?.reason,
                  hasToolCalls,
                })
                usage.inputTokens.total = chunk.value.response.usage.input_tokens
                usage.outputTokens.total = chunk.value.response.usage.output_tokens
                // Capture trace_id and span_id in the responseBody object
                if (chunk.value.response.trace_id !== undefined) {
                  responseBody.trace_id = chunk.value.response.trace_id
                }
                if (chunk.value.response.span_id !== undefined) {
                  responseBody.span_id = chunk.value.response.span_id
                }
                return
              }

              // Extract trace info from response.output_item.done event
              // The endpoint returns trace info in databricks_output.trace.info
              if (chunk.value.type === 'response.output_item.done') {
                const traceInfo = (chunk.value as any).databricks_output?.trace?.info
                if (traceInfo?.trace_id) {
                  responseBody.trace_id = traceInfo.trace_id
                  // Store full trace info for advanced use cases
                  responseBody.databricks_trace_info = traceInfo
                }
              }

              // Track tool call IDs to names from response.output_item.done function_call events
              if (
                chunk.value.type === 'response.output_item.done' &&
                chunk.value.item.type === 'function_call'
              ) {
                toolNamesByCallId.set(chunk.value.item.call_id, chunk.value.item.name)
              }

              const parts = convertResponsesAgentChunkToMessagePart(chunk.value, {
                useRemoteToolCalling,
                toolNamesByCallId,
              })

              allParts.push(...parts)
              /**
               * Check if the last chunk was a tool result without a tool call
               * This is a special case for MCP approval requests where the tool result
               * is sent in a separate call after the tool call was approved/denied.
               */
              if (parts.length === 0) {
                return
              }
              const part = parts[0]
              if (part.type === 'tool-result') {
                // First check if the tool call is in the current stream parts
                const matchingToolCallInParts = parts.find(
                  (c) => c.type === 'tool-call' && c.toolCallId === part.toolCallId
                )
                // Also check if the tool call was emitted earlier in this stream
                const matchingToolCallInStream = allParts.find(
                  (c) => c.type === 'tool-call' && c.toolCallId === part.toolCallId
                )
                if (!matchingToolCallInParts && !matchingToolCallInStream) {
                  // Find the tool call in the prompt (previous messages)
                  const toolCallFromPreviousMessages = options.prompt
                    .flatMap((message) => {
                      if (typeof message.content === 'string') return []
                      return message.content.filter(
                        (p): p is typeof p & { type: 'tool-call' } => p.type === 'tool-call'
                      )
                    })
                    .find((p) => p.toolCallId === part.toolCallId)
                  if (!toolCallFromPreviousMessages) {
                    throw new Error('No matching tool call found in previous message')
                  }
                  controller.enqueue({
                    type: 'tool-call',
                    toolCallId: toolCallFromPreviousMessages.toolCallId,
                    toolName: toolCallFromPreviousMessages.toolName,
                    input: JSON.stringify(toolCallFromPreviousMessages.input),
                    // Mark as provider-executed so AI SDK doesn't try to validate the tool
                    providerExecuted: true,
                    dynamic: true,
                  })
                }
              }
              // Dedupe logic for messages sent via response.output_item.done
              // MAS relies on sending text via response.output_item.done ONLY without any delta chunks
              // We have to decide when to display these messages in the UI
              if (shouldDedupeOutputItemDone(parts, allParts.slice(0, -parts.length))) {
                return
              }
              for (const part of parts) {
                controller.enqueue(part)
              }
            },

            flush(controller) {
              // When useRemoteToolCalling=false, find all tool calls that don't have
              // matching tool results and re-emit them with providerExecuted: true
              // so the AI SDK doesn't expect a client-side result for them.
              // When useRemoteToolCalling=true, tool calls already have providerExecuted: true.
              // Skip MCP approval requests since they intentionally pause for user approval.
              if (!useRemoteToolCalling) {
                const toolCalls = allParts.filter(
                  (p): p is Extract<LanguageModelV3StreamPart, { type: 'tool-call' }> =>
                    p.type === 'tool-call'
                )
                const toolResults = allParts.filter(
                  (p): p is Extract<LanguageModelV3StreamPart, { type: 'tool-result' }> =>
                    p.type === 'tool-result'
                )

                for (const toolCall of toolCalls) {
                  // Skip MCP approval requests - they intentionally wait for user approval
                  const isMcpApprovalRequest =
                    toolCall.providerMetadata?.databricks?.approvalRequestId != null
                  if (isMcpApprovalRequest) {
                    continue
                  }

                  const hasResult = toolResults.some((r) => r.toolCallId === toolCall.toolCallId)
                  if (!hasResult) {
                    // Re-emit the tool call with providerExecuted: true and dynamic: true
                    // This tells the AI SDK not to expect a client-side result
                    controller.enqueue({
                      ...toolCall,
                      providerExecuted: true,
                      dynamic: true,
                    })
                  }
                }
              }

              controller.enqueue({
                type: 'finish',
                finishReason,
                usage,
              })
            },
          })
        )
        .pipeThrough(getDatabricksLanguageModelTransformStream()),
      request: { body: networkArgs.body },
      response: {
        headers: responseHeaders,
        body: responseBody,
      },
    }
  }

  private async getArgs({
    config,
    options,
    stream,
    modelId,
  }: {
    options: LanguageModelV3CallOptions
    config: DatabricksLanguageModelConfig
    stream: boolean
    modelId: string
  }) {
    const { input } = await convertToResponsesInput({
      prompt: options.prompt,
      systemMessageMode: 'system',
    })

    // Prepare tools for the Responses API format
    const { tools, toolChoice } = prepareResponsesTools({
      tools: options.tools,
      toolChoice: options.toolChoice,
    })

    // Convert call options to Responses API args
    const { args: callArgs, warnings } = callOptionsToResponsesArgs(options)

    return {
      url: config.url({
        path: '/responses',
      }),
      headers: combineHeaders(config.headers(), options.headers),
      body: {
        model: modelId,
        input,
        stream,
        ...(tools ? { tools } : {}),
        ...(toolChoice && tools ? { tool_choice: toolChoice } : {}),
        ...callArgs,
      },
      warnings,
      fetch: config.fetch,
    }
  }
}

export function shouldDedupeOutputItemDone(
  incomingParts: LanguageModelV3StreamPart[],
  previousParts: LanguageModelV3StreamPart[]
): boolean {
  // Determine if the incoming parts contain a text-delta that is a response.output_item.done
  const doneTextDelta = incomingParts.find(
    (p) =>
      p.type === 'text-delta' &&
      p.providerMetadata?.databricks?.itemType === 'response.output_item.done'
  )

  // If the incoming parts do not contain a text-delta that is a response.output_item.done, return false
  if (!doneTextDelta || doneTextDelta.type !== 'text-delta' || !doneTextDelta.id) {
    return false
  }

  /**
   * To determine if the text in response.output_item.done is a duplicate, we need to reconstruct the text from the
   * previous consecutive text-deltas and check if the .done text is already present in what we've streamed.
   *
   * The caveat is that the response.output_item.done text uses GFM footnote syntax, where as the streamed content
   * uses response.output_text.delta and response.output_text.annotation.added events. So we reconstruct all the
   * delta text and check if the .done text is contained in it (meaning we've already streamed it).
   *
   * We only consider text-deltas that came AFTER the last response.output_item.done event, since each .done
   * corresponds to a specific message and we should only compare against text streamed for that message.
   */
  // 1. Find the index after the last response.output_item.done event using findLastIndex
  const lastDoneIndex = previousParts.findLastIndex(
    (part) =>
      part.type === 'text-delta' &&
      part.providerMetadata?.databricks?.itemType === 'response.output_item.done'
  )
  const partsAfterLastDone = previousParts.slice(lastDoneIndex + 1)

  // 2. Reconstruct text blocks from parts after the last .done event, separated by non-text-delta parts
  const { texts: reconstructuredTexts, current } = partsAfterLastDone.reduce<{
    texts: string[]
    current: string
  }>(
    (acc, part) => {
      if (part.type === 'text-delta') {
        return { ...acc, current: acc.current + part.delta }
      } else if (acc.current.trim().length > 0) {
        return { texts: [...acc.texts, acc.current.trim()], current: '' }
      }
      return acc
    },
    { texts: [], current: '' }
  )
  // Only push current if it has content (avoid pushing empty strings)
  if (current.length > 0) {
    reconstructuredTexts.push(current)
  }

  // 3. Check if the .done text contains all reconstructed text blocks in order
  // If there are no text-deltas to compare against, don't dedupe - this is new content
  if (reconstructuredTexts.length === 0) {
    return false
  }

  const allTextsFoundInOrder = reconstructuredTexts.reduce<{ found: boolean; lastIndex: number }>(
    (acc, text) => {
      if (!acc.found) return acc
      const index = doneTextDelta.delta.indexOf(text, acc.lastIndex)
      if (index === -1) return { found: false, lastIndex: acc.lastIndex }
      return { found: true, lastIndex: index + text.length }
    },
    { found: true, lastIndex: 0 }
  )

  return allTextsFoundInOrder.found
}
