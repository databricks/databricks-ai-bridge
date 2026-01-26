import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
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
}): LanguageModelV2FinishReason {
  switch (finishReason) {
    case undefined:
    case null:
      return hasToolCalls ? 'tool-calls' : 'stop'
    case 'max_output_tokens':
      return 'length'
    case 'content_filter':
      return 'content-filter'
    default:
      return hasToolCalls ? 'tool-calls' : 'other'
  }
}

export class DatabricksResponsesAgentLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = 'v2'

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
    options: Parameters<LanguageModelV2['doGenerate']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV2['doGenerate']>>> {
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

    return {
      content,
      finishReason: mapResponsesFinishReason({
        finishReason: response.incomplete_details?.reason,
        hasToolCalls,
      }),
      usage: {
        inputTokens: response.usage?.input_tokens ?? 0,
        outputTokens: response.usage?.output_tokens ?? 0,
        totalTokens: response.usage?.total_tokens ?? 0,
      },
      warnings,
    }
  }

  async doStream(
    options: Parameters<LanguageModelV2['doStream']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV2['doStream']>>> {
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

    let finishReason: LanguageModelV2FinishReason = 'unknown'
    const usage: LanguageModelV2Usage = {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    }

    const allParts: LanguageModelV2StreamPart[] = []

    return {
      stream: response
        .pipeThrough(
          new TransformStream<
            ParseResult<z.infer<typeof responsesAgentChunkSchema>>,
            LanguageModelV2StreamPart
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
                finishReason = 'error'
                controller.enqueue({ type: 'error', error: chunk.error })
                return
              }

              if (chunk.value.type === 'responses.completed') {
                const hasToolCalls = allParts.some((p) => p.type === 'tool-call')
                finishReason = mapResponsesFinishReason({
                  finishReason: chunk.value.response.incomplete_details?.reason,
                  hasToolCalls,
                })
                usage.inputTokens = chunk.value.response.usage.input_tokens
                usage.outputTokens = chunk.value.response.usage.output_tokens
                usage.totalTokens = chunk.value.response.usage.total_tokens
                return
              }

              const parts = convertResponsesAgentChunkToMessagePart(chunk.value)

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
                      return message.content
                    })
                    .find((p) => p.type === 'tool-call' && p.toolCallId === part.toolCallId)
                  if (!toolCallFromPreviousMessages) {
                    throw new Error('No matching tool call found in previous message')
                  }
                  if (toolCallFromPreviousMessages.type === 'tool-call') {
                    controller.enqueue({
                      ...toolCallFromPreviousMessages,
                      input: JSON.stringify(toolCallFromPreviousMessages.input),
                    })
                  }
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
      response: { headers: responseHeaders },
    }
  }

  private async getArgs({
    config,
    options,
    stream,
    modelId,
  }: {
    options: LanguageModelV2CallOptions
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
  incomingParts: LanguageModelV2StreamPart[],
  previousParts: LanguageModelV2StreamPart[]
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
  reconstructuredTexts.push(current)

  // 3. Check if the .done text contains all reconstructed text blocks in order
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
