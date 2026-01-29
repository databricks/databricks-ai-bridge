import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3FinishReason,
  LanguageModelV3FunctionTool,
  LanguageModelV3ProviderTool,
  LanguageModelV3StreamPart,
  LanguageModelV3ToolChoice,
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
import { fmapiChunkSchema, fmapiResponseSchema } from './fmapi-schema'
import {
  convertFmapiChunkToMessagePart,
  convertFmapiResponseToMessagePart,
  type FmapiConvertOptions,
} from './fmapi-convert-to-message-parts'
import { convertPromptToFmapiMessages } from './fmapi-convert-to-input'
import { getDatabricksLanguageModelTransformStream } from '../stream-transformers/databricks-stream-transformer'
import { mapFmapiFinishReason } from './fmapi-finish-reason'
import { callOptionsToFmapiArgs } from './call-options-to-fmapi-args'

export class DatabricksFmapiLanguageModel implements LanguageModelV3 {
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
      successfulResponseHandler: createJsonResponseHandler(fmapiResponseSchema),
      failedResponseHandler: createJsonErrorResponseHandler({
        errorSchema: z.any(),
        errorToMessage: (error) => JSON.stringify(error),
        isRetryable: () => false,
      }),
    })

    // Determine finish reason from response
    const choice = response.choices[0]
    const finishReason = mapFmapiFinishReason(choice?.finish_reason)

    const useRemoteToolCalling = this.config.useRemoteToolCalling ?? true
    return {
      content: convertFmapiResponseToMessagePart(response, { useRemoteToolCalling }),
      finishReason,
      usage: {
        inputTokens: {
          total: response.usage?.prompt_tokens ?? 0,
          noCache: 0,
          cacheRead: 0,
          cacheWrite: 0,
        },
        outputTokens: {
          total: response.usage?.completion_tokens ?? 0,
          text: 0,
          reasoning: 0,
        },
      },
      warnings,
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
      successfulResponseHandler: createEventSourceResponseHandler(fmapiChunkSchema),
      abortSignal: options.abortSignal,
    })

    let finishReason: LanguageModelV3FinishReason = {
      raw: undefined,
      unified: 'other',
    }
    let usage: LanguageModelV3Usage = {
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
    // Track tool call IDs by index for streaming (OpenAI only sends ID in first chunk)
    const toolCallIdsByIndex = new Map<number, string>()
    // Track tool call names by ID
    const toolCallNamesById = new Map<string, string>()
    // Track accumulated tool call inputs by ID
    const toolCallInputsById = new Map<string, string>()
    const useRemoteToolCalling = this.config.useRemoteToolCalling ?? true

    return {
      stream: response
        .pipeThrough(
          new TransformStream<
            ParseResult<z.infer<typeof fmapiChunkSchema>>,
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
                finishReason = {
                  raw: undefined,
                  unified: 'error',
                }
                controller.enqueue({ type: 'error', error: chunk.error })
                return
              }

              // Track finish reason from chunk
              const choice = chunk.value.choices[0]
              finishReason = mapFmapiFinishReason(choice?.finish_reason)

              // Track usage from chunk
              if (chunk.value.usage) {
                usage = {
                  inputTokens: {
                    total: chunk.value.usage.prompt_tokens ?? 0,
                    noCache: 0,
                    cacheRead: 0,
                    cacheWrite: 0,
                  },
                  outputTokens: {
                    total: chunk.value.usage.completion_tokens ?? 0,
                    text: 0,
                    reasoning: 0,
                  },
                }
              }

              const parts = convertFmapiChunkToMessagePart(chunk.value, toolCallIdsByIndex)
              for (const part of parts) {
                // Track tool call info for later emission
                if (part.type === 'tool-input-start') {
                  toolCallNamesById.set(part.id, part.toolName)
                  toolCallInputsById.set(part.id, '')
                } else if (part.type === 'tool-input-delta') {
                  const current = toolCallInputsById.get(part.id) ?? ''
                  toolCallInputsById.set(part.id, current + part.delta)
                }
                controller.enqueue(part)
              }
            },

            flush(controller) {
              // Emit complete tool-call events for all accumulated tool calls
              for (const [toolCallId, inputText] of toolCallInputsById) {
                const toolName = toolCallNamesById.get(toolCallId)
                if (toolName) {
                  // Emit tool-input-end to signal streaming is complete
                  controller.enqueue({ type: 'tool-input-end', id: toolCallId })

                  // Emit a complete tool-call with actual tool name
                  controller.enqueue({
                    type: 'tool-call',
                    toolCallId,
                    toolName,
                    input: inputText,
                    ...(useRemoteToolCalling && {
                      dynamic: true,
                      providerExecuted: true,
                    }),
                  })
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
      response: { headers: responseHeaders },
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
    // Convert tools to OpenAI format (filter out provider-defined tools)
    const tools = options.tools
      ?.map((tool) => convertToolToOpenAIFormat(tool))
      .filter((tool): tool is NonNullable<typeof tool> => tool !== undefined)

    // Convert tool choice to OpenAI format
    const toolChoice = options.toolChoice
      ? convertToolChoiceToOpenAIFormat(options.toolChoice)
      : undefined

    const { messages } = await convertPromptToFmapiMessages(options.prompt)

    // Convert call options to FMAPI args
    const { args: callArgs, warnings } = callOptionsToFmapiArgs(options)

    return {
      url: config.url({
        path: '/chat/completions',
      }),
      headers: combineHeaders(config.headers(), options.headers),
      body: {
        messages,
        stream,
        model: modelId,
        ...(tools && tools.length > 0 ? { tools } : {}),
        ...(toolChoice && tools && tools.length > 0 ? { tool_choice: toolChoice } : {}),
        ...callArgs,
      },
      warnings,
      fetch: config.fetch,
    }
  }
}

/**
 * Convert AI SDK tool to OpenAI format
 */
function convertToolToOpenAIFormat(
  tool: LanguageModelV3FunctionTool | LanguageModelV3ProviderTool
):
  | { type: 'function'; function: { name: string; description?: string; parameters?: unknown } }
  | undefined {
  if (tool.type === 'provider') {
    // Skip provider-defined tools as they're not supported in OpenAI format
    return undefined
  }
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema,
    },
  }
}

/**
 * Convert AI SDK tool choice to OpenAI format
 */
function convertToolChoiceToOpenAIFormat(
  toolChoice: LanguageModelV3ToolChoice
): 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } } {
  if (toolChoice.type === 'auto') {
    return 'auto'
  }
  if (toolChoice.type === 'none') {
    return 'none'
  }
  if (toolChoice.type === 'required') {
    return 'required'
  }
  if (toolChoice.type === 'tool') {
    return {
      type: 'function',
      function: { name: toolChoice.toolName },
    }
  }
  return 'auto'
}
