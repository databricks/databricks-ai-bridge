import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2FinishReason,
  LanguageModelV2FunctionTool,
  LanguageModelV2StreamPart,
  LanguageModelV2ToolChoice,
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
} from './fmapi-convert-to-message-parts'
import { convertPromptToFmapiMessages } from './fmapi-convert-to-input'
import { getDatabricksLanguageModelTransformStream } from '../stream-transformers/databricks-stream-transformer'
import { DATABRICKS_TOOL_CALL_ID } from '../tools'

export class DatabricksFmapiLanguageModel implements LanguageModelV2 {
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
    const networkArgs = this.getArgs({
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
    let finishReason: LanguageModelV2FinishReason = 'stop'
    if (choice?.finish_reason === 'tool_calls') {
      finishReason = 'tool-calls'
    }

    return {
      content: convertFmapiResponseToMessagePart(response),
      finishReason,
      usage: {
        inputTokens: response.usage?.prompt_tokens ?? 0,
        outputTokens: response.usage?.completion_tokens ?? 0,
        totalTokens: response.usage?.total_tokens ?? 0,
      },
      warnings: [],
    }
  }

  async doStream(
    options: Parameters<LanguageModelV2['doStream']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV2['doStream']>>> {
    const networkArgs = this.getArgs({
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

    let finishReason: LanguageModelV2FinishReason = 'unknown'
    let usage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 }

    // Track tool call IDs by index for streaming (OpenAI only sends ID in first chunk)
    const toolCallIdsByIndex = new Map<number, string>()
    // Track tool call names by ID
    const toolCallNamesById = new Map<string, string>()
    // Track accumulated tool call inputs by ID
    const toolCallInputsById = new Map<string, string>()

    return {
      stream: response
        .pipeThrough(
          new TransformStream<
            ParseResult<z.infer<typeof fmapiChunkSchema>>,
            LanguageModelV2StreamPart
          >({
            start(controller) {
              controller.enqueue({ type: 'stream-start', warnings: [] })
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

              // Track finish reason from chunk
              const choice = chunk.value.choices[0]
              if (choice?.finish_reason === 'stop') {
                finishReason = 'stop'
              } else if (choice?.finish_reason === 'tool_calls') {
                finishReason = 'tool-calls'
              }

              // Track usage from chunk
              if (chunk.value.usage) {
                usage = {
                  inputTokens: chunk.value.usage.prompt_tokens ?? 0,
                  outputTokens: chunk.value.usage.completion_tokens ?? 0,
                  totalTokens: chunk.value.usage.total_tokens ?? 0,
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
              const toolCalls: Array<{ toolCallId: string; toolName: string; input: string }> = []
              for (const [toolCallId, inputText] of toolCallInputsById) {
                const toolName = toolCallNamesById.get(toolCallId)
                if (toolName) {
                  toolCalls.push({ toolCallId, toolName, input: inputText })
                  // Emit tool-input-end to signal streaming is complete
                  controller.enqueue({ type: 'tool-input-end', id: toolCallId })

                  // Emit a complete tool-call with raw input string
                  // (AI SDK will parse it internally)
                  controller.enqueue({
                    type: 'tool-call',
                    toolCallId,
                    toolName,
                    input: inputText,
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

  private getArgs({
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
    // Convert tools to OpenAI format (filter out provider-defined tools)
    const tools = options.tools
      ?.map((tool) => convertToolToOpenAIFormat(tool))
      .filter((tool): tool is NonNullable<typeof tool> => tool !== undefined)

    // Convert tool choice to OpenAI format
    const toolChoice = options.toolChoice
      ? convertToolChoiceToOpenAIFormat(options.toolChoice)
      : undefined

    return {
      url: config.url({
        path: '/chat/completions',
      }),
      headers: combineHeaders(config.headers(), options.headers),
      body: {
        messages: convertPromptToFmapiMessages(options.prompt).messages,
        stream,
        model: modelId,
        ...(tools && tools.length > 0 ? { tools } : {}),
        ...(toolChoice && tools && tools.length > 0 ? { tool_choice: toolChoice } : {}),
        ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
        ...(options.maxOutputTokens !== undefined ? { max_tokens: options.maxOutputTokens } : {}),
        ...(options.stopSequences && options.stopSequences.length > 0
          ? { stop: options.stopSequences }
          : {}),
      },
      fetch: config.fetch,
    }
  }
}

/**
 * Convert AI SDK tool to OpenAI format
 */
function convertToolToOpenAIFormat(
  tool: LanguageModelV2FunctionTool | { type: 'provider-defined'; id: string }
):
  | { type: 'function'; function: { name: string; description?: string; parameters?: unknown } }
  | undefined {
  if (tool.type === 'provider-defined' || tool.name === DATABRICKS_TOOL_CALL_ID) {
    // Skip provider-defined tools as they're not supported in OpenAI format
    // or tools that are orchestrated by Databricks' agents
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
  toolChoice: LanguageModelV2ToolChoice
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
