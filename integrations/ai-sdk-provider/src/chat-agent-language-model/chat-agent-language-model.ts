import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
} from '@ai-sdk/provider'
import {
  type ParseResult,
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  createJsonErrorResponseHandler,
  postJsonToApi,
} from '@ai-sdk/provider-utils'
import { z } from 'zod/v4'
import type { DatabricksLanguageModelConfig } from '../databricks-provider'
import { chatAgentResponseSchema, chatAgentChunkSchema } from './chat-agent-schema'
import {
  convertChatAgentChunkToMessagePart,
  convertChatAgentResponseToMessagePart,
} from './chat-agent-convert-to-message-parts'
import { convertLanguageModelV3PromptToChatAgentResponse } from './chat-agent-convert-to-input'
import { getDatabricksLanguageModelTransformStream } from '../stream-transformers/databricks-stream-transformer'

export class DatabricksChatAgentLanguageModel implements LanguageModelV3 {
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
    const networkArgs = this.getArgs({
      config: this.config,
      options,
      stream: false,
      modelId: this.modelId,
    })

    const { value: response } = await postJsonToApi({
      ...networkArgs,
      successfulResponseHandler: createJsonResponseHandler(chatAgentResponseSchema),
      failedResponseHandler: createJsonErrorResponseHandler({
        errorSchema: z.any(),
        errorToMessage: (error) => JSON.stringify(error),
        isRetryable: () => false,
      }),
    })

    return {
      content: convertChatAgentResponseToMessagePart(response),
      finishReason: { raw: undefined, unified: 'stop' },
      usage: {
        inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 0, text: 0, reasoning: 0 },
      },
      warnings: [],
    }
  }

  async doStream(
    options: Parameters<LanguageModelV3['doStream']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV3['doStream']>>> {
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
      successfulResponseHandler: createEventSourceResponseHandler(chatAgentChunkSchema),
    })

    let finishReason: LanguageModelV3FinishReason = { raw: undefined, unified: 'other' }

    return {
      stream: response
        .pipeThrough(
          new TransformStream<
            ParseResult<z.infer<typeof chatAgentChunkSchema>>,
            LanguageModelV3StreamPart
          >({
            start(controller) {
              controller.enqueue({ type: 'stream-start', warnings: [] })
            },

            transform(chunk, controller) {
              if (options.includeRawChunks) {
                controller.enqueue({ type: 'raw', rawValue: chunk.rawValue })
              }

              // // handle failed chunk parsing / validation:
              if (!chunk.success) {
                finishReason = { raw: undefined, unified: 'error' }
                controller.enqueue({ type: 'error', error: chunk.error })
                return
              }

              const parts = convertChatAgentChunkToMessagePart(chunk.value)
              for (const part of parts) {
                controller.enqueue(part)
              }
            },

            flush(controller) {
              controller.enqueue({
                type: 'finish',
                finishReason,
                usage: {
                  inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
                  outputTokens: { total: 0, text: 0, reasoning: 0 },
                },
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
    options: LanguageModelV3CallOptions
    config: DatabricksLanguageModelConfig
    stream: boolean
    modelId: string
  }) {
    return {
      body: {
        model: modelId,
        stream,
        messages: convertLanguageModelV3PromptToChatAgentResponse(options.prompt),
      },
      url: config.url({
        path: '/completions',
      }),
      headers: combineHeaders(config.headers(), options.headers),
      fetch: config.fetch,
      abortSignal: options.abortSignal,
    }
  }
}
