import {
  Output,
  generateText,
  streamText,
  type Agent,
  type GenerateTextResult,
  type ModelMessage,
  type StreamTextResult,
} from 'ai'
import type {
  LanguageModelV3,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
} from '@ai-sdk/provider'
import {
  createDatabricksGenieConversationClient,
  type DatabricksGenieConversationEvent,
  type DatabricksGenieAskResult,
  type DatabricksGenieConversationClient,
  type DatabricksGeniePollingSettings,
  type DatabricksGenieSettings,
} from './conversation-client'

export interface DatabricksGenieAgentCallOptions extends DatabricksGeniePollingSettings {
  conversationId?: string
  fetchQueryResult?: boolean
  headers?: Record<string, string>
}

export type DatabricksGenieAgentResult = GenerateTextResult<
  Record<string, never>,
  ReturnType<typeof Output.text>
> & {
  genie: DatabricksGenieAskResult
}

export type DatabricksGenieAgentStreamResult = StreamTextResult<
  Record<string, never>,
  ReturnType<typeof Output.text>
> & {
  genie: Promise<DatabricksGenieAskResult>
}

export interface DatabricksGenieAgent
  extends Agent<
    DatabricksGenieAgentCallOptions,
    Record<string, never>,
    ReturnType<typeof Output.text>
  > {
  readonly genie: DatabricksGenieConversationClient
  generate(
    options: Parameters<
      Agent<
        DatabricksGenieAgentCallOptions,
        Record<string, never>,
        ReturnType<typeof Output.text>
      >['generate']
    >[0]
  ): Promise<DatabricksGenieAgentResult>
  stream(
    options: Parameters<
      Agent<
        DatabricksGenieAgentCallOptions,
        Record<string, never>,
        ReturnType<typeof Output.text>
      >['stream']
    >[0]
  ): Promise<DatabricksGenieAgentStreamResult>
}

function extractLatestUserText(messages: ModelMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message.role !== 'user') {
      continue
    }

    if (typeof message.content === 'string' && message.content.trim().length > 0) {
      return message.content.trim()
    }

    if (Array.isArray(message.content)) {
      const text = message.content
        .filter(
          (
            part
          ): part is Extract<(typeof message.content)[number], { type: 'text'; text: string }> =>
            typeof part === 'object' &&
            part !== null &&
            'type' in part &&
            part.type === 'text' &&
            'text' in part &&
            typeof part.text === 'string'
        )
        .map((part) => part.text.trim())
        .filter((part) => part.length > 0)
        .join('\n')

      if (text.length > 0) {
        return text
      }
    }
  }

  throw new Error('Databricks Genie agent requires at least one user message with text content')
}

function resolvePromptMessages(prompt?: string | ModelMessage[], messages?: ModelMessage[]): ModelMessage[] {
  if (typeof prompt === 'string') {
    return [{ role: 'user', content: prompt }]
  }

  if (Array.isArray(prompt)) {
    return prompt
  }

  if (Array.isArray(messages)) {
    return messages
  }

  throw new Error('Databricks Genie agent requires either prompt or messages')
}

function resolveAgentText(result: DatabricksGenieAskResult): string {
  if (result.text.trim().length > 0) {
    return result.text
  }

  if (result.sql) {
    return 'Genie generated a SQL query for this request.'
  }

  return 'Genie completed the request.'
}

function formatGenieStatus(status: string): string {
  return status
    .toLowerCase()
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

class DatabricksGenieLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = 'v3'
  readonly modelId = 'genie-conversation'
  readonly supportedUrls: Record<string, RegExp[]> = {}

  private geniePromise?: Promise<DatabricksGenieAskResult>
  private streamEventListener?: (event: DatabricksGenieConversationEvent) => void

  constructor(
    private readonly helper: DatabricksGenieConversationClient,
    private readonly question: string,
    private readonly requestOptions: DatabricksGenieAgentCallOptions,
    private readonly abortSignal?: AbortSignal
  ) {}

  get provider(): string {
    return 'databricks'
  }

  get result(): Promise<DatabricksGenieAskResult> {
    if (!this.geniePromise) {
      this.geniePromise = this.helper.ask(this.question, {
        ...this.requestOptions,
        abortSignal: this.abortSignal,
      })
    }

    return this.geniePromise
  }

  prepareStreamingResult(onEvent?: (event: DatabricksGenieConversationEvent) => void) {
    this.streamEventListener = onEvent

    if (!this.geniePromise) {
      this.geniePromise = (async () => {
        let completedResult: DatabricksGenieAskResult | undefined

        for await (const event of this.helper.streamConversation(this.question, {
          ...this.requestOptions,
          abortSignal: this.abortSignal,
        })) {
          this.streamEventListener?.(event)

          if (event.type === 'complete') {
            completedResult = event.result
          }
        }

        if (!completedResult) {
          throw new Error('Databricks Genie conversation ended without a completed result')
        }

        return completedResult
      })()
    }

    return this.geniePromise
  }

  async doGenerate(
    _options: Parameters<LanguageModelV3['doGenerate']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV3['doGenerate']>>> {
    const genie = await this.result

    return {
      content: [{ type: 'text', text: resolveAgentText(genie) }],
      finishReason: { raw: 'stop', unified: 'stop' },
      usage: {
        inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 0, text: 0, reasoning: 0 },
      },
      warnings: [],
      response: {
        body: genie,
      },
    }
  }

  doStream(
    _options: Parameters<LanguageModelV3['doStream']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV3['doStream']>>> {
    let finishReason: LanguageModelV3FinishReason = { raw: 'stop', unified: 'stop' }
    const textId = 'genie-text-0'
    const reasoningId = 'genie-reasoning-0'

    return Promise.resolve({
      stream: new ReadableStream<LanguageModelV3StreamPart>({
        start: async (controller) => {
          controller.enqueue({ type: 'stream-start', warnings: [] })
          controller.enqueue({ type: 'reasoning-start', id: reasoningId })

          try {
            const geniePromise = this.prepareStreamingResult((event) => {
              if (event.type === 'status') {
                controller.enqueue({
                  type: 'reasoning-delta',
                  id: reasoningId,
                  delta: `${formatGenieStatus(event.status)}\n`,
                })
              }
            })
            const genie = await geniePromise

            controller.enqueue({ type: 'reasoning-end', id: reasoningId })

            controller.enqueue({ type: 'text-start', id: textId })
            controller.enqueue({
              type: 'text-delta',
              id: textId,
              delta: resolveAgentText(genie),
            })
            controller.enqueue({ type: 'text-end', id: textId })
          } catch (error) {
            finishReason = { raw: 'error', unified: 'error' }
            controller.enqueue({ type: 'reasoning-end', id: reasoningId })
            controller.enqueue({ type: 'error', error })
          } finally {
            controller.enqueue({
              type: 'finish',
              finishReason,
              usage: {
                inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
                outputTokens: { total: 0, text: 0, reasoning: 0 },
              },
            })
            controller.close()
          }
        },
      }),
      request: {
        body: {
          question: this.question,
        },
      },
      response: {
        headers: {},
      },
    })
  }
}

class DefaultDatabricksGenieAgent implements DatabricksGenieAgent {
  readonly version = 'agent-v1' as const
  readonly tools = {}
  readonly genie: DatabricksGenieConversationClient
  readonly id: string | undefined

  constructor(settings: DatabricksGenieSettings & { id?: string }) {
    this.id = settings.id
    this.genie = createDatabricksGenieConversationClient(settings)
  }

  async generate({
    prompt,
    messages,
    options,
    abortSignal,
  }: Parameters<DatabricksGenieAgent['generate']>[0]): Promise<DatabricksGenieAgentResult> {
    const promptMessages = resolvePromptMessages(prompt, messages)
    const question = extractLatestUserText(promptMessages)
    const model = new DatabricksGenieLanguageModel(this.genie, question, options ?? {}, abortSignal)
    const result = await generateText({
      model,
      prompt: question,
      abortSignal,
    })

    return Object.assign(result, {
      genie: await model.result,
    }) as unknown as DatabricksGenieAgentResult
  }

  stream({
    prompt,
    messages,
    options,
    abortSignal,
    experimental_transform,
  }: Parameters<DatabricksGenieAgent['stream']>[0]): Promise<DatabricksGenieAgentStreamResult> {
    const promptMessages = resolvePromptMessages(prompt, messages)
    const question = extractLatestUserText(promptMessages)
    const model = new DatabricksGenieLanguageModel(this.genie, question, options ?? {}, abortSignal)
    const geniePromise = model.prepareStreamingResult()
    const result = streamText({
      model,
      prompt: question,
      abortSignal,
      experimental_transform,
    })

    return Promise.resolve(Object.assign(result, {
      genie: geniePromise,
    }) as unknown as DatabricksGenieAgentStreamResult)
  }
}

export function createDatabricksGenieAgent(
  settings: DatabricksGenieSettings & { id?: string }
): DatabricksGenieAgent {
  return new DefaultDatabricksGenieAgent(settings)
}
