import { DatabricksGenieApiClient, type DatabricksGenieRequestOptions } from './client'
import {
  DATABRICKS_GENIE_TERMINAL_MESSAGE_STATUSES,
  type DatabricksGenieAttachment,
  type DatabricksGenieCreateConversationMessageRequest,
  type DatabricksGenieDeleteConversationRequest,
  type DatabricksGenieExecuteMessageAttachmentQueryRequest,
  type DatabricksGenieGenerateDownloadFullQueryResultRequest,
  type DatabricksGenieGenerateDownloadFullQueryResultResponse,
  type DatabricksGenieGetConversationMessageRequest,
  type DatabricksGenieGetDownloadFullQueryResultRequest,
  type DatabricksGenieGetDownloadFullQueryResultResponse,
  type DatabricksGenieGetMessageAttachmentQueryResultRequest,
  type DatabricksGenieGetQueryResultByAttachmentRequest,
  type DatabricksGenieGetSpaceRequest,
  type DatabricksGenieListConversationMessagesRequest,
  type DatabricksGenieListConversationMessagesResponse,
  type DatabricksGenieListConversationsRequest,
  type DatabricksGenieListConversationsResponse,
  type DatabricksGenieMessage,
  type DatabricksGenieMessageStatus,
  type DatabricksGenieNormalizedAttachment,
  type DatabricksGenieQueryResult,
  type DatabricksGenieSampleQuestion,
  type DatabricksGenieSpace,
  type DatabricksGenieStartConversationRequest,
  type DatabricksGenieStartConversationResponse,
} from './types'
import {
  DEFAULT_GENIE_POLLING_SETTINGS,
  type DatabricksGeniePollingSettings,
  type DatabricksGenieSettings,
  resolveDatabricksGenieSettings,
  sleep,
} from './shared'

export interface DatabricksGenieAskOptions extends DatabricksGeniePollingSettings {
  conversationId?: string
  fetchQueryResult?: boolean
  headers?: Record<string, string>
  abortSignal?: AbortSignal
  onProgress?: (message: DatabricksGenieMessage) => void
}

export interface DatabricksGenieAskResult {
  attachmentEntries: DatabricksGenieAttachment[]
  attachmentId?: string
  attachments: DatabricksGenieNormalizedAttachment[]
  conversationId: string
  message: DatabricksGenieMessage
  messageId: string
  queryResult?: DatabricksGenieQueryResult
  sql?: string
  status: DatabricksGenieMessageStatus
  suggestedQuestions: string[]
  text: string
}

export interface DatabricksGenieConversationEventBase {
  conversationId: string
  message: DatabricksGenieMessage
  messageId: string
}

export interface DatabricksGenieStatusEvent extends DatabricksGenieConversationEventBase {
  status: DatabricksGenieMessageStatus
  type: 'status'
}

export interface DatabricksGenieAttachmentEvent extends DatabricksGenieConversationEventBase {
  attachment: DatabricksGenieNormalizedAttachment
  type: 'attachment'
}

export interface DatabricksGenieQueryResultEvent extends DatabricksGenieConversationEventBase {
  attachmentId: string
  queryResult: DatabricksGenieQueryResult
  type: 'query-result'
}

export interface DatabricksGenieCompleteEvent extends DatabricksGenieConversationEventBase {
  result: DatabricksGenieAskResult
  type: 'complete'
}

export type DatabricksGenieConversationEvent =
  | DatabricksGenieStatusEvent
  | DatabricksGenieAttachmentEvent
  | DatabricksGenieQueryResultEvent
  | DatabricksGenieCompleteEvent

export interface DatabricksGenieConversationClient {
  getSpace(
    request?: DatabricksGenieGetSpaceRequest | DatabricksGenieRequestOptions,
    options?: DatabricksGenieRequestOptions
  ): Promise<DatabricksGenieSpace>
  getSampleQuestions(options?: DatabricksGenieRequestOptions): Promise<DatabricksGenieSampleQuestion[]>
  startConversation(
    question: string,
    options?: DatabricksGenieRequestOptions
  ): Promise<DatabricksGenieStartConversationResponse>
  createMessage(
    conversationId: string,
    question: string,
    options?: DatabricksGenieRequestOptions
  ): Promise<DatabricksGenieMessage>
  getMessage(
    conversationId: string,
    messageId: string,
    options?: DatabricksGenieRequestOptions
  ): Promise<DatabricksGenieMessage>
  listConversationMessages(
    request: DatabricksGenieListConversationMessagesRequest,
    options?: DatabricksGenieRequestOptions
  ): Promise<DatabricksGenieListConversationMessagesResponse>
  listConversations(
    request?: DatabricksGenieListConversationsRequest,
    options?: DatabricksGenieRequestOptions
  ): Promise<DatabricksGenieListConversationsResponse>
  deleteConversation(
    conversationId: string,
    options?: DatabricksGenieRequestOptions
  ): Promise<void>
  waitForCompletion(
    conversationId: string,
    messageId: string,
    options?: DatabricksGenieAskOptions
  ): Promise<DatabricksGenieMessage>
  getQueryResult(
    conversationId: string,
    messageId: string,
    attachmentId: string,
    options?: DatabricksGenieRequestOptions
  ): Promise<DatabricksGenieQueryResult>
  getQueryResultByAttachment(
    request: DatabricksGenieGetQueryResultByAttachmentRequest,
    options?: DatabricksGenieRequestOptions
  ): Promise<DatabricksGenieQueryResult>
  executeQueryAttachment(
    request: DatabricksGenieExecuteMessageAttachmentQueryRequest,
    options?: DatabricksGenieRequestOptions
  ): Promise<DatabricksGenieQueryResult>
  generateDownloadFullQueryResult(
    request: DatabricksGenieGenerateDownloadFullQueryResultRequest,
    options?: DatabricksGenieRequestOptions
  ): Promise<DatabricksGenieGenerateDownloadFullQueryResultResponse>
  getDownloadFullQueryResult(
    request: DatabricksGenieGetDownloadFullQueryResultRequest,
    options?: DatabricksGenieRequestOptions
  ): Promise<DatabricksGenieGetDownloadFullQueryResultResponse>
  streamConversation(
    question: string,
    options?: DatabricksGenieAskOptions
  ): AsyncIterable<DatabricksGenieConversationEvent>
  ask(question: string, options?: DatabricksGenieAskOptions): Promise<DatabricksGenieAskResult>
}

function isGenieRequestOptions(
  value: DatabricksGenieGetSpaceRequest | DatabricksGenieRequestOptions
): value is DatabricksGenieRequestOptions {
  return 'headers' in value || 'abortSignal' in value
}

function extractAskResult(
  message: DatabricksGenieMessage,
  queryResult?: DatabricksGenieQueryResult
): DatabricksGenieAskResult {
  const textAttachments = message.normalizedAttachments.filter(
    (attachment) => attachment.type === 'text'
  )
  const queryAttachments = message.normalizedAttachments.filter(
    (attachment) => attachment.type === 'query'
  )
  const suggestedQuestionsAttachments = message.normalizedAttachments.filter(
    (attachment) => attachment.type === 'suggested_questions'
  )
  const queryAttachment = queryAttachments[0]
  const preferredTextAttachments = textAttachments.filter(
    (attachment) => attachment.text?.purpose !== 'FOLLOW_UP_QUESTION'
  )
  const textSourceAttachments =
    preferredTextAttachments.length > 0 ? preferredTextAttachments : textAttachments
  const text = textSourceAttachments
    .map((attachment) => attachment.text?.content?.trim())
    .filter((value): value is string => typeof value === 'string' && value.length > 0)
    .join('\n\n')
  const suggestedQuestions = suggestedQuestionsAttachments.flatMap(
    (attachment) => attachment.suggestedQuestions?.questions ?? []
  )

  return {
    attachmentEntries: message.attachments,
    attachmentId: queryAttachment?.attachmentId,
    attachments: message.normalizedAttachments,
    conversationId: message.conversationId,
    message,
    messageId: message.messageId,
    queryResult,
    sql: queryAttachment?.query?.query,
    status: message.status,
    suggestedQuestions,
    text,
  }
}

function getAttachmentEventKey(attachment: DatabricksGenieNormalizedAttachment): string {
  switch (attachment.type) {
    case 'query':
      return JSON.stringify({
        attachmentId: attachment.attachmentId,
        description: attachment.query?.description,
        parameters: attachment.query?.parameters ?? [],
        query: attachment.query?.query,
        queryResultMetadata: attachment.query?.queryResultMetadata,
        statementId: attachment.query?.statementId,
        thoughts: attachment.query?.thoughts ?? [],
        title: attachment.query?.title,
        type: attachment.type,
      })
    case 'suggested_questions':
      return JSON.stringify({
        attachmentId: attachment.attachmentId,
        questions: attachment.suggestedQuestions?.questions ?? [],
        type: attachment.type,
      })
    case 'text':
      return JSON.stringify({
        attachmentId: attachment.attachmentId,
        content: attachment.text?.content,
        purpose: attachment.text?.purpose,
        type: attachment.type,
      })
    default:
      return JSON.stringify({
        attachmentId: attachment.attachmentId,
        type: attachment.type,
      })
  }
}

function isRetryableQueryResultFetchError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false
  }

  return (
    error.message.includes('404') ||
    error.message.includes('409') ||
    error.message.includes('425') ||
    error.message.includes('not ready') ||
    error.message.includes('not available')
  )
}

function isSerializedSpaceUnavailableError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false
  }

  return (
    error.message.includes('400 Bad Request') &&
    error.message.includes('Serialized export is not available to integrated Genie spaces')
  )
}

class DefaultDatabricksGenieConversationClient implements DatabricksGenieConversationClient {
  private readonly client: DatabricksGenieApiClient
  private readonly polling: Required<DatabricksGeniePollingSettings>

  constructor(settings: DatabricksGenieSettings) {
    const resolvedSettings = resolveDatabricksGenieSettings(settings)
    this.client = new DatabricksGenieApiClient(resolvedSettings)
    this.polling = resolvedSettings.polling
  }

  async getSpace(
    requestOrOptions: DatabricksGenieGetSpaceRequest | DatabricksGenieRequestOptions = {},
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieSpace> {
    const request: DatabricksGenieGetSpaceRequest = isGenieRequestOptions(requestOrOptions)
      ? { includeSerializedSpace: true }
      : requestOrOptions
    const requestOptions = isGenieRequestOptions(requestOrOptions) ? requestOrOptions : options

    return this.client.getSpace(
      {
        includeSerializedSpace: request.includeSerializedSpace ?? true,
      },
      requestOptions
    )
  }

  async getSampleQuestions(
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieSampleQuestion[]> {
    try {
      const space = await this.getSpace({ includeSerializedSpace: true }, options)
      return space.serializedSpace?.sampleQuestions ?? []
    } catch (error) {
      if (isSerializedSpaceUnavailableError(error)) {
        return []
      }

      throw error
    }
  }

  async startConversation(
    question: string,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieStartConversationResponse> {
    const request: DatabricksGenieStartConversationRequest = { content: question }
    return this.client.startConversation(request, options)
  }

  async createMessage(
    conversationId: string,
    question: string,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieMessage> {
    const request: DatabricksGenieCreateConversationMessageRequest = {
      content: question,
      conversationId,
    }

    return this.client.createMessage(request, options)
  }

  async getMessage(
    conversationId: string,
    messageId: string,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieMessage> {
    const request: DatabricksGenieGetConversationMessageRequest = {
      conversationId,
      messageId,
    }

    return this.client.getMessage(request, options)
  }

  async listConversationMessages(
    request: DatabricksGenieListConversationMessagesRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieListConversationMessagesResponse> {
    return this.client.listConversationMessages(request, options)
  }

  async listConversations(
    request: DatabricksGenieListConversationsRequest = {},
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieListConversationsResponse> {
    return this.client.listConversations(request, options)
  }

  async deleteConversation(
    conversationId: string,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<void> {
    const request: DatabricksGenieDeleteConversationRequest = {
      conversationId,
    }

    await this.client.deleteConversation(request, options)
  }

  async waitForCompletion(
    conversationId: string,
    messageId: string,
    options: DatabricksGenieAskOptions = {}
  ): Promise<DatabricksGenieMessage> {
    const timeoutMs = options.timeoutMs ?? this.polling.timeoutMs
    const initialPollIntervalMs =
      options.initialPollIntervalMs ?? this.polling.initialPollIntervalMs
    const maxPollIntervalMs = options.maxPollIntervalMs ?? this.polling.maxPollIntervalMs
    const backoffMultiplier = options.backoffMultiplier ?? this.polling.backoffMultiplier

    let pollIntervalMs = initialPollIntervalMs
    const startedAt = Date.now()
    let lastStatus: DatabricksGenieMessageStatus | undefined

    for (;;) {
      if (Date.now() - startedAt >= timeoutMs) {
        throw new Error(
          `Databricks Genie polling timed out after ${Math.floor(timeoutMs / 1000)}s${
            lastStatus ? ` (last status: ${lastStatus})` : ''
          }`
        )
      }

      const message = await this.getMessage(conversationId, messageId, options)
      lastStatus = message.status
      options.onProgress?.(message)

      if (
        DATABRICKS_GENIE_TERMINAL_MESSAGE_STATUSES.includes(
          message.status as (typeof DATABRICKS_GENIE_TERMINAL_MESSAGE_STATUSES)[number]
        )
      ) {
        if (message.status === 'FAILED') {
          throw new Error(
            `Databricks Genie message failed: ${message.error?.error ?? 'Unknown error'}${
              message.error?.type ? ` (${message.error.type})` : ''
            }`
          )
        }

        if (message.status === 'CANCELLED') {
          throw new Error('Databricks Genie message was cancelled')
        }

        if (message.status === 'QUERY_RESULT_EXPIRED') {
          throw new Error('Databricks Genie query result expired')
        }

        return message
      }

      await sleep(pollIntervalMs, options.abortSignal)
      pollIntervalMs = Math.min(
        Math.max(pollIntervalMs * backoffMultiplier, initialPollIntervalMs),
        maxPollIntervalMs
      )
    }
  }

  async getQueryResult(
    conversationId: string,
    messageId: string,
    attachmentId: string,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieQueryResult> {
    const request: DatabricksGenieGetMessageAttachmentQueryResultRequest = {
      attachmentId,
      conversationId,
      messageId,
    }

    return this.client.getMessageAttachmentQueryResult(request, options)
  }

  async getQueryResultByAttachment(
    request: DatabricksGenieGetQueryResultByAttachmentRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieQueryResult> {
    return this.client.getQueryResultByAttachment(request, options)
  }

  async executeQueryAttachment(
    request: DatabricksGenieExecuteMessageAttachmentQueryRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieQueryResult> {
    return this.client.executeMessageAttachmentQuery(request, options)
  }

  async generateDownloadFullQueryResult(
    request: DatabricksGenieGenerateDownloadFullQueryResultRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieGenerateDownloadFullQueryResultResponse> {
    return this.client.generateDownloadFullQueryResult(request, options)
  }

  async getDownloadFullQueryResult(
    request: DatabricksGenieGetDownloadFullQueryResultRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieGetDownloadFullQueryResultResponse> {
    return this.client.getDownloadFullQueryResult(request, options)
  }

  async *streamConversation(
    question: string,
    options: DatabricksGenieAskOptions = {}
  ): AsyncIterable<DatabricksGenieConversationEvent> {
    const initialMessage = options.conversationId
      ? await this.createMessage(options.conversationId, question, options)
      : await this.startConversation(question, options).then((response) => response.message)

    let lastStatus: DatabricksGenieMessageStatus | undefined
    let queryResult: DatabricksGenieQueryResult | undefined
    let queryResultAttachmentId: string | undefined
    const emittedAttachmentKeys = new Set<string>()

    const emitMessageProgress = async function* (
      client: DefaultDatabricksGenieConversationClient,
      message: DatabricksGenieMessage,
      shouldFetchQueryResult: boolean,
      allowRetryableQueryResultErrors: boolean
    ): AsyncIterable<DatabricksGenieConversationEvent> {
      options.onProgress?.(message)

      if (lastStatus !== message.status) {
        lastStatus = message.status
        yield {
          type: 'status',
          conversationId: message.conversationId,
          messageId: message.messageId,
          message,
          status: message.status,
        }
      }

      for (const attachment of message.normalizedAttachments) {
        const attachmentKey = getAttachmentEventKey(attachment)

        if (!emittedAttachmentKeys.has(attachmentKey)) {
          emittedAttachmentKeys.add(attachmentKey)
          yield {
            type: 'attachment',
            attachment,
            conversationId: message.conversationId,
            messageId: message.messageId,
            message,
          }
        }
      }

      const queryAttachment = message.normalizedAttachments.find(
        (attachment) => attachment.type === 'query' && attachment.attachmentId
      )

      if (
        shouldFetchQueryResult &&
        queryAttachment?.attachmentId &&
        queryResultAttachmentId !== queryAttachment.attachmentId
      ) {
        try {
          queryResult = await client.getQueryResult(
            message.conversationId,
            message.messageId,
            queryAttachment.attachmentId,
            options
          )
          queryResultAttachmentId = queryAttachment.attachmentId
          yield {
            type: 'query-result',
            attachmentId: queryAttachment.attachmentId,
            conversationId: message.conversationId,
            messageId: message.messageId,
            message,
            queryResult,
          }
        } catch (error) {
          if (!allowRetryableQueryResultErrors || !isRetryableQueryResultFetchError(error)) {
            throw error
          }
        }
      }
    }

    let currentMessage = initialMessage
    yield* emitMessageProgress(this, currentMessage, Boolean(options.fetchQueryResult), true)

    if (!DATABRICKS_GENIE_TERMINAL_MESSAGE_STATUSES.includes(currentMessage.status as never)) {
      const timeoutMs = options.timeoutMs ?? this.polling.timeoutMs
      const initialPollIntervalMs =
        options.initialPollIntervalMs ?? this.polling.initialPollIntervalMs
      const maxPollIntervalMs = options.maxPollIntervalMs ?? this.polling.maxPollIntervalMs
      const backoffMultiplier = options.backoffMultiplier ?? this.polling.backoffMultiplier
      const startedAt = Date.now()
      let pollIntervalMs = initialPollIntervalMs

      for (;;) {
        if (Date.now() - startedAt >= timeoutMs) {
          throw new Error(
            `Databricks Genie polling timed out after ${Math.floor(timeoutMs / 1000)}s${
              lastStatus ? ` (last status: ${lastStatus})` : ''
            }`
          )
        }

        await sleep(pollIntervalMs, options.abortSignal)
        currentMessage = await this.getMessage(
          initialMessage.conversationId,
          initialMessage.messageId,
          options
        )
        yield* emitMessageProgress(this, currentMessage, Boolean(options.fetchQueryResult), true)

        if (
          DATABRICKS_GENIE_TERMINAL_MESSAGE_STATUSES.includes(currentMessage.status as never)
        ) {
          break
        }

        pollIntervalMs = Math.min(
          Math.max(pollIntervalMs * backoffMultiplier, initialPollIntervalMs),
          maxPollIntervalMs
        )
      }
    }

    if (currentMessage.status === 'FAILED') {
      throw new Error(
        `Databricks Genie message failed: ${currentMessage.error?.error ?? 'Unknown error'}${
          currentMessage.error?.type ? ` (${currentMessage.error.type})` : ''
        }`
      )
    }

    if (currentMessage.status === 'CANCELLED') {
      throw new Error('Databricks Genie message was cancelled')
    }

    if (currentMessage.status === 'QUERY_RESULT_EXPIRED') {
      throw new Error('Databricks Genie query result expired')
    }

    const completedMessage =
      currentMessage.status === 'COMPLETED'
        ? await this.getMessage(currentMessage.conversationId, currentMessage.messageId, options)
        : currentMessage

    yield* emitMessageProgress(this, completedMessage, Boolean(options.fetchQueryResult), false)

    const result = extractAskResult(completedMessage, queryResult)
    yield {
      type: 'complete',
      conversationId: completedMessage.conversationId,
      messageId: completedMessage.messageId,
      message: completedMessage,
      result,
    }
  }

  async ask(question: string, options: DatabricksGenieAskOptions = {}): Promise<DatabricksGenieAskResult> {
    let completedResult: DatabricksGenieAskResult | undefined

    for await (const event of this.streamConversation(question, options)) {
      if (event.type === 'complete') {
        completedResult = event.result
      }
    }

    if (!completedResult) {
      throw new Error('Databricks Genie conversation ended without a completed result')
    }

    return completedResult
  }
}

export function createDatabricksGenieConversationClient(
  settings: DatabricksGenieSettings
): DatabricksGenieConversationClient {
  return new DefaultDatabricksGenieConversationClient(settings)
}

export { DEFAULT_GENIE_POLLING_SETTINGS }
export type { DatabricksGeniePollingSettings, DatabricksGenieSettings }
export * from './types'
