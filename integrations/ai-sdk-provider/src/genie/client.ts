import type { FetchFunction } from '@ai-sdk/provider-utils'
import {
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
  type DatabricksGenieQueryResult,
  type DatabricksGenieSpace,
  type DatabricksGenieStartConversationRequest,
  type DatabricksGenieStartConversationResponse,
} from './types'
import {
  ensureOkResponse,
  getGenieHeaders,
  logGenieDebugEvent,
  parseJsonResponse,
  sanitizeGenieHeaders,
  type DatabricksGenieResolvedSettings,
} from './shared'
import {
  parseGenerateDownloadFullQueryResultResponse,
  parseGetDownloadFullQueryResultResponse,
  parseListConversationMessagesResponse,
  parseListConversationsResponse,
  parseMessage,
  parseQueryResult,
  parseSpace,
  parseStartConversationResponse,
} from './parser'

export interface DatabricksGenieRequestOptions {
  headers?: Record<string, string>
  abortSignal?: AbortSignal
}

type DatabricksGenieQueryValue = string | number | boolean | undefined

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isParsedGenieMessage(value: unknown): value is DatabricksGenieMessage {
  return (
    isRecord(value) &&
    typeof value.messageId === 'string' &&
    typeof value.conversationId === 'string' &&
    Array.isArray(value.normalizedAttachments) &&
    Array.isArray(value.attachments)
  )
}

function collectTextAttachmentDebugMetadata(parsed: unknown): unknown[] {
  const messages = collectMessagesFromParsedValue(parsed)

  return messages.flatMap((message) =>
    message.normalizedAttachments
      .filter((attachment) => attachment.type === 'text')
      .map((attachment) => ({
        messageId: message.messageId,
        conversationId: message.conversationId,
        attachmentId: attachment.attachmentId,
        normalizedText: {
          id: attachment.text?.id,
          content: attachment.text?.content,
          purpose: attachment.text?.purpose,
        },
        rawAttachmentKeys: Object.keys(attachment.raw),
        rawTextKeys: attachment.text?.raw ? Object.keys(attachment.text.raw) : [],
        rawAttachment: attachment.raw,
        rawText: attachment.text?.raw,
      }))
  )
}

function collectMessagesFromParsedValue(parsed: unknown): DatabricksGenieMessage[] {
  if (!isRecord(parsed)) {
    return []
  }

  if (Array.isArray(parsed.messages)) {
    return parsed.messages.filter(isParsedGenieMessage)
  }

  if (isParsedGenieMessage(parsed.message)) {
    return [parsed.message]
  }

  if (isParsedGenieMessage(parsed)) {
    return [parsed]
  }

  return []
}

function buildUrl(
  settings: DatabricksGenieResolvedSettings,
  path: string,
  query?: Record<string, DatabricksGenieQueryValue>
): string {
  const url = new URL(settings.url(path))

  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined) {
        continue
      }

      url.searchParams.set(key, String(value))
    }
  }

  return url.toString()
}

async function performJsonRequest({
  settings,
  fetch,
  url,
  method,
  headers,
  body,
  abortSignal,
}: {
  settings: DatabricksGenieResolvedSettings
  fetch: FetchFunction
  url: string
  method: string
  headers: Record<string, string>
  body?: unknown
  abortSignal?: AbortSignal
}): Promise<unknown> {
  logGenieDebugEvent(settings, {
    phase: 'request',
    method,
    url,
    headers: sanitizeGenieHeaders(headers),
    body: settings.debug.logBodies ? body : undefined,
  })

  const response = await fetch(url, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: abortSignal,
  })

  let responseBody: unknown
  if (settings.debug.logBodies) {
    try {
      responseBody = await response.clone().json()
    } catch {
      try {
        responseBody = await response.clone().text()
      } catch {
        responseBody = undefined
      }
    }
  }

  logGenieDebugEvent(settings, {
    phase: 'response',
    method,
    url,
    status: response.status,
    statusText: response.statusText,
    body: responseBody,
  })

  return parseJsonResponse(response)
}

async function performRequest({
  settings,
  fetch,
  url,
  method,
  headers,
  body,
  abortSignal,
}: {
  settings: DatabricksGenieResolvedSettings
  fetch: FetchFunction
  url: string
  method: string
  headers: Record<string, string>
  body?: unknown
  abortSignal?: AbortSignal
}): Promise<void> {
  logGenieDebugEvent(settings, {
    phase: 'request',
    method,
    url,
    headers: sanitizeGenieHeaders(headers),
    body: settings.debug.logBodies ? body : undefined,
  })

  const response = await fetch(url, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: abortSignal,
  })

  let responseBody: unknown
  if (settings.debug.logBodies) {
    try {
      responseBody = await response.clone().json()
    } catch {
      try {
        responseBody = await response.clone().text()
      } catch {
        responseBody = undefined
      }
    }
  }

  logGenieDebugEvent(settings, {
    phase: 'response',
    method,
    url,
    status: response.status,
    statusText: response.statusText,
    body: responseBody,
  })

  await ensureOkResponse(response)
}

export class DatabricksGenieApiClient {
  private static readonly GENIE_SPACES_API_PREFIX = '/api/2.0/genie/spaces'

  constructor(private readonly settings: DatabricksGenieResolvedSettings) {}

  private spacePath(suffix = ''): string {
    return `${DatabricksGenieApiClient.GENIE_SPACES_API_PREFIX}/${this.settings.spaceId}${suffix}`
  }

  private conversationPath(conversationId: string, suffix = ''): string {
    return this.spacePath(`/conversations/${conversationId}${suffix}`)
  }

  private messagePath(conversationId: string, messageId: string, suffix = ''): string {
    return this.conversationPath(conversationId, `/messages/${messageId}${suffix}`)
  }

  private attachmentPath(
    conversationId: string,
    messageId: string,
    attachmentId: string,
    suffix = ''
  ): string {
    return this.messagePath(
      conversationId,
      messageId,
      `/attachments/${attachmentId}${suffix}`
    )
  }

  private async requestJson<T>({
    method,
    path,
    parse,
    options = {},
    query,
    body,
  }: {
    method: string
    path: string
    parse: (value: unknown) => T
    options?: DatabricksGenieRequestOptions
    query?: Record<string, DatabricksGenieQueryValue>
    body?: unknown
  }): Promise<T> {
    const response = await performJsonRequest({
      settings: this.settings,
      fetch: this.settings.fetch,
      url: buildUrl(this.settings, path, query),
      method,
      headers: getGenieHeaders(this.settings, options.headers),
      body,
      abortSignal: options.abortSignal,
    })

    const parsed = parse(response)
    const textAttachmentMetadata = collectTextAttachmentDebugMetadata(parsed)

    if (textAttachmentMetadata.length > 0) {
      logGenieDebugEvent(this.settings, {
        phase: 'response',
        method,
        url: buildUrl(this.settings, path, query),
        status: 200,
        statusText: 'Parsed Text Attachments',
        metadata: {
          textAttachments: textAttachmentMetadata,
        },
      })
    }

    return parsed
  }

  async getSpace(
    request: DatabricksGenieGetSpaceRequest = {},
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieSpace> {
    return this.requestJson({
      method: 'GET',
      path: this.spacePath(),
      query: {
        include_serialized_space: request.includeSerializedSpace,
      },
      options,
      parse: parseSpace,
    })
  }

  async startConversation(
    request: DatabricksGenieStartConversationRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieStartConversationResponse> {
    return this.requestJson({
      method: 'POST',
      path: this.spacePath('/start-conversation'),
      body: request,
      options,
      parse: parseStartConversationResponse,
    })
  }

  async createMessage(
    request: DatabricksGenieCreateConversationMessageRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieMessage> {
    return this.requestJson({
      method: 'POST',
      path: this.conversationPath(request.conversationId, '/messages'),
      body: { content: request.content },
      options,
      parse: parseMessage,
    })
  }

  async getMessage(
    request: DatabricksGenieGetConversationMessageRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieMessage> {
    return this.requestJson({
      method: 'GET',
      path: this.messagePath(request.conversationId, request.messageId),
      options,
      parse: parseMessage,
    })
  }

  async listConversationMessages(
    request: DatabricksGenieListConversationMessagesRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieListConversationMessagesResponse> {
    return this.requestJson({
      method: 'GET',
      path: this.conversationPath(request.conversationId, '/messages'),
      query: {
        page_size: request.pageSize,
        page_token: request.pageToken,
      },
      options,
      parse: parseListConversationMessagesResponse,
    })
  }

  async listConversations(
    request: DatabricksGenieListConversationsRequest = {},
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieListConversationsResponse> {
    return this.requestJson({
      method: 'GET',
      path: this.spacePath('/conversations'),
      query: {
        include_all: request.includeAll,
        page_size: request.pageSize,
        page_token: request.pageToken,
      },
      options,
      parse: parseListConversationsResponse,
    })
  }

  async deleteConversation(
    request: DatabricksGenieDeleteConversationRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<void> {
    await performRequest({
      settings: this.settings,
      fetch: this.settings.fetch,
      url: buildUrl(this.settings, this.conversationPath(request.conversationId)),
      method: 'DELETE',
      headers: getGenieHeaders(this.settings, options.headers),
      abortSignal: options.abortSignal,
    })
  }

  async getMessageAttachmentQueryResult(
    request: DatabricksGenieGetMessageAttachmentQueryResultRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieQueryResult> {
    return this.requestJson({
      method: 'GET',
      path: this.attachmentPath(
        request.conversationId,
        request.messageId,
        request.attachmentId,
        '/query-result'
      ),
      options,
      parse: parseQueryResult,
    })
  }

  async getQueryResultByAttachment(
    request: DatabricksGenieGetQueryResultByAttachmentRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieQueryResult> {
    return this.getMessageAttachmentQueryResult(request, options)
  }

  async executeMessageAttachmentQuery(
    request: DatabricksGenieExecuteMessageAttachmentQueryRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieQueryResult> {
    return this.requestJson({
      method: 'POST',
      path: this.attachmentPath(
        request.conversationId,
        request.messageId,
        request.attachmentId,
        '/execute-query'
      ),
      options,
      parse: parseQueryResult,
    })
  }

  async generateDownloadFullQueryResult(
    request: DatabricksGenieGenerateDownloadFullQueryResultRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieGenerateDownloadFullQueryResultResponse> {
    return this.requestJson({
      method: 'POST',
      path: this.attachmentPath(
        request.conversationId,
        request.messageId,
        request.attachmentId,
        '/downloads'
      ),
      options,
      parse: parseGenerateDownloadFullQueryResultResponse,
    })
  }

  async getDownloadFullQueryResult(
    request: DatabricksGenieGetDownloadFullQueryResultRequest,
    options: DatabricksGenieRequestOptions = {}
  ): Promise<DatabricksGenieGetDownloadFullQueryResultResponse> {
    return this.requestJson({
      method: 'GET',
      path: this.attachmentPath(
        request.conversationId,
        request.messageId,
        request.attachmentId,
        `/downloads/${request.downloadId}`
      ),
      query: {
        download_id_signature: request.downloadIdSignature,
      },
      options,
      parse: parseGetDownloadFullQueryResultResponse,
    })
  }
}
