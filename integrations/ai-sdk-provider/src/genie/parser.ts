import {
  type DatabricksGenieAttachment,
  type DatabricksGenieConversation,
  type DatabricksGenieConversationSummary,
  type DatabricksGenieFeedback,
  type DatabricksGenieGenerateDownloadFullQueryResultResponse,
  type DatabricksGenieGetDownloadFullQueryResultResponse,
  type DatabricksGenieListConversationMessagesResponse,
  type DatabricksGenieListConversationsResponse,
  type DatabricksGenieMessage,
  type DatabricksGenieMessageError,
  type DatabricksGenieNormalizedAttachment,
  type DatabricksGenieQueryAttachment,
  type DatabricksGenieQueryAttachmentParameter,
  type DatabricksGenieQueryAttachmentThought,
  type DatabricksGenieQueryResult,
  type DatabricksGenieQueryResultColumn,
  type DatabricksGenieResultMetadata,
  type DatabricksGenieSampleQuestion,
  type DatabricksGenieSerializedSpace,
  type DatabricksGenieSpace,
  type DatabricksGenieStartConversationResponse,
  type DatabricksGenieSuggestedQuestionsAttachment,
  type DatabricksGenieTextAttachment,
} from './types'
import { isRecord, isStringArray } from './shared'

function assertRecord(value: unknown, context: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`Expected ${context} to be an object`)
  }

  return value
}

function getOptionalString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function getRequiredString(value: unknown, fieldName: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`Expected ${fieldName} to be a non-empty string`)
  }

  return value
}

function getOptionalNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function getOptionalBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined
}

function parseQueryAttachmentParameter(
  value: unknown
): DatabricksGenieQueryAttachmentParameter | null {
  if (!isRecord(value)) {
    return null
  }

  return {
    keyword: getOptionalString(value.keyword),
    sqlType: getOptionalString(value.sql_type),
    value: getOptionalString(value.value),
    raw: value,
  }
}

function parseQueryAttachmentThought(
  value: unknown
): DatabricksGenieQueryAttachmentThought | null {
  if (!isRecord(value)) {
    return null
  }

  return {
    content: getOptionalString(value.content),
    thoughtType: getOptionalString(value.thought_type),
    raw: value,
  }
}

function parseResultMetadata(value: unknown): DatabricksGenieResultMetadata | undefined {
  if (!isRecord(value)) {
    return undefined
  }

  return {
    isTruncated: getOptionalBoolean(value.is_truncated),
    rowCount: getOptionalNumber(value.row_count),
    statementId: getOptionalString(value.statement_id),
    statementIdSignature: getOptionalString(value.statement_id_signature),
    raw: value,
  }
}

function parseQueryAttachment(
  value: unknown,
  attachmentId?: string
): DatabricksGenieQueryAttachment | undefined {
  if (!isRecord(value)) {
    return undefined
  }

  return {
    attachmentId,
    description: getOptionalString(value.description),
    id: getOptionalString(value.id),
    lastUpdatedTimestamp: getOptionalNumber(value.last_updated_timestamp),
    parameters: Array.isArray(value.parameters)
      ? value.parameters
          .map(parseQueryAttachmentParameter)
          .filter((item): item is DatabricksGenieQueryAttachmentParameter => item !== null)
      : [],
    query: getOptionalString(value.query),
    queryResultMetadata: parseResultMetadata(value.query_result_metadata),
    statementId: getOptionalString(value.statement_id),
    thoughts: Array.isArray(value.thoughts)
      ? value.thoughts
          .map(parseQueryAttachmentThought)
          .filter((item): item is DatabricksGenieQueryAttachmentThought => item !== null)
      : [],
    title: getOptionalString(value.title),
    raw: value,
  }
}

function parseTextAttachment(value: unknown, attachmentId?: string): DatabricksGenieTextAttachment | undefined {
  if (!isRecord(value)) {
    return undefined
  }

  return {
    attachmentId,
    content: getOptionalString(value.content),
    id: getOptionalString(value.id),
    purpose: getOptionalString(value.purpose),
    raw: value,
  }
}

function parseSuggestedQuestionsAttachment(
  value: unknown,
  attachmentId?: string
): DatabricksGenieSuggestedQuestionsAttachment | undefined {
  if (!isRecord(value)) {
    return undefined
  }

  return {
    attachmentId,
    questions: isStringArray(value.questions) ? value.questions : [],
    raw: value,
  }
}

function flattenAttachment(attachment: DatabricksGenieAttachment): DatabricksGenieNormalizedAttachment[] {
  const normalized: DatabricksGenieNormalizedAttachment[] = []

  if (attachment.query) {
    normalized.push({
      attachmentId: attachment.attachmentId,
      type: 'query',
      query: attachment.query,
      raw: attachment.raw,
    })
  }

  if (attachment.text) {
    normalized.push({
      attachmentId: attachment.attachmentId,
      type: 'text',
      text: attachment.text,
      raw: attachment.raw,
    })
  }

  if (attachment.suggestedQuestions) {
    normalized.push({
      attachmentId: attachment.attachmentId,
      type: 'suggested_questions',
      suggestedQuestions: attachment.suggestedQuestions,
      raw: attachment.raw,
    })
  }

  return normalized
}

function parseAttachment(value: unknown): DatabricksGenieAttachment | null {
  if (!isRecord(value)) {
    return null
  }

  const attachmentId = getOptionalString(value.attachment_id)

  return {
    attachmentId,
    query: parseQueryAttachment(value.query, attachmentId),
    suggestedQuestions: parseSuggestedQuestionsAttachment(value.suggested_questions, attachmentId),
    text: parseTextAttachment(value.text, attachmentId),
    raw: value,
  }
}

function parseMessageError(value: unknown): DatabricksGenieMessageError | undefined {
  if (!isRecord(value)) {
    return undefined
  }

  return {
    error: getOptionalString(value.error),
    type: getOptionalString(value.type),
    raw: value,
  }
}

function parseFeedback(value: unknown): DatabricksGenieFeedback | undefined {
  if (!isRecord(value)) {
    return undefined
  }

  return {
    rating: getOptionalString(value.rating),
    raw: value,
  }
}

export function parseMessage(value: unknown): DatabricksGenieMessage {
  const record = assertRecord(value, 'Databricks Genie message')
  const attachments = Array.isArray(record.attachments)
    ? record.attachments
        .map(parseAttachment)
        .filter((item): item is DatabricksGenieAttachment => item !== null)
    : []
  const messageId =
    getOptionalString(record.message_id) ?? getRequiredString(record.id, 'message.id')
  const legacyId = getOptionalString(record.id)

  return {
    attachments,
    normalizedAttachments: attachments.flatMap(flattenAttachment),
    content: getOptionalString(record.content),
    conversationId: getRequiredString(record.conversation_id, 'message.conversation_id'),
    createdTimestamp: getOptionalNumber(record.created_timestamp),
    error: parseMessageError(record.error),
    feedback: parseFeedback(record.feedback),
    id: messageId,
    lastUpdatedTimestamp: getOptionalNumber(record.last_updated_timestamp),
    legacyId,
    messageId,
    queryResult: parseResultMetadata(record.query_result),
    spaceId: getOptionalString(record.space_id),
    status: getRequiredString(record.status, 'message.status'),
    userId: getOptionalNumber(record.user_id),
    raw: record,
  }
}

export function parseConversation(value: unknown): DatabricksGenieConversation {
  const record = assertRecord(value, 'Databricks Genie conversation')
  const conversationId =
    getOptionalString(record.conversation_id) ?? getRequiredString(record.id, 'conversation.id')
  const legacyId = getOptionalString(record.id)

  return {
    conversationId,
    createdTimestamp: getOptionalNumber(record.created_timestamp),
    id: conversationId,
    lastUpdatedTimestamp: getOptionalNumber(record.last_updated_timestamp),
    legacyId,
    spaceId: getOptionalString(record.space_id),
    title: getOptionalString(record.title),
    userId: getOptionalNumber(record.user_id),
    raw: record,
  }
}

export function parseConversationSummary(value: unknown): DatabricksGenieConversationSummary {
  const record = assertRecord(value, 'Databricks Genie conversation summary')

  return {
    conversationId: getRequiredString(record.conversation_id, 'conversation_summary.conversation_id'),
    createdTimestamp: getOptionalNumber(record.created_timestamp),
    title: getOptionalString(record.title),
    raw: record,
  }
}

function parseSerializedSpaceSampleQuestion(value: unknown): DatabricksGenieSampleQuestion | null {
  if (!isRecord(value)) {
    return null
  }

  const question = Array.isArray(value.question)
    ? value.question.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
    : []

  if (question.length === 0) {
    return null
  }

  return {
    id: getOptionalString(value.id),
    text: question.join(' ').trim(),
    raw: value,
  }
}

function parseSerializedSpace(value: unknown): DatabricksGenieSerializedSpace | undefined {
  if (typeof value !== 'string' || value.trim().length === 0) {
    return undefined
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(value)
  } catch (error) {
    throw new Error(
      `Failed to parse Databricks Genie serialized_space: ${
        error instanceof Error ? error.message : String(error)
      }`
    )
  }

  const record = assertRecord(parsed, 'Databricks Genie serialized_space')
  const config = isRecord(record.config) ? record.config : {}
  const sampleQuestions = Array.isArray(config.sample_questions)
    ? config.sample_questions
        .map(parseSerializedSpaceSampleQuestion)
        .filter((item): item is DatabricksGenieSampleQuestion => item !== null)
    : []

  return {
    sampleQuestions,
    raw: record,
  }
}

export function parseSpace(value: unknown): DatabricksGenieSpace {
  const record = assertRecord(value, 'Databricks Genie space')
  const spaceId =
    getOptionalString(record.space_id) ?? getRequiredString(record.id, 'space.space_id')

  return {
    description: getOptionalString(record.description),
    id: spaceId,
    parentPath: getOptionalString(record.parent_path),
    serializedSpace: parseSerializedSpace(record.serialized_space),
    serializedSpaceText: getOptionalString(record.serialized_space),
    spaceId,
    title: getOptionalString(record.title),
    warehouseId: getOptionalString(record.warehouse_id),
    raw: record,
  }
}

export function parseStartConversationResponse(value: unknown): DatabricksGenieStartConversationResponse {
  const record = assertRecord(value, 'Databricks Genie start conversation response')
  const conversation = parseConversation(record.conversation)
  const message = parseMessage(record.message)

  return {
    conversation,
    conversationId:
      getOptionalString(record.conversation_id) ?? conversation.conversationId,
    message,
    messageId: getOptionalString(record.message_id) ?? message.messageId,
    raw: record,
  }
}

export function parseListConversationMessagesResponse(
  value: unknown
): DatabricksGenieListConversationMessagesResponse {
  const record = assertRecord(value, 'Databricks Genie list conversation messages response')

  return {
    messages: Array.isArray(record.messages) ? record.messages.map(parseMessage) : [],
    nextPageToken: getOptionalString(record.next_page_token),
    raw: record,
  }
}

export function parseListConversationsResponse(
  value: unknown
): DatabricksGenieListConversationsResponse {
  const record = assertRecord(value, 'Databricks Genie list conversations response')

  return {
    conversations: Array.isArray(record.conversations)
      ? record.conversations.map(parseConversationSummary)
      : [],
    nextPageToken: getOptionalString(record.next_page_token),
    raw: record,
  }
}

export function parseQueryResult(value: unknown): DatabricksGenieQueryResult {
  const record = assertRecord(value, 'Databricks Genie query result response')
  const statementResponse = isRecord(record.statement_response)
    ? record.statement_response
    : undefined
  const manifest = statementResponse && isRecord(statementResponse.manifest)
    ? statementResponse.manifest
    : undefined
  const schema = manifest && isRecord(manifest.schema) ? manifest.schema : undefined
  const columnsValue = schema && Array.isArray(schema.columns) ? schema.columns : []
  const resultValue = statementResponse && isRecord(statementResponse.result)
    ? statementResponse.result
    : undefined
  const rowsValue = resultValue && Array.isArray(resultValue.data_array) ? resultValue.data_array : []
  const rowCount =
    getOptionalNumber(record.row_count) ??
    getOptionalNumber(manifest?.total_row_count) ??
    rowsValue.length
  const truncated =
    getOptionalBoolean(record.is_truncated) ??
    getOptionalBoolean(manifest?.truncated) ??
    rowCount > rowsValue.length

  return {
    columns: columnsValue.reduce<DatabricksGenieQueryResultColumn[]>((result, column) => {
      if (!isRecord(column) || typeof column.name !== "string") {
        return result
      }

      result.push({
        name: column.name,
        typeName: getOptionalString(column.type_name),
      })

      return result
    }, []),
    rows: rowsValue.filter((row): row is unknown[][][number] => Array.isArray(row)),
    rowCount,
    truncated,
    statementResponse,
    raw: record,
  }
}

export function parseGenerateDownloadFullQueryResultResponse(
  value: unknown
): DatabricksGenieGenerateDownloadFullQueryResultResponse {
  const record = assertRecord(value, 'Databricks Genie generate download response')

  return {
    downloadId: getRequiredString(record.download_id, 'download.download_id'),
    downloadIdSignature: getOptionalString(record.download_id_signature),
    raw: record,
  }
}

export function parseGetDownloadFullQueryResultResponse(
  value: unknown
): DatabricksGenieGetDownloadFullQueryResultResponse {
  const record = assertRecord(value, 'Databricks Genie get download response')

  return {
    queryResult: parseQueryResult(record),
    raw: record,
  }
}
