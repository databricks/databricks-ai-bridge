export const DATABRICKS_GENIE_TERMINAL_MESSAGE_STATUSES = [
  'COMPLETED',
  'FAILED',
  'CANCELLED',
  'QUERY_RESULT_EXPIRED',
] as const

export const DATABRICKS_GENIE_MESSAGE_STATUSES = [
  'FETCHING_METADATA',
  'FILTERING_CONTEXT',
  'ASKING_AI',
  'PENDING_WAREHOUSE',
  'EXECUTING_QUERY',
  'FAILED',
  'COMPLETED',
  'SUBMITTED',
  'QUERY_RESULT_EXPIRED',
  'CANCELLED',
] as const

export const DATABRICKS_GENIE_TEXT_ATTACHMENT_PURPOSES = ['FOLLOW_UP_QUESTION'] as const

export const DATABRICKS_GENIE_FEEDBACK_RATINGS = ['NEGATIVE', 'NONE', 'POSITIVE'] as const

export type DatabricksGenieAttachmentType = 'query' | 'text' | 'suggested_questions'

export type DatabricksGenieMessageStatus =
  | (typeof DATABRICKS_GENIE_MESSAGE_STATUSES)[number]
  | (string & {})

export type DatabricksGenieTextAttachmentPurpose =
  | (typeof DATABRICKS_GENIE_TEXT_ATTACHMENT_PURPOSES)[number]
  | (string & {})

export type DatabricksGenieFeedbackRating =
  | (typeof DATABRICKS_GENIE_FEEDBACK_RATINGS)[number]
  | (string & {})

export interface DatabricksGenieQueryAttachmentParameter {
  keyword?: string
  sqlType?: string
  value?: string
  raw: Record<string, unknown>
}

export interface DatabricksGenieQueryAttachmentThought {
  content?: string
  raw: Record<string, unknown>
  thoughtType?: string
}

export interface DatabricksGenieResultMetadata {
  isTruncated?: boolean
  rowCount?: number
  statementId?: string
  statementIdSignature?: string
  raw: Record<string, unknown>
}

export interface DatabricksGenieQueryAttachment {
  attachmentId?: string
  description?: string
  id?: string
  lastUpdatedTimestamp?: number
  parameters: DatabricksGenieQueryAttachmentParameter[]
  query?: string
  queryResultMetadata?: DatabricksGenieResultMetadata
  statementId?: string
  thoughts: DatabricksGenieQueryAttachmentThought[]
  title?: string
  raw: Record<string, unknown>
}

export interface DatabricksGenieTextAttachment {
  attachmentId?: string
  content?: string
  id?: string
  purpose?: DatabricksGenieTextAttachmentPurpose
  raw: Record<string, unknown>
}

export interface DatabricksGenieSuggestedQuestionsAttachment {
  attachmentId?: string
  questions: string[]
  raw: Record<string, unknown>
}

export interface DatabricksGenieAttachment {
  attachmentId?: string
  query?: DatabricksGenieQueryAttachment
  suggestedQuestions?: DatabricksGenieSuggestedQuestionsAttachment
  text?: DatabricksGenieTextAttachment
  raw: Record<string, unknown>
}

export interface DatabricksGenieNormalizedAttachment {
  attachmentId?: string
  type: DatabricksGenieAttachmentType
  query?: DatabricksGenieQueryAttachment
  suggestedQuestions?: DatabricksGenieSuggestedQuestionsAttachment
  text?: DatabricksGenieTextAttachment
  raw: Record<string, unknown>
}

export interface DatabricksGenieMessageError {
  error?: string
  type?: string
  raw: Record<string, unknown>
}

export interface DatabricksGenieFeedback {
  rating?: DatabricksGenieFeedbackRating
  raw: Record<string, unknown>
}

export interface DatabricksGenieMessage {
  attachments: DatabricksGenieAttachment[]
  normalizedAttachments: DatabricksGenieNormalizedAttachment[]
  content?: string
  conversationId: string
  createdTimestamp?: number
  error?: DatabricksGenieMessageError
  feedback?: DatabricksGenieFeedback
  id: string
  lastUpdatedTimestamp?: number
  legacyId?: string
  messageId: string
  queryResult?: DatabricksGenieResultMetadata
  spaceId?: string
  status: DatabricksGenieMessageStatus
  userId?: number
  raw: Record<string, unknown>
}

export interface DatabricksGenieConversation {
  conversationId: string
  createdTimestamp?: number
  id: string
  lastUpdatedTimestamp?: number
  legacyId?: string
  spaceId?: string
  title?: string
  userId?: number
  raw: Record<string, unknown>
}

export interface DatabricksGenieConversationSummary {
  conversationId: string
  createdTimestamp?: number
  title?: string
  raw: Record<string, unknown>
}

export interface DatabricksGenieSampleQuestion {
  id?: string
  text: string
  raw: Record<string, unknown>
}

export interface DatabricksGenieSerializedSpace {
  sampleQuestions: DatabricksGenieSampleQuestion[]
  raw: Record<string, unknown>
}

export interface DatabricksGenieSpace {
  description?: string
  id: string
  parentPath?: string
  serializedSpace?: DatabricksGenieSerializedSpace
  serializedSpaceText?: string
  spaceId: string
  title?: string
  warehouseId?: string
  raw: Record<string, unknown>
}

export interface DatabricksGenieStartConversationResponse {
  conversation: DatabricksGenieConversation
  conversationId: string
  message: DatabricksGenieMessage
  messageId: string
  raw: Record<string, unknown>
}

export interface DatabricksGenieListConversationMessagesResponse {
  messages: DatabricksGenieMessage[]
  nextPageToken?: string
  raw: Record<string, unknown>
}

export interface DatabricksGenieListConversationsResponse {
  conversations: DatabricksGenieConversationSummary[]
  nextPageToken?: string
  raw: Record<string, unknown>
}

export interface DatabricksGenieQueryResultColumn {
  name: string
  typeName?: string
}

export interface DatabricksGenieQueryResult {
  columns: DatabricksGenieQueryResultColumn[]
  rows: unknown[][]
  rowCount: number
  truncated: boolean
  statementResponse?: Record<string, unknown>
  raw: Record<string, unknown>
}

export interface DatabricksGenieGenerateDownloadFullQueryResultResponse {
  downloadId: string
  downloadIdSignature?: string
  raw: Record<string, unknown>
}

export interface DatabricksGenieGetDownloadFullQueryResultResponse {
  queryResult: DatabricksGenieQueryResult
  raw: Record<string, unknown>
}

export interface DatabricksGenieStartConversationRequest {
  content: string
}

export interface DatabricksGenieCreateConversationMessageRequest {
  content: string
  conversationId: string
}

export interface DatabricksGenieGetConversationMessageRequest {
  conversationId: string
  messageId: string
}

export interface DatabricksGenieListConversationMessagesRequest {
  conversationId: string
  pageSize?: number
  pageToken?: string
}

export interface DatabricksGenieListConversationsRequest {
  includeAll?: boolean
  pageSize?: number
  pageToken?: string
}

export interface DatabricksGenieDeleteConversationRequest {
  conversationId: string
}

export interface DatabricksGenieGetMessageAttachmentQueryResultRequest {
  attachmentId: string
  conversationId: string
  messageId: string
}

export interface DatabricksGenieExecuteMessageAttachmentQueryRequest {
  attachmentId: string
  conversationId: string
  messageId: string
}

export interface DatabricksGenieGenerateDownloadFullQueryResultRequest {
  attachmentId: string
  conversationId: string
  messageId: string
}

export interface DatabricksGenieGetDownloadFullQueryResultRequest {
  attachmentId: string
  conversationId: string
  downloadId: string
  downloadIdSignature?: string
  messageId: string
}

export interface DatabricksGenieGetQueryResultByAttachmentRequest {
  attachmentId: string
  conversationId: string
  messageId: string
}

export interface DatabricksGenieGetSpaceRequest {
  includeSerializedSpace?: boolean
}
