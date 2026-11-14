import type { DatabricksGenieNormalizedAttachment } from './types'

const FOLLOW_UP_QUESTION_PREFIXES = [
  /^would you like\b/i,
  /^would you also like\b/i,
  /^do you want\b/i,
  /^do you also want\b/i,
  /^should i\b/i,
  /^shall i\b/i,
  /^want me to\b/i,
  /^would it help if\b/i,
] as const

export interface DatabricksGenieOrderedAttachments {
  answerTextAttachments: DatabricksGenieNormalizedAttachment[]
  followUpQuestionAttachments: DatabricksGenieNormalizedAttachment[]
  orderedAttachments: DatabricksGenieNormalizedAttachment[]
  queryAttachments: DatabricksGenieNormalizedAttachment[]
  suggestedQuestionAttachments: DatabricksGenieNormalizedAttachment[]
}

function getTextContent(attachment: DatabricksGenieNormalizedAttachment): string {
  return attachment.type === 'text' ? attachment.text?.content?.trim() ?? '' : ''
}

function isQuestionText(text: string): boolean {
  return text.endsWith('?')
}

function matchesFollowUpPrefix(text: string): boolean {
  return FOLLOW_UP_QUESTION_PREFIXES.some((pattern) => pattern.test(text))
}

export function isLikelyDatabricksGenieFollowUpQuestionText(text: string): boolean {
  const normalizedText = text.trim()

  if (!normalizedText || !isQuestionText(normalizedText)) {
    return false
  }

  return matchesFollowUpPrefix(normalizedText)
}

export function isLikelyDatabricksGenieFollowUpQuestionAttachment(
  attachment: DatabricksGenieNormalizedAttachment
): boolean {
  if (attachment.type !== 'text') {
    return false
  }

  if (attachment.text?.purpose === 'FOLLOW_UP_QUESTION') {
    return true
  }

  const content = getTextContent(attachment)

  return isLikelyDatabricksGenieFollowUpQuestionText(content)
}

export function orderDatabricksGenieAttachments(
  attachments: DatabricksGenieNormalizedAttachment[]
): DatabricksGenieOrderedAttachments {
  const answerTextAttachments: DatabricksGenieNormalizedAttachment[] = []
  const followUpQuestionAttachments: DatabricksGenieNormalizedAttachment[] = []
  const queryAttachments: DatabricksGenieNormalizedAttachment[] = []
  const suggestedQuestionAttachments: DatabricksGenieNormalizedAttachment[] = []

  for (const attachment of attachments) {
    if (attachment.type === 'query') {
      queryAttachments.push(attachment)
      continue
    }

    if (attachment.type === 'suggested_questions') {
      suggestedQuestionAttachments.push(attachment)
      continue
    }

    if (attachment.type === 'text') {
      if (isLikelyDatabricksGenieFollowUpQuestionAttachment(attachment)) {
        followUpQuestionAttachments.push(attachment)
      } else {
        answerTextAttachments.push(attachment)
      }
    }
  }

  return {
    answerTextAttachments,
    followUpQuestionAttachments,
    orderedAttachments: [
      ...answerTextAttachments,
      ...queryAttachments,
      ...followUpQuestionAttachments,
      ...suggestedQuestionAttachments,
    ],
    queryAttachments,
    suggestedQuestionAttachments,
  }
}
