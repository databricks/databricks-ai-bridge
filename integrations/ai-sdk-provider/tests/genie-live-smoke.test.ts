import { beforeAll, describe, expect, it } from 'vitest'
import { createDatabricksGenieAgent, createDatabricksGenieConversationClient } from '../src'

const RUN_GENIE_LIVE_TEST = process.env.npm_lifecycle_event === 'test:genie:live'

function getRequiredEnv(name: string): string {
  const value = process.env[name]?.trim()
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`)
  }

  return value
}

function getLiveSettings() {
  return {
    baseURL: getRequiredEnv('DATABRICKS_HOST'),
    spaceId: getRequiredEnv('DATABRICKS_GENIE_SPACE_ID'),
    headers: {
      Authorization: `Bearer ${getRequiredEnv('DATABRICKS_TOKEN')}`,
    },
  }
}

function logGenieResult(label: string, result: {
  conversationId: string
  messageId: string
  status: string
  text: string
  sql?: string
  attachmentId?: string
  suggestedQuestions?: string[]
  queryResult?: {
    rowCount: number
    columns: Array<{ name: string; typeName?: string }>
    rows: unknown[][]
  }
}) {
  console.log(`\n[${label}]`)
  console.log(`conversationId: ${result.conversationId}`)
  console.log(`messageId: ${result.messageId}`)
  console.log(`status: ${result.status}`)
  console.log(`text: ${result.text || '<empty>'}`)

  if (result.sql) {
    console.log(`sql: ${result.sql}`)
  }

  if (result.attachmentId) {
    console.log(`attachmentId: ${result.attachmentId}`)
  }

  if (result.suggestedQuestions && result.suggestedQuestions.length > 0) {
    console.log(`suggestedQuestions: ${result.suggestedQuestions.join(' | ')}`)
  }

  if (result.queryResult) {
    console.log(`queryResult.rowCount: ${result.queryResult.rowCount}`)
    console.log(
      `queryResult.columns: ${result.queryResult.columns
        .map((column) => `${column.name}${column.typeName ? ` (${column.typeName})` : ''}`)
        .join(', ')}`
    )
    console.log(`queryResult.previewRows: ${JSON.stringify(result.queryResult.rows.slice(0, 5), null, 2)}`)
  }
}

describe.skipIf(!RUN_GENIE_LIVE_TEST)('Databricks Genie live smoke test', () => {
  const question =
    process.env.DATABRICKS_GENIE_TEST_QUESTION?.trim() ??
    'Give me a short summary of the data available in this Genie space.'

  const helper = createDatabricksGenieConversationClient(getLiveSettings())
  const agent = createDatabricksGenieAgent(getLiveSettings())

  let initialResult: Awaited<ReturnType<typeof helper.ask>>

  beforeAll(async () => {
    initialResult = await helper.ask(question, {
      fetchQueryResult: false,
    })

    if (initialResult.attachmentId && !initialResult.queryResult) {
      try {
        initialResult = {
          ...initialResult,
          queryResult: await helper.getQueryResult(
            initialResult.conversationId,
            initialResult.messageId,
            initialResult.attachmentId
          ),
        }
      } catch (error) {
        console.log('[helper] query-result fetch was not available:', error instanceof Error ? error.message : error)
      }
    }

    logGenieResult('helper', initialResult)
    console.log(
      '[helper.attachments]',
      JSON.stringify(
        initialResult.message.normalizedAttachments.map((attachment) => ({
          type: attachment.type,
          attachmentId: attachment.attachmentId,
          rawKeys: Object.keys(attachment.raw),
          text: attachment.text?.content,
          sql: attachment.query?.query,
          suggestedQuestions: attachment.suggestedQuestions?.questions,
        })),
        null,
        2
      )
    )
  }, 120_000)

  it.sequential(
    'maps raw Genie attachments into normalized helper output',
    async () => {
      expect(initialResult.conversationId.length).toBeGreaterThan(0)
      expect(initialResult.messageId.length).toBeGreaterThan(0)
      expect(initialResult.status).toBe('COMPLETED')
      expect(initialResult.message.raw).toBeDefined()
      expect(initialResult.message.attachments.length).toBeGreaterThan(0)

      const textAttachment = initialResult.message.normalizedAttachments.find(
        (attachment) => attachment.type === 'text'
      )
      if (textAttachment?.text?.content) {
        expect(initialResult.text).toBe(textAttachment.text.content)
      }

      const queryAttachment = initialResult.message.normalizedAttachments.find(
        (attachment) => attachment.type === 'query'
      )
      if (queryAttachment?.query?.query) {
        expect(initialResult.sql).toBe(queryAttachment.query.query)
        expect(initialResult.attachmentId).toBe(queryAttachment.attachmentId)
      }

      const suggestionAttachment = initialResult.message.normalizedAttachments.find(
        (attachment) => attachment.type === 'suggested_questions'
      )
      if (suggestionAttachment?.suggestedQuestions?.questions) {
        expect(initialResult.suggestedQuestions).toEqual(suggestionAttachment.suggestedQuestions.questions)
      }

      if (initialResult.queryResult) {
        expect(initialResult.queryResult.rowCount).toBeGreaterThanOrEqual(0)
        expect(Array.isArray(initialResult.queryResult.columns)).toBe(true)
        expect(Array.isArray(initialResult.queryResult.rows)).toBe(true)
      }
    },
    120_000
  )

  it.sequential(
    'uses Genie suggested questions for helper and agent follow-ups',
    async () => {
      expect(initialResult.suggestedQuestions.length).toBeGreaterThan(0)

      const helperFollowUpQuestion = initialResult.suggestedQuestions[0]
      let helperFollowUpResult = await helper.ask(helperFollowUpQuestion, {
        conversationId: initialResult.conversationId,
      })

      if (helperFollowUpResult.attachmentId && !helperFollowUpResult.queryResult) {
        helperFollowUpResult = {
          ...helperFollowUpResult,
          queryResult: await helper.getQueryResult(
            helperFollowUpResult.conversationId,
            helperFollowUpResult.messageId,
            helperFollowUpResult.attachmentId
          ),
        }
      }

      logGenieResult('helper-follow-up', helperFollowUpResult)

      expect(helperFollowUpResult.conversationId).toBe(initialResult.conversationId)
      expect(helperFollowUpResult.messageId).not.toBe(initialResult.messageId)
      expect(
        helperFollowUpResult.text.length > 0 ||
          Boolean(helperFollowUpResult.sql) ||
          helperFollowUpResult.suggestedQuestions.length > 0
      ).toBe(true)

      const agentFollowUpQuestion =
        initialResult.suggestedQuestions[1] ??
        process.env.DATABRICKS_GENIE_TEST_FOLLOW_UP_QUESTION?.trim() ??
        helperFollowUpQuestion

      const agentResult = await agent.generate({
        messages: [
          { role: 'user', content: question },
          { role: 'assistant', content: initialResult.text || 'Genie returned metadata without text.' },
          { role: 'user', content: agentFollowUpQuestion },
        ],
        options: {
          conversationId: initialResult.conversationId,
          fetchQueryResult: true,
        },
      })

      logGenieResult('agent-follow-up', agentResult.genie)

      expect(agentResult.genie.conversationId).toBe(initialResult.conversationId)
      expect(agentResult.genie.messageId.length).toBeGreaterThan(0)
      expect(agentResult.genie.status).toBe('COMPLETED')
      expect(
        agentResult.text.length > 0 ||
          Boolean(agentResult.genie.sql) ||
          agentResult.genie.suggestedQuestions.length > 0
      ).toBe(true)
    },
    180_000
  )
})
