import { createAgentUIStreamResponse, type UIMessage } from 'ai'
import { describe, expect, it, vi } from 'vitest'
import { createDatabricksGenieAgent } from '../src/genie/agent'

type MockRoute = {
  method: string
  path: string
  handler: (request: Request, body: unknown) => unknown
}

function extractMessagePayload(payload: unknown): Record<string, unknown> | null {
  if (!payload || typeof payload !== 'object') {
    return null
  }

  const record = payload as Record<string, unknown>

  if (
    typeof record.id === 'string' &&
    typeof record.conversation_id === 'string' &&
    typeof record.status === 'string'
  ) {
    return record
  }

  const nestedMessage = record.message
  if (!nestedMessage || typeof nestedMessage !== 'object') {
    return null
  }

  const messageRecord = nestedMessage as Record<string, unknown>
  if (
    typeof messageRecord.id === 'string' &&
    typeof messageRecord.conversation_id === 'string' &&
    typeof messageRecord.status === 'string'
  ) {
    return messageRecord
  }

  return null
}

function createMockFetch(routes: MockRoute[]): typeof globalThis.fetch {
  const completedMessages = new Map<string, unknown>()

  return vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(input.toString(), init)
    const url = new URL(request.url)
    let route = routes.find(
      (candidate) => candidate.method === request.method && candidate.path === url.pathname
    )

    if (!route && request.method === 'GET') {
      const match = url.pathname.match(
        /^\/api\/2\.0\/genie\/spaces\/[^/]+\/conversations\/([^/]+)\/messages\/([^/]+)$/
      )

      if (match) {
        const cacheKey = `${match[1]}:${match[2]}`
        const cachedPayload = completedMessages.get(cacheKey)

        if (cachedPayload) {
          return new Response(JSON.stringify(cachedPayload), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          })
        }
      }
    }

    if (!route) {
      return new Response(`Unhandled route: ${request.method} ${url.pathname}`, { status: 404 })
    }

    const bodyText = request.method === 'GET' ? '' : await request.text()
    const body = bodyText.length > 0 ? (JSON.parse(bodyText) as unknown) : undefined
    const payload = route.handler(request, body)
    const messagePayload = extractMessagePayload(payload)

    if (
      messagePayload &&
      messagePayload.status === 'COMPLETED' &&
      typeof messagePayload.id === 'string' &&
      typeof messagePayload.conversation_id === 'string'
    ) {
      completedMessages.set(
        `${messagePayload.conversation_id}:${messagePayload.id}`,
        messagePayload
      )
    }

    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }) as unknown as typeof globalThis.fetch
}

function createAgent(mockFetch: typeof globalThis.fetch) {
  return createDatabricksGenieAgent({
    id: 'genie-agent',
    baseURL: 'https://example.databricks.com',
    spaceId: 'space-123',
    headers: {
      Authorization: 'Bearer test-token',
    },
    fetch: mockFetch,
    initialPollIntervalMs: 0,
  })
}

describe('createDatabricksGenieAgent', () => {
  it('uses only the latest user message as the Genie question', async () => {
    const agent = createAgent(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: (_request, body) => {
            expect(body).toEqual({ content: 'Use this one' })

            return {
              conversation: { id: 'conversation-1' },
              message: {
                id: 'message-1',
                conversation_id: 'conversation-1',
                status: 'COMPLETED',
                attachments: [
                  {
                    text: { content: 'Genie answer' },
                  },
                ],
              },
            }
          },
        },
      ])
    )

    const result = await agent.generate({
      messages: [
        { role: 'user', content: 'Ignore this older message' },
        { role: 'assistant', content: 'Old response' },
        { role: 'user', content: 'Use this one' },
      ],
      options: {},
    })

    expect(result.text).toBe('Genie answer')
    expect(result.genie.conversationId).toBe('conversation-1')
  })

  it('extracts text from multipart user content', async () => {
    const agent = createAgent(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: (_request, body) => {
            expect(body).toEqual({ content: 'First line\nSecond line' })

            return {
              conversation: { id: 'conversation-1' },
              message: {
                id: 'message-1',
                conversation_id: 'conversation-1',
                status: 'COMPLETED',
                attachments: [
                  {
                    text: { content: 'Multipart answer' },
                  },
                ],
              },
            }
          },
        },
      ])
    )

    const result = await agent.generate({
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'First line' },
            { type: 'text', text: 'Second line' },
          ],
        },
      ],
      options: {},
    })

    expect(result.text).toBe('Multipart answer')
  })

  it('supports generate with structured Genie metadata', async () => {
    const agent = createAgent(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: { id: 'conversation-1' },
            message: {
              id: 'message-1',
              conversation_id: 'conversation-1',
              status: 'COMPLETED',
              attachments: [
                {
                  attachment_id: 'attachment-1',
                  text: { content: 'Sales are up 8%.' },
                },
                {
                  attachment_id: 'attachment-1',
                  query: { query: 'SELECT * FROM sales' },
                },
              ],
            },
          }),
        },
      ])
    )

    const result = await agent.generate({
      prompt: 'What happened to sales?',
      options: {},
    })

    expect(result.text).toBe('Sales are up 8%.')
    expect(result.genie.sql).toBe('SELECT * FROM sales')
    expect(result.genie.attachmentId).toBe('attachment-1')
    expect(result.genie.attachments).toHaveLength(2)
    expect(result.finishReason).toEqual('stop')
    expect(result.usage).toMatchObject({
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      reasoningTokens: 0,
    })
  })

  it('preserves suggested questions when Genie returns a combined attachment payload', async () => {
    const agent = createAgent(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: { id: 'conversation-1' },
            message: {
              id: 'message-1',
              conversation_id: 'conversation-1',
              status: 'COMPLETED',
              attachments: [
                {
                  attachment_id: 'attachment-1',
                  text: { content: 'Sales are strongest on weekends.' },
                  query: {
                    query: 'SELECT day_of_week, COUNT(*) AS trip_count FROM trips GROUP BY day_of_week',
                  },
                  suggested_questions: {
                    questions: ['Show the same view by month'],
                  },
                },
              ],
            },
          }),
        },
      ])
    )

    const result = await agent.generate({
      prompt: 'What is the distribution of trips by day of the week?',
      options: {},
    })

    expect(result.text).toBe('Sales are strongest on weekends.')
    expect(result.genie.sql).toBe(
      'SELECT day_of_week, COUNT(*) AS trip_count FROM trips GROUP BY day_of_week'
    )
    expect(result.genie.suggestedQuestions).toEqual(['Show the same view by month'])
    expect(result.genie.attachments.map((attachment) => attachment.type)).toEqual([
      'query',
      'text',
      'suggested_questions',
    ])
  })

  it('supports stream results for createAgentUIStreamResponse', async () => {
    const agent = createAgent(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: { id: 'conversation-1' },
            message: {
              id: 'message-1',
              conversation_id: 'conversation-1',
              status: 'COMPLETED',
              attachments: [
                {
                  text: { content: 'Streaming Genie answer' },
                },
              ],
            },
          }),
        },
      ])
    )

    const streamResult = await agent.stream({
      prompt: 'Summarize the dashboard',
      options: {},
    })

    const chunks: string[] = []
    for await (const chunk of streamResult.textStream) {
      chunks.push(chunk)
    }

    expect(chunks.join('')).toBe('Streaming Genie answer')
    await expect(streamResult.genie).resolves.toMatchObject({
      text: 'Streaming Genie answer',
    })

    const response = await createAgentUIStreamResponse({
      agent,
      uiMessages: [
        {
          id: 'msg-1',
          role: 'user',
          parts: [{ type: 'text', text: 'Summarize the dashboard' }],
        } satisfies UIMessage,
      ],
    })

    expect(response).toBeInstanceOf(Response)
  })

  it('emits the expected AI SDK full stream lifecycle and finish metadata', async () => {
    const agent = createAgent(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: { id: 'conversation-1' },
            message: {
              id: 'message-1',
              conversation_id: 'conversation-1',
              status: 'COMPLETED',
              attachments: [
                {
                  text: { content: 'Lifecycle answer' },
                },
              ],
            },
          }),
        },
      ])
    )

    const streamResult = await agent.stream({
      prompt: 'Show the full stream lifecycle',
      options: {},
    })

    const parts: Array<{ type: string; [key: string]: unknown }> = []
    for await (const part of streamResult.fullStream) {
      parts.push(part as { type: string; [key: string]: unknown })
    }

    expect(parts[0]).toMatchObject({ type: 'start' })
    expect(parts[1]).toMatchObject({ type: 'start-step' })
    expect(parts).toContainEqual(expect.objectContaining({ type: 'text-start', id: 'genie-text-0' }))
    expect(parts).toContainEqual(
      expect.objectContaining({
        type: 'text-delta',
        id: 'genie-text-0',
        text: 'Lifecycle answer',
      })
    )
    expect(parts).toContainEqual(expect.objectContaining({ type: 'text-end', id: 'genie-text-0' }))

    const finishPart = parts.find((part) => part.type === 'finish')
    expect(finishPart).toMatchObject({
      type: 'finish',
      finishReason: 'stop',
      totalUsage: {
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
      },
    })
  })

  it('emits AI SDK error and error finish metadata when Genie streaming fails', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    const fetch = vi.fn(async () => new Response('boom', { status: 500, statusText: 'Internal Server Error' }))
    const agent = createAgent(fetch as unknown as typeof globalThis.fetch)
    try {
      const streamResult = await agent.stream({
        prompt: 'Trigger a streaming error',
        options: {},
      })

      const parts: Array<{ type: string; [key: string]: unknown }> = []
      for await (const part of streamResult.fullStream) {
        parts.push(part as { type: string; [key: string]: unknown })
      }

      expect(parts).toContainEqual(expect.objectContaining({ type: 'error' }))

      const finishPart = parts.find((part) => part.type === 'finish')
      expect(finishPart).toMatchObject({
        type: 'finish',
        finishReason: 'error',
        totalUsage: {
          inputTokens: 0,
          outputTokens: 0,
          totalTokens: 0,
        },
      })
    } finally {
      consoleError.mockRestore()
    }
  })

  it('fetches query results through the agent only when explicitly enabled', async () => {
    const fetch = createMockFetch([
      {
        method: 'POST',
        path: '/api/2.0/genie/spaces/space-123/start-conversation',
        handler: () => ({
          conversation: { id: 'conversation-1' },
          message: {
            id: 'message-1',
            conversation_id: 'conversation-1',
            status: 'COMPLETED',
            attachments: [
              {
                attachment_id: 'attachment-1',
                text: { content: 'Query ready' },
              },
              {
                attachment_id: 'attachment-1',
                query: { query: 'SELECT * FROM dashboard' },
              },
            ],
          },
        }),
      },
      {
        method: 'GET',
        path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages/message-1/attachments/attachment-1/query-result',
        handler: () => ({
          statement_response: {
            manifest: {
              total_row_count: 2,
              truncated: false,
              schema: {
                columns: [
                  { name: 'metric', type_name: 'STRING' },
                  { name: 'value', type_name: 'DOUBLE' },
                ],
              },
            },
            result: {
              data_array: [
                ['revenue', '100'],
                ['margin', '20'],
              ],
            },
          },
        }),
      },
    ])

    const agent = createAgent(fetch)
    const result = await agent.generate({
      prompt: 'Fetch the query result too',
      options: {
        fetchQueryResult: true,
      },
    })

    expect(result.genie.queryResult).toMatchObject({
      rowCount: 2,
      columns: [
        { name: 'metric', typeName: 'STRING' },
        { name: 'value', typeName: 'DOUBLE' },
      ],
      rows: [
        ['revenue', '100'],
        ['margin', '20'],
      ],
    })
    expect(fetch).toHaveBeenCalledTimes(3)
  })

  it('uses explicit conversation ids for follow-up questions', async () => {
    const agent = createAgent(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-9/messages',
          handler: (_request, body) => {
            expect(body).toEqual({ content: 'Follow up question' })

            return {
              id: 'message-2',
              conversation_id: 'conversation-9',
              status: 'COMPLETED',
              attachments: [
                {
                  text: { content: 'Follow-up answer' },
                },
              ],
            }
          },
        },
      ])
    )

    const result = await agent.generate({
      prompt: 'Follow up question',
      options: {
        conversationId: 'conversation-9',
      },
    })

    expect(result.genie.conversationId).toBe('conversation-9')
    expect(result.text).toBe('Follow-up answer')
  })

  it('does not fetch query results unless explicitly enabled', async () => {
    const fetch = createMockFetch([
      {
        method: 'POST',
        path: '/api/2.0/genie/spaces/space-123/start-conversation',
        handler: () => ({
          conversation: { id: 'conversation-1' },
          message: {
            id: 'message-1',
            conversation_id: 'conversation-1',
            status: 'COMPLETED',
            attachments: [
              {
                attachment_id: 'attachment-1',
                text: { content: 'Dashboard updated' },
              },
              {
                attachment_id: 'attachment-1',
                query: { query: 'SELECT * FROM dashboard' },
              },
            ],
          },
        }),
      },
    ])

    const agent = createAgent(fetch)
    const result = await agent.generate({
      prompt: 'What changed?',
      options: {},
    })

    expect(result.genie.queryResult).toBeUndefined()
    expect(fetch).toHaveBeenCalledTimes(2)
  })

  it('returns fallback text when Genie only provides SQL metadata', async () => {
    const agent = createAgent(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: { id: 'conversation-1' },
            message: {
              id: 'message-1',
              conversation_id: 'conversation-1',
              status: 'COMPLETED',
              attachments: [
                {
                  attachment_id: 'attachment-1',
                  query: { query: 'SELECT * FROM sales' },
                },
              ],
            },
          }),
        },
      ])
    )

    const result = await agent.generate({
      prompt: 'Generate SQL',
      options: {},
    })

    expect(result.text).toBe('Genie generated a SQL query for this request.')
    expect(result.genie.sql).toBe('SELECT * FROM sales')
  })

  it('throws when no user text is available', async () => {
    const agent = createAgent(createMockFetch([]))

    await expect(
      agent.generate({
        messages: [{ role: 'assistant', content: 'No user input here' }],
        options: {},
      })
    ).rejects.toThrow('requires at least one user message with text content')
  })
})
