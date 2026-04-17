import { describe, expect, it, vi } from 'vitest'
import { createDatabricksGenieConversationClient } from '../src/genie/conversation-client'

type MockRoute = {
  method: string
  path: string
  handler: (request: Request, body: unknown) => unknown
}

function createMockFetch(routes: MockRoute[]): typeof globalThis.fetch {
  return vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(input.toString(), init)
    const url = new URL(request.url)
    const route = routes.find(
      (candidate) => candidate.method === request.method && candidate.path === url.pathname
    )

    if (!route) {
      return new Response(`Unhandled route: ${request.method} ${url.pathname}`, { status: 404 })
    }

    const bodyText = request.method === 'GET' ? '' : await request.text()
    const body = bodyText.length > 0 ? (JSON.parse(bodyText) as unknown) : undefined
    const payload = route.handler(request, body)

    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }) as unknown as typeof globalThis.fetch
}

function createHelper(mockFetch: typeof globalThis.fetch) {
  return createDatabricksGenieConversationClient({
    baseURL: 'https://example.databricks.com',
    spaceId: 'space-123',
    headers: {
      Authorization: 'Bearer test-token',
    },
    fetch: mockFetch,
    initialPollIntervalMs: 0,
  })
}

describe('createDatabricksGenieConversationClient', () => {
  it('parses serialized space sample questions', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123',
          handler: () => ({
            space_id: 'space-123',
            title: 'Taxi analytics',
            serialized_space: JSON.stringify({
              version: 2,
              config: {
                sample_questions: [
                  {
                    id: 'sample-1',
                    question: ['What is the total number of trips?'],
                  },
                  {
                    id: 'sample-2',
                    question: ['Show the top 5 routes by revenue'],
                  },
                ],
              },
            }),
          }),
        },
      ])
    )

    const space = await helper.getSpace()
    const sampleQuestions = await helper.getSampleQuestions()

    expect(space.id).toBe('space-123')
    expect(space.serializedSpace?.sampleQuestions).toEqual([
      {
        id: 'sample-1',
        text: 'What is the total number of trips?',
        raw: {
          id: 'sample-1',
          question: ['What is the total number of trips?'],
        },
      },
      {
        id: 'sample-2',
        text: 'Show the top 5 routes by revenue',
        raw: {
          id: 'sample-2',
          question: ['Show the top 5 routes by revenue'],
        },
      },
    ])
    expect(sampleQuestions.map((question) => question.text)).toEqual([
      'What is the total number of trips?',
      'Show the top 5 routes by revenue',
    ])
  })

  it('returns no sample questions for integrated Genie spaces that do not expose serialized export', async () => {
    const fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          error_code: 'BAD_REQUEST',
          message: 'Serialized export is not available to integrated Genie spaces.',
        }),
        {
          status: 400,
          statusText: 'Bad Request',
          headers: { 'Content-Type': 'application/json' },
        }
      )
    ) as unknown as typeof globalThis.fetch

    const helper = createHelper(fetch)

    await expect(helper.getSampleQuestions()).resolves.toEqual([])
  })

  it('parses nested start conversation responses', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: {
              id: 'conversation-1',
              title: 'Initial question',
            },
            message: {
              id: 'message-1',
              conversation_id: 'conversation-1',
              status: 'IN_PROGRESS',
              attachments: null,
            },
          }),
        },
      ])
    )

    const result = await helper.startConversation('How are sales?')

    expect(result.conversation.id).toBe('conversation-1')
    expect(result.message.id).toBe('message-1')
    expect(result.message.status).toBe('IN_PROGRESS')
  })

  it('parses canonical Java SDK message and conversation fields', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: {
              conversation_id: 'conversation-10',
              id: 'legacy-conversation-10',
              created_timestamp: 1710000000000,
              last_updated_timestamp: 1710000001000,
              space_id: 'space-123',
              title: 'Revenue analysis',
              user_id: 42,
            },
            conversation_id: 'conversation-10',
            message: {
              message_id: 'message-10',
              id: 'legacy-message-10',
              conversation_id: 'conversation-10',
              created_timestamp: 1710000000001,
              last_updated_timestamp: 1710000001001,
              space_id: 'space-123',
              user_id: 42,
              status: 'COMPLETED',
              feedback: {
                rating: 'POSITIVE',
              },
              query_result: {
                is_truncated: true,
                row_count: 99,
                statement_id: 'stmt-legacy',
                statement_id_signature: 'sig-legacy',
              },
              attachments: [
                {
                  attachment_id: 'attachment-10',
                  query: {
                    id: 'query-10',
                    description: 'Revenue by route',
                    last_updated_timestamp: 1710000002000,
                    parameters: [
                      {
                        keyword: 'limit',
                        sql_type: 'BIGINT',
                        value: '5',
                      },
                    ],
                    query: 'SELECT * FROM revenue LIMIT 5',
                    query_result_metadata: {
                      is_truncated: false,
                      row_count: 5,
                    },
                    statement_id: 'stmt-10',
                    thoughts: [
                      {
                        thought_type: 'THOUGHT_TYPE_DESCRIPTION',
                        content: 'Summarize revenue by route.',
                      },
                    ],
                    title: 'Top routes',
                  },
                  text: {
                    id: 'text-10',
                    content: 'Here are the top routes.',
                  },
                },
              ],
            },
            message_id: 'message-10',
          }),
        },
      ])
    )

    const result = await helper.startConversation('Show me top routes')

    expect(result.conversation.id).toBe('conversation-10')
    expect(result.conversation.legacyId).toBe('legacy-conversation-10')
    expect(result.conversation.conversationId).toBe('conversation-10')
    expect(result.message.id).toBe('message-10')
    expect(result.message.legacyId).toBe('legacy-message-10')
    expect(result.message.messageId).toBe('message-10')
    expect(result.message.feedback?.rating).toBe('POSITIVE')
    expect(result.message.queryResult).toMatchObject({
      isTruncated: true,
      rowCount: 99,
      statementId: 'stmt-legacy',
    })
    expect(result.message.attachments[0]?.query).toMatchObject({
      id: 'query-10',
      description: 'Revenue by route',
      statementId: 'stmt-10',
      title: 'Top routes',
      parameters: [{ keyword: 'limit', sqlType: 'BIGINT', value: '5' }],
      thoughts: [
        {
          thoughtType: 'THOUGHT_TYPE_DESCRIPTION',
          content: 'Summarize revenue by route.',
        },
      ],
      queryResultMetadata: {
        isTruncated: false,
        rowCount: 5,
      },
    })
    expect(result.message.normalizedAttachments.map((attachment) => attachment.type)).toEqual([
      'query',
      'text',
    ])
  })

  it('supports follow-up conversations and attachment parsing when Genie returns split attachments', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages',
          handler: (_request, body) => {
            expect(body).toEqual({ content: 'Show me the same data for this quarter' })

            return {
              id: 'message-2',
              conversation_id: 'conversation-1',
              status: 'IN_PROGRESS',
              attachments: null,
            }
          },
        },
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages/message-2',
          handler: () => ({
            id: 'message-2',
            conversation_id: 'conversation-1',
            status: 'COMPLETED',
            attachments: [
              {
                attachment_id: 'attachment-1',
                text: {
                  content: 'Revenue increased 12% quarter over quarter.',
                },
              },
              {
                attachment_id: 'attachment-1',
                query: {
                  query: 'SELECT * FROM revenue',
                  description: 'Revenue by quarter',
                },
              },
              {
                suggested_questions: {
                  questions: ['Break this down by region'],
                },
              },
            ],
          }),
        },
      ])
    )

    const result = await helper.ask('Show me the same data for this quarter', {
      conversationId: 'conversation-1',
    })

    expect(result.conversationId).toBe('conversation-1')
    expect(result.messageId).toBe('message-2')
    expect(result.text).toBe('Revenue increased 12% quarter over quarter.')
    expect(result.sql).toBe('SELECT * FROM revenue')
    expect(result.attachmentId).toBe('attachment-1')
    expect(result.suggestedQuestions).toEqual(['Break this down by region'])
    expect(result.attachments).toHaveLength(3)
    expect(result.attachments[1]).toMatchObject({
      type: 'query',
      query: {
        description: 'Revenue by quarter',
      },
    })
  })

  it('parses suggested questions when Genie returns query, text, and suggestions in one attachment', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: {
              id: 'conversation-2',
            },
            message: {
              id: 'message-3',
              conversation_id: 'conversation-2',
              status: 'COMPLETED',
              attachments: [
                {
                  attachment_id: 'attachment-2',
                  query: {
                    query: 'SELECT day_of_week, COUNT(*) FROM trips GROUP BY day_of_week',
                    description: 'Trip counts by day of week',
                  },
                  text: {
                    content: 'Saturday has the most trips.',
                  },
                  suggested_questions: {
                    questions: [
                      'Break this down by pickup zip code',
                      'Show the same distribution by hour of day',
                    ],
                  },
                },
              ],
            },
          }),
        },
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-2/messages/message-3',
          handler: () => ({
            id: 'message-3',
            conversation_id: 'conversation-2',
            status: 'COMPLETED',
            attachments: [
              {
                attachment_id: 'attachment-2',
                query: {
                  query: 'SELECT day_of_week, COUNT(*) FROM trips GROUP BY day_of_week',
                  description: 'Trip counts by day of week',
                },
                text: {
                  content: 'Saturday has the most trips.',
                },
                suggested_questions: {
                  questions: [
                    'Break this down by pickup zip code',
                    'Show the same distribution by hour of day',
                  ],
                },
              },
            ],
          }),
        },
      ])
    )

    const result = await helper.ask('What is the distribution of trips by day of the week?')

    expect(result.text).toBe('Saturday has the most trips.')
    expect(result.sql).toBe('SELECT day_of_week, COUNT(*) FROM trips GROUP BY day_of_week')
    expect(result.attachmentId).toBe('attachment-2')
    expect(result.suggestedQuestions).toEqual([
      'Break this down by pickup zip code',
      'Show the same distribution by hour of day',
    ])
    expect(result.attachments).toHaveLength(3)
    expect(result.attachments.map((attachment) => attachment.type)).toEqual([
      'query',
      'text',
      'suggested_questions',
    ])
  })

  it('aggregates multiple text and suggested-question attachments', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: {
              id: 'conversation-4',
            },
            message: {
              id: 'message-5',
              conversation_id: 'conversation-4',
              status: 'COMPLETED',
              attachments: [
                {
                  attachment_id: 'attachment-5a',
                  text: {
                    content: 'Here are the top 5 routes by revenue.',
                  },
                },
              ],
            },
          }),
        },
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-4/messages/message-5',
          handler: () => ({
            id: 'message-5',
            conversation_id: 'conversation-4',
            status: 'COMPLETED',
            attachments: [
              {
                attachment_id: 'attachment-5a',
                text: {
                  content: 'Here are the top 5 routes by revenue.',
                },
              },
              {
                attachment_id: 'attachment-5b',
                text: {
                  content: 'The highest revenue route is 10023-10023.',
                },
              },
              {
                attachment_id: 'attachment-5c',
                suggested_questions: {
                  questions: ['Show the top 5 routes by number of trips'],
                },
              },
              {
                attachment_id: 'attachment-5d',
                suggested_questions: {
                  questions: ['Break this down by pickup zip code'],
                },
              },
            ],
          }),
        },
      ])
    )

    const result = await helper.ask('What are the top 5 routes by total revenue?')

    expect(result.text).toBe(
      'Here are the top 5 routes by revenue.\n\nThe highest revenue route is 10023-10023.'
    )
    expect(result.suggestedQuestions).toEqual([
      'Show the top 5 routes by number of trips',
      'Break this down by pickup zip code',
    ])
  })

  it('prefers final-summary text attachments over follow-up-question text attachments', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: {
              id: 'conversation-5',
            },
            message: {
              id: 'message-6',
              conversation_id: 'conversation-5',
              status: 'COMPLETED',
              attachments: [
                {
                  attachment_id: 'attachment-6',
                  text: {
                    content: 'Initial placeholder',
                  },
                },
              ],
            },
          }),
        },
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-5/messages/message-6',
          handler: () => ({
            id: 'message-6',
            conversation_id: 'conversation-5',
            status: 'COMPLETED',
            attachments: [
              {
                attachment_id: 'attachment-6a',
                text: {
                  id: 'text-summary',
                  content: 'The top 5 routes by revenue are listed below.',
                },
              },
              {
                attachment_id: 'attachment-6b',
                text: {
                  id: 'text-follow-up',
                  content:
                    'Would you like to see the top 5 routes strictly by total revenue without including ties?',
                  purpose: 'FOLLOW_UP_QUESTION',
                },
              },
            ],
          }),
        },
      ])
    )

    const result = await helper.ask('What are the top 5 routes by total revenue?')

    expect(result.text).toBe('The top 5 routes by revenue are listed below.')
    expect(
      result.attachments.find(
        (attachment) => attachment.type === 'text' && attachment.text?.purpose === 'FOLLOW_UP_QUESTION'
      )?.text
    ).toMatchObject({
      id: 'text-follow-up',
      purpose: 'FOLLOW_UP_QUESTION',
    })
  })

  it('re-fetches the completed message to use the finalized attachments', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: {
              id: 'conversation-3',
            },
            message: {
              id: 'message-4',
              conversation_id: 'conversation-3',
              status: 'COMPLETED',
              attachments: [
                {
                  attachment_id: 'attachment-stale',
                  text: {
                    content: 'Interim response',
                  },
                },
              ],
            },
          }),
        },
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-3/messages/message-4',
          handler: () => ({
            id: 'message-4',
            conversation_id: 'conversation-3',
            status: 'COMPLETED',
            attachments: [
              {
                attachment_id: 'attachment-final',
                text: {
                  content: 'Final response with complete data',
                },
                query: {
                  query: 'SELECT * FROM trips LIMIT 5',
                },
                suggested_questions: {
                  questions: ['Show the same for revenue'],
                },
              },
            ],
          }),
        },
      ])
    )

    const result = await helper.ask('Give me a final answer')

    expect(result.text).toBe('Final response with complete data')
    expect(result.sql).toBe('SELECT * FROM trips LIMIT 5')
    expect(result.suggestedQuestions).toEqual(['Show the same for revenue'])
    expect(result.attachmentId).toBe('attachment-final')
  })

  it('retries until a message reaches completion', async () => {
    let attempts = 0

    const helper = createHelper(
      createMockFetch([
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages/message-1',
          handler: () => {
            attempts += 1

            if (attempts === 1) {
              return {
                id: 'message-1',
                conversation_id: 'conversation-1',
                status: 'IN_PROGRESS',
                attachments: null,
              }
            }

            return {
              id: 'message-1',
              conversation_id: 'conversation-1',
              status: 'COMPLETED',
              attachments: [],
            }
          },
        },
      ])
    )

    const result = await helper.waitForCompletion('conversation-1', 'message-1')

    expect(result.status).toBe('COMPLETED')
    expect(attempts).toBe(2)
  })

  it('raises terminal errors for failed messages', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages/message-1',
          handler: () => ({
            id: 'message-1',
            conversation_id: 'conversation-1',
            status: 'FAILED',
            error: {
              error: 'Warehouse unavailable',
              type: 'TRANSIENT',
            },
            attachments: null,
          }),
        },
      ])
    )

    await expect(helper.waitForCompletion('conversation-1', 'message-1')).rejects.toThrow(
      'Warehouse unavailable'
    )
  })

  it('raises terminal errors for cancelled messages', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages/message-1',
          handler: () => ({
            id: 'message-1',
            conversation_id: 'conversation-1',
            status: 'CANCELLED',
            attachments: null,
          }),
        },
      ])
    )

    await expect(helper.waitForCompletion('conversation-1', 'message-1')).rejects.toThrow(
      'cancelled'
    )
  })

  it('raises terminal errors for expired query results', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages/message-1',
          handler: () => ({
            id: 'message-1',
            conversation_id: 'conversation-1',
            status: 'QUERY_RESULT_EXPIRED',
            attachments: null,
          }),
        },
      ])
    )

    await expect(helper.waitForCompletion('conversation-1', 'message-1')).rejects.toThrow(
      'query result expired'
    )
  })

  it('reports polling progress for intermediate and terminal messages', async () => {
    const onProgress = vi.fn()
    let attempts = 0

    const helper = createHelper(
      createMockFetch([
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages/message-1',
          handler: () => {
            attempts += 1

            return attempts === 1
              ? {
                  id: 'message-1',
                  conversation_id: 'conversation-1',
                  status: 'IN_PROGRESS',
                  attachments: null,
                }
              : {
                  id: 'message-1',
                  conversation_id: 'conversation-1',
                  status: 'COMPLETED',
                  attachments: [],
                }
          },
        },
      ])
    )

    const result = await helper.waitForCompletion('conversation-1', 'message-1', {
      onProgress,
    })

    expect(result.status).toBe('COMPLETED')
    expect(onProgress).toHaveBeenCalledTimes(2)
    expect(onProgress.mock.calls[0]?.[0]).toMatchObject({ status: 'IN_PROGRESS' })
    expect(onProgress.mock.calls[1]?.[0]).toMatchObject({ status: 'COMPLETED' })
  })

  it('times out if polling never completes', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages/message-1',
          handler: () => ({
            id: 'message-1',
            conversation_id: 'conversation-1',
            status: 'IN_PROGRESS',
            attachments: null,
          }),
        },
      ])
    )

    await expect(
      helper.waitForCompletion('conversation-1', 'message-1', {
        timeoutMs: 1,
      })
    ).rejects.toThrow('timed out')
  })

  it('fetches query results only when explicitly enabled', async () => {
    const fetch = createMockFetch([
      {
        method: 'POST',
        path: '/api/2.0/genie/spaces/space-123/start-conversation',
        handler: () => ({
          conversation: { id: 'conversation-1' },
          message: {
            id: 'message-1',
            conversation_id: 'conversation-1',
            status: 'IN_PROGRESS',
            attachments: null,
          },
        }),
      },
      {
        method: 'GET',
        path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages/message-1',
        handler: () => ({
          id: 'message-1',
          conversation_id: 'conversation-1',
          status: 'COMPLETED',
          attachments: [
            {
              attachment_id: 'attachment-1',
              text: {
                content: 'Here is the data you requested.',
              },
            },
            {
              attachment_id: 'attachment-1',
              query: {
                query: 'SELECT region, revenue FROM quarterly_revenue',
              },
            },
          ],
        }),
      },
      {
        method: 'GET',
        path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages/message-1/attachments/attachment-1/query-result',
        handler: () => ({
          statement_response: {
            manifest: {
              schema: {
                columns: [
                  { name: 'region', type_name: 'STRING' },
                  { name: 'revenue', type_name: 'DOUBLE' },
                ],
              },
            },
            result: {
              data_array: [
                ['North America', '100'],
                ['Europe', '90'],
              ],
            },
          },
        }),
      },
    ])

    const helper = createHelper(fetch)
    const result = await helper.ask('Show quarterly revenue', {
      fetchQueryResult: true,
    })

    expect(result.queryResult).toBeDefined()
    expect(result.queryResult?.columns).toEqual([
      { name: 'region', typeName: 'STRING' },
      { name: 'revenue', typeName: 'DOUBLE' },
    ])
    expect(result.queryResult?.rows).toEqual([
      ['North America', '100'],
      ['Europe', '90'],
    ])
    expect(result.attachments).toHaveLength(2)
    expect(fetch).toHaveBeenCalledTimes(4)
  })

  it('re-fetches once when the initial message already completed', async () => {
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
                text: {
                  content: 'Immediate answer',
                },
              },
            ],
          },
        }),
      },
      {
        method: 'GET',
        path: '/api/2.0/genie/spaces/space-123/conversations/conversation-1/messages/message-1',
        handler: () => ({
          id: 'message-1',
          conversation_id: 'conversation-1',
          status: 'COMPLETED',
          attachments: [
            {
              text: {
                content: 'Immediate answer',
              },
            },
          ],
        }),
      },
    ])

    const helper = createHelper(fetch)
    const result = await helper.ask('Answer immediately')

    expect(result.text).toBe('Immediate answer')
    expect(fetch).toHaveBeenCalledTimes(2)
  })

  it('lists conversation messages with pagination metadata', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-20/messages',
          handler: () => ({
            messages: [
              {
                message_id: 'message-20',
                conversation_id: 'conversation-20',
                status: 'COMPLETED',
                attachments: [
                  {
                    attachment_id: 'attachment-20',
                    text: {
                      content: 'Message page content',
                    },
                  },
                ],
              },
            ],
            next_page_token: 'page-2',
          }),
        },
      ])
    )

    const result = await helper.listConversationMessages({
      conversationId: 'conversation-20',
      pageSize: 50,
    })

    expect(result.nextPageToken).toBe('page-2')
    expect(result.messages[0]?.messageId).toBe('message-20')
    expect(result.messages[0]?.normalizedAttachments[0]).toMatchObject({
      type: 'text',
      text: {
        content: 'Message page content',
      },
    })
  })

  it('supports full query result download helpers', async () => {
    const helper = createHelper(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-30/messages/message-30/attachments/attachment-30/downloads',
          handler: () => ({
            download_id: 'download-30',
            download_id_signature: 'signature-30',
          }),
        },
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-30/messages/message-30/attachments/attachment-30/downloads/download-30',
          handler: () => ({
            statement_response: {
              manifest: {
                total_row_count: 2,
                schema: {
                  columns: [
                    { name: 'route', type_name: 'STRING' },
                    { name: 'revenue', type_name: 'DOUBLE' },
                  ],
                },
              },
              result: {
                data_array: [
                  ['A-B', 12.5],
                  ['C-D', 10.0],
                ],
              },
            },
          }),
        },
      ])
    )

    const download = await helper.generateDownloadFullQueryResult({
      attachmentId: 'attachment-30',
      conversationId: 'conversation-30',
      messageId: 'message-30',
    })

    expect(download).toMatchObject({
      downloadId: 'download-30',
      downloadIdSignature: 'signature-30',
    })

    const result = await helper.getDownloadFullQueryResult({
      attachmentId: 'attachment-30',
      conversationId: 'conversation-30',
      downloadId: download.downloadId,
      downloadIdSignature: download.downloadIdSignature,
      messageId: 'message-30',
    })

    expect(result.queryResult.columns).toEqual([
      { name: 'route', typeName: 'STRING' },
      { name: 'revenue', typeName: 'DOUBLE' },
    ])
    expect(result.queryResult.rows).toEqual([
      ['A-B', 12.5],
      ['C-D', 10.0],
    ])
    expect(result.queryResult.rowCount).toBe(2)
  })

  it('deletes conversations through the Genie API', async () => {
    const fetch = createMockFetch([
      {
        method: 'DELETE',
        path: '/api/2.0/genie/spaces/space-123/conversations/conversation-50',
        handler: () => ({}),
      },
    ])

    const helper = createHelper(fetch)
    await expect(helper.deleteConversation('conversation-50')).resolves.toBeUndefined()
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('streams progressive Genie status, attachments, query results, and the final result', async () => {
    let messagePollCount = 0

    const helper = createHelper(
      createMockFetch([
        {
          method: 'POST',
          path: '/api/2.0/genie/spaces/space-123/start-conversation',
          handler: () => ({
            conversation: { id: 'conversation-40' },
            message: {
              id: 'message-40',
              conversation_id: 'conversation-40',
              status: 'SUBMITTED',
              attachments: [],
            },
          }),
        },
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-40/messages/message-40',
          handler: () => {
            messagePollCount += 1

            if (messagePollCount === 1) {
              return {
                id: 'message-40',
                conversation_id: 'conversation-40',
                status: 'FETCHING_METADATA',
                attachments: [],
              }
            }

            if (messagePollCount === 2) {
              return {
                id: 'message-40',
                conversation_id: 'conversation-40',
                status: 'EXECUTING_QUERY',
                attachments: [
                  {
                    attachment_id: 'attachment-40',
                    query: {
                      description: 'Revenue by route',
                      query: 'SELECT route, revenue FROM trips ORDER BY revenue DESC LIMIT 5',
                    },
                  },
                ],
              }
            }

            return {
              id: 'message-40',
              conversation_id: 'conversation-40',
              status: 'COMPLETED',
              attachments: [
                {
                  attachment_id: 'attachment-40',
                  query: {
                    description: 'Revenue by route',
                    query: 'SELECT route, revenue FROM trips ORDER BY revenue DESC LIMIT 5',
                    thoughts: [
                      {
                        thought_type: 'THOUGHT_TYPE_DESCRIPTION',
                        content: 'Summarize the highest revenue routes first.',
                      },
                      {
                        thought_type: 'THOUGHT_TYPE_STEPS',
                        content: '- Rank routes by revenue\n- Keep the top 5 rows',
                      },
                    ],
                  },
                },
                {
                  attachment_id: 'attachment-text-40',
                  text: {
                    content: 'The top 5 routes are shown below.',
                  },
                },
                {
                  attachment_id: 'attachment-suggested-40',
                  suggested_questions: {
                    questions: ['Show the same ranking by month'],
                  },
                },
              ],
            }
          },
        },
        {
          method: 'GET',
          path: '/api/2.0/genie/spaces/space-123/conversations/conversation-40/messages/message-40/attachments/attachment-40/query-result',
          handler: () => ({
            statement_response: {
              manifest: {
                total_row_count: 2,
                schema: {
                  columns: [
                    { name: 'route', type_name: 'STRING' },
                    { name: 'revenue', type_name: 'DOUBLE' },
                  ],
                },
              },
              result: {
                data_array: [
                  ['A-B', 12.5],
                  ['C-D', 10.0],
                ],
              },
            },
          }),
        },
      ])
    )

    const events = []
    for await (const event of helper.streamConversation('Show the top 5 routes by total revenue', {
      fetchQueryResult: true,
    })) {
      events.push(event)
    }

    expect(events.map((event) => event.type)).toEqual([
      'status',
      'status',
      'status',
      'attachment',
      'query-result',
      'status',
      'attachment',
      'attachment',
      'attachment',
      'complete',
    ])
    expect(events.filter((event) => event.type === 'status').map((event) => event.status)).toEqual([
      'SUBMITTED',
      'FETCHING_METADATA',
      'EXECUTING_QUERY',
      'COMPLETED',
    ])
    expect(events.find((event) => event.type === 'query-result')).toMatchObject({
      type: 'query-result',
      attachmentId: 'attachment-40',
      queryResult: {
        columns: [
          { name: 'route', typeName: 'STRING' },
          { name: 'revenue', typeName: 'DOUBLE' },
        ],
        rows: [
          ['A-B', 12.5],
          ['C-D', 10.0],
        ],
      },
    })
    expect(
      events.find(
        (event) =>
          event.type === 'attachment' &&
          event.attachment.type === 'query' &&
          (event.attachment.query?.thoughts.length ?? 0) > 0
      )
    ).toMatchObject({
        type: 'attachment',
        attachment: {
          type: 'query',
          query: {
            thoughts: [
              {
                thoughtType: 'THOUGHT_TYPE_DESCRIPTION',
                content: 'Summarize the highest revenue routes first.',
              },
              {
                thoughtType: 'THOUGHT_TYPE_STEPS',
                content: '- Rank routes by revenue\n- Keep the top 5 rows',
              },
            ],
          },
        },
      })
    expect(events.at(-1)).toMatchObject({
      type: 'complete',
      result: {
        text: 'The top 5 routes are shown below.',
        suggestedQuestions: ['Show the same ranking by month'],
      },
    })
  })
})
