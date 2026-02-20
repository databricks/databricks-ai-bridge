import { describe, it, expect } from 'vitest'
import { DatabricksResponsesAgentLanguageModel } from '../src/responses-agent-language-model/responses-agent-language-model'

/**
 * Creates a mock fetch function that returns a streaming response with trace_id
 */
function createMockFetchWithTrace(traceId: string): typeof fetch {
  return async () => {
    const encoder = new TextEncoder()

    const events = [
      `data: ${JSON.stringify({
        type: 'response.output_text.delta',
        item_id: 'msg_123',
        delta: 'Hello World',
      })}\n\n`,
      `data: ${JSON.stringify({
        type: 'responses.completed',
        response: {
          id: 'resp_123',
          status: 'completed',
          usage: {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
          },
          trace_id: traceId,
        },
      })}\n\n`,
    ]

    const stream = new ReadableStream({
      start(controller) {
        for (const event of events) {
          controller.enqueue(encoder.encode(event))
        }
        controller.close()
      },
    })

    return new Response(stream, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
      },
    })
  }
}

/**
 * Creates a mock fetch function that returns a non-streaming response with trace_id
 */
function createMockFetchNonStreamingWithTrace(traceId: string): typeof fetch {
  return async () => {
    const responseBody = {
      id: 'resp_456',
      model: 'test-model',
      output: [
        {
          type: 'message',
          role: 'assistant',
          id: 'msg_456',
          content: [
            {
              type: 'output_text',
              text: 'Hello World',
              annotations: [],
            },
          ],
        },
      ],
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
      },
      trace_id: traceId,
    }

    return new Response(JSON.stringify(responseBody), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
      },
    })
  }
}

/**
 * Creates a mock fetch function that returns streaming SSE with trace_id in
 * response.output_item.done.databricks_output.trace.info.trace_id
 */
function createMockFetchWithDatabricksOutputTrace(traceId: string): typeof fetch {
  return async () => {
    const encoder = new TextEncoder()

    const events = [
      `data: ${JSON.stringify({
        type: 'response.output_text.delta',
        item_id: 'msg_789',
        delta: 'Hello from trace',
      })}\n\n`,
      `data: ${JSON.stringify({
        type: 'response.output_item.done',
        output_index: 0,
        item: {
          type: 'message',
          role: 'assistant',
          id: 'msg_789',
          content: [{ type: 'output_text', text: 'Hello from trace', annotations: [] }],
        },
        databricks_output: {
          trace: {
            info: {
              trace_id: traceId,
            },
          },
        },
      })}\n\n`,
      `data: ${JSON.stringify({
        type: 'responses.completed',
        response: {
          id: 'resp_789',
          status: 'completed',
          usage: {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
          },
        },
      })}\n\n`,
    ]

    const stream = new ReadableStream({
      start(controller) {
        for (const event of events) {
          controller.enqueue(encoder.encode(event))
        }
        controller.close()
      },
    })

    return new Response(stream, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
      },
    })
  }
}

describe('Trace ID in Response', () => {
  it('should include trace_id in streaming response body', async () => {
    const expectedTraceId = 'trace_abc123'

    const mockFetch = createMockFetchWithTrace(expectedTraceId)
    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    })

    // Consume the stream
    const reader = result.stream.getReader()
    while (true) {
      const { done } = await reader.read()
      if (done) break
    }

    // Verify trace_id is in response.body
    const responseBody = (result.response as any)?.body
    expect(responseBody).toBeDefined()
    expect(responseBody?.trace_id).toBe(expectedTraceId)
  })

  it('should include trace_id in non-streaming response body', async () => {
    const expectedTraceId = 'trace_def456'

    const mockFetch = createMockFetchNonStreamingWithTrace(expectedTraceId)
    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    })

    // Verify trace_id is in response.body
    const responseBody = (result.response as any)?.body
    expect(responseBody).toBeDefined()
    expect(responseBody?.trace_id).toBe(expectedTraceId)
  })

  it('should not fail when trace_id is not present', async () => {
    const mockFetch: typeof fetch = async () => {
      const responseBody = {
        id: 'resp_789',
        output: [
          {
            type: 'message',
            role: 'assistant',
            id: 'msg_789',
            content: [
              {
                type: 'output_text',
                text: 'Hello',
                annotations: [],
              },
            ],
          },
        ],
        usage: {
          input_tokens: 5,
          output_tokens: 3,
          total_tokens: 8,
        },
        // No trace_id
      }

      return new Response(JSON.stringify(responseBody), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
        },
      })
    }

    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    })

    // Should still have response.body but without trace_id
    expect(result.response?.body).toBeDefined()
    expect((result.response?.body as any)?.trace_id).toBeUndefined()
  })

  it('should capture trace_id from response.output_item.done.databricks_output.trace.info.trace_id', async () => {
    const expectedTraceId = 'trace_from_databricks_output'

    const mockFetch = createMockFetchWithDatabricksOutputTrace(expectedTraceId)
    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    })

    // Consume the stream
    const reader = result.stream.getReader()
    while (true) {
      const { done } = await reader.read()
      if (done) break
    }

    // Verify trace_id is captured from databricks_output path
    const responseBody = (result.response as any)?.body
    expect(responseBody).toBeDefined()
    expect(responseBody?.trace_id).toBe(expectedTraceId)
  })
})
