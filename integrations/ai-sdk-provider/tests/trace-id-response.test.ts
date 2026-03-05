import { describe, it, expect } from 'vitest'
import type { LanguageModelV3StreamPart } from '@ai-sdk/provider'
import { DatabricksResponsesAgentLanguageModel } from '../src/responses-agent-language-model/responses-agent-language-model'

/**
 * Creates a mock fetch that returns a streaming SSE response from a plain
 * array of JSON-serializable event payloads. Each payload is emitted as a
 * `data: <json>\n\n` SSE event.
 */
function makeSseFetch(events: object[]): typeof fetch {
  // eslint-disable-next-line @typescript-eslint/require-await
  return async () => {
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      start(controller) {
        for (const event of events) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
        }
        controller.close()
      },
    })
    return new Response(stream, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    })
  }
}

/**
 * Regression test for the looseResponseAgentChunkSchema fix.
 *
 * The MLflow AgentServer (used via API_PROXY / local agent) emits a standalone
 * `{"trace_id":"..."}` SSE event that has NO `type` field. Before the fix,
 * `looseResponseAgentChunkSchema` used `z.object({ type: z.string() }).loose()`
 * as its fallback, which rejected this event and caused duplicate streaming or
 * errors. The fix changed the fallback to `z.object({}).loose()` so any
 * unknown chunk is silently dropped rather than crashing the stream.
 */
describe('trace_id SSE event (no type field) — regression for looseResponseAgentChunkSchema', () => {
  it('does not emit error parts when stream contains a {"trace_id":"..."} event', async () => {
    const mockFetch = makeSseFetch([
      // Normal streaming text delta
      { type: 'response.output_text.delta', item_id: 'item_001', delta: 'Hello' },
      // Item-done event carrying the completed message
      {
        type: 'response.output_item.done',
        output_index: 0,
        item: {
          type: 'message',
          role: 'assistant',
          id: 'item_001',
          content: [{ type: 'output_text', text: 'Hello', annotations: [] }],
        },
      },
      // Completion event with usage
      {
        type: 'responses.completed',
        response: {
          id: 'resp_001',
          usage: { input_tokens: 5, output_tokens: 1, total_tokens: 6 },
        },
      },
      // Standalone trace_id event emitted by MLflow AgentServer — no "type" field.
      // This is the event that triggered the bug before the fix.
      { trace_id: 'tr-abc123' },
    ])

    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
    })

    const parts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      parts.push(value)
    }

    // No error parts should be emitted
    const errorParts = parts.filter((p) => p.type === 'error')
    expect(errorParts).toHaveLength(0)

    // Text delta should appear exactly once (not duplicated)
    const textDeltas = parts.filter((p) => p.type === 'text-delta')
    expect(textDeltas).toHaveLength(1)
    expect((textDeltas[0] as { type: 'text-delta'; delta: string }).delta).toBe('Hello')
  })
})

describe('doGenerate response body pass-through', () => {
  it('passes the raw response body through to the caller', async () => {
    const mockFetch: typeof fetch = async () => {
      return new Response(
        JSON.stringify({
          id: 'resp_123',
          model: 'test-model',
          output: [
            {
              type: 'message',
              role: 'assistant',
              id: 'msg_123',
              content: [{ type: 'output_text', text: 'Hello', annotations: [] }],
            },
          ],
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
          trace_id: 'trace_abc',
          databricks_output: { trace: { info: { trace_id: 'trace_abc' } } },
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
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

    // The raw parsed response is passed through as-is; callers can read
    // body.trace_id (MLflow) or body.databricks_output.trace.info.trace_id
    // (Databricks endpoint) directly.
    const body = result.response?.body as any
    expect(body).toBeDefined()
    expect(body.trace_id).toBe('trace_abc')
    expect(body.databricks_output?.trace?.info?.trace_id).toBe('trace_abc')
  })
})
