import { describe, it, expect } from 'vitest'
import { DatabricksResponsesAgentLanguageModel } from '../src/responses-agent-language-model/responses-agent-language-model'

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
