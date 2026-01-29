import { describe, it, expect } from 'vitest'
import type { LanguageModelV3StreamPart } from '@ai-sdk/provider'
import { DatabricksFmapiLanguageModel } from '../src/fmapi-language-model/fmapi-language-model'
import {
  FMAPI_BASIC_TEXT_OUTPUT,
  FMAPI_WITH_OPENAI_STREAMING_TOOL_CALLS,
  FMAPI_WITH_PARALLEL_STREAMING_TOOL_CALLS,
  FMAPI_RESPONSE_WITH_TOOL_CALLS,
  FMAPI_RESPONSE_WITH_PARALLEL_TOOL_CALLS,
} from './__fixtures__/fmapi-fixtures'

/**
 * Removes trailing commas from JSON strings (JavaScript JSON.parse doesn't allow them)
 */
function removeTrailingCommas(jsonStr: string): string {
  // Remove commas before closing braces or brackets (repeat until no more matches)
  let result = jsonStr
  let prev: string
  do {
    prev = result
    result = result.replace(/,(\s*[}\]])/g, '$1')
  } while (result !== prev)
  return result
}

/**
 * Converts JavaScript object notation (with single quotes) to valid JSON (with double quotes)
 */
function convertToValidJSON(jsStr: string): string {
  let result = jsStr

  // Pattern: "key": '...' where ... can contain anything except unescaped single quotes
  const regex = /"([^"]+)":\s*'((?:[^'\\]|\\.)*)'/g

  result = result.replace(regex, (_match, key, value: string) => {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
    const escapedValue = value
      .replace(/\\"/g, '###ESCAPED_QUOTE###') // Temporarily protect already-escaped quotes
      .replace(/"/g, '\\"') // Escape unescaped quotes
      .replace(/###ESCAPED_QUOTE###/g, '\\"') // Restore escaped quotes
    return `"${key}": "${escapedValue}"`
  })

  return result
}

/**
 * Creates a mock fetch function that returns the fixture SSE stream
 */
function createMockFetch(sseContent: string): typeof fetch {
  return () => {
    const encoder = new TextEncoder()

    // Split by "data:" and filter out empty strings
    const rawEvents = sseContent.split(/\ndata:/).filter((s) => s.trim())

    const events: string[] = []
    for (let rawEvent of rawEvents) {
      // Remove leading "data:" if present (first event might have it)
      rawEvent = rawEvent.replace(/^data:\s*/, '').trim()

      try {
        // Convert JS notation to valid JSON, then remove trailing commas
        let cleanedJson = convertToValidJSON(rawEvent)
        cleanedJson = removeTrailingCommas(cleanedJson)
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        const parsed = JSON.parse(cleanedJson)
        events.push(`data: ${JSON.stringify(parsed)}`)
      } catch (e) {
        // If it's not JSON (like "[DONE]"), keep as is
        // eslint-disable-next-line no-console
        console.log('Failed to parse JSON:', e, rawEvent.substring(0, 100))
        events.push(`data: ${rawEvent}`)
      }
    }

    const stream = new ReadableStream({
      start(controller) {
        for (const event of events) {
          // SSE format: each event followed by double newline
          controller.enqueue(encoder.encode(`${event}\n\n`))
        }
        controller.close()
      },
    })

    return Promise.resolve(
      new Response(stream, {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
        },
      })
    )
  }
}

describe('DatabricksFmapiLanguageModel', () => {
  it('correctly converts basic text output', async () => {
    const mockFetch = createMockFetch(FMAPI_BASIC_TEXT_OUTPUT.in)

    const model = new DatabricksFmapiLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    })

    const streamParts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()

    try {
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        streamParts.push(value)
      }
    } catch (error) {
      console.error('Error reading stream:', error)
      throw error
    }

    // Filter out stream-start and finish parts for comparison
    const contentParts = streamParts.filter(
      (part) => part.type !== 'stream-start' && part.type !== 'finish' && part.type !== 'raw'
    )

    // Verify the number of parts matches expected
    expect(contentParts.length).toBe(FMAPI_BASIC_TEXT_OUTPUT.out.length)

    // Verify each part matches the expected output
    for (let i = 0; i < contentParts.length; i++) {
      expect(contentParts[i]).toMatchObject(FMAPI_BASIC_TEXT_OUTPUT.out[i])
    }
  })

  it('correctly handles OpenAI-format streaming tool calls with ID tracking', async () => {
    const mockFetch = createMockFetch(FMAPI_WITH_OPENAI_STREAMING_TOOL_CALLS.in)

    const model = new DatabricksFmapiLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'What is the weather in San Francisco?' }] }],
    })

    const streamParts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      streamParts.push(value)
    }

    const contentParts = streamParts.filter(
      (part) => part.type !== 'stream-start' && part.type !== 'finish' && part.type !== 'raw'
    )

    // Verify tool-input-start is emitted
    const toolInputStart = contentParts.find((part) => part.type === 'tool-input-start')
    expect(toolInputStart).toBeDefined()
    expect(toolInputStart).toMatchObject({
      type: 'tool-input-start',
      id: 'call_abc123',
      toolName: 'get_weather',
    })

    // Verify tool-input-delta events are emitted
    const toolInputDeltas = contentParts.filter((part) => part.type === 'tool-input-delta')
    expect(toolInputDeltas.length).toBe(2)
    expect(toolInputDeltas[0]).toMatchObject({
      type: 'tool-input-delta',
      id: 'call_abc123',
      delta: '{"location":',
    })
    expect(toolInputDeltas[1]).toMatchObject({
      type: 'tool-input-delta',
      id: 'call_abc123',
      delta: '"San Francisco"}',
    })

    // Verify tool-input-end is emitted
    const toolInputEnd = contentParts.find((part) => part.type === 'tool-input-end')
    expect(toolInputEnd).toBeDefined()
    expect(toolInputEnd).toMatchObject({
      type: 'tool-input-end',
      id: 'call_abc123',
    })

    // Verify complete tool-call is emitted (from flush handler)
    const toolCallPart = contentParts.find((part) => part.type === 'tool-call')
    expect(toolCallPart).toBeDefined()
    expect(toolCallPart).toMatchObject({
      type: 'tool-call',
      toolCallId: 'call_abc123',
      toolName: 'get_weather',
      input: '{"location":"San Francisco"}',
      dynamic: true,
    })
  })

  it('correctly handles parallel OpenAI-format streaming tool calls', async () => {
    const mockFetch = createMockFetch(FMAPI_WITH_PARALLEL_STREAMING_TOOL_CALLS.in)

    const model = new DatabricksFmapiLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [
        { role: 'user', content: [{ type: 'text', text: 'Call both tool_a and tool_b' }] },
      ],
    })

    const streamParts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      streamParts.push(value)
    }

    const contentParts = streamParts.filter(
      (part) => part.type !== 'stream-start' && part.type !== 'finish' && part.type !== 'raw'
    )

    // Verify both tool-input-start events
    const toolInputStarts = contentParts.filter((part) => part.type === 'tool-input-start')
    expect(toolInputStarts.length).toBe(2)
    expect(toolInputStarts[0]).toMatchObject({ id: 'call_tool_a', toolName: 'tool_a' })
    expect(toolInputStarts[1]).toMatchObject({ id: 'call_tool_b', toolName: 'tool_b' })

    // Verify deltas use tracked IDs (not fallback IDs)
    const toolInputDeltas = contentParts.filter((part) => part.type === 'tool-input-delta')
    expect(toolInputDeltas.length).toBe(2)
    // Tool A delta should use tracked ID
    expect(toolInputDeltas[0]).toMatchObject({ id: 'call_tool_a', delta: '{"x":1}' })
    // Tool B delta should use tracked ID (from index 1)
    expect(toolInputDeltas[1]).toMatchObject({ id: 'call_tool_b', delta: '{"y":2}' })

    // Verify both tool-input-end events
    const toolInputEnds = contentParts.filter((part) => part.type === 'tool-input-end')
    expect(toolInputEnds.length).toBe(2)

    // Verify both complete tool-call events
    const toolCalls = contentParts.filter((part) => part.type === 'tool-call')
    expect(toolCalls.length).toBe(2)
    expect(toolCalls[0]).toMatchObject({
      toolCallId: 'call_tool_a',
      toolName: 'tool_a',
      input: '{"x":1}',
      dynamic: true,
    })
    expect(toolCalls[1]).toMatchObject({
      toolCallId: 'call_tool_b',
      toolName: 'tool_b',
      input: '{"y":2}',
      dynamic: true,
    })
  })

  it('correctly reports tool-calls finish reason for streaming tool calls', async () => {
    const mockFetch = createMockFetch(FMAPI_WITH_OPENAI_STREAMING_TOOL_CALLS.in)

    const model = new DatabricksFmapiLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Call a tool' }] }],
    })

    const streamParts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      streamParts.push(value)
    }

    // Verify finish event has correct reason
    const finishPart = streamParts.find((part) => part.type === 'finish')
    expect(finishPart).toBeDefined()
    expect(finishPart).toMatchObject({
      type: 'finish',
      finishReason: { raw: 'tool_calls', unified: 'tool-calls' },
    })
  })

  describe('doGenerate (non-streaming)', () => {
    /**
     * Creates a mock fetch function that returns a non-streaming JSON response
     */
    function createMockJsonFetch(responseData: unknown): typeof fetch {
      return () => {
        return Promise.resolve(
          new Response(JSON.stringify(responseData), {
            status: 200,
            headers: {
              'Content-Type': 'application/json',
            },
          })
        )
      }
    }

    it('correctly handles tool_calls in non-streaming response', async () => {
      const mockFetch = createMockJsonFetch(FMAPI_RESPONSE_WITH_TOOL_CALLS)

      const model = new DatabricksFmapiLanguageModel('test-model', {
        provider: 'databricks',
        headers: () => ({ Authorization: 'Bearer test-token' }),
        url: () => 'http://test.example.com/api',
        fetch: mockFetch,
      })

      const result = await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'What is the weather in New York?' }] }],
      })

      // Verify finish reason
      expect(result.finishReason).toEqual({ raw: 'tool_calls', unified: 'tool-calls' })

      // Verify tool calls are returned
      expect(result.content).toHaveLength(1)
      expect(result.content[0]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call_weather_123',
        toolName: 'get_weather',
        input: '{"location":"New York","unit":"celsius"}',
        dynamic: true,
      })

      // Verify usage
      expect(result.usage).toEqual({
        inputTokens: { total: 50, noCache: 0, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 25, text: 0, reasoning: 0 },
      })
    })

    it('correctly handles multiple parallel tool_calls in non-streaming response', async () => {
      const mockFetch = createMockJsonFetch(FMAPI_RESPONSE_WITH_PARALLEL_TOOL_CALLS)

      const model = new DatabricksFmapiLanguageModel('test-model', {
        provider: 'databricks',
        headers: () => ({ Authorization: 'Bearer test-token' }),
        url: () => 'http://test.example.com/api',
        fetch: mockFetch,
      })

      const result = await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'What is the weather and time in Paris?' }] }],
      })

      // Verify finish reason
      expect(result.finishReason).toEqual({ raw: 'tool_calls', unified: 'tool-calls' })

      // Verify both tool calls are returned
      expect(result.content).toHaveLength(2)
      expect(result.content[0]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call_tool_1',
        toolName: 'get_weather',
        input: '{"location":"Paris"}',
        dynamic: true,
      })
      expect(result.content[1]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call_tool_2',
        toolName: 'get_time',
        input: '{"timezone":"Europe/Paris"}',
        dynamic: true,
      })

      // Verify usage
      expect(result.usage).toEqual({
        inputTokens: { total: 60, noCache: 0, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 40, text: 0, reasoning: 0 },
      })
    })
  })
})
