import { describe, it, expect } from 'vitest'
import type { LanguageModelV2StreamPart } from '@ai-sdk/provider'
import { DatabricksFmapiLanguageModel } from '../src/fmapi-language-model/fmapi-language-model'
import {
  FMAPI_BASIC_TEXT_OUTPUT,
  FMAPI_WITH_TOOL_CALLS,
  FMAPI_WITH_LEGACY_TAGS,
  FMAPI_WITH_OPENAI_STREAMING_TOOL_CALLS,
  FMAPI_WITH_PARALLEL_STREAMING_TOOL_CALLS,
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

    const streamParts: LanguageModelV2StreamPart[] = []
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

  it('correctly parses tool calls from XML tags', async () => {
    const mockFetch = createMockFetch(FMAPI_WITH_TOOL_CALLS.in)

    const model = new DatabricksFmapiLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'What is the weather?' }] }],
    })

    const streamParts: LanguageModelV2StreamPart[] = []
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

    // Verify the number of parts
    expect(contentParts.length).toBe(FMAPI_WITH_TOOL_CALLS.out.length)

    // Verify each part
    for (let i = 0; i < contentParts.length; i++) {
      expect(contentParts[i]).toMatchObject(FMAPI_WITH_TOOL_CALLS.out[i])
    }

    // Verify tool call has correct structure and providerExecuted flag
    const toolCallPart = contentParts.find((part) => part.type === 'tool-call')
    expect(toolCallPart).toBeDefined()
    expect(toolCallPart).toMatchObject({
      type: 'tool-call',
      toolCallId: 'call_weather_001',
      toolName: 'get_weather',
      providerExecuted: true,
    })

    // Verify tool result
    const toolResultPart = contentParts.find((part) => part.type === 'tool-result')
    expect(toolResultPart).toBeDefined()
    expect(toolResultPart).toMatchObject({
      type: 'tool-result',
      toolCallId: 'call_weather_001',
      toolName: 'databricks-tool-call',
    })
  })

  it('correctly handles legacy UC function call tags', async () => {
    const mockFetch = createMockFetch(FMAPI_WITH_LEGACY_TAGS.in)

    const model = new DatabricksFmapiLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Calculate 2 + 2' }] }],
    })

    const streamParts: LanguageModelV2StreamPart[] = []
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

    // Verify the number of parts
    expect(contentParts.length).toBe(FMAPI_WITH_LEGACY_TAGS.out.length)

    // Verify legacy tags are parsed correctly
    const toolCallPart = contentParts.find((part) => part.type === 'tool-call')
    expect(toolCallPart).toBeDefined()
    expect(toolCallPart).toMatchObject({
      type: 'tool-call',
      toolCallId: 'calc_001',
      toolName: 'calculate',
      providerExecuted: true,
    })

    // Verify legacy result tag
    const toolResultPart = contentParts.find((part) => part.type === 'tool-result')
    expect(toolResultPart).toBeDefined()
    expect(toolResultPart).toMatchObject({
      type: 'tool-result',
      toolCallId: 'calc_001',
      toolName: 'databricks-tool-call',
      result: '4',
    })

    // Verify text parts before and after tool calls
    const textParts = contentParts.filter((part) => part.type === 'text-delta')
    expect(textParts.length).toBe(2)
    expect(textParts[0].delta).toBe("I'll execute that calculation. ")
    expect(textParts[1].delta).toBe('The result is 4.')
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

    const streamParts: LanguageModelV2StreamPart[] = []
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

    const streamParts: LanguageModelV2StreamPart[] = []
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
    })
    expect(toolCalls[1]).toMatchObject({
      toolCallId: 'call_tool_b',
      toolName: 'tool_b',
      input: '{"y":2}',
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

    const streamParts: LanguageModelV2StreamPart[] = []
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
      finishReason: 'tool-calls',
    })
  })
})
