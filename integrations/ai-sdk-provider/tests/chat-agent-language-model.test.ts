import { describe, it, expect } from 'vitest'
import type { LanguageModelV2StreamPart } from '@ai-sdk/provider'
import { DatabricksChatAgentLanguageModel } from '../src/chat-agent-language-model/chat-agent-language-model'
import {
  CHAT_AGENT_BASIC_TEXT_OUTPUT,
  CHAT_AGENT_WITH_TOOL_CALLS,
  CHAT_AGENT_MULTI_TOOL_CALLS,
} from './__fixtures__/chat-agent-fixtures'

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

describe('DatabricksChatAgentLanguageModel', () => {
  it('correctly converts basic text output', async () => {
    const mockFetch = createMockFetch(CHAT_AGENT_BASIC_TEXT_OUTPUT.in)

    const model = new DatabricksChatAgentLanguageModel('test-model', {
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
    expect(contentParts.length).toBe(CHAT_AGENT_BASIC_TEXT_OUTPUT.out.length)

    // Verify each part matches the expected output
    for (let i = 0; i < contentParts.length; i++) {
      expect(contentParts[i]).toMatchObject(CHAT_AGENT_BASIC_TEXT_OUTPUT.out[i])
    }
  })

  it('correctly parses tool calls from structured JSON', async () => {
    const mockFetch = createMockFetch(CHAT_AGENT_WITH_TOOL_CALLS.in)

    const model = new DatabricksChatAgentLanguageModel('test-model', {
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
    expect(contentParts.length).toBe(CHAT_AGENT_WITH_TOOL_CALLS.out.length)

    // Verify each part
    for (let i = 0; i < contentParts.length; i++) {
      expect(contentParts[i]).toMatchObject(CHAT_AGENT_WITH_TOOL_CALLS.out[i])
    }

    // Verify tool call has correct structure
    const toolCallPart = contentParts.find((part) => part.type === 'tool-call')
    expect(toolCallPart).toBeDefined()
    expect(toolCallPart).toMatchObject({
      type: 'tool-call',
      toolCallId: 'call_abc123',
      toolName: 'get_weather',
      input: '{"location": "San Francisco", "unit": "celsius"}',
    })

    // Verify tool result
    const toolResultPart = contentParts.find((part) => part.type === 'tool-result')
    expect(toolResultPart).toBeDefined()
    expect(toolResultPart).toMatchObject({
      type: 'tool-result',
      toolCallId: 'call_abc123',
      toolName: 'databricks-tool-call',
    })
  })

  it('correctly handles multiple tool calls', async () => {
    const mockFetch = createMockFetch(CHAT_AGENT_MULTI_TOOL_CALLS.in)

    const model = new DatabricksChatAgentLanguageModel('test-model', {
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
    expect(contentParts.length).toBe(CHAT_AGENT_MULTI_TOOL_CALLS.out.length)

    // Verify multiple tool calls are present
    const toolCalls = contentParts.filter((part) => part.type === 'tool-call')
    expect(toolCalls.length).toBe(2)

    // Verify first tool call
    expect(toolCalls[0]).toMatchObject({
      type: 'tool-call',
      toolCallId: 'call_weather_123',
      toolName: 'get_weather',
    })

    // Verify second tool call
    expect(toolCalls[1]).toMatchObject({
      type: 'tool-call',
      toolCallId: 'call_time_456',
      toolName: 'get_current_time',
    })
  })
})
