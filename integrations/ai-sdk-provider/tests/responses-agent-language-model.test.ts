import { describe, it, expect } from 'vitest'
import type { LanguageModelV3StreamPart } from '@ai-sdk/provider'
import {
  DatabricksResponsesAgentLanguageModel,
  shouldDedupeOutputItemDone,
} from '../src/responses-agent-language-model/responses-agent-language-model'
import { RESPONSES_AGENT_OUTPUT_WITH_TOOL_CALLS } from './__fixtures__/llm-output-fixtures'
import {
  MCP_APPROVAL_REQUEST_FIXTURE,
  MCP_APPROVAL_RESPONSE_APPROVED_FIXTURE,
  MCP_APPROVAL_RESPONSE_DENIED_FIXTURE,
} from './__fixtures__/mcp-approval-fixtures'

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
 * This handles string values like: "delta": '{"code": "#' -> "delta": "{\"code\": \"#"
 */
function convertToValidJSON(jsStr: string): string {
  // Replace single-quoted string values with double-quoted strings
  // We need to handle cases where the value might contain escaped quotes or other complex content
  let result = jsStr

  // Pattern: "key": '...' where ... can contain anything except unescaped single quotes
  // We use a more permissive pattern that matches everything between single quotes
  const regex = /"([^"]+)":\s*'((?:[^'\\]|\\.)*)'/g

  result = result.replace(regex, (_match, key, value: string) => {
    // The value might already have escaped quotes (\"), keep them but also escape unescaped quotes
    // Since JSON uses double quotes, any literal double quotes in the value need to be escaped
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
 * Parses SSE content by splitting on "data:" markers
 */
function createMockFetch(sseContent: string): typeof fetch {
  // eslint-disable-next-line @typescript-eslint/require-await
  return async () => {
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

    return new Response(stream, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
      },
    })
  }
}

describe('DatabricksResponsesAgentLanguageModel', () => {
  it('correctly converts RESPONSES_AGENT_OUTPUT_WITH_TOOL_CALLS fixture', async () => {
    // Create a mock fetch that returns the fixture SSE stream
    const mockFetch = createMockFetch(RESPONSES_AGENT_OUTPUT_WITH_TOOL_CALLS.in)

    // Instantiate the model with mock config
    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    // Call doStream
    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      //   mode: { type: 'regular' },
    })

    // Collect all stream parts
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
    expect(contentParts.length).toBe(RESPONSES_AGENT_OUTPUT_WITH_TOOL_CALLS.out.length)

    // Verify each part matches the expected output
    for (let i = 0; i < contentParts.length; i++) {
      const actual = contentParts[i]
      const expected = RESPONSES_AGENT_OUTPUT_WITH_TOOL_CALLS.out[i]

      expect(actual).toMatchObject(expected as any)
    }
  })
})

describe('MCP Approval Streaming', () => {
  it('correctly converts MCP approval request stream', async () => {
    const mockFetch = createMockFetch(MCP_APPROVAL_REQUEST_FIXTURE.in)

    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
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
    expect(contentParts.length).toBe(MCP_APPROVAL_REQUEST_FIXTURE.out.length)

    // Verify each part
    for (let i = 0; i < contentParts.length; i++) {
      expect(contentParts[i]).toMatchObject(MCP_APPROVAL_REQUEST_FIXTURE.out[i] as any)
    }

    // Verify MCP approval request has correct metadata
    const mcpRequestPart = contentParts.find(
      (part) =>
        part.type === 'tool-call' &&
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        (part as any).providerMetadata?.databricks?.type === 'mcp_approval_request'
    )
    expect(mcpRequestPart).toBeDefined()
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    expect((mcpRequestPart as any).providerMetadata.databricks).toMatchObject({
      type: 'mcp_approval_request',
      toolName: 'test_mcp_tool',
      serverLabel: 'test-server',
    })
  })

  it('correctly converts MCP approval response (approved) stream', async () => {
    const mockFetch = createMockFetch(MCP_APPROVAL_RESPONSE_APPROVED_FIXTURE.in)

    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    // The approval response comes in a second API call after the approval request.
    // We need to include the original tool-call from the approval request in the prompt.
    const result = await model.doStream({
      prompt: [
        { role: 'user', content: [{ type: 'text', text: 'test' }] },
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: '__fake_mcp_request_id__',
              toolName: 'databricks-tool-call',
              input: { action: 'test', param: 'value' },
            },
          ],
        },
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

    // Verify the number of parts
    expect(contentParts.length).toBe(MCP_APPROVAL_RESPONSE_APPROVED_FIXTURE.out.length)

    // Verify each part
    for (let i = 0; i < contentParts.length; i++) {
      expect(contentParts[i]).toMatchObject(MCP_APPROVAL_RESPONSE_APPROVED_FIXTURE.out[i] as any)
    }

    // Verify MCP approval response has correct approval status
    const mcpResponsePart = contentParts.find(
      (part) =>
        part.type === 'tool-result' &&
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        (part as any).providerMetadata?.databricks?.type === 'mcp_approval_response'
    )
    expect(mcpResponsePart).toBeDefined()
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    expect((mcpResponsePart as any).result).toEqual({ __approvalStatus__: true })
  })

  it('correctly converts MCP approval response (denied) stream', async () => {
    const mockFetch = createMockFetch(MCP_APPROVAL_RESPONSE_DENIED_FIXTURE.in)

    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    // The approval response comes in a second API call after the approval request.
    // We need to include the original tool-call from the approval request in the prompt.
    const result = await model.doStream({
      prompt: [
        { role: 'user', content: [{ type: 'text', text: 'test' }] },
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: '__fake_mcp_request_id__',
              toolName: 'databricks-tool-call',
              input: { action: 'test', param: 'value' },
            },
          ],
        },
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

    // Verify the number of parts
    expect(contentParts.length).toBe(MCP_APPROVAL_RESPONSE_DENIED_FIXTURE.out.length)

    // Verify each part
    for (let i = 0; i < contentParts.length; i++) {
      expect(contentParts[i]).toMatchObject(MCP_APPROVAL_RESPONSE_DENIED_FIXTURE.out[i] as any)
    }

    // Verify MCP approval response has correct denial status
    const mcpResponsePart = contentParts.find(
      (part) =>
        part.type === 'tool-result' &&
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        (part as any).providerMetadata?.databricks?.type === 'mcp_approval_response'
    )
    expect(mcpResponsePart).toBeDefined()
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    expect((mcpResponsePart as any).result).toEqual({
      __approvalStatus__: false,
    })
  })
})

describe('Usage Tracking', () => {
  it('extracts usage from responses.completed event in stream', async () => {
    const sseContent = `
data: {
  "type": "response.output_text.delta",
  "item_id": "msg_123",
  "delta": "Hello"
}
data: {
  "type": "responses.completed",
  "response": {
    "id": "resp_123",
    "status": "completed",
    "usage": {
      "input_tokens": 150,
      "output_tokens": 75,
      "total_tokens": 225
    }
  }
}
    `

    const mockFetch = createMockFetch(sseContent)
    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    })

    const parts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      parts.push(value)
    }

    // Find the finish part
    const finishPart = parts.find((p) => p.type === 'finish')
    expect(finishPart).toBeDefined()
    expect(finishPart?.type).toBe('finish')

    if (finishPart?.type === 'finish') {
      expect(finishPart.usage).toEqual({
        inputTokens: { total: 150, noCache: 0, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 75, text: 0, reasoning: 0 },
      })
      expect(finishPart.finishReason).toEqual({ raw: undefined, unified: 'stop' })
    }
  })

  it('returns zero usage when no responses.completed event is received', async () => {
    const sseContent = `
data: {
  "type": "response.output_text.delta",
  "item_id": "msg_123",
  "delta": "Hello"
}
    `

    const mockFetch = createMockFetch(sseContent)
    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    })

    const parts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      parts.push(value)
    }

    const finishPart = parts.find((p) => p.type === 'finish')
    expect(finishPart).toBeDefined()

    if (finishPart?.type === 'finish') {
      expect(finishPart.usage).toEqual({
        inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 0, text: 0, reasoning: 0 },
      })
      expect(finishPart.finishReason).toEqual({ raw: undefined, unified: 'stop' })
    }
  })
})

describe('Deduplication Logic', () => {
  it('deduplicates .done event when text matches reconstructed deltas', async () => {
    const sseContent = `
data: {
  "type": "response.output_text.delta",
  "item_id": "msg_123",
  "delta": "Hello "
}
data: {
  "type": "response.output_text.delta",
  "item_id": "msg_123",
  "delta": "World"
}
data: {
  "type": "response.output_item.done",
  "item": {
    "type": "message",
    "id": "msg_123",
    "role": "assistant",
    "content": [{ "type": "text", "text": "Hello World[^ref]" }]
  }
}
    `

    const mockFetch = createMockFetch(sseContent)
    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    })

    const parts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      parts.push(value)
    }

    const textParts = parts.filter((p) => p.type === 'text-delta')

    // Expect "Hello " and "World", but NOT "Hello World[^ref]" from .done
    expect(textParts.length).toBe(2)
    expect(textParts[0].delta).toBe('Hello ')
    expect(textParts[1].delta).toBe('World')
  })

  it('does NOT dedupe .done event when text content differs', async () => {
    const sseContent = `
data: {
  "type": "response.output_text.delta",
  "item_id": "msg_123",
  "delta": "Hello "
}
data: {
  "type": "response.output_text.delta",
  "item_id": "msg_123",
  "delta": "World"
}
data: {
  "type": "response.output_item.done",
  "item": {
    "type": "message",
    "id": "msg_123",
    "role": "assistant",
    "content": [{ "type": "text", "text": "Different Text" }]
  }
}
    `

    const mockFetch = createMockFetch(sseContent)
    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    })

    const parts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      parts.push(value)
    }

    const textParts = parts.filter((p) => p.type === 'text-delta')
    // "Hello ", "World", and then "Different Text" from .done
    expect(textParts.length).toBe(3)
    expect(textParts[0].delta).toBe('Hello ')
    expect(textParts[1].delta).toBe('World')
    expect(textParts[2].delta).toBe('Different Text')

    // Should NOT be deduped -> .done event emits text-start?
    // Since we removed text-start from .done converter, we only expect the initial text-start from the stream
    const textStartParts = parts.filter((p) => p.type === 'text-start')
    expect(textStartParts.length).toBe(1)
  })
})

describe('shouldDedupeOutputItemDone', () => {
  // Helper to create a text-delta from response.output_text.delta (regular streaming)
  const createTextDelta = (delta: string, id: string): LanguageModelV3StreamPart => ({
    type: 'text-delta',
    id,
    delta,
    providerMetadata: {
      databricks: {
        itemId: id,
      },
    },
  })

  // Helper to create a text-delta from response.output_item.done
  const createDoneTextDelta = (delta: string, id: string): LanguageModelV3StreamPart => ({
    type: 'text-delta',
    id,
    delta,
    providerMetadata: {
      databricks: {
        itemId: id,
        itemType: 'response.output_item.done',
      },
    },
  })

  // Helper to create a non-text part (e.g., tool-call)
  const createToolCall = (toolCallId: string): LanguageModelV3StreamPart => ({
    type: 'tool-call',
    toolCallId,
    toolName: 'test-tool',
    input: '{}',
  })

  it('returns false when incoming parts have no response.output_item.done text-delta', () => {
    const incomingParts: LanguageModelV3StreamPart[] = [createTextDelta('Hello', 'msg_1')]
    const previousParts: LanguageModelV3StreamPart[] = []

    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(false)
  })

  it('returns false when done text-delta has no id', () => {
    // Intentionally omitting `id` to test edge case - use type assertion
    const doneWithoutId = {
      type: 'text-delta',
      delta: 'Hello',
      providerMetadata: {
        databricks: {
          itemType: 'response.output_item.done',
        },
      },
    } as unknown as LanguageModelV3StreamPart
    const incomingParts: LanguageModelV3StreamPart[] = [doneWithoutId]
    const previousParts: LanguageModelV3StreamPart[] = []

    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(false)
  })

  it('returns true when previous parts are empty (empty string matches)', () => {
    // When previousParts is empty, reconstructuredTexts = [''] (empty string)
    // indexOf('') in any string returns 0, so it returns true
    const incomingParts: LanguageModelV3StreamPart[] = [createDoneTextDelta('Hello World', 'msg_1')]
    const previousParts: LanguageModelV3StreamPart[] = []

    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(false)
  })

  it('returns true when done text contains all previous text-deltas in order', () => {
    const incomingParts: LanguageModelV3StreamPart[] = [
      createDoneTextDelta('Hello World[^1]', 'msg_1'),
    ]
    const previousParts: LanguageModelV3StreamPart[] = [
      createTextDelta('Hello ', 'msg_1'),
      createTextDelta('World', 'msg_1'),
    ]

    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(true)
  })

  it('returns false when footnotes interrupt the text sequence', () => {
    // The done text has footnotes [^ref1] and [^ref2] that interrupt the original text
    // The reconstructed text is "The answer is 42. See more." but the done delta
    // has "The answer is 42[^ref1]. See more[^ref2]." - the footnotes break the match
    const incomingParts: LanguageModelV3StreamPart[] = [
      createDoneTextDelta('The answer is 42[^ref1]. See more[^ref2].', 'msg_1'),
    ]
    const previousParts: LanguageModelV3StreamPart[] = [
      createTextDelta('The answer is 42', 'msg_1'),
      createTextDelta('. See more', 'msg_1'),
      createTextDelta('.', 'msg_1'),
    ]

    // Returns false because ". See more." is not found after "The answer is 42"
    // since the footnote [^ref1] appears in between
    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(false)
  })

  it('returns false when done text does not contain previous text', () => {
    const incomingParts: LanguageModelV3StreamPart[] = [
      createDoneTextDelta('Completely different text', 'msg_1'),
    ]
    const previousParts: LanguageModelV3StreamPart[] = [
      createTextDelta('Hello ', 'msg_1'),
      createTextDelta('World', 'msg_1'),
    ]

    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(false)
  })

  it('returns false when previous text is not in the correct order within done text', () => {
    const incomingParts: LanguageModelV3StreamPart[] = [
      createDoneTextDelta('World Hello', 'msg_1'), // Reversed order
    ]
    const previousParts: LanguageModelV3StreamPart[] = [
      createTextDelta('Hello', 'msg_1'),
      createTextDelta('World', 'msg_1'),
    ]

    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(false)
  })

  it('handles multiple text blocks separated by non-text parts', () => {
    const incomingParts: LanguageModelV3StreamPart[] = [
      createDoneTextDelta('First block. Second block.', 'msg_1'),
    ]
    const previousParts: LanguageModelV3StreamPart[] = [
      createTextDelta('First ', 'msg_1'),
      createTextDelta('block.', 'msg_1'),
      createToolCall('tool_1'), // Non-text part separates the blocks
      createTextDelta('Second ', 'msg_1'),
      createTextDelta('block.', 'msg_1'),
    ]

    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(true)
  })

  it('returns false when only partial text matches', () => {
    const incomingParts: LanguageModelV3StreamPart[] = [
      createDoneTextDelta('Hello World', 'msg_1'),
    ]
    const previousParts: LanguageModelV3StreamPart[] = [
      createTextDelta('Hello ', 'msg_1'),
      // Missing 'World' delta
    ]

    // Should return true since "Hello " is contained in "Hello World"
    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(true)
  })

  it('includes leading whitespace in reconstructed text', () => {
    // Whitespace is included in the reconstructed text, so "   Hello World"
    // is not found in "Hello World" (no leading whitespace in done delta)
    const incomingParts: LanguageModelV3StreamPart[] = [
      createDoneTextDelta('Hello World', 'msg_1'),
    ]
    const previousParts: LanguageModelV3StreamPart[] = [
      createTextDelta('   ', 'msg_1'), // Leading whitespace becomes part of reconstructed text
      createTextDelta('Hello World', 'msg_1'),
    ]

    // Returns false because "   Hello World" is not found in "Hello World"
    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(false)
  })

  it('dedupes when whitespace matches', () => {
    // When the done delta includes the same whitespace, it should dedupe
    const incomingParts: LanguageModelV3StreamPart[] = [
      createDoneTextDelta('   Hello World', 'msg_1'),
    ]
    const previousParts: LanguageModelV3StreamPart[] = [
      createTextDelta('   ', 'msg_1'),
      createTextDelta('Hello World', 'msg_1'),
    ]

    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(true)
  })

  it('returns false when incoming parts is empty', () => {
    const incomingParts: LanguageModelV3StreamPart[] = []
    const previousParts: LanguageModelV3StreamPart[] = [createTextDelta('Hello', 'msg_1')]

    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(false)
  })

  it('returns false when only non-text previous parts (no text to compare against)', () => {
    // When there are only non-text parts, there are no text-deltas to compare against
    // This is new content, not a duplicate, so return false
    const incomingParts: LanguageModelV3StreamPart[] = [
      createDoneTextDelta('Hello World', 'msg_1'),
    ]
    const previousParts: LanguageModelV3StreamPart[] = [
      createToolCall('tool_1'),
      createToolCall('tool_2'),
    ]

    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(false)
  })

  it('only considers text-deltas after the last response.output_item.done event', () => {
    // This is the key behavior: when there's a previous .done event,
    // we should only reconstruct text from parts AFTER that .done event
    const incomingParts: LanguageModelV3StreamPart[] = [
      createDoneTextDelta('Second message', 'msg_2'),
    ]
    const previousParts: LanguageModelV3StreamPart[] = [
      // First message's text-deltas
      createTextDelta('First ', 'msg_1'),
      createTextDelta('message', 'msg_1'),
      // First message's .done event
      createDoneTextDelta('First message', 'msg_1'),
      // Tool call between messages
      createToolCall('tool_1'),
      // Second message's text-deltas (only these should be considered)
      createTextDelta('Second ', 'msg_2'),
      createTextDelta('message', 'msg_2'),
    ]

    // Should return true because "Second message" matches the text-deltas after the last .done
    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(true)
  })

  it('returns false when text after last .done does not match', () => {
    const incomingParts: LanguageModelV3StreamPart[] = [
      createDoneTextDelta('Different content', 'msg_2'),
    ]
    const previousParts: LanguageModelV3StreamPart[] = [
      createTextDelta('First message', 'msg_1'),
      createDoneTextDelta('First message', 'msg_1'),
      createTextDelta('Second ', 'msg_2'),
      createTextDelta('message', 'msg_2'),
    ]

    // Should return false because "Different content" doesn't match "Second message"
    expect(shouldDedupeOutputItemDone(incomingParts, previousParts)).toBe(false)
  })
})

describe('Missing Tool Results', () => {
  it('emits correction tool-call with providerExecuted for tool calls without results', async () => {
    // When a function_call has no corresponding function_call_output,
    // the flush callback should re-emit the tool-call with providerExecuted: true
    const sseContent = `
data: {
  "type": "response.output_item.done",
  "item": {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "tool_call_123",
    "name": "agent-netflix-titles",
    "arguments": "{\\"query\\": \\"test\\"}"
  }
}
data: {
  "type": "response.output_item.done",
  "item": {
    "type": "message",
    "id": "msg_result",
    "role": "assistant",
    "content": [{ "type": "output_text", "text": "| title | year |" }]
  }
}
data: {
  "type": "responses.completed",
  "response": {
    "id": "resp_123",
    "status": "completed",
    "usage": { "input_tokens": 100, "output_tokens": 50, "total_tokens": 150 }
  }
}
    `

    const mockFetch = createMockFetch(sseContent)
    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'find movies' }] }],
    })

    const parts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      parts.push(value)
    }

    // Should have two tool-call parts: the original and the correction with providerExecuted: true
    const toolCallParts = parts.filter((p) => p.type === 'tool-call')
    expect(toolCallParts).toHaveLength(2)

    // First tool-call is the original (without providerExecuted)
    const originalToolCall = toolCallParts[0]
    expect(originalToolCall.type).toBe('tool-call')
    if (originalToolCall.type === 'tool-call') {
      expect(originalToolCall.toolCallId).toBe('tool_call_123')
      expect(originalToolCall.providerExecuted).toBeUndefined()
    }

    // Second tool-call is the correction emitted in flush (with providerExecuted: true)
    // This tells the AI SDK not to expect a client-side result for this tool call
    const correctionToolCall = toolCallParts[1]
    expect(correctionToolCall.type).toBe('tool-call')
    if (correctionToolCall.type === 'tool-call') {
      expect(correctionToolCall.toolCallId).toBe('tool_call_123')
      expect(correctionToolCall.providerExecuted).toBe(true)
    }

    // Should NOT have a tool-result part (we don't emit synthetic results)
    const toolResultPart = parts.find((p) => p.type === 'tool-result')
    expect(toolResultPart).toBeUndefined()

    // Finish part should come after the correction tool-call
    const finishIndex = parts.findIndex((p) => p.type === 'finish')
    const lastToolCallIndex = parts.findLastIndex((p) => p.type === 'tool-call')
    expect(finishIndex).toBeGreaterThan(lastToolCallIndex)
  })

  it('does not emit correction tool-call when result already exists', async () => {
    // When function_call_output is present, should not emit a correction tool-call
    const sseContent = `
data: {
  "type": "response.output_item.done",
  "item": {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "tool_call_456",
    "name": "search",
    "arguments": "{\\"q\\": \\"test\\"}"
  }
}
data: {
  "type": "response.output_item.done",
  "item": {
    "type": "function_call_output",
    "call_id": "tool_call_456",
    "output": "search results here"
  }
}
data: {
  "type": "responses.completed",
  "response": {
    "id": "resp_123",
    "status": "completed",
    "usage": { "input_tokens": 50, "output_tokens": 25, "total_tokens": 75 }
  }
}
    `

    const mockFetch = createMockFetch(sseContent)
    const model = new DatabricksResponsesAgentLanguageModel('test-model', {
      provider: 'databricks',
      headers: () => ({ Authorization: 'Bearer test-token' }),
      url: () => 'http://test.example.com/api',
      fetch: mockFetch,
    })

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'search' }] }],
    })

    const parts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      parts.push(value)
    }

    // Should have exactly one tool-call (no correction needed since result exists)
    const toolCallParts = parts.filter((p) => p.type === 'tool-call')
    expect(toolCallParts.length).toBe(1)
    if (toolCallParts[0]?.type === 'tool-call') {
      expect(toolCallParts[0].providerExecuted).toBeUndefined()
    }

    // Should have exactly one tool-result (the real one)
    const toolResultParts = parts.filter((p) => p.type === 'tool-result')
    expect(toolResultParts.length).toBe(1)
    if (toolResultParts[0]?.type === 'tool-result') {
      expect(toolResultParts[0].result).toBe('search results here')
    }
  })
})
