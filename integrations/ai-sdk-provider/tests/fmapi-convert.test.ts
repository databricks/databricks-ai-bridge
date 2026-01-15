import { describe, it, expect } from 'vitest'
import type { LanguageModelV2Message } from '@ai-sdk/provider'
import { convertPromptToFmapiMessages } from '../src/fmapi-language-model/fmapi-convert-to-input'
import {
  convertFmapiChunkToMessagePart,
  convertFmapiResponseToMessagePart,
} from '../src/fmapi-language-model/fmapi-convert-to-message-parts'
import type { FmapiChunk, FmapiResponse } from '../src/fmapi-language-model/fmapi-schema'
import { DATABRICKS_TOOL_CALL_ID } from '../src/tools'

// ============================================================================
// Tests for convertPromptToFmapiMessages (fmapi-convert-to-input.ts)
// ============================================================================

describe('convertPromptToFmapiMessages', () => {
  describe('system messages', () => {
    it('should convert system message to user role with text content', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'system',
          content: 'You are a helpful assistant.',
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].role).toBe('user')
      expect(result.messages[0].content).toEqual([
        { type: 'text', text: 'You are a helpful assistant.' },
      ])
    })

    it('should handle empty system message', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'system',
          content: '',
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].role).toBe('user')
      expect(result.messages[0].content).toEqual([{ type: 'text', text: '' }])
    })
  })

  describe('user messages', () => {
    it('should convert user message with text parts', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Hello, world!' }],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].role).toBe('user')
      expect(result.messages[0].content).toEqual([{ type: 'text', text: 'Hello, world!' }])
    })

    it('should convert user message with multiple text parts', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'First part' },
            { type: 'text', text: 'Second part' },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].content).toEqual([
        { type: 'text', text: 'First part' },
        { type: 'text', text: 'Second part' },
      ])
    })

    it('should convert user message with file/image part using URL', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'user',
          content: [
            {
              type: 'file',
              mediaType: 'image/png',
              data: new URL('https://example.com/image.png'),
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].content).toEqual([
        { type: 'image', image_url: 'https://example.com/image.png' },
      ])
    })

    it('should convert user message with file/image part using string URL', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'user',
          content: [
            {
              type: 'file',
              mediaType: 'image/jpeg',
              data: 'https://example.com/photo.jpg',
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].content).toEqual([
        { type: 'image', image_url: 'https://example.com/photo.jpg' },
      ])
    })

    it('should convert user message with file/image part using http URL', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'user',
          content: [
            {
              type: 'file',
              mediaType: 'image/gif',
              data: 'http://example.com/animation.gif',
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].content).toEqual([
        { type: 'image', image_url: 'http://example.com/animation.gif' },
      ])
    })

    it('should ignore file part with base64 data (non-URL)', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'user',
          content: [
            {
              type: 'file',
              mediaType: 'image/png',
              data: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA',
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      // Base64 data doesn't start with http:// or https://, so it should be skipped
      expect(result.messages[0].content).toBe('')
    })

    it('should ignore file part with Uint8Array data', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'user',
          content: [
            {
              type: 'file',
              mediaType: 'image/png',
              data: new Uint8Array([137, 80, 78, 71]),
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].content).toBe('')
    })

    it('should ignore non-image file types', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'user',
          content: [
            {
              type: 'file',
              mediaType: 'application/pdf',
              data: new URL('https://example.com/document.pdf'),
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].content).toBe('')
    })

    it('should handle empty user content', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'user',
          content: [],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].content).toBe('')
    })

    it('should handle mixed text and image content', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'What is in this image?' },
            {
              type: 'file',
              mediaType: 'image/png',
              data: new URL('https://example.com/image.png'),
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].content).toEqual([
        { type: 'text', text: 'What is in this image?' },
        { type: 'image', image_url: 'https://example.com/image.png' },
      ])
    })
  })

  describe('assistant messages', () => {
    it('should convert assistant message with text', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'assistant',
          content: [{ type: 'text', text: 'Hello! How can I help you today?' }],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].role).toBe('assistant')
      expect(result.messages[0].content).toEqual([
        { type: 'text', text: 'Hello! How can I help you today?' },
      ])
    })

    it('should convert assistant message with reasoning', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'assistant',
          content: [
            {
              type: 'reasoning',
              text: 'Let me think about this step by step...',
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].role).toBe('assistant')
      expect(result.messages[0].content).toEqual([
        {
          type: 'reasoning',
          summary: [
            {
              type: 'summary_text',
              text: 'Let me think about this step by step...',
            },
          ],
        },
      ])
    })

    it('should convert assistant message with tool-call (serialized as tagged text)', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call-123',
              toolName: 'get_weather',
              input: { location: 'San Francisco' },
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].role).toBe('assistant')
      expect(result.messages[0].content).toHaveLength(1)

      const content = result.messages[0].content as Array<{ type: string; text?: string }>
      expect(content[0].type).toBe('text')
      expect(content[0].text).toContain('<tool_call>')
      expect(content[0].text).toContain('</tool_call>')
      expect(content[0].text).toContain('"id":"call-123"')
      expect(content[0].text).toContain('"name":"get_weather"')
      expect(content[0].text).toContain('"arguments"')
    })

    it('should convert assistant message with tool-result (serialized as tagged text)', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-123',
              toolName: 'get_weather',
              output: {
                type: 'json',
                value: { temperature: 72, unit: 'F' },
              },
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].role).toBe('assistant')
      expect(result.messages[0].content).toHaveLength(1)

      const content = result.messages[0].content as Array<{ type: string; text?: string }>
      expect(content[0].type).toBe('text')
      expect(content[0].text).toContain('<tool_call_result>')
      expect(content[0].text).toContain('</tool_call_result>')
      expect(content[0].text).toContain('"id":"call-123"')
    })

    it('should convert assistant message with file/image using URL', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'assistant',
          content: [
            {
              type: 'file',
              mediaType: 'image/png',
              data: new URL('https://example.com/generated.png'),
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].content).toEqual([
        { type: 'image', image_url: 'https://example.com/generated.png' },
      ])
    })

    it('should handle empty assistant content', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'assistant',
          content: [],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].content).toBe('')
    })

    it('should convert assistant message with mixed content', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'Let me help you with that.' },
            {
              type: 'reasoning',
              text: 'I need to call the weather API.',
            },
            {
              type: 'tool-call',
              toolCallId: 'call-456',
              toolName: 'weather_api',
              input: { city: 'NYC' },
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      const content = result.messages[0].content as Array<{
        type: string
        text?: string
        summary?: unknown
      }>
      expect(content).toHaveLength(3)
      expect(content[0].type).toBe('text')
      expect(content[1].type).toBe('reasoning')
      expect(content[2].type).toBe('text') // tool-call is serialized as text
      expect(content[2].text).toContain('<tool_call>')
    })
  })

  describe('tool messages', () => {
    it('should convert tool message with tool-result parts', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-789',
              toolName: 'weather_api',
              output: {
                type: 'text',
                value: 'The weather is sunny with a high of 75F.',
              },
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].role).toBe('tool')
      expect(result.messages[0].content).toHaveLength(1)

      const content = result.messages[0].content as Array<{ type: string; text?: string }>
      expect(content[0].type).toBe('text')
      expect(content[0].text).toContain('<tool_call_result>')
      expect(content[0].text).toContain('</tool_call_result>')
      expect(content[0].text).toContain('"id":"call-789"')
    })

    it('should convert tool message with multiple tool-result parts', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-1',
              toolName: 'test_tool',
              output: {
                type: 'json',
                value: { result: 'first' },
              },
            },
            {
              type: 'tool-result',
              toolCallId: 'call-2',
              toolName: 'test_tool',
              output: {
                type: 'json',
                value: { result: 'second' },
              },
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      const content = result.messages[0].content as Array<{ type: string; text?: string }>
      expect(content).toHaveLength(2)
      expect(content[0].text).toContain('"id":"call-1"')
      expect(content[1].text).toContain('"id":"call-2"')
    })

    it('should handle tool-result with error-text output', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-error',
              toolName: 'test_tool',
              output: {
                type: 'error-text',
                value: 'API rate limit exceeded',
              },
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      const content = result.messages[0].content as Array<{ type: string; text?: string }>
      expect(content[0].text).toContain('API rate limit exceeded')
    })

    it('should handle tool-result with error-json output', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-error-json',
              toolName: 'test_tool',
              output: {
                type: 'error-json',
                value: { error: 'Not found', code: 404 },
              },
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      const content = result.messages[0].content as Array<{ type: string; text?: string }>
      expect(content[0].text).toContain('"error":"Not found"')
    })

    it('should handle tool-result with content output', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-content',
              toolName: 'test_tool',
              output: {
                type: 'content',
                value: [{ type: 'text', text: 'Complex content' }],
              },
            },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      const content = result.messages[0].content as Array<{ type: string; text?: string }>
      expect(content[0].text).toContain('<tool_call_result>')
    })

    it('should handle empty tool message content', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'tool',
          content: [],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].content).toBe('')
    })
  })

  describe('multiple messages', () => {
    it('should convert a full conversation', () => {
      const prompt: LanguageModelV2Message[] = [
        {
          role: 'system',
          content: 'You are a helpful weather assistant.',
        },
        {
          role: 'user',
          content: [{ type: 'text', text: "What's the weather in Paris?" }],
        },
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'weather-call',
              toolName: 'get_weather',
              input: { city: 'Paris' },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'weather-call',
              toolName: 'get_weather',
              output: {
                type: 'json',
                value: { temp: 18, condition: 'cloudy' },
              },
            },
          ],
        },
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'The weather in Paris is currently 18C and cloudy.' },
          ],
        },
      ]

      const result = convertPromptToFmapiMessages(prompt)

      expect(result.messages).toHaveLength(5)
      expect(result.messages[0].role).toBe('user') // system converted to user
      expect(result.messages[1].role).toBe('user')
      expect(result.messages[2].role).toBe('assistant')
      expect(result.messages[3].role).toBe('tool')
      expect(result.messages[4].role).toBe('assistant')
    })
  })
})

// ============================================================================
// Tests for convertFmapiChunkToMessagePart (fmapi-convert-to-message-parts.ts)
// ============================================================================

describe('convertFmapiChunkToMessagePart', () => {
  const createChunk = (content: string | FmapiChunk['choices'][0]['delta']['content']): FmapiChunk => ({
    id: 'chunk-123',
    created: Date.now(),
    model: 'test-model',
    object: 'chat.completion.chunk',
    choices: [
      {
        index: 0,
        delta: {
          role: 'assistant',
          content,
        },
      },
    ],
  })

  const createToolCallChunk = (
    toolCalls: FmapiChunk['choices'][0]['delta']['tool_calls'],
    content?: string | null
  ): FmapiChunk => ({
    id: 'chunk-tool',
    created: Date.now(),
    model: 'test-model',
    object: 'chat.completion.chunk',
    choices: [
      {
        index: 0,
        delta: {
          role: 'assistant',
          content: content ?? null,
          tool_calls: toolCalls,
        },
      },
    ],
  })

  describe('string content', () => {
    it('should convert plain text to text-delta', () => {
      const chunk = createChunk('Hello, world!')

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(1)
      expect(result[0]).toEqual({
        type: 'text-delta',
        id: 'chunk-123',
        delta: 'Hello, world!',
      })
    })

    it('should convert string content with tool call tags to tool-call', () => {
      const toolCallContent = '<tool_call>{"id":"call-1","name":"get_weather","arguments":{"city":"NYC"}}</tool_call>'
      const chunk = createChunk(toolCallContent)

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(1)
      expect(result[0].type).toBe('tool-call')
      expect(result[0]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call-1',
        toolName: 'get_weather',
        providerExecuted: true,
      })
    })

    it('should convert string content with legacy tool call tags', () => {
      const toolCallContent = '<uc_function_call>{"id":"call-legacy","name":"old_tool","arguments":{}}</uc_function_call>'
      const chunk = createChunk(toolCallContent)

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(1)
      expect(result[0].type).toBe('tool-call')
      expect(result[0]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call-legacy',
        toolName: 'old_tool',
      })
    })

    it('should convert string content with tool result tags to tool-result', () => {
      const toolResultContent = '<tool_call_result>{"id":"call-1","content":"Sunny, 75F"}</tool_call_result>'
      const chunk = createChunk(toolResultContent)

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(1)
      expect(result[0].type).toBe('tool-result')
      expect(result[0]).toMatchObject({
        type: 'tool-result',
        toolCallId: 'call-1',
        toolName: DATABRICKS_TOOL_CALL_ID,
        result: 'Sunny, 75F',
      })
    })

    it('should convert string content with legacy tool result tags', () => {
      const toolResultContent = '<uc_function_result>{"id":"call-legacy","content":{"data":"result"}}</uc_function_result>'
      const chunk = createChunk(toolResultContent)

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(1)
      expect(result[0].type).toBe('tool-result')
      expect(result[0]).toMatchObject({
        type: 'tool-result',
        toolCallId: 'call-legacy',
        result: { data: 'result' },
      })
    })

    it('should handle mixed content with text and tool tags', () => {
      const mixedContent = 'Here is the result: <tool_call>{"id":"call-mix","name":"test","arguments":{}}</tool_call> and more text'
      const chunk = createChunk(mixedContent)

      const result = convertFmapiChunkToMessagePart(chunk)

      // Should split into text, tool-call, and text
      expect(result.length).toBeGreaterThanOrEqual(2)

      const textDeltas = result.filter((p) => p.type === 'text-delta')
      const toolCalls = result.filter((p) => p.type === 'tool-call')

      expect(textDeltas.length).toBeGreaterThan(0)
      expect(toolCalls).toHaveLength(1)
    })

    it('should handle empty string content', () => {
      const chunk = createChunk('')

      const result = convertFmapiChunkToMessagePart(chunk)

      // Empty string should be filtered out
      expect(result).toHaveLength(0)
    })

    it('should handle tool call with string arguments', () => {
      const toolCallContent = '<tool_call>{"id":"call-str","name":"echo","arguments":"hello world"}</tool_call>'
      const chunk = createChunk(toolCallContent)

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(1)
      expect(result[0].type).toBe('tool-call')
      const toolCall = result[0] as { type: 'tool-call'; input: string }
      expect(toolCall.input).toBe('hello world')
    })

    it('should handle tool call with object arguments', () => {
      const toolCallContent = '<tool_call>{"id":"call-obj","name":"complex","arguments":{"nested":{"key":"value"}}}</tool_call>'
      const chunk = createChunk(toolCallContent)

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(1)
      expect(result[0].type).toBe('tool-call')
      const toolCall = result[0] as { type: 'tool-call'; input: string }
      expect(toolCall.input).toContain('nested')
    })
  })

  describe('array content', () => {
    it('should convert array content with text items', () => {
      const chunk = createChunk([
        { type: 'text' as const, text: 'Hello from array!' },
      ])

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(1)
      expect(result[0]).toEqual({
        type: 'text-delta',
        id: 'chunk-123',
        delta: 'Hello from array!',
      })
    })

    it('should convert array content with multiple text items', () => {
      const chunk = createChunk([
        { type: 'text' as const, text: 'First ' },
        { type: 'text' as const, text: 'Second' },
      ])

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(2)
      expect(result[0]).toMatchObject({ type: 'text-delta', delta: 'First ' })
      expect(result[1]).toMatchObject({ type: 'text-delta', delta: 'Second' })
    })

    it('should convert array content with reasoning items', () => {
      const chunk = createChunk([
        {
          type: 'reasoning' as const,
          summary: [
            { type: 'summary_text' as const, text: 'Thinking step 1...' },
          ],
        },
      ])

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(1)
      expect(result[0]).toEqual({
        type: 'reasoning-delta',
        id: 'chunk-123',
        delta: 'Thinking step 1...',
      })
    })

    it('should convert array content with multiple reasoning summaries', () => {
      const chunk = createChunk([
        {
          type: 'reasoning' as const,
          summary: [
            { type: 'summary_text' as const, text: 'Step 1' },
            { type: 'summary_text' as const, text: 'Step 2' },
          ],
        },
      ])

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(2)
      expect(result[0]).toMatchObject({ type: 'reasoning-delta', delta: 'Step 1' })
      expect(result[1]).toMatchObject({ type: 'reasoning-delta', delta: 'Step 2' })
    })

    it('should ignore encrypted reasoning summaries', () => {
      const chunk = createChunk([
        {
          type: 'reasoning' as const,
          summary: [
            { type: 'summary_encrypted_text' as const, data: 'encrypted-data' },
          ],
        },
      ])

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(0)
    })

    it('should ignore image items in array content', () => {
      const chunk = createChunk([
        { type: 'image' as const, image_url: 'https://example.com/image.png' },
      ])

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(0)
    })

    it('should handle mixed array content', () => {
      const chunk = createChunk([
        { type: 'text' as const, text: 'Some text' },
        {
          type: 'reasoning' as const,
          summary: [{ type: 'summary_text' as const, text: 'Some reasoning' }],
        },
        { type: 'image' as const, image_url: 'https://example.com/img.png' },
      ])

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(2)
      expect(result[0].type).toBe('text-delta')
      expect(result[1].type).toBe('reasoning-delta')
    })
  })

  describe('empty choices array', () => {
    it('should return empty array when choices is empty', () => {
      const chunk: FmapiChunk = {
        id: 'chunk-empty',
        created: Date.now(),
        model: 'test-model',
        object: 'chat.completion.chunk',
        choices: [],
      }

      const result = convertFmapiChunkToMessagePart(chunk)

      expect(result).toHaveLength(0)
    })
  })

  describe('streaming tool calls (OpenAI format)', () => {
    it('should emit tool-input-start when tool call starts with id and name', () => {
      const chunk = createToolCallChunk([
        {
          index: 0,
          id: 'call-123',
          type: 'function',
          function: {
            name: 'get_weather',
            arguments: '',
          },
        },
      ])

      const toolCallIdsByIndex = new Map<number, string>()
      const result = convertFmapiChunkToMessagePart(chunk, toolCallIdsByIndex)

      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({
        type: 'tool-input-start',
        id: 'call-123',
        toolName: 'get_weather',
      })
      // Should track the ID by index
      expect(toolCallIdsByIndex.get(0)).toBe('call-123')
    })

    it('should emit tool-input-delta with tracked ID when arguments stream', () => {
      const toolCallIdsByIndex = new Map<number, string>()
      toolCallIdsByIndex.set(0, 'call-123') // Pre-populate from previous chunk

      const chunk = createToolCallChunk([
        {
          index: 0,
          // No id in subsequent chunks (OpenAI behavior)
          function: {
            arguments: '{"location":',
          },
        },
      ])

      const result = convertFmapiChunkToMessagePart(chunk, toolCallIdsByIndex)

      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({
        type: 'tool-input-delta',
        id: 'call-123', // Should use tracked ID
        delta: '{"location":',
      })
    })

    it('should emit both tool-input-start and tool-input-delta when first chunk has args', () => {
      const chunk = createToolCallChunk([
        {
          index: 0,
          id: 'call-456',
          type: 'function',
          function: {
            name: 'search',
            arguments: '{"query":"hello"}',
          },
        },
      ])

      const toolCallIdsByIndex = new Map<number, string>()
      const result = convertFmapiChunkToMessagePart(chunk, toolCallIdsByIndex)

      expect(result).toHaveLength(2)
      expect(result[0]).toMatchObject({
        type: 'tool-input-start',
        id: 'call-456',
        toolName: 'search',
      })
      expect(result[1]).toMatchObject({
        type: 'tool-input-delta',
        id: 'call-456',
        delta: '{"query":"hello"}',
      })
    })

    it('should handle multiple tool calls in parallel', () => {
      const chunk = createToolCallChunk([
        {
          index: 0,
          id: 'call-a',
          type: 'function',
          function: {
            name: 'tool_a',
            arguments: '{"a":1}',
          },
        },
        {
          index: 1,
          id: 'call-b',
          type: 'function',
          function: {
            name: 'tool_b',
            arguments: '{"b":2}',
          },
        },
      ])

      const toolCallIdsByIndex = new Map<number, string>()
      const result = convertFmapiChunkToMessagePart(chunk, toolCallIdsByIndex)

      expect(result).toHaveLength(4) // 2 starts + 2 deltas
      expect(result[0]).toMatchObject({ type: 'tool-input-start', id: 'call-a', toolName: 'tool_a' })
      expect(result[1]).toMatchObject({ type: 'tool-input-delta', id: 'call-a', delta: '{"a":1}' })
      expect(result[2]).toMatchObject({ type: 'tool-input-start', id: 'call-b', toolName: 'tool_b' })
      expect(result[3]).toMatchObject({ type: 'tool-input-delta', id: 'call-b', delta: '{"b":2}' })
      expect(toolCallIdsByIndex.get(0)).toBe('call-a')
      expect(toolCallIdsByIndex.get(1)).toBe('call-b')
    })

    it('should track IDs across multiple chunks for parallel tool calls', () => {
      const toolCallIdsByIndex = new Map<number, string>()

      // First chunk: starts both tool calls
      const chunk1 = createToolCallChunk([
        { index: 0, id: 'call-x', type: 'function', function: { name: 'tool_x', arguments: '' } },
        { index: 1, id: 'call-y', type: 'function', function: { name: 'tool_y', arguments: '' } },
      ])
      convertFmapiChunkToMessagePart(chunk1, toolCallIdsByIndex)

      // Second chunk: arguments for tool at index 0 (no id)
      const chunk2 = createToolCallChunk([{ index: 0, function: { arguments: '{"x":' } }])
      const result2 = convertFmapiChunkToMessagePart(chunk2, toolCallIdsByIndex)

      expect(result2).toHaveLength(1)
      expect(result2[0]).toMatchObject({
        type: 'tool-input-delta',
        id: 'call-x', // Should use tracked ID for index 0
        delta: '{"x":',
      })

      // Third chunk: arguments for tool at index 1 (no id)
      const chunk3 = createToolCallChunk([{ index: 1, function: { arguments: '{"y":' } }])
      const result3 = convertFmapiChunkToMessagePart(chunk3, toolCallIdsByIndex)

      expect(result3).toHaveLength(1)
      expect(result3[0]).toMatchObject({
        type: 'tool-input-delta',
        id: 'call-y', // Should use tracked ID for index 1
        delta: '{"y":',
      })
    })

    it('should use fallback ID when toolCallIdsByIndex is not provided', () => {
      const chunk = createToolCallChunk([
        {
          index: 0,
          // No id provided
          function: {
            arguments: '{"test":true}',
          },
        },
      ])

      const result = convertFmapiChunkToMessagePart(chunk) // No map provided

      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({
        type: 'tool-input-delta',
        id: 'tool-call-0', // Fallback ID
        delta: '{"test":true}',
      })
    })

    it('should use fallback ID when index not in map', () => {
      const toolCallIdsByIndex = new Map<number, string>()
      // Map is empty, no ID tracked for index 2

      const chunk = createToolCallChunk([
        {
          index: 2,
          function: {
            arguments: '{"data":"value"}',
          },
        },
      ])

      const result = convertFmapiChunkToMessagePart(chunk, toolCallIdsByIndex)

      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({
        type: 'tool-input-delta',
        id: 'tool-call-2', // Fallback using index
        delta: '{"data":"value"}',
      })
    })
  })
})

// ============================================================================
// Tests for convertFmapiResponseToMessagePart (fmapi-convert-to-message-parts.ts)
// ============================================================================

describe('convertFmapiResponseToMessagePart', () => {
  const createResponse = (content: string | FmapiResponse['choices'][0]['message']['content']): FmapiResponse => ({
    id: 'resp-123',
    created: Date.now(),
    model: 'test-model',
    choices: [
      {
        message: {
          role: 'assistant',
          content,
        },
      },
    ],
  })

  describe('string content', () => {
    it('should convert plain text string content', () => {
      const response = createResponse('Hello, this is the complete response.')

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(1)
      expect(result[0]).toEqual({
        type: 'text',
        text: 'Hello, this is the complete response.',
      })
    })

    it('should convert string content with tool call tags', () => {
      const toolCallContent = '<tool_call>{"id":"call-resp","name":"final_tool","arguments":{"key":"value"}}</tool_call>'
      const response = createResponse(toolCallContent)

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call-resp',
        toolName: 'final_tool',
        providerExecuted: true,
      })
    })

    it('should convert string content with legacy tool call tags', () => {
      const toolCallContent = '<uc_function_call>{"id":"legacy-resp","name":"legacy_fn","arguments":{}}</uc_function_call>'
      const response = createResponse(toolCallContent)

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'legacy-resp',
        toolName: 'legacy_fn',
      })
    })

    it('should convert string content with tool result tags', () => {
      const toolResultContent = '<tool_call_result>{"id":"result-resp","content":{"status":"success"}}</tool_call_result>'
      const response = createResponse(toolResultContent)

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({
        type: 'tool-result',
        toolCallId: 'result-resp',
        toolName: DATABRICKS_TOOL_CALL_ID,
        result: { status: 'success' },
      })
    })

    it('should handle empty string content', () => {
      const response = createResponse('')

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(1)
      expect(result[0]).toEqual({ type: 'text', text: '' })
    })
  })

  describe('array content', () => {
    it('should convert array content with text items', () => {
      const response = createResponse([
        { type: 'text' as const, text: 'Response in array format' },
      ])

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(1)
      expect(result[0]).toEqual({
        type: 'text',
        text: 'Response in array format',
      })
    })

    it('should convert array content with multiple text items', () => {
      const response = createResponse([
        { type: 'text' as const, text: 'Part 1' },
        { type: 'text' as const, text: 'Part 2' },
      ])

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(2)
      expect(result[0]).toEqual({ type: 'text', text: 'Part 1' })
      expect(result[1]).toEqual({ type: 'text', text: 'Part 2' })
    })

    it('should convert reasoning content items', () => {
      const response = createResponse([
        {
          type: 'reasoning' as const,
          summary: [
            { type: 'summary_text' as const, text: 'Complete reasoning output' },
          ],
        },
      ])

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(1)
      expect(result[0]).toEqual({
        type: 'reasoning',
        text: 'Complete reasoning output',
      })
    })

    it('should convert reasoning with multiple summary items', () => {
      const response = createResponse([
        {
          type: 'reasoning' as const,
          summary: [
            { type: 'summary_text' as const, text: 'Reasoning 1' },
            { type: 'summary_text' as const, text: 'Reasoning 2' },
          ],
        },
      ])

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(2)
      expect(result[0]).toEqual({ type: 'reasoning', text: 'Reasoning 1' })
      expect(result[1]).toEqual({ type: 'reasoning', text: 'Reasoning 2' })
    })

    it('should ignore encrypted reasoning summaries', () => {
      const response = createResponse([
        {
          type: 'reasoning' as const,
          summary: [
            { type: 'summary_encrypted_text' as const, data: 'encrypted' },
          ],
        },
      ])

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(0)
    })

    it('should ignore image items', () => {
      const response = createResponse([
        { type: 'image' as const, image_url: 'https://example.com/result.png' },
      ])

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(0)
    })

    it('should handle mixed array content', () => {
      const response = createResponse([
        { type: 'text' as const, text: 'Final answer' },
        {
          type: 'reasoning' as const,
          summary: [{ type: 'summary_text' as const, text: 'My reasoning' }],
        },
        { type: 'image' as const, image_url: 'https://example.com/chart.png' },
      ])

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(2)
      expect(result[0]).toEqual({ type: 'text', text: 'Final answer' })
      expect(result[1]).toEqual({ type: 'reasoning', text: 'My reasoning' })
    })

    it('should handle undefined content', () => {
      const response: FmapiResponse = {
        id: 'resp-undef',
        created: Date.now(),
        model: 'test-model',
        choices: [
          {
            message: {
              role: 'assistant',
              content: undefined,
            },
          },
        ],
      }

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(0)
    })
  })

  describe('OpenAI-format tool_calls', () => {
    it('should convert single tool_call to tool-call content', () => {
      const response: FmapiResponse = {
        id: 'resp-tool',
        created: Date.now(),
        model: 'test-model',
        choices: [
          {
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                {
                  id: 'call-weather-1',
                  type: 'function',
                  function: {
                    name: 'get_weather',
                    arguments: '{"location":"Tokyo"}',
                  },
                },
              ],
            },
          },
        ],
      }

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call-weather-1',
        toolName: 'get_weather',
        input: '{"location":"Tokyo"}',
      })
    })

    it('should convert multiple tool_calls', () => {
      const response: FmapiResponse = {
        id: 'resp-multi-tool',
        created: Date.now(),
        model: 'test-model',
        choices: [
          {
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                {
                  id: 'call-1',
                  type: 'function',
                  function: {
                    name: 'tool_a',
                    arguments: '{"a":1}',
                  },
                },
                {
                  id: 'call-2',
                  type: 'function',
                  function: {
                    name: 'tool_b',
                    arguments: '{"b":2}',
                  },
                },
              ],
            },
          },
        ],
      }

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(2)
      expect(result[0]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call-1',
        toolName: 'tool_a',
        input: '{"a":1}',
      })
      expect(result[1]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call-2',
        toolName: 'tool_b',
        input: '{"b":2}',
      })
    })

    it('should include text content alongside tool_calls', () => {
      const response: FmapiResponse = {
        id: 'resp-tool-text',
        created: Date.now(),
        model: 'test-model',
        choices: [
          {
            message: {
              role: 'assistant',
              content: 'Let me help you with that.',
              tool_calls: [
                {
                  id: 'call-helper',
                  type: 'function',
                  function: {
                    name: 'helper',
                    arguments: '{}',
                  },
                },
              ],
            },
          },
        ],
      }

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(2)
      expect(result[0]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call-helper',
        toolName: 'helper',
      })
      expect(result[1]).toMatchObject({
        type: 'text',
        text: 'Let me help you with that.',
      })
    })

    it('should handle tool_calls with complex JSON arguments', () => {
      const response: FmapiResponse = {
        id: 'resp-complex',
        created: Date.now(),
        model: 'test-model',
        choices: [
          {
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                {
                  id: 'call-complex',
                  type: 'function',
                  function: {
                    name: 'process_data',
                    arguments: '{"items":[1,2,3],"config":{"nested":true,"values":["a","b"]}}',
                  },
                },
              ],
            },
          },
        ],
      }

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call-complex',
        toolName: 'process_data',
        input: '{"items":[1,2,3],"config":{"nested":true,"values":["a","b"]}}',
      })
    })

    it('should handle empty tool_calls array by falling back to content', () => {
      const response: FmapiResponse = {
        id: 'resp-empty-tools',
        created: Date.now(),
        model: 'test-model',
        choices: [
          {
            message: {
              role: 'assistant',
              content: 'Just a regular response',
              tool_calls: [],
            },
          },
        ],
      }

      const result = convertFmapiResponseToMessagePart(response)

      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({
        type: 'text',
        text: 'Just a regular response',
      })
    })
  })
})
