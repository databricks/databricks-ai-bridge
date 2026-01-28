import { describe, it, expect } from 'vitest'
import type { LanguageModelV3Prompt } from '@ai-sdk/provider'
import { convertLanguageModelV3PromptToChatAgentResponse } from '../src/chat-agent-language-model/chat-agent-convert-to-input'

describe('convertLanguageModelV3PromptToChatAgentResponse', () => {
  describe('System messages', () => {
    it('should skip system messages and not include them in output', () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'system', content: 'You are a helpful assistant.' },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([])
    })

    it('should skip multiple system messages', () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'system', content: 'System prompt 1' },
        { role: 'system', content: 'System prompt 2' },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([])
    })

    it('should skip system messages but process other message types', () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toHaveLength(1)
      expect(result[0]).toEqual({
        role: 'user',
        content: 'Hello',
        id: 'user-0',
      })
    })
  })

  describe('User messages', () => {
    it('should convert simple text content', () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'user', content: [{ type: 'text', text: 'Hello, how are you?' }] },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'user',
          content: 'Hello, how are you?',
          id: 'user-0',
        },
      ])
    })

    it('should join multiple text parts with newline', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'First line' },
            { type: 'text', text: 'Second line' },
            { type: 'text', text: 'Third line' },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'user',
          content: 'First line\nSecond line\nThird line',
          id: 'user-0',
        },
      ])
    })

    it('should handle empty content array', () => {
      const prompt: LanguageModelV3Prompt = [{ role: 'user', content: [] }]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'user',
          content: '',
          id: 'user-0',
        },
      ])
    })

    it('should filter non-text parts from user content', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Text content' },
            { type: 'file', data: new Uint8Array(), mediaType: 'image/png' },
            { type: 'text', text: 'More text' },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'user',
          content: 'Text content\nMore text',
          id: 'user-0',
        },
      ])
    })

    it('should handle multiple user messages with correct IDs', () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'user', content: [{ type: 'text', text: 'First message' }] },
        { role: 'user', content: [{ type: 'text', text: 'Second message' }] },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        { role: 'user', content: 'First message', id: 'user-0' },
        { role: 'user', content: 'Second message', id: 'user-1' },
      ])
    })
  })

  describe('Assistant messages', () => {
    it('should convert text content', () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'assistant', content: [{ type: 'text', text: 'Hello! How can I help?' }] },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'assistant',
          content: 'Hello! How can I help?',
          id: 'assistant-0',
          tool_calls: undefined,
        },
      ])
    })

    it('should merge reasoning content with text', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            { type: 'reasoning', text: 'Let me think about this...' },
            { type: 'text', text: 'The answer is 42.' },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'assistant',
          content: 'Let me think about this...\nThe answer is 42.',
          id: 'assistant-0',
          tool_calls: undefined,
        },
      ])
    })

    it('should extract tool calls', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'I will search for that.' },
            {
              type: 'tool-call',
              toolCallId: 'call-123',
              toolName: 'search',
              input: { query: 'test' },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'assistant',
          content: 'I will search for that.',
          id: 'assistant-0',
          tool_calls: [
            {
              type: 'function',
              id: 'call-123',
              function: {
                name: 'search',
                arguments: '{"query":"test"}',
              },
            },
          ],
        },
      ])
    })

    it('should handle tool calls with string arguments', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call-456',
              toolName: 'calculator',
              input: '{"operation":"add","a":1,"b":2}',
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'assistant',
          content: '',
          id: 'assistant-0',
          tool_calls: [
            {
              type: 'function',
              id: 'call-456',
              function: {
                name: 'calculator',
                arguments: '{"operation":"add","a":1,"b":2}',
              },
            },
          ],
        },
      ])
    })

    it('should handle tool calls with object arguments', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call-789',
              toolName: 'weather',
              input: { city: 'London', units: 'metric' },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'assistant',
          content: '',
          id: 'assistant-0',
          tool_calls: [
            {
              type: 'function',
              id: 'call-789',
              function: {
                name: 'weather',
                arguments: '{"city":"London","units":"metric"}',
              },
            },
          ],
        },
      ])
    })

    it('should handle tool calls with null/undefined arguments', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call-empty',
              toolName: 'noArgs',
              input: undefined as unknown as Record<string, unknown>,
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result[0]).toMatchObject({
        role: 'assistant',
        tool_calls: [
          {
            type: 'function',
            id: 'call-empty',
            function: {
              name: 'noArgs',
              arguments: '{}',
            },
          },
        ],
      })
    })

    it('should return undefined for tool_calls when no tool calls present', () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'assistant', content: [{ type: 'text', text: 'Just text, no tools.' }] },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result[0]).toEqual({
        role: 'assistant',
        content: 'Just text, no tools.',
        id: 'assistant-0',
        tool_calls: undefined,
      })
    })

    it('should handle multiple tool calls', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call-1',
              toolName: 'search',
              input: { q: 'a' },
            },
            {
              type: 'tool-call',
              toolCallId: 'call-2',
              toolName: 'fetch',
              input: { url: 'http://example.com' },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result[0]).toMatchObject({
        tool_calls: [
          { id: 'call-1', function: { name: 'search' } },
          { id: 'call-2', function: { name: 'fetch' } },
        ],
      })
    })

    it('should convert embedded tool-result parts to separate tool messages', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'Using tool...' },
            {
              type: 'tool-call',
              toolCallId: 'call-123',
              toolName: 'calculator',
              input: { a: 1, b: 2 },
            },
            {
              type: 'tool-result',
              toolCallId: 'call-123',
              toolName: 'calculator',
              output: { type: 'text', value: 'Result: 3' },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toHaveLength(2)
      expect(result[0]).toMatchObject({
        role: 'assistant',
        content: 'Using tool...',
        id: 'assistant-0',
      })
      expect(result[1]).toEqual({
        role: 'tool',
        name: 'calculator',
        content: 'Result: 3',
        tool_call_id: 'call-123',
        id: 'tool-1',
      })
    })

    it('should handle empty content array', () => {
      const prompt: LanguageModelV3Prompt = [{ role: 'assistant', content: [] }]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'assistant',
          content: '',
          id: 'assistant-0',
          tool_calls: undefined,
        },
      ])
    })
  })

  describe('Tool messages', () => {
    it('should convert tool result with text output', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-123',
              toolName: 'search',
              output: { type: 'text', value: 'Found 10 results' },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'tool',
          name: 'search',
          content: 'Found 10 results',
          tool_call_id: 'call-123',
          id: 'tool-0',
        },
      ])
    })

    it('should convert tool result with json output', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-123',
              toolName: 'api',
              output: { type: 'json', value: { data: [1, 2, 3], status: 'ok' } },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'tool',
          name: 'api',
          content: '{"data":[1,2,3],"status":"ok"}',
          tool_call_id: 'call-123',
          id: 'tool-0',
        },
      ])
    })

    it('should convert tool result with error-text output', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-123',
              toolName: 'network',
              output: { type: 'error-text', value: 'Connection timeout' },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'tool',
          name: 'network',
          content: 'Connection timeout',
          tool_call_id: 'call-123',
          id: 'tool-0',
        },
      ])
    })

    it('should convert tool result with error-json output', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-123',
              toolName: 'api',
              output: {
                type: 'error-json',
                value: { code: 500, message: 'Internal server error' },
              },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'tool',
          name: 'api',
          content: '{"code":500,"message":"Internal server error"}',
          tool_call_id: 'call-123',
          id: 'tool-0',
        },
      ])
    })

    it('should convert tool result with content output (array of parts)', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-123',
              toolName: 'multi',
              output: {
                type: 'content',
                value: [
                  { type: 'text', text: 'Part 1' },
                  { type: 'text', text: 'Part 2' },
                  { type: 'image-data', data: 'base64data', mediaType: 'image/png' },
                  { type: 'text', text: 'Part 3' },
                ],
              },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'tool',
          name: 'multi',
          content: 'Part 1\nPart 2\nPart 3',
          tool_call_id: 'call-123',
          id: 'tool-0',
        },
      ])
    })

    it('should handle content output with only non-text parts', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-123',
              toolName: 'image-gen',
              output: {
                type: 'content',
                value: [{ type: 'image-data', data: 'base64data', mediaType: 'image/png' }],
              },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'tool',
          name: 'image-gen',
          content: '',
          tool_call_id: 'call-123',
          id: 'tool-0',
        },
      ])
    })

    it('should handle multiple tool results in one message', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-1',
              toolName: 'tool1',
              output: { type: 'text', value: 'Result 1' },
            },
            {
              type: 'tool-result',
              toolCallId: 'call-2',
              toolName: 'tool2',
              output: { type: 'text', value: 'Result 2' },
            },
            {
              type: 'tool-result',
              toolCallId: 'call-3',
              toolName: 'tool3',
              output: { type: 'json', value: { x: 1 } },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'tool',
          name: 'tool1',
          content: 'Result 1',
          tool_call_id: 'call-1',
          id: 'tool-0',
        },
        {
          role: 'tool',
          name: 'tool2',
          content: 'Result 2',
          tool_call_id: 'call-2',
          id: 'tool-1',
        },
        {
          role: 'tool',
          name: 'tool3',
          content: '{"x":1}',
          tool_call_id: 'call-3',
          id: 'tool-2',
        },
      ])
    })

    it('should skip non-tool-result parts in tool messages', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-1',
              toolName: 'valid',
              output: { type: 'text', value: 'Valid result' },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({
        role: 'tool',
        name: 'valid',
      })
    })

    it('should handle empty tool content array', () => {
      const prompt: LanguageModelV3Prompt = [{ role: 'tool', content: [] }]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([])
    })
  })

  describe('Message ID generation', () => {
    it('should generate unique IDs with correct prefixes', () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
        { role: 'assistant', content: [{ type: 'text', text: 'Hi!' }] },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'c1',
              toolName: 't1',
              output: { type: 'text', value: 'r' },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result[0].id).toBe('user-0')
      expect(result[1].id).toBe('assistant-1')
      expect(result[2].id).toBe('tool-2')
    })

    it('should continue ID sequence across message types', () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'user', content: [{ type: 'text', text: 'A' }] },
        { role: 'user', content: [{ type: 'text', text: 'B' }] },
        { role: 'assistant', content: [{ type: 'text', text: 'C' }] },
        { role: 'user', content: [{ type: 'text', text: 'D' }] },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result.map((m) => m.id)).toEqual(['user-0', 'user-1', 'assistant-2', 'user-3'])
    })

    it('should not increment ID for skipped system messages', () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'system', content: 'System' },
        { role: 'user', content: [{ type: 'text', text: 'User' }] },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result[0].id).toBe('user-0')
    })

    it('should increment ID for embedded tool results in assistant messages', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call-1',
              toolName: 'tool1',
              input: {},
            },
            {
              type: 'tool-result',
              toolCallId: 'call-1',
              toolName: 'tool1',
              output: { type: 'text', value: 'result' },
            },
          ],
        },
        { role: 'user', content: [{ type: 'text', text: 'Next' }] },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result[0].id).toBe('assistant-0')
      expect(result[1].id).toBe('tool-1')
      expect(result[2].id).toBe('user-2')
    })
  })

  describe('Edge cases', () => {
    it('should handle empty prompt array', () => {
      const prompt: LanguageModelV3Prompt = []

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([])
    })

    it('should handle mixed message types in realistic conversation', () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: [{ type: 'text', text: "What's the weather in Paris?" }] },
        {
          role: 'assistant',
          content: [
            { type: 'text', text: "I'll check the weather for you." },
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
              output: { type: 'json', value: { temp: 22, condition: 'sunny' } },
            },
          ],
        },
        {
          role: 'assistant',
          content: [{ type: 'text', text: "It's 22C and sunny in Paris!" }],
        },
        { role: 'user', content: [{ type: 'text', text: 'Thanks!' }] },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toHaveLength(5)
      expect(result[0]).toMatchObject({ role: 'user', id: 'user-0' })
      expect(result[1]).toMatchObject({
        role: 'assistant',
        id: 'assistant-1',
        tool_calls: [{ function: { name: 'get_weather' } }],
      })
      expect(result[2]).toMatchObject({ role: 'tool', id: 'tool-2' })
      expect(result[3]).toMatchObject({ role: 'assistant', id: 'assistant-3' })
      expect(result[4]).toMatchObject({ role: 'user', id: 'user-4' })
    })

    it('should handle assistant message with only reasoning content', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [{ type: 'reasoning', text: 'Thinking about the problem...' }],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'assistant',
          content: 'Thinking about the problem...',
          id: 'assistant-0',
          tool_calls: undefined,
        },
      ])
    })

    it('should handle user message with single empty text part', () => {
      const prompt: LanguageModelV3Prompt = [{ role: 'user', content: [{ type: 'text', text: '' }] }]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'user',
          content: '',
          id: 'user-0',
        },
      ])
    })

    it('should handle unknown output type in tool result (default case)', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-123',
              toolName: 'unknown',
              output: { type: 'unknown-type' as 'text', value: 'something' },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toEqual([
        {
          role: 'tool',
          name: 'unknown',
          content: '',
          tool_call_id: 'call-123',
          id: 'tool-0',
        },
      ])
    })

    it('should handle embedded tool result with unknown output type', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-123',
              toolName: 'unknown',
              output: { type: 'unknown-type' as 'text', value: 'something' },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toHaveLength(2)
      expect(result[1]).toMatchObject({
        role: 'tool',
        content: '',
      })
    })

    it('should handle multiple embedded tool results in assistant message', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call-1',
              toolName: 'tool1',
              input: {},
            },
            {
              type: 'tool-call',
              toolCallId: 'call-2',
              toolName: 'tool2',
              input: {},
            },
            {
              type: 'tool-result',
              toolCallId: 'call-1',
              toolName: 'tool1',
              output: { type: 'text', value: 'Result 1' },
            },
            {
              type: 'tool-result',
              toolCallId: 'call-2',
              toolName: 'tool2',
              output: { type: 'json', value: { data: 'Result 2' } },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result).toHaveLength(3)
      expect(result[0]).toMatchObject({
        role: 'assistant',
        tool_calls: [
          { id: 'call-1', function: { name: 'tool1' } },
          { id: 'call-2', function: { name: 'tool2' } },
        ],
      })
      expect(result[1]).toMatchObject({
        role: 'tool',
        tool_call_id: 'call-1',
        content: 'Result 1',
      })
      expect(result[2]).toMatchObject({
        role: 'tool',
        tool_call_id: 'call-2',
        content: '{"data":"Result 2"}',
      })
    })

    it('should handle content output with empty text parts filtered out', () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call-123',
              toolName: 'multi',
              output: {
                type: 'content',
                value: [
                  { type: 'text', text: '' },
                  { type: 'text', text: 'Non-empty' },
                  { type: 'text', text: '' },
                ],
              },
            },
          ],
        },
      ]

      const result = convertLanguageModelV3PromptToChatAgentResponse(prompt)

      expect(result[0]).toMatchObject({
        content: 'Non-empty',
      })
    })
  })
})
