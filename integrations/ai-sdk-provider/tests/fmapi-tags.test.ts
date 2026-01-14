import { describe, it, expect } from 'vitest'
import {
  parseTaggedToolCall,
  parseTaggedToolResult,
  serializeToolCall,
  serializeToolResult,
  tagSplitRegex,
  ParsedToolCall,
  ParsedToolResult,
} from '../src/fmapi-language-model/fmapi-tags'

describe('FMAPI Tags', () => {
  describe('parseTaggedToolCall', () => {
    describe('valid tag formats', () => {
      it('parses valid legacy tool call with all fields', () => {
        const text = '<uc_function_call>{"id":"call-123","name":"myTool","arguments":{"key":"value"}}</uc_function_call>'
        const result = parseTaggedToolCall(text)
        expect(result).toEqual({
          id: 'call-123',
          name: 'myTool',
          arguments: { key: 'value' },
        })
      })

      it('parses legacy tool call without arguments field', () => {
        const text = '<uc_function_call>{"id":"call-789","name":"noArgsTool"}</uc_function_call>'
        const result = parseTaggedToolCall(text)
        expect(result).toEqual({
          id: 'call-789',
          name: 'noArgsTool',
          arguments: undefined,
        })
      })

      it('parses valid new tool call with all fields', () => {
        const text = '<tool_call>{"id":"tc-123","name":"newTool","arguments":{"param":"test"}}</tool_call>'
        const result = parseTaggedToolCall(text)
        expect(result).toEqual({
          id: 'tc-123',
          name: 'newTool',
          arguments: { param: 'test' },
        })
      })

      it('parses new tool call without arguments field', () => {
        const text = '<tool_call>{"id":"tc-789","name":"newNoArgsTool"}</tool_call>'
        const result = parseTaggedToolCall(text)
        expect(result).toEqual({
          id: 'tc-789',
          name: 'newNoArgsTool',
          arguments: undefined,
        })
      })
    })

    describe('whitespace and formatting', () => {
      it('parses tool call with leading and trailing whitespace', () => {
        const text = '   <tool_call>{"id":"ws-3","name":"tool"}</tool_call>   '
        const result = parseTaggedToolCall(text)
        expect(result).toEqual({
          id: 'ws-3',
          name: 'tool',
          arguments: undefined,
        })
      })
    })

    describe('invalid cases', () => {
      it('returns null for empty string', () => {
        expect(parseTaggedToolCall('')).toBeNull()
      })

      it('returns null for invalid JSON inside tags', () => {
        const text = '<tool_call>not valid json</tool_call>'
        expect(parseTaggedToolCall(text)).toBeNull()
      })

      it('returns null for missing id field', () => {
        const text = '<tool_call>{"name":"tool","arguments":{}}</tool_call>'
        expect(parseTaggedToolCall(text)).toBeNull()
      })

      it('returns null for missing name field', () => {
        const text = '<tool_call>{"id":"test-id","arguments":{}}</tool_call>'
        expect(parseTaggedToolCall(text)).toBeNull()
      })
    })

    describe('type coercion', () => {
      it('converts numeric id to string', () => {
        const text = '<tool_call>{"id":123,"name":"tool"}</tool_call>'
        const result = parseTaggedToolCall(text)
        expect(result).toEqual({
          id: '123',
          name: 'tool',
          arguments: undefined,
        })
      })
    })
  })

  describe('parseTaggedToolResult', () => {
    describe('valid tag formats', () => {
      it('parses valid legacy tool result with content', () => {
        const text = '<uc_function_result>{"id":"result-123","content":"success"}</uc_function_result>'
        const result = parseTaggedToolResult(text)
        expect(result).toEqual({
          id: 'result-123',
          content: 'success',
        })
      })

      it('parses legacy tool result without content field', () => {
        const text = '<uc_function_result>{"id":"result-no-content"}</uc_function_result>'
        const result = parseTaggedToolResult(text)
        expect(result).toEqual({
          id: 'result-no-content',
          content: undefined,
        })
      })

      it('parses valid new tool result with content', () => {
        const text = '<tool_call_result>{"id":"tr-123","content":"new result"}</tool_call_result>'
        const result = parseTaggedToolResult(text)
        expect(result).toEqual({
          id: 'tr-123',
          content: 'new result',
        })
      })

      it('parses new tool result with numeric content', () => {
        const text = '<tool_call_result>{"id":"tr-num","content":42}</tool_call_result>'
        const result = parseTaggedToolResult(text)
        expect(result).toEqual({
          id: 'tr-num',
          content: 42,
        })
      })
    })

    describe('invalid cases', () => {
      it('returns null for empty string', () => {
        expect(parseTaggedToolResult('')).toBeNull()
      })

      it('returns null for invalid JSON inside tags', () => {
        const text = '<tool_call_result>not valid json</tool_call_result>'
        expect(parseTaggedToolResult(text)).toBeNull()
      })
    })
  })

  describe('serializeToolCall', () => {
    it('serializes tool call with object arguments to new tag format', () => {
      const toolCall: ParsedToolCall = {
        id: 'serialize-1',
        name: 'testTool',
        arguments: { key: 'value', num: 42 },
      }
      const result = serializeToolCall(toolCall)
      expect(result).toBe('<tool_call>{"id":"serialize-1","name":"testTool","arguments":{"key":"value","num":42}}</tool_call>')
    })

    it('serializes tool call with nested object arguments', () => {
      const toolCall: ParsedToolCall = {
        id: 'serialize-6',
        name: 'nestedTool',
        arguments: { level1: { level2: { level3: 'deep' } } },
      }
      const result = serializeToolCall(toolCall)
      expect(result).toBe('<tool_call>{"id":"serialize-6","name":"nestedTool","arguments":{"level1":{"level2":{"level3":"deep"}}}}</tool_call>')
    })

    it('serializes tool call with undefined arguments', () => {
      const toolCall: ParsedToolCall = {
        id: 'serialize-4',
        name: 'undefinedTool',
        arguments: undefined,
      }
      const result = serializeToolCall(toolCall)
      // undefined values are omitted in JSON.stringify
      expect(result).toBe('<tool_call>{"id":"serialize-4","name":"undefinedTool"}</tool_call>')
    })
  })

  describe('serializeToolResult', () => {
    it('serializes tool result with string content to new tag format', () => {
      const toolResult: ParsedToolResult = {
        id: 'result-1',
        content: 'success message',
      }
      const result = serializeToolResult(toolResult)
      expect(result).toBe('<tool_call_result>{"id":"result-1","content":"success message"}</tool_call_result>')
    })

    it('serializes tool result with object content', () => {
      const toolResult: ParsedToolResult = {
        id: 'result-2',
        content: { status: 'ok', data: [1, 2, 3] },
      }
      const result = serializeToolResult(toolResult)
      expect(result).toBe('<tool_call_result>{"id":"result-2","content":{"status":"ok","data":[1,2,3]}}</tool_call_result>')
    })

    it('serializes tool result with undefined content', () => {
      const toolResult: ParsedToolResult = {
        id: 'result-4',
        content: undefined,
      }
      const result = serializeToolResult(toolResult)
      // undefined values are omitted in JSON.stringify
      expect(result).toBe('<tool_call_result>{"id":"result-4"}</tool_call_result>')
    })
  })

  describe('tagSplitRegex', () => {
    it('splits text with single new tool call tag', () => {
      const text = 'before<tool_call>{"id":"2","name":"newTool"}</tool_call>after'
      const parts = text.split(tagSplitRegex)
      expect(parts).toEqual([
        'before',
        '<tool_call>{"id":"2","name":"newTool"}</tool_call>',
        'after',
      ])
    })

    it('splits text with multiple tags', () => {
      const text = 'intro<tool_call>{"id":"a","name":"first"}</tool_call>middle<tool_call>{"id":"b","name":"second"}</tool_call>outro'
      const parts = text.split(tagSplitRegex)
      expect(parts).toEqual([
        'intro',
        '<tool_call>{"id":"a","name":"first"}</tool_call>',
        'middle',
        '<tool_call>{"id":"b","name":"second"}</tool_call>',
        'outro',
      ])
    })
  })

  describe('edge cases', () => {
    it('parseTaggedToolCall does not match result tags', () => {
      const text = '<tool_call_result>{"id":"test","name":"tool"}</tool_call_result>'
      expect(parseTaggedToolCall(text)).toBeNull()
    })

    it('parseTaggedToolResult does not match call tags', () => {
      const text = '<tool_call>{"id":"test","content":"value"}</tool_call>'
      expect(parseTaggedToolResult(text)).toBeNull()
    })

    it('handles unicode in arguments', () => {
      const text = '<tool_call>{"id":"unicode","name":"tool","arguments":{"greeting":"Hello, 世界!"}}</tool_call>'
      const result = parseTaggedToolCall(text)
      expect(result).toEqual({
        id: 'unicode',
        name: 'tool',
        arguments: { greeting: 'Hello, 世界!' },
      })
    })
  })
})
