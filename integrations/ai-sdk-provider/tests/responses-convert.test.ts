import { describe, it, expect } from 'vitest'
import type { LanguageModelV3Prompt } from '@ai-sdk/provider'
import { convertToResponsesInput } from '../src/responses-agent-language-model/responses-convert-to-input'
import {
  convertResponsesAgentChunkToMessagePart,
  convertResponsesAgentResponseToMessagePart,
} from '../src/responses-agent-language-model/responses-convert-to-message-parts'

// Helper to create options for conversion functions
const defaultChunkOptions = { useRemoteToolCalling: true, toolNamesByCallId: new Map<string, string>() }
const defaultResponseOptions = { useRemoteToolCalling: true }

// ============================================================================
// Tests for convertToResponsesInput
// ============================================================================

describe('convertToResponsesInput', () => {
  describe('system message modes', () => {
    it('converts system message with mode "system"', async () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'system', content: 'You are a helpful assistant.' },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([{ role: 'system', content: 'You are a helpful assistant.' }])
      expect(warnings).toHaveLength(0)
    })

    it('converts system message with mode "developer"', async () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'system', content: 'You are a helpful assistant.' },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'developer',
      })

      expect(input).toEqual([{ role: 'developer', content: 'You are a helpful assistant.' }])
      expect(warnings).toHaveLength(0)
    })

    it('removes system message with mode "remove" and adds warning', async () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'system', content: 'You are a helpful assistant.' },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'remove',
      })

      expect(input).toEqual([])
      expect(warnings).toHaveLength(1)
      expect(warnings[0]).toEqual({
        type: 'other',
        message: 'system messages are removed for this model',
      })
    })
  })

  describe('user messages', () => {
    it('converts user message with text parts', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Hello, ' },
            { type: 'text', text: 'how are you?' },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          role: 'user',
          content: [
            { type: 'input_text', text: 'Hello, ' },
            { type: 'input_text', text: 'how are you?' },
          ],
        },
      ])
      expect(warnings).toHaveLength(0)
    })

    it('throws error for unsupported user content types', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'user',
          // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-explicit-any
          content: [
            {
              type: 'image',
              image: new Uint8Array([1, 2, 3]),
              mimeType: 'image/png',
            } as any,
          ],
        },
      ]

      await expect(
        convertToResponsesInput({
          prompt,
          systemMessageMode: 'system',
        })
      ).rejects.toThrow()
    })
  })

  describe('assistant messages', () => {
    it('converts assistant text message', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [{ type: 'text', text: 'Hello! How can I help you?' }],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          role: 'assistant',
          content: [{ type: 'output_text', text: 'Hello! How can I help you?' }],
          id: undefined,
        },
      ])
      expect(warnings).toHaveLength(0)
    })

    it('converts assistant text message with itemId from provider options', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'text',
              text: 'Hello!',
              providerOptions: {
                databricks: { itemId: 'msg_123' },
              },
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          role: 'assistant',
          content: [{ type: 'output_text', text: 'Hello!' }],
          id: 'msg_123',
        },
      ])
      expect(warnings).toHaveLength(0)
    })

    it('converts assistant tool-call', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call_123',
              toolName: 'calculator',
              input: { operation: 'add', a: 1, b: 2 },
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          type: 'function_call',
          call_id: 'call_123',
          name: 'calculator',
          arguments: '{"operation":"add","a":1,"b":2}',
          id: undefined,
        },
      ])
      expect(warnings).toHaveLength(0)
    })

    it('converts assistant tool-call with custom toolName from provider options', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call_123',
              toolName: 'system__ai__python_exec',
              input: { code: 'print(1)' },
              providerOptions: {
                databricks: {
                  itemId: 'item_456',
                },
              },
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          type: 'function_call',
          call_id: 'call_123',
          name: 'system__ai__python_exec',
          arguments: '{"code":"print(1)"}',
          id: 'item_456',
        },
      ])
      expect(warnings).toHaveLength(0)
    })

    it('converts assistant tool-call with tool result', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call_123',
              toolName: 'calculator',
              input: { operation: 'add', a: 1, b: 2 },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call_123',
              toolName: 'add',
              output: { type: 'text', value: '3' },
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          type: 'function_call',
          call_id: 'call_123',
          name: 'calculator',
          arguments: '{"operation":"add","a":1,"b":2}',
          id: undefined,
        },
        {
          type: 'function_call_output',
          call_id: 'call_123',
          output: '3',
        },
      ])
      expect(warnings).toHaveLength(0)
    })

    it('includes tool results with any output value', async () => {
      // All tool results should be included - we no longer filter synthetic results
      // because we use dynamic: true instead of synthetic results
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call_123',
              toolName: 'some-tool',
              input: { query: 'test' },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call_123',
              toolName: 'some-tool',
              output: { type: 'text', value: '' },
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      // Should have both function_call AND function_call_output
      expect(input).toEqual([
        {
          type: 'function_call',
          call_id: 'call_123',
          name: 'some-tool',
          arguments: '{"query":"test"}',
          id: undefined,
        },
        {
          type: 'function_call_output',
          call_id: 'call_123',
          output: '',
        },
      ])
      expect(warnings).toHaveLength(0)
    })
  })

  describe('MCP approval request handling', () => {
    it('converts MCP approval request from tool-call with approvalRequestId', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'mcp_req_123',
              toolName: 'filesystem_read',
              input: { action: 'read_file', path: '/etc/hosts' },
              providerOptions: {
                databricks: {
                  approvalRequestId: 'mcp_req_123',
                  serverLabel: 'fs-server',
                },
              },
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          type: 'mcp_approval_request',
          id: 'mcp_req_123',
          name: 'filesystem_read',
          arguments: '{"action":"read_file","path":"/etc/hosts"}',
          server_label: 'fs-server',
        },
      ])
      expect(warnings).toHaveLength(0)
    })

    it('converts MCP approval request with tool-approval-response (approved)', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'mcp_req_123',
              toolName: 'filesystem_read',
              input: { action: 'read_file' },
              providerOptions: {
                databricks: {
                  approvalRequestId: 'mcp_req_123',
                  serverLabel: 'fs-server',
                },
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-approval-response',
              approvalId: 'mcp_req_123',
              approved: true,
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          type: 'mcp_approval_request',
          id: 'mcp_req_123',
          name: 'filesystem_read',
          arguments: '{"action":"read_file"}',
          server_label: 'fs-server',
        },
        {
          type: 'mcp_approval_response',
          approval_request_id: 'mcp_req_123',
          approve: true,
        },
      ])
      expect(warnings).toHaveLength(0)
    })

    it('converts MCP approval request with tool-approval-response (denied)', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'mcp_req_123',
              toolName: 'filesystem_delete',
              input: { action: 'delete_file' },
              providerOptions: {
                databricks: {
                  approvalRequestId: 'mcp_req_123',
                  serverLabel: 'fs-server',
                },
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-approval-response',
              approvalId: 'mcp_req_123',
              approved: false,
              reason: 'User denied',
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          type: 'mcp_approval_request',
          id: 'mcp_req_123',
          name: 'filesystem_delete',
          arguments: '{"action":"delete_file"}',
          server_label: 'fs-server',
        },
        {
          type: 'mcp_approval_response',
          approval_request_id: 'mcp_req_123',
          approve: false,
          reason: 'User denied',
        },
      ])
      expect(warnings).toHaveLength(0)
    })
  })

  describe('MCP approval response handling', () => {
    it('converts tool-approval-response from tool role', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-approval-response',
              approvalId: 'mcp_req_123',
              approved: true,
              reason: 'User approved',
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          type: 'mcp_approval_response',
          approval_request_id: 'mcp_req_123',
          approve: true,
          reason: 'User approved',
        },
      ])
      expect(warnings).toHaveLength(0)
    })

    it('converts tool-approval-response with denial', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-approval-response',
              approvalId: 'mcp_req_123',
              approved: false,
              reason: 'Security concern',
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          type: 'mcp_approval_response',
          approval_request_id: 'mcp_req_123',
          approve: false,
          reason: 'Security concern',
        },
      ])
      expect(warnings).toHaveLength(0)
    })

    it('deduplicates repeated tool-approval-response parts', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-approval-response',
              approvalId: 'mcp_req_123',
              approved: true,
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-approval-response',
              approvalId: 'mcp_req_123',
              approved: true,
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      // Should only have one approval response despite two in input
      expect(input).toEqual([
        {
          type: 'mcp_approval_response',
          approval_request_id: 'mcp_req_123',
          approve: true,
        },
      ])
      expect(warnings).toHaveLength(0)
    })
  })

  describe('reasoning content handling', () => {
    it('converts reasoning content with itemId', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'reasoning',
              text: 'Let me think about this...',
              providerOptions: {
                databricks: { itemId: 'reasoning_123' },
              },
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([
        {
          type: 'reasoning',
          id: 'reasoning_123',
          summary: [{ type: 'summary_text', text: 'Let me think about this...' }],
        },
      ])
      expect(warnings).toHaveLength(0)
    })

    it('skips reasoning content without itemId', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'reasoning',
              text: 'Let me think about this...',
            },
          ],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toEqual([])
      expect(warnings).toHaveLength(0)
    })
  })

  describe('tool results with different output types', () => {
    it('converts tool result with text output', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call_123',
              toolName: 'search',
              input: { query: 'test' },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call_123',
              toolName: 'search',
              output: { type: 'text', value: 'Search results here' },
            },
          ],
        },
      ]

      const { input } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input[1]).toEqual({
        type: 'function_call_output',
        call_id: 'call_123',
        output: 'Search results here',
      })
    })

    it('converts tool result with json output', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call_123',
              toolName: 'search',
              input: { query: 'test' },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call_123',
              toolName: 'query',
              output: { type: 'json', value: { results: ['a', 'b', 'c'] } },
            },
          ],
        },
      ]

      const { input } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input[1]).toEqual({
        type: 'function_call_output',
        call_id: 'call_123',
        output: '{"results":["a","b","c"]}',
      })
    })

    it('converts tool result with error-text output', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call_123',
              toolName: 'search',
              input: { query: 'test' },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call_123',
              toolName: 'fetch_data',
              output: { type: 'error-text', value: 'Connection timeout' },
            },
          ],
        },
      ]

      const { input } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input[1]).toEqual({
        type: 'function_call_output',
        call_id: 'call_123',
        output: 'Connection timeout',
      })
    })

    it('converts tool result with error-json output', async () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call_123',
              toolName: 'search',
              input: { query: 'test' },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call_123',
              toolName: 'api_call',
              output: { type: 'error-json', value: { code: 500, message: 'Server error' } },
            },
          ],
        },
      ]

      const { input } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input[1]).toEqual({
        type: 'function_call_output',
        call_id: 'call_123',
        output: '{"code":500,"message":"Server error"}',
      })
    })
  })

  describe('complex conversation flows', () => {
    it('converts a full conversation with multiple message types', async () => {
      const prompt: LanguageModelV3Prompt = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: [{ type: 'text', text: 'What is 2+2?' }] },
        {
          role: 'assistant',
          content: [{ type: 'text', text: "I'll calculate that for you." }],
        },
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'calc_1',
              toolName: 'calculator',
              input: { op: 'add', a: 2, b: 2 },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'calc_1',
              toolName: 'calculate',
              output: { type: 'text', value: '4' },
            },
          ],
        },
        {
          role: 'assistant',
          content: [{ type: 'text', text: 'The answer is 4.' }],
        },
      ]

      const { input, warnings } = await convertToResponsesInput({
        prompt,
        systemMessageMode: 'system',
      })

      expect(input).toHaveLength(6)
      expect(input[0]).toEqual({ role: 'system', content: 'You are a helpful assistant.' })
      expect(input[1]).toEqual({
        role: 'user',
        content: [{ type: 'input_text', text: 'What is 2+2?' }],
      })
      expect(input[2]).toEqual({
        role: 'assistant',
        content: [{ type: 'output_text', text: "I'll calculate that for you." }],
        id: undefined,
      })
      expect(input[3]).toEqual({
        type: 'function_call',
        call_id: 'calc_1',
        name: 'calculator',
        arguments: '{"op":"add","a":2,"b":2}',
        id: undefined,
      })
      // Tool result is inserted right after the tool call
      expect(input[4]).toEqual({
        type: 'function_call_output',
        call_id: 'calc_1',
        output: '4',
      })
      // The final assistant message is included after the tool result
      expect(input[5]).toEqual({
        role: 'assistant',
        content: [{ type: 'output_text', text: 'The answer is 4.' }],
        id: undefined,
      })
      expect(warnings).toHaveLength(0)
    })
  })
})

// ============================================================================
// Tests for convertResponsesAgentChunkToMessagePart
// ============================================================================

describe('convertResponsesAgentChunkToMessagePart', () => {
  describe('response.output_text.delta events', () => {
    it('converts text delta to text-delta part', () => {
      const chunk = {
        type: 'response.output_text.delta' as const,
        item_id: 'msg_123',
        delta: 'Hello, world!',
      }

      const parts = convertResponsesAgentChunkToMessagePart(chunk)

      expect(parts).toEqual([
        {
          type: 'text-delta',
          id: 'msg_123',
          delta: 'Hello, world!',
          providerMetadata: {
            databricks: { itemId: 'msg_123' },
          },
        },
      ])
    })

    it('handles empty delta', () => {
      const chunk = {
        type: 'response.output_text.delta' as const,
        item_id: 'msg_123',
        delta: '',
      }

      const parts = convertResponsesAgentChunkToMessagePart(chunk)

      expect(parts).toEqual([
        {
          type: 'text-delta',
          id: 'msg_123',
          delta: '',
          providerMetadata: {
            databricks: { itemId: 'msg_123' },
          },
        },
      ])
    })
  })

  describe('response.reasoning_summary_text.delta events', () => {
    it('converts reasoning delta to reasoning-delta part', () => {
      const chunk = {
        type: 'response.reasoning_summary_text.delta' as const,
        item_id: 'reasoning_123',
        summary_index: 0,
        delta: 'Thinking about this...',
      }

      const parts = convertResponsesAgentChunkToMessagePart(chunk)

      expect(parts).toEqual([
        {
          type: 'reasoning-delta',
          id: 'reasoning_123',
          delta: 'Thinking about this...',
          providerMetadata: {
            databricks: { itemId: 'reasoning_123' },
          },
        },
      ])
    })
  })

  describe('function_call_output events', () => {
    it('converts function call output to tool-result part', () => {
      const chunk = {
        type: 'function_call_output' as const,
        call_id: 'call_123',
        output: '{"result": 42}',
      }

      // Set up tool name mapping
      const toolNamesByCallId = new Map([['call_123', 'calculator']])
      const parts = convertResponsesAgentChunkToMessagePart(chunk, {
        useRemoteToolCalling: true,
        toolNamesByCallId,
      })

      expect(parts).toEqual([
        {
          type: 'tool-result',
          toolCallId: 'call_123',
          result: '{"result": 42}',
          toolName: 'calculator',
        },
      ])
    })
  })

  describe('response.output_item.done events', () => {
    describe('message type', () => {
      it('converts completed message to text-delta part', () => {
        const chunk = {
          type: 'response.output_item.done' as const,
          output_index: 0,
          item: {
            type: 'message' as const,
            role: 'assistant' as const,
            id: 'msg_123',
            content: [
              {
                type: 'output_text' as const,
                text: 'Hello!',
                logprobs: null,
                annotations: [],
              },
            ],
          },
        }

        const parts = convertResponsesAgentChunkToMessagePart(chunk)

        expect(parts).toEqual([
          {
            type: 'text-delta',
            id: 'msg_123',
            delta: 'Hello!',
            providerMetadata: {
              databricks: {
                itemId: 'msg_123',
                itemType: 'response.output_item.done',
              },
            },
          },
        ])
      })
    })

    describe('function_call type', () => {
      it('converts completed function call to tool-call part', () => {
        const chunk = {
          type: 'response.output_item.done' as const,
          output_index: 1,
          item: {
            type: 'function_call' as const,
            call_id: 'call_456',
            name: 'python_exec',
            arguments: '{"code": "print(1)"}',
            id: 'item_789',
          },
        }

        const parts = convertResponsesAgentChunkToMessagePart(chunk, defaultChunkOptions)

        expect(parts).toEqual([
          {
            type: 'tool-call',
            toolCallId: 'call_456',
            toolName: 'python_exec',
            input: '{"code": "print(1)"}',
            dynamic: true,
            providerExecuted: true,
            providerMetadata: {
              databricks: {
                itemId: 'item_789',
              },
            },
          },
        ])
      })
    })

    describe('function_call_output type', () => {
      it('converts completed function call output to tool-result part', () => {
        const chunk = {
          type: 'response.output_item.done' as const,
          output_index: 2,
          item: {
            type: 'function_call_output' as const,
            call_id: 'call_456',
            output: 'Output: 1',
          },
        }

        // Set up tool name mapping
        const toolNamesByCallId = new Map([['call_456', 'python_exec']])
        const parts = convertResponsesAgentChunkToMessagePart(chunk, {
          useRemoteToolCalling: true,
          toolNamesByCallId,
        })

        expect(parts).toEqual([
          {
            type: 'tool-result',
            toolCallId: 'call_456',
            result: 'Output: 1',
            toolName: 'python_exec',
          },
        ])
      })
    })

    describe('reasoning type', () => {
      it('converts completed reasoning to reasoning parts', () => {
        const chunk = {
          type: 'response.output_item.done' as const,
          output_index: 0,
          item: {
            type: 'reasoning' as const,
            id: 'reasoning_123',
            summary: [
              {
                type: 'summary_text' as const,
                text: 'I thought about this carefully.',
              },
            ],
          },
        }

        const parts = convertResponsesAgentChunkToMessagePart(chunk)

        expect(parts).toEqual([
          { type: 'reasoning-start', id: 'reasoning_123' },
          {
            type: 'reasoning-delta',
            id: 'reasoning_123',
            delta: 'I thought about this carefully.',
            providerMetadata: {
              databricks: { itemId: 'reasoning_123' },
            },
          },
          { type: 'reasoning-end', id: 'reasoning_123' },
        ])
      })
    })

    describe('mcp_approval_request type', () => {
      it('converts MCP approval request to tool-call and tool-approval-request parts', () => {
        const chunk = {
          type: 'response.output_item.done' as const,
          output_index: 1,
          item: {
            type: 'mcp_approval_request' as const,
            id: 'mcp_req_123',
            name: 'filesystem_read',
            arguments: '{"path": "/etc/hosts"}',
            server_label: 'fs-server',
          },
        }

        const parts = convertResponsesAgentChunkToMessagePart(chunk, defaultChunkOptions)

        expect(parts).toEqual([
          {
            type: 'tool-call',
            toolCallId: 'mcp_req_123',
            toolName: 'filesystem_read',
            input: '{"path": "/etc/hosts"}',
            dynamic: true,
            providerExecuted: true,
            providerMetadata: {
              databricks: {
                itemId: 'mcp_req_123',
                serverLabel: 'fs-server',
                approvalRequestId: 'mcp_req_123',
              },
            },
          },
          {
            type: 'tool-approval-request',
            approvalId: 'mcp_req_123',
            toolCallId: 'mcp_req_123',
          },
        ])
      })
    })

    describe('mcp_approval_response type', () => {
      it('converts MCP approval response (approved) to tool-result part', () => {
        const chunk = {
          type: 'response.output_item.done' as const,
          output_index: 0,
          item: {
            type: 'mcp_approval_response' as const,
            id: 'mcp_resp_123',
            approval_request_id: 'mcp_req_123',
            approve: true,
            reason: null,
          },
        }

        // Set up tool name mapping for the approval request
        const toolNamesByCallId = new Map([['mcp_req_123', 'filesystem_read']])
        const parts = convertResponsesAgentChunkToMessagePart(chunk, {
          useRemoteToolCalling: true,
          toolNamesByCallId,
        })

        expect(parts).toEqual([
          {
            type: 'tool-result',
            toolCallId: 'mcp_req_123',
            toolName: 'filesystem_read',
            result: { approved: true },
            providerMetadata: {
              databricks: {
                itemId: 'mcp_resp_123',
              },
            },
          },
        ])
      })

      it('converts MCP approval response (denied) to tool-result part', () => {
        const chunk = {
          type: 'response.output_item.done' as const,
          output_index: 0,
          item: {
            type: 'mcp_approval_response' as const,
            id: 'mcp_resp_123',
            approval_request_id: 'mcp_req_123',
            approve: false,
            reason: 'User denied',
          },
        }

        // Set up tool name mapping for the approval request
        const toolNamesByCallId = new Map([['mcp_req_123', 'filesystem_read']])
        const parts = convertResponsesAgentChunkToMessagePart(chunk, {
          useRemoteToolCalling: true,
          toolNamesByCallId,
        })

        expect(parts).toEqual([
          {
            type: 'tool-result',
            toolCallId: 'mcp_req_123',
            toolName: 'filesystem_read',
            result: { approved: false },
            providerMetadata: {
              databricks: {
                itemId: 'mcp_resp_123',
              },
            },
          },
        ])
      })

      it('handles MCP approval response without id', () => {
        const chunk = {
          type: 'response.output_item.done' as const,
          output_index: 0,
          item: {
            type: 'mcp_approval_response' as const,
            approval_request_id: 'mcp_req_123',
            approve: true,
            reason: null,
          },
        }

        // Without tool name mapping, falls back to 'mcp_approval'
        const parts = convertResponsesAgentChunkToMessagePart(chunk, defaultChunkOptions)

        expect(parts).toEqual([
          {
            type: 'tool-result',
            toolCallId: 'mcp_req_123',
            toolName: 'mcp_approval',
            result: { approved: true },
            providerMetadata: {
              databricks: {
                itemId: 'mcp_req_123',
              },
            },
          },
        ])
      })
    })
  })

  describe('response.output_text.annotation.added events', () => {
    it('converts URL citation annotation to source part', () => {
      const chunk = {
        type: 'response.output_text.annotation.added' as const,
        annotation: {
          type: 'url_citation' as const,
          url: 'https://example.com/article',
          title: 'Example Article',
        },
      }

      const parts = convertResponsesAgentChunkToMessagePart(chunk)

      expect(parts).toHaveLength(1)
      expect(parts[0]).toMatchObject({
        type: 'source',
        url: 'https://example.com/article',
        title: 'Example Article',
        sourceType: 'url',
      })
      // id is randomly generated, so just check it exists
      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
      expect((parts[0] as any).id).toBeDefined()
    })
  })

  describe('error events', () => {
    it('converts error chunk to error part', () => {
      const chunk = {
        type: 'error' as const,
        code: 'rate_limit_exceeded',
        message: 'Too many requests',
        param: null,
        sequence_number: 5,
      }

      const parts = convertResponsesAgentChunkToMessagePart(chunk)

      expect(parts).toEqual([
        {
          type: 'error',
          error: chunk,
        },
      ])
    })
  })

})


// ============================================================================
// Tests for convertResponsesAgentResponseToMessagePart
// ============================================================================

describe('convertResponsesAgentResponseToMessagePart', () => {
  describe('message output', () => {
    it('converts message with text content', () => {
      const response = {
        id: 'resp_123',
        output: [
          {
            type: 'message' as const,
            role: 'assistant' as const,
            id: 'msg_123',
            content: [
              {
                type: 'output_text' as const,
                text: 'Hello, how can I help?',
                logprobs: null,
                annotations: [],
              },
            ],
          },
        ],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response)

      expect(parts).toEqual([
        {
          type: 'text',
          text: 'Hello, how can I help?',
          providerMetadata: {
            databricks: { itemId: 'msg_123' },
          },
        },
      ])
    })

    it('converts message with multiple text content items', () => {
      const response = {
        id: 'resp_123',
        output: [
          {
            type: 'message' as const,
            role: 'assistant' as const,
            id: 'msg_123',
            content: [
              {
                type: 'output_text' as const,
                text: 'First part.',
                logprobs: null,
                annotations: [],
              },
              {
                type: 'output_text' as const,
                text: 'Second part.',
                logprobs: null,
                annotations: [],
              },
            ],
          },
        ],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response)

      expect(parts).toEqual([
        {
          type: 'text',
          text: 'First part.',
          providerMetadata: {
            databricks: { itemId: 'msg_123' },
          },
        },
        {
          type: 'text',
          text: 'Second part.',
          providerMetadata: {
            databricks: { itemId: 'msg_123' },
          },
        },
      ])
    })
  })

  describe('function_call output', () => {
    it('converts function call', () => {
      const response = {
        id: 'resp_123',
        output: [
          {
            type: 'function_call' as const,
            call_id: 'call_456',
            name: 'calculator',
            arguments: '{"a": 1, "b": 2}',
            id: 'item_789',
          },
        ],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response, defaultResponseOptions)

      expect(parts).toEqual([
        {
          type: 'tool-call',
          toolCallId: 'call_456',
          toolName: 'calculator',
          input: '{"a": 1, "b": 2}',
          dynamic: true,
          providerExecuted: true,
          providerMetadata: {
            databricks: { itemId: 'item_789' },
          },
        },
      ])
    })
  })

  describe('reasoning output', () => {
    it('converts reasoning with summary', () => {
      const response = {
        id: 'resp_123',
        output: [
          {
            type: 'reasoning' as const,
            id: 'reasoning_123',
            summary: [
              {
                type: 'summary_text' as const,
                text: 'I analyzed the problem.',
              },
            ],
          },
        ],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response)

      expect(parts).toEqual([
        {
          type: 'reasoning',
          text: 'I analyzed the problem.',
          providerMetadata: {
            databricks: { itemId: 'reasoning_123' },
          },
        },
      ])
    })

    it('converts reasoning with multiple summary items', () => {
      const response = {
        id: 'resp_123',
        output: [
          {
            type: 'reasoning' as const,
            id: 'reasoning_123',
            summary: [
              {
                type: 'summary_text' as const,
                text: 'First thought.',
              },
              {
                type: 'summary_text' as const,
                text: 'Second thought.',
              },
            ],
          },
        ],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response)

      expect(parts).toEqual([
        {
          type: 'reasoning',
          text: 'First thought.',
          providerMetadata: {
            databricks: { itemId: 'reasoning_123' },
          },
        },
        {
          type: 'reasoning',
          text: 'Second thought.',
          providerMetadata: {
            databricks: { itemId: 'reasoning_123' },
          },
        },
      ])
    })
  })

  describe('function_call_output output', () => {
    it('converts function call output', () => {
      // Response with both function_call and function_call_output so the name can be looked up
      const response = {
        id: 'resp_123',
        output: [
          {
            type: 'function_call' as const,
            call_id: 'call_456',
            name: 'calculator',
            arguments: '{"a": 1}',
            id: 'item_123',
          },
          {
            type: 'function_call_output' as const,
            call_id: 'call_456',
            output: 'Result: 42',
          },
        ],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response, defaultResponseOptions)

      expect(parts[1]).toEqual({
        type: 'tool-result',
        result: 'Result: 42',
        toolCallId: 'call_456',
        toolName: 'calculator',
      })
    })
  })

  describe('mcp_approval_request output', () => {
    it('converts MCP approval request to tool-call and tool-approval-request', () => {
      const response = {
        id: 'resp_123',
        output: [
          {
            type: 'mcp_approval_request' as const,
            id: 'mcp_req_123',
            name: 'filesystem_read',
            arguments: '{"path": "/etc/hosts"}',
            server_label: 'fs-server',
          },
        ],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response, defaultResponseOptions)

      expect(parts).toEqual([
        {
          type: 'tool-call',
          toolCallId: 'mcp_req_123',
          toolName: 'filesystem_read',
          input: '{"path": "/etc/hosts"}',
          dynamic: true,
          providerExecuted: true,
          providerMetadata: {
            databricks: {
              itemId: 'mcp_req_123',
              serverLabel: 'fs-server',
              approvalRequestId: 'mcp_req_123',
            },
          },
        },
        {
          type: 'tool-approval-request',
          approvalId: 'mcp_req_123',
          toolCallId: 'mcp_req_123',
        },
      ])
    })
  })

  describe('mcp_approval_response output', () => {
    it('converts MCP approval response (approved)', () => {
      // Response with mcp_approval_request first so the name can be looked up
      const response = {
        id: 'resp_123',
        output: [
          {
            type: 'mcp_approval_request' as const,
            id: 'mcp_req_123',
            name: 'filesystem_read',
            arguments: '{"path": "/etc/hosts"}',
            server_label: 'fs-server',
          },
          {
            type: 'mcp_approval_response' as const,
            id: 'mcp_resp_123',
            approval_request_id: 'mcp_req_123',
            approve: true,
            reason: null,
          },
        ],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response, defaultResponseOptions)

      // parts[0] is the tool-call, parts[1] is the tool-approval-request, parts[2] is the tool-result
      expect(parts[2]).toEqual({
        type: 'tool-result',
        toolCallId: 'mcp_req_123',
        toolName: 'filesystem_read',
        result: { approved: true },
        providerMetadata: {
          databricks: {
            itemId: 'mcp_resp_123',
          },
        },
      })
    })

    it('converts MCP approval response (denied)', () => {
      // Response with mcp_approval_request first so the name can be looked up
      const response = {
        id: 'resp_123',
        output: [
          {
            type: 'mcp_approval_request' as const,
            id: 'mcp_req_123',
            name: 'filesystem_read',
            arguments: '{"path": "/etc/hosts"}',
            server_label: 'fs-server',
          },
          {
            type: 'mcp_approval_response' as const,
            id: 'mcp_resp_123',
            approval_request_id: 'mcp_req_123',
            approve: false,
            reason: 'User denied',
          },
        ],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response, defaultResponseOptions)

      // parts[0] is the tool-call, parts[1] is the tool-approval-request, parts[2] is the tool-result
      expect(parts[2]).toEqual({
        type: 'tool-result',
        toolCallId: 'mcp_req_123',
        toolName: 'filesystem_read',
        result: { approved: false },
        providerMetadata: {
          databricks: {
            itemId: 'mcp_resp_123',
          },
        },
      })
    })

    it('handles MCP approval response without id', () => {
      // Without a matching mcp_approval_request, falls back to 'mcp_approval'
      const response = {
        id: 'resp_123',
        output: [
          {
            type: 'mcp_approval_response' as const,
            approval_request_id: 'mcp_req_123',
            approve: true,
            reason: null,
          },
        ],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response, defaultResponseOptions)

      expect(parts).toEqual([
        {
          type: 'tool-result',
          toolCallId: 'mcp_req_123',
          toolName: 'mcp_approval',
          result: { approved: true },
          providerMetadata: {
            databricks: {
              itemId: 'mcp_req_123',
            },
          },
        },
      ])
    })
  })

  describe('mixed output types', () => {
    it('converts response with multiple output types', () => {
      const response = {
        id: 'resp_123',
        output: [
          {
            type: 'message' as const,
            role: 'assistant' as const,
            id: 'msg_123',
            content: [
              {
                type: 'output_text' as const,
                text: 'Let me help you.',
                logprobs: null,
                annotations: [],
              },
            ],
          },
          {
            type: 'function_call' as const,
            call_id: 'call_456',
            name: 'search',
            arguments: '{"query": "test"}',
            id: 'item_789',
          },
        ],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response, defaultResponseOptions)

      expect(parts).toHaveLength(2)
      expect(parts[0]).toEqual({
        type: 'text',
        text: 'Let me help you.',
        providerMetadata: {
          databricks: { itemId: 'msg_123' },
        },
      })
      expect(parts[1]).toEqual({
        type: 'tool-call',
        toolCallId: 'call_456',
        toolName: 'search',
        input: '{"query": "test"}',
        dynamic: true,
        providerExecuted: true,
        providerMetadata: {
          databricks: { itemId: 'item_789' },
        },
      })
    })
  })

  describe('empty output', () => {
    it('returns empty array for empty output', () => {
      const response = {
        id: 'resp_123',
        output: [],
      }

      const parts = convertResponsesAgentResponseToMessagePart(response)

      expect(parts).toEqual([])
    })
  })
})
