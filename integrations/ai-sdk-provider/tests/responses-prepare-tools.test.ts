import { describe, it, expect } from 'vitest'
import type { LanguageModelV3FunctionTool, LanguageModelV3ProviderTool, LanguageModelV3ToolChoice } from '@ai-sdk/provider'
import { prepareResponsesTools } from '../src/responses-agent-language-model/responses-prepare-tools'
import { DATABRICKS_TOOL_CALL_ID } from '../src/tools'

describe('prepareResponsesTools', () => {
  describe('tools conversion', () => {
    it('should convert function tools to responses format', () => {
      const tools: LanguageModelV3FunctionTool[] = [
        {
          type: 'function',
          name: 'get_weather',
          description: 'Get the current weather',
          inputSchema: {
            type: 'object',
            properties: {
              location: { type: 'string' },
            },
            required: ['location'],
          },
        },
      ]

      const result = prepareResponsesTools({ tools })

      expect(result.tools).toHaveLength(1)
      expect(result.tools![0]).toEqual({
        type: 'function',
        name: 'get_weather',
        description: 'Get the current weather',
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string' },
          },
          required: ['location'],
        },
      })
    })

    it('should convert multiple tools', () => {
      const tools: LanguageModelV3FunctionTool[] = [
        {
          type: 'function',
          name: 'tool_a',
          description: 'Tool A',
          inputSchema: { type: 'object' },
        },
        {
          type: 'function',
          name: 'tool_b',
          description: 'Tool B',
          inputSchema: { type: 'object' },
        },
      ]

      const result = prepareResponsesTools({ tools })

      expect(result.tools).toHaveLength(2)
      expect(result.tools![0].name).toBe('tool_a')
      expect(result.tools![1].name).toBe('tool_b')
    })

    it('should handle tools without description', () => {
      const tools: LanguageModelV3FunctionTool[] = [
        {
          type: 'function',
          name: 'simple_tool',
          inputSchema: { type: 'object' },
        },
      ]

      const result = prepareResponsesTools({ tools })

      expect(result.tools).toHaveLength(1)
      expect(result.tools![0]).toEqual({
        type: 'function',
        name: 'simple_tool',
        description: undefined,
        parameters: { type: 'object' },
      })
    })

    it('should return undefined for empty tools array', () => {
      const result = prepareResponsesTools({ tools: [] })

      expect(result.tools).toBeUndefined()
      expect(result.toolChoice).toBeUndefined()
    })

    it('should return undefined for undefined tools', () => {
      const result = prepareResponsesTools({})

      expect(result.tools).toBeUndefined()
      expect(result.toolChoice).toBeUndefined()
    })
  })

  describe('provider-defined tools filtering', () => {
    it('should filter out provider-defined tools', () => {
      const tools: Array<LanguageModelV3FunctionTool | LanguageModelV3ProviderTool> = [
        {
          type: 'function',
          name: 'user_tool',
          inputSchema: { type: 'object' },
        },
        {
          type: 'provider',
          id: 'provider.tool-1',
          name: 'provider-tool-1',
          args: {},
        },
      ]

      const result = prepareResponsesTools({ tools })

      expect(result.tools).toHaveLength(1)
      expect(result.tools![0].name).toBe('user_tool')
    })

    it('should filter out Databricks tool call ID', () => {
      const tools: LanguageModelV3FunctionTool[] = [
        {
          type: 'function',
          name: 'user_tool',
          inputSchema: { type: 'object' },
        },
        {
          type: 'function',
          name: DATABRICKS_TOOL_CALL_ID,
          inputSchema: { type: 'object' },
        },
      ]

      const result = prepareResponsesTools({ tools })

      expect(result.tools).toHaveLength(1)
      expect(result.tools![0].name).toBe('user_tool')
    })

    it('should return undefined if all tools are filtered out', () => {
      const tools: Array<LanguageModelV3FunctionTool | LanguageModelV3ProviderTool> = [
        {
          type: 'provider',
          id: 'provider.tool-1',
          name: 'provider-tool-1',
          args: {},
        },
        {
          type: 'function',
          name: DATABRICKS_TOOL_CALL_ID,
          inputSchema: { type: 'object' },
        },
      ]

      const result = prepareResponsesTools({ tools })

      expect(result.tools).toBeUndefined()
      expect(result.toolChoice).toBeUndefined()
    })
  })

  describe('tool choice conversion', () => {
    const sampleTool: LanguageModelV3FunctionTool = {
      type: 'function',
      name: 'sample_tool',
      inputSchema: { type: 'object' },
    }

    it('should convert "auto" tool choice', () => {
      const toolChoice: LanguageModelV3ToolChoice = { type: 'auto' }

      const result = prepareResponsesTools({ tools: [sampleTool], toolChoice })

      expect(result.toolChoice).toBe('auto')
    })

    it('should convert "none" tool choice', () => {
      const toolChoice: LanguageModelV3ToolChoice = { type: 'none' }

      const result = prepareResponsesTools({ tools: [sampleTool], toolChoice })

      expect(result.toolChoice).toBe('none')
    })

    it('should convert "required" tool choice', () => {
      const toolChoice: LanguageModelV3ToolChoice = { type: 'required' }

      const result = prepareResponsesTools({ tools: [sampleTool], toolChoice })

      expect(result.toolChoice).toBe('required')
    })

    it('should convert specific tool choice', () => {
      const toolChoice: LanguageModelV3ToolChoice = {
        type: 'tool',
        toolName: 'specific_tool',
      }

      const result = prepareResponsesTools({ tools: [sampleTool], toolChoice })

      expect(result.toolChoice).toEqual({
        type: 'function',
        name: 'specific_tool',
      })
    })

    it('should return undefined tool choice when not provided', () => {
      const result = prepareResponsesTools({ tools: [sampleTool] })

      expect(result.toolChoice).toBeUndefined()
    })

    it('should not include tool choice when tools are empty', () => {
      const toolChoice: LanguageModelV3ToolChoice = { type: 'auto' }

      const result = prepareResponsesTools({ tools: [], toolChoice })

      expect(result.tools).toBeUndefined()
      expect(result.toolChoice).toBeUndefined()
    })
  })
})
