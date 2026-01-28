import { describe, it, expect } from 'vitest'
import type { LanguageModelV3CallOptions, LanguageModelV3Prompt } from '@ai-sdk/provider'
import { callOptionsToResponsesArgs } from '../src/responses-agent-language-model/call-options-to-responses-args'

const defaultPrompt: LanguageModelV3Prompt = [{ role: 'user', content: [{ type: 'text', text: 'test' }] }]
const createOptions = (overrides: Partial<LanguageModelV3CallOptions> = {}): LanguageModelV3CallOptions => ({
  prompt: defaultPrompt,
  ...overrides,
})

describe('callOptionsToResponsesArgs', () => {
  it('should convert supported options', () => {
    const { args } = callOptionsToResponsesArgs(createOptions({
      maxOutputTokens: 1024,
      temperature: 0.7,
      topP: 0.9,
    }))

    expect(args.max_output_tokens).toBe(1024)
    expect(args.temperature).toBe(0.7)
    expect(args.top_p).toBe(0.9)
  })

  it('should handle response formats', () => {
    expect(callOptionsToResponsesArgs(createOptions({ responseFormat: { type: 'text' } })).args.text)
      .toEqual({ format: { type: 'text' } })

    expect(callOptionsToResponsesArgs(createOptions({ responseFormat: { type: 'json' } })).args.text)
      .toEqual({ format: { type: 'json_object' } })

    const schema = { type: 'object' as const }
    const { args } = callOptionsToResponsesArgs(createOptions({
      responseFormat: { type: 'json', name: 'test', schema },
    }))
    expect(args.text?.format).toMatchObject({
      type: 'json_schema',
      json_schema: { name: 'test', schema, strict: true },
    })
  })

  it('should convert databricks provider options', () => {
    const { args } = callOptionsToResponsesArgs(createOptions({
      providerOptions: {
        databricks: {
          parallelToolCalls: true,
          metadata: { key: 'value' },
          reasoning: { effort: 'high' },
        },
      },
    }))

    expect(args.parallel_tool_calls).toBe(true)
    expect(args.metadata).toEqual({ key: 'value' })
    expect(args.reasoning).toEqual({ effort: 'high' })
  })

  it('should warn for unsupported options', () => {
    const { warnings } = callOptionsToResponsesArgs(createOptions({
      topK: 10,
      presencePenalty: 0.5,
      frequencyPenalty: 0.3,
      seed: 42,
      stopSequences: ['STOP'],
    }))

    expect(warnings).toHaveLength(5)
    const features = warnings.map((w) => w.type === 'unsupported' && w.feature)
    expect(features).toContain('topK')
    expect(features).toContain('presencePenalty')
    expect(features).toContain('frequencyPenalty')
    expect(features).toContain('seed')
    expect(features).toContain('stopSequences')
  })

  it('should handle zero values correctly', () => {
    const { args } = callOptionsToResponsesArgs(createOptions({
      maxOutputTokens: 0,
      temperature: 0,
    }))

    expect(args.max_output_tokens).toBe(0)
    expect(args.temperature).toBe(0)
  })
})
