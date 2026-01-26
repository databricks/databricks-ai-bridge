import { describe, it, expect } from 'vitest'
import type { LanguageModelV2CallOptions, LanguageModelV2Prompt } from '@ai-sdk/provider'
import { callOptionsToResponsesArgs } from '../src/responses-agent-language-model/call-options-to-responses-args'

const defaultPrompt: LanguageModelV2Prompt = [{ role: 'user', content: [{ type: 'text', text: 'test' }] }]
const createOptions = (overrides: Partial<LanguageModelV2CallOptions> = {}): LanguageModelV2CallOptions => ({
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
    const settings = warnings.map((w) => w.type === 'unsupported-setting' && w.setting)
    expect(settings).toContain('topK')
    expect(settings).toContain('presencePenalty')
    expect(settings).toContain('frequencyPenalty')
    expect(settings).toContain('seed')
    expect(settings).toContain('stopSequences')
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
