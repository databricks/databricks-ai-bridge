import { describe, it, expect } from 'vitest'
import type { LanguageModelV2CallOptions, LanguageModelV2Prompt } from '@ai-sdk/provider'
import { callOptionsToFmapiArgs } from '../src/fmapi-language-model/call-options-to-fmapi-args'

const defaultPrompt: LanguageModelV2Prompt = [{ role: 'user', content: [{ type: 'text', text: 'test' }] }]
const createOptions = (overrides: Partial<LanguageModelV2CallOptions> = {}): LanguageModelV2CallOptions => ({
  prompt: defaultPrompt,
  ...overrides,
})

describe('callOptionsToFmapiArgs', () => {
  it('should convert supported options', () => {
    const { args } = callOptionsToFmapiArgs(createOptions({
      maxOutputTokens: 1024,
      temperature: 0.7,
      topP: 0.9,
      topK: 50,
      stopSequences: ['STOP', 'END'],
    }))

    expect(args.max_tokens).toBe(1024)
    expect(args.temperature).toBe(0.7)
    expect(args.top_p).toBe(0.9)
    expect(args.top_k).toBe(50)
    expect(args.stop).toEqual(['STOP', 'END'])
  })

  it('should handle response formats', () => {
    expect(callOptionsToFmapiArgs(createOptions({ responseFormat: { type: 'text' } })).args.response_format)
      .toEqual({ type: 'text' })

    expect(callOptionsToFmapiArgs(createOptions({ responseFormat: { type: 'json' } })).args.response_format)
      .toEqual({ type: 'json_object' })

    const schema = { type: 'object' as const }
    const { args } = callOptionsToFmapiArgs(createOptions({
      responseFormat: { type: 'json', name: 'test', schema },
    }))
    expect(args.response_format).toMatchObject({
      type: 'json_schema',
      json_schema: { name: 'test', schema, strict: true },
    })
  })

  it('should convert databricks provider options', () => {
    const { args } = callOptionsToFmapiArgs(createOptions({
      providerOptions: {
        databricks: {
          topK: 50,
          n: 2,
          logprobs: true,
          topLogprobs: 5,
          reasoningEffort: 'high',
        },
      },
    }))

    expect(args.top_k).toBe(50)
    expect(args.n).toBe(2)
    expect(args.logprobs).toBe(true)
    expect(args.top_logprobs).toBe(5)
    expect(args.reasoning_effort).toBe('high')
  })

  it('should warn for unsupported options', () => {
    const { warnings } = callOptionsToFmapiArgs(createOptions({
      presencePenalty: 0.5,
      frequencyPenalty: 0.3,
      seed: 42,
    }))

    expect(warnings).toHaveLength(3)
    const settings = warnings.map((w) => w.type === 'unsupported-setting' && w.setting)
    expect(settings).toContain('presencePenalty')
    expect(settings).toContain('frequencyPenalty')
    expect(settings).toContain('seed')
  })
})
