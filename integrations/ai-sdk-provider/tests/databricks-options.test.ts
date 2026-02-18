import { describe, it, expect } from 'vitest'
import { callOptionsToResponsesArgs } from '../src/responses-agent-language-model/call-options-to-responses-args'

describe('databricks_options support', () => {
  it('should include databricks_options.return_trace when provided', () => {
    const result = callOptionsToResponsesArgs({
      prompt: [],
      providerOptions: {
        databricks: {
          databricksOptions: {
            return_trace: true,
          },
        },
      },
    })

    expect(result.args.databricks_options).toEqual({
      return_trace: true,
    })
    expect(result.warnings).toEqual([])
  })

  it('should support databricks_options.return_trace set to false', () => {
    const result = callOptionsToResponsesArgs({
      prompt: [],
      providerOptions: {
        databricks: {
          databricksOptions: {
            return_trace: false,
          },
        },
      },
    })

    expect(result.args.databricks_options).toEqual({
      return_trace: false,
    })
  })

  it('should not include databricks_options when not provided', () => {
    const result = callOptionsToResponsesArgs({
      prompt: [],
      providerOptions: {},
    })

    expect(result.args.databricks_options).toBeUndefined()
  })

  it('should not include databricks_options when databricks provider options is empty', () => {
    const result = callOptionsToResponsesArgs({
      prompt: [],
      providerOptions: {
        databricks: {},
      },
    })

    expect(result.args.databricks_options).toBeUndefined()
  })

  it('should work alongside other databricks provider options', () => {
    const result = callOptionsToResponsesArgs({
      prompt: [],
      providerOptions: {
        databricks: {
          parallelToolCalls: true,
          metadata: { key: 'value' },
          databricksOptions: {
            return_trace: true,
          },
        },
      },
    })

    expect(result.args.parallel_tool_calls).toBe(true)
    expect(result.args.metadata).toEqual({ key: 'value' })
    expect(result.args.databricks_options).toEqual({
      return_trace: true,
    })
  })
})
