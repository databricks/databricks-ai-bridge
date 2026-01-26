import type { LanguageModelV2CallOptions, LanguageModelV2CallWarning } from '@ai-sdk/provider'

/**
 * Response body parameters for the Databricks FMAPI (Chat Completions) API.
 * Based on: https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#chat-request
 */
export type FmapiBodyArgs = {
  max_tokens?: number
  temperature?: number
  top_p?: number
  top_k?: number
  stop?: string[]
  n?: number
  response_format?: { type: 'text' } | { type: 'json_object' } | { type: 'json_schema'; json_schema: unknown }
  logprobs?: boolean
  top_logprobs?: number
  reasoning_effort?: 'low' | 'medium' | 'high'
}

/**
 * Databricks-specific provider options for FMAPI
 */
export type DatabricksFmapiProviderOptions = {
  topK?: number
  n?: number
  logprobs?: boolean
  topLogprobs?: number
  reasoningEffort?: 'low' | 'medium' | 'high'
}

/**
 * Converts AI SDK LanguageModelV2CallOptions to Databricks FMAPI body parameters.
 *
 * Inspired by the getArgs method in:
 * https://github.com/vercel/ai/blob/main/packages/openai/src/chat/openai-chat-language-model.ts#L71
 *
 * Complies with the API described in:
 * https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#chat-request
 */
export function callOptionsToFmapiArgs(options: LanguageModelV2CallOptions): {
  args: FmapiBodyArgs
  warnings: LanguageModelV2CallWarning[]
} {
  const warnings: LanguageModelV2CallWarning[] = []

  const databricksOptions = options.providerOptions?.databricks as DatabricksFmapiProviderOptions | undefined

  // Generate warnings for unsupported options
  if (options.presencePenalty != null) {
    warnings.push({
      type: 'unsupported-setting',
      setting: 'presencePenalty',
      details: 'presencePenalty is not supported by the Databricks FMAPI',
    })
  }

  if (options.frequencyPenalty != null) {
    warnings.push({
      type: 'unsupported-setting',
      setting: 'frequencyPenalty',
      details: 'frequencyPenalty is not supported by the Databricks FMAPI',
    })
  }

  if (options.seed != null) {
    warnings.push({
      type: 'unsupported-setting',
      setting: 'seed',
      details: 'seed is not supported by the Databricks FMAPI',
    })
  }

  // Build the args
  const args: FmapiBodyArgs = {}

  if (options.maxOutputTokens != null) {
    args.max_tokens = options.maxOutputTokens
  }

  if (options.temperature != null) {
    args.temperature = options.temperature
  }

  if (options.topP != null) {
    args.top_p = options.topP
  }

  if (options.topK != null) {
    args.top_k = options.topK
  }

  if (options.stopSequences != null && options.stopSequences.length > 0) {
    args.stop = options.stopSequences
  }

  // Handle response format
  if (options.responseFormat != null) {
    switch (options.responseFormat.type) {
      case 'text':
        args.response_format = { type: 'text' }
        break
      case 'json':
        if (options.responseFormat.schema != null) {
          args.response_format = {
            type: 'json_schema',
            json_schema: {
              name: options.responseFormat.name ?? 'response',
              description: options.responseFormat.description,
              schema: options.responseFormat.schema as unknown,
              strict: true,
            },
          }
        } else {
          args.response_format = { type: 'json_object' }
        }
        break
    }
  }

  // Handle Databricks-specific provider options (can override standard options)
  if (databricksOptions?.topK != null) {
    args.top_k = databricksOptions.topK
  }

  if (databricksOptions?.n != null) {
    args.n = databricksOptions.n
  }

  if (databricksOptions?.logprobs != null) {
    args.logprobs = databricksOptions.logprobs
  }

  if (databricksOptions?.topLogprobs != null) {
    args.top_logprobs = databricksOptions.topLogprobs
  }

  if (databricksOptions?.reasoningEffort != null) {
    args.reasoning_effort = databricksOptions.reasoningEffort
  }

  return { args, warnings }
}