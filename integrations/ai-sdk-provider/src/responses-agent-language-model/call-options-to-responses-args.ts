import type { LanguageModelV3CallOptions, SharedV3Warning } from '@ai-sdk/provider'

/**
 * Response body parameters for the Databricks Responses API.
 * Based on: https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#responses-api-request
 */
export type ResponsesBodyArgs = {
  max_output_tokens?: number
  temperature?: number
  top_p?: number
  parallel_tool_calls?: boolean
  metadata?: Record<string, string>
  reasoning?: {
    effort?: 'low' | 'medium' | 'high'
  }
  text?: {
    format?:
      | { type: 'text' }
      | { type: 'json_object' }
      | { type: 'json_schema'; json_schema: unknown }
  }
}

/**
 * Databricks-specific provider options that can be passed via providerOptions.databricks
 */
export type DatabricksProviderOptions = {
  parallelToolCalls?: boolean
  metadata?: Record<string, string>
  reasoning?: {
    effort?: 'low' | 'medium' | 'high'
  }
}

/**
 * Converts AI SDK LanguageModelV3CallOptions to Databricks Responses API body parameters.
 *
 * Inspired by the getArgs method in:
 * https://github.com/vercel/ai/blob/main/packages/openai/src/responses/openai-responses-language-model.ts#L118
 *
 * Complies with the API described in:
 * https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#responses-api-request
 */
export function callOptionsToResponsesArgs(options: LanguageModelV3CallOptions): {
  args: ResponsesBodyArgs
  warnings: SharedV3Warning[]
} {
  const warnings: SharedV3Warning[] = []

  // Extract Databricks-specific provider options
  const databricksOptions = options.providerOptions?.databricks as
    | DatabricksProviderOptions
    | undefined

  // Generate warnings for unsupported options
  if (options.topK != null) {
    warnings.push({
      type: 'unsupported',
      feature: 'topK',
      details: 'topK is not supported by the Databricks Responses API',
    })
  }

  if (options.presencePenalty != null) {
    warnings.push({
      type: 'unsupported',
      feature: 'presencePenalty',
      details: 'presencePenalty is not supported by the Databricks Responses API',
    })
  }

  if (options.frequencyPenalty != null) {
    warnings.push({
      type: 'unsupported',
      feature: 'frequencyPenalty',
      details: 'frequencyPenalty is not supported by the Databricks Responses API',
    })
  }

  if (options.seed != null) {
    warnings.push({
      type: 'unsupported',
      feature: 'seed',
      details: 'seed is not supported by the Databricks Responses API',
    })
  }

  if (options.stopSequences != null && options.stopSequences.length > 0) {
    warnings.push({
      type: 'unsupported',
      feature: 'stopSequences',
      details: 'stopSequences is not supported by the Databricks Responses API',
    })
  }

  // Build the response body args
  const args: ResponsesBodyArgs = {}

  if (options.maxOutputTokens != null) {
    args.max_output_tokens = options.maxOutputTokens
  }

  if (options.temperature != null) {
    args.temperature = options.temperature
  }

  if (options.topP != null) {
    args.top_p = options.topP
  }

  // Handle response format
  if (options.responseFormat != null) {
    switch (options.responseFormat.type) {
      case 'text':
        args.text = { format: { type: 'text' } }
        break
      case 'json':
        if (options.responseFormat.schema != null) {
          args.text = {
            format: {
              type: 'json_schema',
              json_schema: {
                name: options.responseFormat.name ?? 'response',
                description: options.responseFormat.description,
                schema: options.responseFormat.schema as unknown,
                strict: true,
              },
            },
          }
        } else {
          args.text = { format: { type: 'json_object' } }
        }
        break
    }
  }

  // Handle Databricks-specific provider options
  if (databricksOptions?.parallelToolCalls != null) {
    args.parallel_tool_calls = databricksOptions.parallelToolCalls
  }

  if (databricksOptions?.metadata != null) {
    args.metadata = databricksOptions.metadata
  }

  if (databricksOptions?.reasoning != null) {
    args.reasoning = databricksOptions.reasoning
  }

  return { args, warnings }
}
