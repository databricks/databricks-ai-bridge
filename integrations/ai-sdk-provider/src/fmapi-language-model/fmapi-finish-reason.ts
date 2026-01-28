import { LanguageModelV3FinishReason } from '@ai-sdk/provider'

export function mapFmapiFinishReason(
  finishReason: string | null | undefined
): LanguageModelV3FinishReason {
  switch (finishReason) {
    case 'stop':
      return {
        raw: finishReason,
        unified: 'stop',
      }
    case 'length':
      return {
        raw: finishReason,
        unified: 'length',
      }
    case 'content_filter':
      return {
        raw: finishReason,
        unified: 'content-filter',
      }
    case 'function_call':
    case 'tool_calls':
      return {
        raw: finishReason,
        unified: 'tool-calls',
      }
    default:
      return {
        raw: finishReason ?? undefined,
        unified: 'other',
      }
  }
}
