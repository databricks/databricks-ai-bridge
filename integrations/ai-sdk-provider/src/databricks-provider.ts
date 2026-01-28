import type { LanguageModelV3, ProviderV3 } from '@ai-sdk/provider'
import { combineHeaders, type FetchFunction, withoutTrailingSlash } from '@ai-sdk/provider-utils'
import { DatabricksChatAgentLanguageModel } from './chat-agent-language-model/chat-agent-language-model'
import { DatabricksResponsesAgentLanguageModel } from './responses-agent-language-model/responses-agent-language-model'
import { DatabricksFmapiLanguageModel } from './fmapi-language-model/fmapi-language-model'

export type DatabricksLanguageModelConfig = {
  provider: string
  headers: () => Record<string, string | undefined>
  url: (options: { path: string }) => string
  fetch?: FetchFunction
}

export interface DatabricksProvider extends ProviderV3 {
  /** Agents */
  chatAgent(modelId: string): LanguageModelV3 // agent/v2/chat
  responses(modelId: string): LanguageModelV3 // agent/v1/responses

  /** Foundation Models */
  chatCompletions(modelId: string): LanguageModelV3 // llm/v1/chat

  /** Required overrides from ProviderV3 */
  textEmbeddingModel(modelId: string): never
  languageModel(modelId: string): never
}

export interface DatabricksProviderSettings {
  /** Base URL for the Databricks API calls. */
  baseURL: string
  /** Custom headers to include in the requests. */
  headers?: Record<string, string>
  /** Provider name. Overrides the `databricks` default name for 3rd party providers. */
  provider?: string

  /**
   * Custom fetch implementation. You can use it as a middleware to intercept requests,
   * or to provide a custom fetch implementation for e.g. testing.
   * */
  fetch?: FetchFunction

  /**
   * Optional function to format the URL
   */
  formatUrl?: (options: { baseUrl?: string; path: string }) => string
}

export const createDatabricksProvider = (
  settings: DatabricksProviderSettings
): DatabricksProvider => {
  const baseUrl = withoutTrailingSlash(settings.baseURL)
  const getHeaders = () => combineHeaders(settings.headers)
  const fetch = settings.fetch
  const provider = settings.provider ?? 'databricks'

  const formatUrl = ({ path }: { path: string }) =>
    settings.formatUrl?.({ baseUrl, path }) ?? `${baseUrl}${path}`

  const createChatAgent = (modelId: string): LanguageModelV3 =>
    new DatabricksChatAgentLanguageModel(modelId, {
      url: formatUrl,
      headers: getHeaders,
      fetch,
      provider,
    })

  const createResponsesAgent = (modelId: string): LanguageModelV3 =>
    new DatabricksResponsesAgentLanguageModel(modelId, {
      url: formatUrl,
      headers: getHeaders,
      fetch,
      provider,
    })

  const createFmapi = (modelId: string): LanguageModelV3 =>
    new DatabricksFmapiLanguageModel(modelId, {
      url: formatUrl,
      headers: getHeaders,
      fetch,
      provider,
    })

  const notImplemented = (name: string) => {
    return () => {
      throw new Error(`${name} is not supported yet`)
    }
  }

  return {
    specificationVersion: 'v3' as const,
    responses: createResponsesAgent,
    chatCompletions: createFmapi,
    chatAgent: createChatAgent,
    imageModel: notImplemented('ImageModel'),
    textEmbeddingModel: notImplemented('TextEmbeddingModel'),
    embeddingModel: notImplemented('EmbeddingModel'),
    languageModel: notImplemented('LanguageModel'),
  }
}
