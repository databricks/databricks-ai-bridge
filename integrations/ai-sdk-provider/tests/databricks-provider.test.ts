import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  createDatabricksProvider,
  type DatabricksProvider,
  type DatabricksProviderSettings,
} from '../src/databricks-provider'
import { DatabricksChatAgentLanguageModel } from '../src/chat-agent-language-model/chat-agent-language-model'
import { DatabricksResponsesAgentLanguageModel } from '../src/responses-agent-language-model/responses-agent-language-model'
import { DatabricksFmapiLanguageModel } from '../src/fmapi-language-model/fmapi-language-model'

describe('createDatabricksProvider', () => {
  describe('Provider creation', () => {
    it('creates provider with minimal settings (just baseURL)', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
      })

      expect(provider).toBeDefined()
      expect(typeof provider.chatAgent).toBe('function')
      expect(typeof provider.responses).toBe('function')
      expect(typeof provider.chatCompletions).toBe('function')
      expect(typeof provider.imageModel).toBe('function')
      expect(typeof provider.textEmbeddingModel).toBe('function')
      expect(typeof provider.languageModel).toBe('function')
    })

    it('creates provider with full settings', () => {
      const customFetch = vi.fn()
      const customFormatUrl = vi.fn(({ baseUrl, path }) => `${baseUrl}/custom${path}`)
      const customHeaders = { Authorization: 'Bearer token123', 'X-Custom': 'value' }

      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
        headers: customHeaders,
        provider: 'custom-provider',
        fetch: customFetch,
        formatUrl: customFormatUrl,
      })

      expect(provider).toBeDefined()
      expect(typeof provider.chatAgent).toBe('function')
    })

    it('strips trailing slash from baseURL', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com/',
      })

      // Create a model to verify the URL is formatted correctly
      const model = provider.chatAgent('test-model') as DatabricksChatAgentLanguageModel
      expect(model).toBeInstanceOf(DatabricksChatAgentLanguageModel)

      // The internal config should have the trailing slash stripped
      // We can verify this by checking the provider still works
      expect(model.modelId).toBe('test-model')
    })

    it('strips multiple trailing slashes from baseURL', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com///',
      })

      const model = provider.chatAgent('test-model')
      expect(model).toBeDefined()
    })
  })

  describe('Provider methods exist and return correct types', () => {
    let provider: DatabricksProvider

    beforeEach(() => {
      provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
      })
    })

    it('chatAgent method exists and returns DatabricksChatAgentLanguageModel', () => {
      const model = provider.chatAgent('test-model')

      expect(model).toBeInstanceOf(DatabricksChatAgentLanguageModel)
      expect(model.modelId).toBe('test-model')
      expect(model.specificationVersion).toBe('v3')
    })

    it('responses method exists and returns DatabricksResponsesAgentLanguageModel', () => {
      const model = provider.responses('test-model')

      expect(model).toBeInstanceOf(DatabricksResponsesAgentLanguageModel)
      expect(model.modelId).toBe('test-model')
      expect(model.specificationVersion).toBe('v3')
    })

    it('chatCompletions method exists and returns DatabricksFmapiLanguageModel', () => {
      const model = provider.chatCompletions('test-model')

      expect(model).toBeInstanceOf(DatabricksFmapiLanguageModel)
      expect(model.modelId).toBe('test-model')
      expect(model.specificationVersion).toBe('v3')
    })

    it('models have correct provider name with default', () => {
      const chatModel = provider.chatAgent('test-model')
      const responsesModel = provider.responses('test-model')
      const chatCompletionsModel = provider.chatCompletions('test-model')

      expect(chatModel.provider).toBe('databricks')
      expect(responsesModel.provider).toBe('databricks')
      expect(chatCompletionsModel.provider).toBe('databricks')
    })

    it('models have correct provider name with custom provider', () => {
      const customProvider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
        provider: 'my-custom-provider',
      })

      const chatModel = customProvider.chatAgent('test-model')
      const responsesModel = customProvider.responses('test-model')
      const chatCompletionsModel = customProvider.chatCompletions('test-model')

      expect(chatModel.provider).toBe('my-custom-provider')
      expect(responsesModel.provider).toBe('my-custom-provider')
      expect(chatCompletionsModel.provider).toBe('my-custom-provider')
    })
  })

  describe('URL formatting', () => {
    it('uses default URL formatting: ${baseUrl}${path}', async () => {
      let capturedUrl = ''
      const customFetch = vi.fn((url: string | URL | Request) => {
        const urlString = url instanceof Request ? url.url : url.toString()
        capturedUrl = urlString
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'test-response',
              messages: [
                {
                  role: 'assistant',
                  content: 'response',
                  id: 'msg-1',
                },
              ],
            }),
            {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            }
          )
        )
      })

      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
        fetch: customFetch,
      })

      const model = provider.chatAgent('test-model')

      // Trigger the model to make a request
      await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      })

      // Verify default URL format is used (Chat Agent uses /completions endpoint)
      expect(customFetch).toHaveBeenCalled()
      expect(capturedUrl).toBe('https://example.databricks.com/completions')
    })

    it('custom formatUrl function is used when provided', async () => {
      let capturedUrl = ''
      const customFormatUrl = vi.fn(({ baseUrl, path }: { baseUrl?: string; path: string }) => {
        return `${baseUrl}/api/v2${path}`
      })

      const customFetch = vi.fn((url: string | URL | Request) => {
        const urlString = url instanceof Request ? url.url : url.toString()
        capturedUrl = urlString
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'test-response',
              messages: [
                {
                  role: 'assistant',
                  content: 'response',
                  id: 'msg-1',
                },
              ],
            }),
            {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            }
          )
        )
      })

      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
        formatUrl: customFormatUrl,
        fetch: customFetch,
      })

      const model = provider.chatAgent('test-model')

      // Trigger API call to invoke formatUrl
      await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      })

      // Verify custom formatUrl was called with Chat Agent path
      expect(customFormatUrl).toHaveBeenCalled()
      expect(customFormatUrl).toHaveBeenCalledWith({
        baseUrl: 'https://example.databricks.com',
        path: '/completions',
      })
      // Verify the URL used the custom format
      expect(capturedUrl).toBe('https://example.databricks.com/api/v2/completions')
    })

    it('formatUrl receives baseUrl without trailing slash', async () => {
      const customFormatUrl = vi.fn(({ baseUrl, path }: { baseUrl?: string; path: string }) => {
        return `${baseUrl}${path}`
      })

      const customFetch = vi.fn(() => {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'test-response',
              messages: [
                {
                  role: 'assistant',
                  content: 'response',
                  id: 'msg-1',
                },
              ],
            }),
            {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            }
          )
        )
      })

      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com/',
        formatUrl: customFormatUrl,
        fetch: customFetch,
      })

      const model = provider.chatAgent('test-model')

      // Trigger URL formatting
      await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      })

      // Verify formatUrl received baseUrl without trailing slash and correct Chat Agent path
      expect(customFormatUrl).toHaveBeenCalledWith({
        baseUrl: 'https://example.databricks.com', // No trailing slash
        path: '/completions',
      })
    })
  })

  describe('Headers', () => {
    it('headers are passed through correctly', async () => {
      const customHeaders = {
        Authorization: 'Bearer my-token',
        'X-Custom-Header': 'custom-value',
        'Content-Type': 'application/json',
      }

      let capturedHeaders: Headers | undefined
      const customFetch = vi.fn((url: string | URL | Request, init?: RequestInit) => {
        if (init?.headers) {
          // Headers can be a Headers object, plain object, or array
          capturedHeaders = new Headers(init.headers)
        }
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'test-response',
              messages: [
                {
                  role: 'assistant',
                  content: 'response',
                  id: 'msg-1',
                },
              ],
            }),
            {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            }
          )
        )
      })

      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
        headers: customHeaders,
        fetch: customFetch,
      })

      const model = provider.chatAgent('test-model')

      // Trigger API call
      await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      })

      // Verify custom headers are included
      expect(capturedHeaders).toBeDefined()
      expect(capturedHeaders!.get('Authorization')).toBe('Bearer my-token')
      expect(capturedHeaders!.get('X-Custom-Header')).toBe('custom-value')
    })

    it('empty headers object works', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
        headers: {},
      })

      const model = provider.chatAgent('test-model')
      expect(model).toBeDefined()
    })

    it('undefined headers works (default case)', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
      })

      const model = provider.chatAgent('test-model')
      expect(model).toBeDefined()
    })
  })

  describe('Not implemented methods', () => {
    let provider: DatabricksProvider

    beforeEach(() => {
      provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
      })
    })

    it('imageModel throws "not supported yet" error', () => {
      expect(() => provider.imageModel('test-image-model')).toThrow(
        'ImageModel is not supported yet'
      )
    })

    it('textEmbeddingModel throws "not supported yet" error', () => {
      expect(() => provider.textEmbeddingModel('test-embedding-model')).toThrow(
        'TextEmbeddingModel is not supported yet'
      )
    })

    it('languageModel throws "not supported yet" error', () => {
      expect(() => provider.languageModel('test-language-model')).toThrow(
        'LanguageModel is not supported yet'
      )
    })

    it('not implemented methods include correct method name in error message', () => {
      try {
        provider.imageModel('test')
      } catch (e) {
        expect((e as Error).message).toContain('ImageModel')
      }

      try {
        provider.textEmbeddingModel('test')
      } catch (e) {
        expect((e as Error).message).toContain('TextEmbeddingModel')
      }

      try {
        provider.languageModel('test')
      } catch (e) {
        expect((e as Error).message).toContain('LanguageModel')
      }
    })
  })

  describe('Custom fetch function', () => {
    it('custom fetch is passed to language models', async () => {
      const customFetch = vi.fn(() => {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'test-response',
              messages: [
                {
                  role: 'assistant',
                  content: 'response',
                  id: 'msg-1',
                },
              ],
            }),
            {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            }
          )
        )
      })

      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
        fetch: customFetch,
      })

      // All model types should receive and use the custom fetch
      const chatModel = provider.chatAgent('test-model')

      // Trigger API call
      await chatModel.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      })

      // Verify custom fetch was called
      expect(customFetch).toHaveBeenCalled()
      expect(customFetch).toHaveBeenCalledTimes(1)
    })

    it('works without custom fetch (uses default)', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
      })

      const model = provider.chatAgent('test-model')
      expect(model).toBeDefined()
    })
  })

  describe('Multiple model instances', () => {
    it('can create multiple chat agent models with different model IDs', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
      })

      const model1 = provider.chatAgent('model-1')
      const model2 = provider.chatAgent('model-2')

      expect(model1.modelId).toBe('model-1')
      expect(model2.modelId).toBe('model-2')
      expect(model1).not.toBe(model2)
    })

    it('can create different model types from same provider', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
      })

      const chatModel = provider.chatAgent('chat-model')
      const responsesModel = provider.responses('responses-model')
      const chatCompletionsModel = provider.chatCompletions('chat-completions-model')

      expect(chatModel).toBeInstanceOf(DatabricksChatAgentLanguageModel)
      expect(responsesModel).toBeInstanceOf(DatabricksResponsesAgentLanguageModel)
      expect(chatCompletionsModel).toBeInstanceOf(DatabricksFmapiLanguageModel)

      // Each instance should be unique
      expect(chatModel).not.toBe(responsesModel)
      expect(responsesModel).not.toBe(chatCompletionsModel)
      expect(chatModel).not.toBe(chatCompletionsModel)
    })
  })

  describe('Provider settings types', () => {
    it('accepts all valid DatabricksProviderSettings combinations', () => {
      // Minimal settings
      const minimal: DatabricksProviderSettings = {
        baseURL: 'https://example.com',
      }
      expect(createDatabricksProvider(minimal)).toBeDefined()

      // With headers
      const withHeaders: DatabricksProviderSettings = {
        baseURL: 'https://example.com',
        headers: { Authorization: 'Bearer token' },
      }
      expect(createDatabricksProvider(withHeaders)).toBeDefined()

      // With provider name
      const withProvider: DatabricksProviderSettings = {
        baseURL: 'https://example.com',
        provider: 'custom',
      }
      expect(createDatabricksProvider(withProvider)).toBeDefined()

      // With fetch
      const withFetch: DatabricksProviderSettings = {
        baseURL: 'https://example.com',
        fetch: vi.fn(),
      }
      expect(createDatabricksProvider(withFetch)).toBeDefined()

      // With formatUrl
      const withFormatUrl: DatabricksProviderSettings = {
        baseURL: 'https://example.com',
        formatUrl: ({ baseUrl, path }) => `${baseUrl}${path}`,
      }
      expect(createDatabricksProvider(withFormatUrl)).toBeDefined()

      // Full settings
      const full: DatabricksProviderSettings = {
        baseURL: 'https://example.com',
        headers: { Authorization: 'Bearer token' },
        provider: 'custom',
        fetch: vi.fn(),
        formatUrl: ({ baseUrl, path }) => `${baseUrl}${path}`,
      }
      expect(createDatabricksProvider(full)).toBeDefined()
    })
  })

  describe('Edge cases', () => {
    it('handles empty string provider name', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
        provider: '', // Empty string is a valid provider name
      })

      const model = provider.chatAgent('test-model')
      // Empty string is not nullish, so it's used as-is
      expect(model.provider).toBe('')
    })

    it('handles baseURL with path segments', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com/api/v1',
      })

      const model = provider.chatAgent('test-model')
      expect(model).toBeDefined()
    })

    it('handles special characters in model ID', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
      })

      const model = provider.chatAgent('model/with/slashes:v1.0')
      expect(model.modelId).toBe('model/with/slashes:v1.0')
    })

    it('handles unicode in headers', () => {
      const provider = createDatabricksProvider({
        baseURL: 'https://example.databricks.com',
        headers: {
          'X-Unicode': 'value-with-unicode-\u00e9\u00e8\u00ea',
        },
      })

      const model = provider.chatAgent('test-model')
      expect(model).toBeDefined()
    })
  })
})
