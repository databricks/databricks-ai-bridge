import { createDatabricksProvider } from "@databricks/ai-sdk-provider";
import {
  type LanguageModel,
  extractReasoningMiddleware,
  wrapLanguageModel,
} from "ai";
import { getToken, getHostUrl } from "./auth";

export const CONTEXT_HEADER_CONVERSATION_ID = "x-databricks-conversation-id";
export const CONTEXT_HEADER_USER_ID = "x-databricks-user-id";

type CachedProvider = ReturnType<typeof createDatabricksProvider>;

const providerCache = new Map<
  string,
  { provider: CachedProvider; time: number }
>();
const activePromise = new Map<string, Promise<CachedProvider>>();
const PROVIDER_CACHE_MS = 5 * 60 * 1000;

export interface CreateChatProviderOptions {
  apiProxy?: string;
}

const UTILITY_MODEL = "databricks-meta-llama-3-3-70b-instruct";

/**
 * Returns a provider that can resolve language models for chat.
 * Auth is delegated to `./auth.ts` (PAT → OAuth → CLI).
 */
export function createChatProvider(options: CreateChatProviderOptions = {}): {
  languageModel(id: string): Promise<LanguageModel>;
} {
  const apiProxy = options.apiProxy ?? process.env.API_PROXY;
  const cacheKey = apiProxy ?? "__default__";

  async function getOrCreate(): Promise<CachedProvider> {
    const cached = providerCache.get(cacheKey);
    if (cached && Date.now() - cached.time < PROVIDER_CACHE_MS) {
      return cached.provider;
    }
    const pending = activePromise.get(cacheKey);
    if (pending) return pending;

    const promise = (async (): Promise<CachedProvider> => {
      const host = await getHostUrl();
      const baseURL = `${host.replace(/\/$/, "")}/serving-endpoints`;
      return createDatabricksProvider({
        useRemoteToolCalling: true,
        baseURL,
        formatUrl: ({ baseUrl, path }) =>
          apiProxy
            ? `${apiProxy.replace(/\/$/, "")}${path}`
            : `${baseUrl}${path}`,
        fetch: async (input, init) => {
          const freshToken = await getToken();
          const headers = new Headers(init?.headers);
          headers.set("Authorization", `Bearer ${freshToken}`);
          if (apiProxy) {
            headers.set("x-mlflow-return-trace-id", "true");
          }
          return fetch(input, { ...init, headers });
        },
      });
    })();

    activePromise.set(cacheKey, promise);
    const provider = await promise;
    providerCache.set(cacheKey, { provider, time: Date.now() });
    activePromise.delete(cacheKey);
    return provider;
  }

  return {
    async languageModel(id: string): Promise<LanguageModel> {
      const provider = await getOrCreate();
      let model: LanguageModel;
      if (id === "title-model" || id === "artifact-model") {
        model = provider.chatCompletions(UTILITY_MODEL);
      } else if (apiProxy) {
        model = provider.responses(id);
      } else {
        const endpoint =
          id === "chat-model-reasoning" || id === "chat-model"
            ? (process.env.DATABRICKS_SERVING_ENDPOINT ?? id)
            : id;
        model = provider.chatCompletions(endpoint);
      }
      return wrapLanguageModel({
        model,
        middleware: [extractReasoningMiddleware({ tagName: "think" })],
      });
    },
  };
}
