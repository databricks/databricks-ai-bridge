import { useChatAgentContext } from "../context.js";
import type { ChatFeatures } from "../types.js";

export function useConfig(): {
  apiBase: string;
  basePath: string;
  features: ChatFeatures;
  isLoading: boolean;
} {
  const ctx = useChatAgentContext();
  return {
    apiBase: ctx.apiBase,
    basePath: ctx.basePath,
    features: ctx.features,
    isLoading: ctx.isLoading,
  };
}
