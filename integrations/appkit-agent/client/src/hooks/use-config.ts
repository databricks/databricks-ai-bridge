import { useChatContext, type ChatFeatures } from '@/contexts/ChatProvider';

/**
 * Public headless hook — returns resolved feature flags and API config
 * from ChatProvider. No UI dependency.
 */
export function useConfig(): {
  apiBase: string;
  basePath: string;
  features: ChatFeatures;
  isLoading: boolean;
} {
  const ctx = useChatContext();
  return {
    apiBase: ctx.apiBase,
    basePath: ctx.basePath,
    features: ctx.features,
    isLoading: ctx.isLoading,
  };
}
