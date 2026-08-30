import { createContext, useContext, type ReactNode } from 'react';
import useSWR from 'swr';
import { fetcher } from '@/lib/utils';
import { apiUrl } from '@/lib/config';
import { ChatContext } from './ChatProvider';

interface ConfigResponse {
  features: {
    chatHistory: boolean;
    feedback: boolean;
  };
}

interface AppConfigContextType {
  config: ConfigResponse | undefined;
  isLoading: boolean;
  error: Error | undefined;
  chatHistoryEnabled: boolean;
  feedbackEnabled: boolean;
}

const AppConfigContext = createContext<AppConfigContextType | undefined>(
  undefined,
);

export function AppConfigProvider({ children }: { children: ReactNode }) {
  const { data, error, isLoading } = useSWR<ConfigResponse>(
    apiUrl('/config'),
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
      dedupingInterval: 60000,
    },
  );

  const value: AppConfigContextType = {
    config: data,
    isLoading,
    error,
    chatHistoryEnabled: data?.features.chatHistory ?? true,
    feedbackEnabled: data?.features.feedback ?? false,
  };

  return (
    <AppConfigContext.Provider value={value}>
      {children}
    </AppConfigContext.Provider>
  );
}

/**
 * Read feature-flag config. Transparently delegates to ChatProvider when
 * present in the tree, otherwise falls back to the legacy AppConfigContext.
 */
export function useAppConfig(): AppConfigContextType {
  const chatCtx = useContext(ChatContext);
  if (chatCtx) {
    return {
      config: { features: chatCtx.features },
      isLoading: chatCtx.isLoading,
      error: undefined,
      chatHistoryEnabled: chatCtx.chatHistoryEnabled,
      feedbackEnabled: chatCtx.feedbackEnabled,
    };
  }

  const legacy = useContext(AppConfigContext);
  if (!legacy) {
    throw new Error(
      'useAppConfig must be used within a ChatProvider or AppConfigProvider',
    );
  }
  return legacy;
}
