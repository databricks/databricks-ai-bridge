import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  type ReactNode,
} from 'react';
import useSWR from 'swr';
import { ThemeProvider } from '../components/theme-provider';
import type { ClientSession } from '@/types';

export interface ChatFeatures {
  chatHistory: boolean;
  feedback: boolean;
}

export interface ChatProviderProps {
  /** Base URL for the chat API (e.g. "/api/chat") */
  apiBase?: string;
  /** Base path for client-side routing (e.g. "/chat") */
  basePath?: string;
  /** Feature flags — auto-fetched from the /config endpoint when omitted */
  features?: Partial<ChatFeatures>;
  /** User session — auto-fetched from the /session endpoint when omitted */
  session?: ClientSession;
  /** Theme preference (default: "system") */
  theme?: 'light' | 'dark' | 'system';
  children: ReactNode;
}

export interface ChatContextValue {
  apiBase: string;
  basePath: string;
  features: ChatFeatures;
  chatHistoryEnabled: boolean;
  feedbackEnabled: boolean;
  session: ClientSession | null;
  isLoading: boolean;
}

export const ChatContext = createContext<ChatContextValue | undefined>(
  undefined,
);

const jsonFetcher = async (url: string) => {
  const res = await fetch(url, { credentials: 'include' });
  if (!res.ok) throw new Error(`Fetch failed: ${url}`);
  return res.json();
};

export function ChatProvider({
  apiBase = '/api/chat',
  basePath = '/',
  features: featuresProp,
  session: sessionProp,
  theme = 'system',
  children,
}: ChatProviderProps) {
  // Synchronously set window config so apiUrl() works from the first child render
  if (typeof window !== 'undefined') {
    window.__CHAT_CONFIG__ = { apiBase, basePath };
  }

  const shouldFetchConfig = featuresProp === undefined;
  const { data: serverConfig, isLoading: configLoading } = useSWR<{
    features: ChatFeatures;
  }>(shouldFetchConfig ? `${apiBase}/config` : null, jsonFetcher, {
    revalidateOnFocus: false,
    revalidateOnReconnect: false,
    dedupingInterval: 60000,
  });

  const shouldFetchSession = sessionProp === undefined;
  const { data: serverSession, isLoading: sessionLoading } =
    useSWR<ClientSession>(
      shouldFetchSession ? `${apiBase}/session` : null,
      jsonFetcher,
      { revalidateOnFocus: false },
    );

  const features: ChatFeatures = useMemo(
    () => ({
      chatHistory:
        featuresProp?.chatHistory ??
        serverConfig?.features?.chatHistory ??
        true,
      feedback:
        featuresProp?.feedback ?? serverConfig?.features?.feedback ?? false,
    }),
    [featuresProp, serverConfig],
  );

  const session = sessionProp ?? serverSession ?? null;
  const isLoading =
    (shouldFetchConfig && configLoading) ||
    (shouldFetchSession && sessionLoading);

  const value: ChatContextValue = useMemo(
    () => ({
      apiBase,
      basePath,
      features,
      chatHistoryEnabled: features.chatHistory,
      feedbackEnabled: features.feedback,
      session,
      isLoading,
    }),
    [apiBase, basePath, features, session, isLoading],
  );

  return (
    <ThemeProvider
      attribute="class"
      defaultTheme={theme}
      enableSystem
      disableTransitionOnChange
    >
      <ChatContext.Provider value={value}>{children}</ChatContext.Provider>
    </ThemeProvider>
  );
}

export function useChatContext(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) {
    throw new Error('useChatContext must be used within a ChatProvider');
  }
  return ctx;
}
