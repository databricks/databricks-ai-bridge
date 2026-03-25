import { useEffect, type ReactNode } from 'react';
import {
  ChatAgentProvider,
  useChatAgentContext,
  type ChatAgentContextValue,
} from '@databricks/appkit-agent/react';
import { ThemeProvider } from '../components/theme-provider';
import type { ClientSession, ChatFeatures } from '@databricks/appkit-agent/react';

export type { ChatFeatures, ClientSession };

export type ChatContextValue = ChatAgentContextValue;

export interface ChatProviderProps {
  apiBase?: string;
  basePath?: string;
  features?: Partial<ChatFeatures>;
  session?: ClientSession;
  theme?: 'light' | 'dark' | 'system';
  onNavigate?: (chatId: string) => void;
  children: ReactNode;
}

export function ChatProvider({
  apiBase = '/api/chat',
  basePath = '/',
  features,
  session,
  theme = 'system',
  onNavigate,
  children,
}: ChatProviderProps) {
  if (typeof window !== 'undefined') {
    window.__CHAT_CONFIG__ = { apiBase, basePath };
  }

  useEffect(() => {
    window.__CHAT_CONFIG__ = { apiBase, basePath };
  }, [apiBase, basePath]);

  return (
    <ChatAgentProvider
      apiBase={apiBase}
      basePath={basePath}
      features={features}
      session={session}
      onNavigate={onNavigate}
    >
      <ThemeProvider
        attribute="class"
        defaultTheme={theme}
        enableSystem
        disableTransitionOnChange
      >
        {children}
      </ThemeProvider>
    </ChatAgentProvider>
  );
}

export function useChatContext(): ChatContextValue {
  return useChatAgentContext();
}

export { ChatAgentContext as ChatContext } from '@databricks/appkit-agent/react';
