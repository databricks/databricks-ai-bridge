import { type ReactNode } from "react";
import { SWRConfig } from "swr";
import {
  ChatAgentContext,
  type ChatAgentContextValue,
} from "../../react/context";

const defaultContext: ChatAgentContextValue = {
  apiBase: "/api/chat",
  basePath: "/",
  features: { chatHistory: true, feedback: false },
  chatHistoryEnabled: true,
  feedbackEnabled: false,
  session: null,
  isLoading: false,
};

export function createWrapper(
  overrides: Partial<ChatAgentContextValue> = {},
) {
  const value = { ...defaultContext, ...overrides };
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <SWRConfig
        value={{
          provider: () => new Map(),
          dedupingInterval: 0,
          shouldRetryOnError: false,
        }}
      >
        <ChatAgentContext.Provider value={value}>
          {children}
        </ChatAgentContext.Provider>
      </SWRConfig>
    );
  };
}
