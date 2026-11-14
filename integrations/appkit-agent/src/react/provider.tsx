import { useEffect, useMemo, type ReactNode } from "react";
import useSWR, { SWRConfig } from "swr";
import { ChatAgentContext, type ChatAgentContextValue } from "./context.js";
import type { ChatFeatures, ClientSession } from "./types.js";

export interface ChatAgentProviderProps {
  /** Base URL for the chat API (e.g. "/api/chat") */
  apiBase?: string;
  /** Base path for client-side routing (e.g. "/chat") */
  basePath?: string;
  /** Feature flags — auto-fetched from the /config endpoint when omitted */
  features?: Partial<ChatFeatures>;
  /** User session — auto-fetched from the /session endpoint when omitted */
  session?: ClientSession;
  /** Called when the library wants to navigate to a chat (e.g. after creating a new one) */
  onNavigate?: (chatId: string) => void;
  children: ReactNode;
}

const jsonFetcher = async (url: string) => {
  const res = await fetch(url, { credentials: "include" });
  if (!res.ok) throw new Error(`Fetch failed: ${url}`);
  return res.json();
};

export function ChatAgentProvider({
  apiBase = "/api/chat",
  basePath = "/",
  features: featuresProp,
  session: sessionProp,
  onNavigate,
  children,
}: ChatAgentProviderProps) {
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

  const value: ChatAgentContextValue = useMemo(
    () => ({
      apiBase,
      basePath,
      features,
      chatHistoryEnabled: features.chatHistory,
      feedbackEnabled: features.feedback,
      session,
      isLoading,
      onNavigate,
    }),
    [apiBase, basePath, features, session, isLoading, onNavigate],
  );

  return (
    <SWRConfig value={{ dedupingInterval: 2000 }}>
      <ChatAgentContext.Provider value={value}>
        {children}
      </ChatAgentContext.Provider>
    </SWRConfig>
  );
}
