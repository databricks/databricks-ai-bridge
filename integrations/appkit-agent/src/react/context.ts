import { createContext, useContext } from "react";
import type { ChatFeatures, ClientSession } from "./types.js";

export interface ChatAgentContextValue {
  apiBase: string;
  basePath: string;
  features: ChatFeatures;
  chatHistoryEnabled: boolean;
  feedbackEnabled: boolean;
  session: ClientSession | null;
  isLoading: boolean;
  onNavigate?: (chatId: string) => void;
}

export const ChatAgentContext = createContext<
  ChatAgentContextValue | undefined
>(undefined);

export function useChatAgentContext(): ChatAgentContextValue {
  const ctx = useContext(ChatAgentContext);
  if (!ctx) {
    throw new Error(
      "useChatAgentContext must be used within a ChatAgentProvider",
    );
  }
  return ctx;
}
