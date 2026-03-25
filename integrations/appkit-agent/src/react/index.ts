// Provider
export { ChatAgentProvider, type ChatAgentProviderProps } from "./provider.js";

// Context (for advanced use-cases)
export {
  ChatAgentContext,
  useChatAgentContext,
  type ChatAgentContextValue,
} from "./context.js";

// Hooks
export { useChat, type UseChatOptions, type UseChatReturn } from "./hooks/use-chat.js";
export { useHistory, getChatHistoryPaginationKey } from "./hooks/use-history.js";
export { useSession } from "./hooks/use-session.js";
export { useConfig } from "./hooks/use-config.js";
export { useChatData } from "./hooks/use-chat-data.js";

// Headless components
export {
  Conversation,
  type ConversationProps,
  type ConversationRenderProps,
} from "./headless/conversation.js";
export {
  HistoryList,
  type HistoryListProps,
  type HistoryListRenderProps,
} from "./headless/history-list.js";
export {
  ChatInput,
  type ChatInputProps,
  type ChatInputRenderProps,
} from "./headless/chat-input.js";

// Types
export type {
  ChatMessage,
  ChatFeatures,
  ClientSession,
  Chat,
  Attachment,
  VisibilityType,
  Feedback,
  FeedbackMap,
  ChatHistoryPage,
  ErrorCode,
  CustomUIDataTypes,
} from "./types.js";
export { ChatSDKError } from "./types.js";

// Utilities
export { generateUUID } from "./lib/utils.js";
