import type { ChatStatus, LanguageModelUsage, UIMessageChunk } from "ai";
import { useChat as useAiChat, type UseChatHelpers } from "@ai-sdk/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSWRConfig } from "swr";
import { unstable_serialize } from "swr/infinite";
import {
  fetchWithErrorHandlers,
  generateUUID,
  apiUrl,
  isCredentialErrorMessage,
} from "../lib/utils.js";
import { ChatSDKError, type ChatMessage, type FeedbackMap, type VisibilityType } from "../types.js";
import { ChatTransport } from "../lib/transport.js";
import { useChatAgentContext } from "../context.js";
import { getChatHistoryPaginationKey } from "./use-history.js";

export type UseChatReturn = UseChatHelpers<ChatMessage> & {
  id: string;
  title: string | undefined;
  isTitleLoading: boolean;
  isReadonly: boolean;
  feedback: FeedbackMap;
  visibilityType: VisibilityType;
  model: string;
};

export interface UseChatOptions {
  id?: string;
  initialMessages?: ChatMessage[];
  model?: string;
  initialVisibility?: VisibilityType;
  isReadonly?: boolean;
  feedback?: FeedbackMap;
  title?: string;
  onError?: (error: Error) => void;
  onTitleGenerated?: (title: string) => void;
}

export function useChat(options: UseChatOptions = {}): UseChatReturn {
  const {
    id: providedId,
    initialMessages = [],
    model = "chat-model",
    initialVisibility = "private",
    isReadonly = false,
    feedback = {},
    title: externalTitle,
    onError: onErrorCb,
    onTitleGenerated,
  } = options;

  const [id] = useState(() => providedId ?? generateUUID());
  const { chatHistoryEnabled, apiBase, onNavigate } = useChatAgentContext();

  const [visibilityType] = useState<VisibilityType>(initialVisibility);

  const { mutate } = useSWRConfig();

  const [_usage, setUsage] = useState<LanguageModelUsage | undefined>();
  const [lastPart, setLastPart] = useState<UIMessageChunk | undefined>();
  const lastPartRef = useRef<UIMessageChunk | undefined>(lastPart);
  lastPartRef.current = lastPart;

  const resumeAttemptCountRef = useRef(0);
  const maxResumeAttempts = 3;

  const abortController = useRef<AbortController | null>(
    new AbortController(),
  );
  useEffect(() => {
    return () => {
      abortController.current?.abort("ABORT_SIGNAL");
    };
  }, []);

  const fetchWithAbort = useMemo(() => {
    return async (input: RequestInfo | URL, init?: RequestInit) => {
      const signal = abortController.current?.signal;
      return fetchWithErrorHandlers(input, { ...init, signal });
    };
  }, []);

  const stop = useCallback(async () => {
    abortController.current?.abort("USER_ABORT_SIGNAL");
    abortController.current = new AbortController();
  }, []);

  const isNewChat = initialMessages.length === 0;
  const didFetchHistoryOnNewChat = useRef(false);
  const fetchChatHistory = useCallback(() => {
    mutate(
      unstable_serialize((pageIndex: number, previousPageData: unknown) =>
        getChatHistoryPaginationKey(apiBase, pageIndex, previousPageData),
      ),
    );
  }, [mutate, apiBase]);

  const [streamTitle, setStreamTitle] = useState<string | undefined>();
  const [titlePending, setTitlePending] = useState(false);
  const displayTitle = externalTitle ?? streamTitle;

  const chatApiUrl = apiUrl(apiBase, "/");
  const chatResult = useAiChat<ChatMessage>({
    id,
    messages: initialMessages,
    experimental_throttle: 100,
    generateId: generateUUID,
    resume: id !== undefined && initialMessages.length > 0,
    transport: new ChatTransport({
      onStreamPart: (part) => {
        if (isNewChat && !didFetchHistoryOnNewChat.current) {
          fetchChatHistory();
          if (chatHistoryEnabled) {
            setTitlePending(true);
          }
          didFetchHistoryOnNewChat.current = true;

          if (chatHistoryEnabled && onNavigate) {
            onNavigate(id);
          }
        }
        resumeAttemptCountRef.current = 0;
        setLastPart(part);
      },
      api: chatApiUrl,
      fetch: fetchWithAbort,
      prepareSendMessagesRequest({ messages, id: msgId, body }) {
        const lastMessage = messages.at(-1);
        const isUserMessage = lastMessage?.role === "user";
        const needsPreviousMessages =
          !chatHistoryEnabled || !isUserMessage;

        return {
          body: {
            id: msgId,
            ...(isUserMessage ? { message: lastMessage } : {}),
            selectedChatModel: model,
            selectedVisibilityType: visibilityType,
            nextMessageId: generateUUID(),
            ...(needsPreviousMessages
              ? {
                  previousMessages: isUserMessage
                    ? messages.slice(0, -1)
                    : messages,
                }
              : {}),
            ...body,
          },
        };
      },
      prepareReconnectToStreamRequest({ id: streamId }) {
        return {
          api: apiUrl(apiBase, `/${streamId}/stream`),
          credentials: "include",
        };
      },
    }),
    onData: (dataPart) => {
      if (dataPart.type === "data-usage") {
        setUsage(dataPart.data as LanguageModelUsage);
      }
      if (dataPart.type === "data-title") {
        const title = dataPart.data as string;
        setStreamTitle(title);
        setTitlePending(false);
        fetchChatHistory();
        onTitleGenerated?.(title);
      }
    },
    onFinish: ({
      isAbort,
      isDisconnect,
      isError,
      messages: finishedMessages,
    }) => {
      didFetchHistoryOnNewChat.current = false;
      setTitlePending(false);

      if (isAbort) {
        fetchChatHistory();
        return;
      }

      const lastMessage = finishedMessages?.at(-1);
      const hasOAuthError = lastMessage?.parts?.some(
        (part) =>
          part.type === "data-error" &&
          typeof part.data === "string" &&
          isCredentialErrorMessage(part.data),
      );

      if (hasOAuthError) {
        fetchChatHistory();
        chatResult.clearError();
        return;
      }

      const streamIncomplete = lastPartRef.current?.type !== "finish";
      const shouldResume =
        streamIncomplete &&
        (isDisconnect || isError || lastPartRef.current === undefined);

      if (
        shouldResume &&
        resumeAttemptCountRef.current < maxResumeAttempts
      ) {
        resumeAttemptCountRef.current++;
        queueMicrotask(() => {
          chatResult.resumeStream();
        });
      } else {
        if (resumeAttemptCountRef.current >= maxResumeAttempts) {
          console.warn("[useChat] Max resume attempts reached");
        }
        fetchChatHistory();
      }
    },
    onError: (error) => {
      if (error instanceof ChatSDKError) {
        console.warn("[useChat] Chat error:", error.message);
      } else {
        console.warn("[useChat] Error during streaming:", error.message);
      }
      onErrorCb?.(error);
    },
  });

  return {
    ...chatResult,
    stop,
    id,
    title: displayTitle,
    isTitleLoading: titlePending && !displayTitle,
    isReadonly,
    feedback,
    visibilityType,
    model,
  };
}
