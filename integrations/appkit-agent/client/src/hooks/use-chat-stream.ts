import type { LanguageModelUsage, UIMessageChunk } from 'ai';
import { useChat } from '@ai-sdk/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSWRConfig } from 'swr';
import { unstable_serialize } from 'swr/infinite';
import { fetchWithErrorHandlers, generateUUID } from '@/lib/utils';
import { ChatSDKError } from '@/types';
import type {
  Attachment,
  ChatMessage,
  FeedbackMap,
  VisibilityType,
} from '@/types';
import { apiUrl } from '@/lib/config';
import { getChatHistoryPaginationKey } from '@/hooks/use-chat-history';
import { toast } from '@/components/toast';
import { useChatVisibility } from '@/hooks/use-chat-visibility';
import { isCredentialErrorMessage } from '@/lib/oauth-error-utils';
import { ChatTransport } from '@/lib/ChatTransport';
import { useChatContext } from '@/contexts/ChatProvider';

export interface UseChatStreamOptions {
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

export function useChatStream(options: UseChatStreamOptions = {}) {
  const {
    id: providedId,
    initialMessages = [],
    model = 'chat-model',
    initialVisibility = 'private',
    isReadonly = false,
    feedback = {},
    title: externalTitle,
    onError: onErrorCb,
    onTitleGenerated,
  } = options;

  const [id] = useState(() => providedId ?? generateUUID());
  const { chatHistoryEnabled } = useChatContext();

  const { visibilityType } = useChatVisibility({
    chatId: id,
    initialVisibilityType: initialVisibility,
  });

  const { mutate } = useSWRConfig();

  const [_usage, setUsage] = useState<LanguageModelUsage | undefined>();
  const [lastPart, setLastPart] = useState<UIMessageChunk | undefined>();
  const lastPartRef = useRef<UIMessageChunk | undefined>(lastPart);
  lastPartRef.current = lastPart;

  const resumeAttemptCountRef = useRef(0);
  const maxResumeAttempts = 3;

  const abortController = useRef<AbortController | null>(new AbortController());
  useEffect(() => {
    return () => {
      abortController.current?.abort('ABORT_SIGNAL');
    };
  }, []);

  const fetchWithAbort = useMemo(() => {
    return async (input: RequestInfo | URL, init?: RequestInit) => {
      const signal = abortController.current?.signal;
      return fetchWithErrorHandlers(input, { ...init, signal });
    };
  }, []);

  const stop = useCallback(() => {
    abortController.current?.abort('USER_ABORT_SIGNAL');
  }, []);

  const isNewChat = initialMessages.length === 0;
  const didFetchHistoryOnNewChat = useRef(false);
  const fetchChatHistory = useCallback(() => {
    mutate(unstable_serialize(getChatHistoryPaginationKey));
  }, [mutate]);

  const [streamTitle, setStreamTitle] = useState<string | undefined>();
  const [titlePending, setTitlePending] = useState(false);
  const displayTitle = externalTitle ?? streamTitle;

  const chatResult = useChat<ChatMessage>({
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
        }
        resumeAttemptCountRef.current = 0;
        setLastPart(part);
      },
      api: apiUrl('/'),
      fetch: fetchWithAbort,
      prepareSendMessagesRequest({ messages, id, body }) {
        const lastMessage = messages.at(-1);
        const isUserMessage = lastMessage?.role === 'user';
        const needsPreviousMessages = !chatHistoryEnabled || !isUserMessage;

        return {
          body: {
            id,
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
      prepareReconnectToStreamRequest({ id }) {
        return {
          api: apiUrl(`/${id}/stream`),
          credentials: 'include',
        };
      },
    }),
    onData: (dataPart) => {
      if (dataPart.type === 'data-usage') {
        setUsage(dataPart.data as LanguageModelUsage);
      }
      if (dataPart.type === 'data-title') {
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
          part.type === 'data-error' &&
          typeof part.data === 'string' &&
          isCredentialErrorMessage(part.data),
      );

      if (hasOAuthError) {
        fetchChatHistory();
        chatResult.clearError();
        return;
      }

      const streamIncomplete = lastPartRef.current?.type !== 'finish';
      const shouldResume =
        streamIncomplete &&
        (isDisconnect || isError || lastPartRef.current === undefined);

      if (shouldResume && resumeAttemptCountRef.current < maxResumeAttempts) {
        resumeAttemptCountRef.current++;
        queueMicrotask(() => {
          chatResult.resumeStream();
        });
      } else {
        if (resumeAttemptCountRef.current >= maxResumeAttempts) {
          console.warn('[useChatStream] Max resume attempts reached');
        }
        fetchChatHistory();
      }
    },
    onError: (error) => {
      if (error instanceof ChatSDKError) {
        toast({ type: 'error', description: error.message });
      } else {
        console.warn('[useChatStream] Error during streaming:', error.message);
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
