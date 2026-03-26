import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { ReactNode } from 'react';
import { useChatStream, type UseChatStreamOptions } from '@/hooks/use-chat-stream';
import { ChatHeader, type ChatHeaderProps } from './chat-header';
import { Messages } from './messages';
import { ChatComposer } from './ChatComposer';
import { Greeting } from './greeting';
import { softNavigateToChatId } from '@/lib/navigation';
import { useChatContext } from '@/contexts/ChatProvider';
import type { ChatMessage, FeedbackMap, VisibilityType, ClientSession } from '@/types';

export interface ChatPanelProps extends UseChatStreamOptions {
  /** Render a custom header. Receives default props so you can spread or override. */
  renderHeader?: (props: ChatHeaderProps) => ReactNode;
  /** Render a custom composer. Receives the chat state so you can build your own input. */
  renderComposer?: (props: {
    chatId: string;
    status: string;
    messages: ChatMessage[];
    setMessages: any;
    sendMessage: any;
    stop: () => void;
    visibilityType: VisibilityType;
  }) => ReactNode;
  /** @deprecated Legacy prop — pass session via ChatProvider instead */
  session?: ClientSession;
}

/**
 * Pre-assembled chat panel that wires useChatStream with ChatHeader,
 * Messages, and ChatComposer. Drop this into a ChatProvider and you
 * get a fully functional chat. Supports render-prop slots for header
 * and composer customization.
 */
export function ChatPanel({
  renderHeader,
  renderComposer,
  ...streamOptions
}: ChatPanelProps) {
  const { chatHistoryEnabled } = useChatContext();

  const chat = useChatStream(streamOptions);

  // Handle query param auto-send (for ?query=... deep-links)
  let searchParams: URLSearchParams | undefined;
  try {
    [searchParams] = useSearchParams();
  } catch {
    // Not inside a router — skip query param handling
  }

  const query = searchParams?.get('query');
  const [hasAppendedQuery, setHasAppendedQuery] = useState(false);

  useEffect(() => {
    if (query && !hasAppendedQuery) {
      chat.sendMessage({
        role: 'user' as const,
        parts: [{ type: 'text', text: query }],
      });
      setHasAppendedQuery(true);
      softNavigateToChatId(chat.id, chatHistoryEnabled);
    }
  }, [query, chat.sendMessage, hasAppendedQuery, chat.id, chatHistoryEnabled]);

  const headerProps: ChatHeaderProps = {
    title: chat.title,
    isLoadingTitle: chat.isTitleLoading,
    empty: chat.messages.length === 0,
  };

  const header = renderHeader ? renderHeader(headerProps) : <ChatHeader {...headerProps} />;

  const composerProps = {
    chatId: chat.id,
    status: chat.status,
    messages: chat.messages,
    setMessages: chat.setMessages,
    sendMessage: chat.sendMessage,
    stop: chat.stop,
    visibilityType: chat.visibilityType,
  };

  const composer = renderComposer
    ? renderComposer(composerProps)
    : <ChatComposer {...composerProps} />;

  if (chat.messages.length === 0) {
    return (
      <div className="flex h-dvh min-w-0 flex-col bg-background">
        <ChatHeader empty />
        <div className="flex min-h-0 flex-1 overflow-y-auto overscroll-contain touch-pan-y p-4">
          <div className="m-auto flex w-full max-w-4xl flex-col">
            <Greeting />
            {composer}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="overscroll-behavior-contain flex h-dvh min-w-0 touch-pan-y flex-col bg-background">
      {header}

      <Messages
        status={chat.status}
        messages={chat.messages}
        setMessages={chat.setMessages}
        addToolApprovalResponse={chat.addToolApprovalResponse}
        regenerate={chat.regenerate}
        sendMessage={chat.sendMessage}
        isReadonly={chat.isReadonly}
        selectedModelId={chat.model}
        feedback={chat.feedback}
      />

      <div className="sticky bottom-0 z-1 mx-auto flex w-full max-w-4xl gap-2 border-t-0 bg-background px-2 pb-3 md:px-4 md:pb-4">
        {!chat.isReadonly && composer}
      </div>
    </div>
  );
}
