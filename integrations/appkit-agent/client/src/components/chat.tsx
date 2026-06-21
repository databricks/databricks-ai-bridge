import type { LanguageModelUsage } from 'ai';
import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ChatHeader } from '@/components/chat-header';
import { useChatStream } from '@/hooks/use-chat-stream';
import { softNavigateToChatId } from '@/lib/navigation';
import { useAppConfig } from '@/contexts/AppConfigContext';
import type {
  Attachment,
  ChatMessage,
  FeedbackMap,
  VisibilityType,
  ClientSession,
} from '@/types';
import { Greeting } from './greeting';
import { Messages } from './messages';
import { MultimodalInput } from './multimodal-input';

export function Chat({
  id,
  initialMessages,
  initialChatModel,
  initialVisibilityType,
  isReadonly,
  session: _session,
  initialLastContext: _initialLastContext,
  feedback = {},
  title,
}: {
  id: string;
  initialMessages: ChatMessage[];
  initialChatModel: string;
  initialVisibilityType: VisibilityType;
  isReadonly: boolean;
  session: ClientSession;
  initialLastContext?: LanguageModelUsage;
  feedback?: FeedbackMap;
  title?: string;
}) {
  const { chatHistoryEnabled } = useAppConfig();

  const chat = useChatStream({
    id,
    initialMessages,
    model: initialChatModel,
    initialVisibility: initialVisibilityType,
    isReadonly,
    feedback,
    title,
  });

  const [searchParams] = useSearchParams();
  const query = searchParams.get('query');

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

  const [input, setInput] = useState<string>('');
  const [attachments, setAttachments] = useState<Array<Attachment>>([]);

  const inputElement = <MultimodalInput
    chatId={chat.id}
    input={input}
    setInput={setInput}
    status={chat.status}
    stop={chat.stop}
    attachments={attachments}
    setAttachments={setAttachments}
    messages={chat.messages}
    setMessages={chat.setMessages}
    sendMessage={chat.sendMessage}
    selectedVisibilityType={chat.visibilityType}
  />

  if (chat.messages.length === 0) {
    return (
      <div className="flex h-dvh min-w-0 flex-col bg-background">
        <ChatHeader empty />
        <div className="flex min-h-0 flex-1 overflow-y-auto overscroll-contain touch-pan-y p-4">
          <div className="m-auto flex w-full max-w-4xl flex-col">
            <Greeting />
            {inputElement}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="overscroll-behavior-contain flex h-dvh min-w-0 touch-pan-y flex-col bg-background">
      <ChatHeader title={chat.title} isLoadingTitle={chat.isTitleLoading} />

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
        {!chat.isReadonly && inputElement}
      </div>
    </div>
  );
}
