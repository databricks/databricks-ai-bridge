import { useState } from 'react';
import { MultimodalInput } from './multimodal-input';
import type { UseChatHelpers } from '@ai-sdk/react';
import type { Attachment, ChatMessage, VisibilityType } from '@/types';
import type { UIMessage } from 'ai';

export interface ChatComposerProps {
  chatId: string;
  status: UseChatHelpers<ChatMessage>['status'];
  messages: Array<UIMessage>;
  setMessages: UseChatHelpers<ChatMessage>['setMessages'];
  sendMessage: UseChatHelpers<ChatMessage>['sendMessage'];
  stop: () => void;
  visibilityType?: VisibilityType;
}

/**
 * Composable chat input area — wraps MultimodalInput with a clean,
 * self-contained API. Manages its own input text and attachment state.
 */
export function ChatComposer({
  chatId,
  status,
  messages,
  setMessages,
  sendMessage,
  stop,
  visibilityType = 'private',
}: ChatComposerProps) {
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<Array<Attachment>>([]);

  return (
    <MultimodalInput
      chatId={chatId}
      input={input}
      setInput={setInput}
      status={status}
      stop={stop}
      attachments={attachments}
      setAttachments={setAttachments}
      messages={messages}
      setMessages={setMessages}
      sendMessage={sendMessage}
      selectedVisibilityType={visibilityType}
    />
  );
}
