import {
  useChat,
  ChatSDKError,
  type ChatMessage,
  type FeedbackMap,
  type VisibilityType,
} from '@databricks/appkit-agent/react';
import { toast } from '@/components/toast';

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
  return useChat({
    ...options,
    onError: (error) => {
      if (error instanceof ChatSDKError) {
        toast({ type: 'error', description: error.message });
      }
      options.onError?.(error);
    },
  });
}
