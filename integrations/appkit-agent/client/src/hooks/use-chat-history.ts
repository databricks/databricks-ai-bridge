import {
  useHistory,
  getChatHistoryPaginationKey as _getChatHistoryPaginationKey,
} from '@databricks/appkit-agent/react';
import { useChatContext } from '@/contexts/ChatProvider';
import { toast } from 'sonner';
import { apiUrl } from '@/lib/config';
import type { ChatHistoryPage } from '@databricks/appkit-agent/react';

export type ChatHistory = ChatHistoryPage;

export function getChatHistoryPaginationKey(
  pageIndex: number,
  previousPageData: ChatHistory,
) {
  const config = window.__CHAT_CONFIG__ ?? { apiBase: '/api/chat', basePath: '/' };
  return _getChatHistoryPaginationKey(config.apiBase, pageIndex, previousPageData);
}

export function useChatHistory() {
  const history = useHistory();

  const deleteChatWithToast = async (chatId: string) => {
    const deletePromise = history.deleteChat(chatId);

    toast.promise(deletePromise, {
      loading: 'Deleting chat...',
      success: 'Chat deleted successfully',
      error: 'Failed to delete chat',
    });
  };

  return {
    ...history,
    deleteChat: deleteChatWithToast,
  };
}
