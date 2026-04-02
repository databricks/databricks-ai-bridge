import { useCallback } from 'react';
import useSWRInfinite from 'swr/infinite';
import { fetcher } from '@/lib/utils';
import { apiUrl } from '@/lib/config';
import { useChatContext } from '@/contexts/ChatProvider';
import type { Chat } from '@/types';
import { toast } from 'sonner';

export interface ChatHistory {
  chats: Array<Chat>;
  hasMore: boolean;
}

const PAGE_SIZE = 20;

export function getChatHistoryPaginationKey(
  pageIndex: number,
  previousPageData: ChatHistory,
) {
  if (previousPageData && previousPageData.hasMore === false) {
    return null;
  }
  if (pageIndex === 0) return apiUrl(`/history?limit=${PAGE_SIZE}`);

  const lastChat = previousPageData.chats.at(-1);
  if (!lastChat) return null;

  return apiUrl(
    `/history?ending_before=${lastChat.id}&limit=${PAGE_SIZE}`,
  );
}

export function useChatHistory() {
  const { chatHistoryEnabled } = useChatContext();

  const {
    data: pages,
    setSize,
    isValidating,
    isLoading,
    mutate,
  } = useSWRInfinite<ChatHistory>(
    chatHistoryEnabled ? getChatHistoryPaginationKey : () => null,
    fetcher,
    { fallbackData: [] },
  );

  const chats = pages?.flatMap((page) => page.chats) ?? [];
  const hasMore = pages ? !pages.some((p) => p.hasMore === false) : true;
  const isEmpty = pages ? pages.every((p) => p.chats.length === 0) : false;

  const loadMore = useCallback(() => {
    if (!isValidating && hasMore) {
      setSize((s) => s + 1);
    }
  }, [isValidating, hasMore, setSize]);

  const deleteChat = useCallback(
    async (chatId: string) => {
      const deletePromise = fetch(apiUrl(`/${chatId}`), { method: 'DELETE' });

      toast.promise(deletePromise, {
        loading: 'Deleting chat...',
        success: () => {
          mutate((histories) =>
            histories?.map((h) => ({
              ...h,
              chats: h.chats.filter((c) => c.id !== chatId),
            })),
          );
          return 'Chat deleted successfully';
        },
        error: 'Failed to delete chat',
      });
    },
    [mutate],
  );

  const renameChat = useCallback(
    async (chatId: string, title: string) => {
      await fetch(apiUrl(`/${chatId}`), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      });
      mutate();
    },
    [mutate],
  );

  return {
    chats,
    isLoading,
    isValidating,
    hasMore,
    isEmpty,
    loadMore,
    deleteChat,
    renameChat,
    mutate,
  };
}
