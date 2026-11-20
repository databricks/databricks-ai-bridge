import { useCallback } from "react";
import useSWRInfinite from "swr/infinite";
import { fetcher, apiUrl } from "../lib/utils.js";
import { useChatAgentContext } from "../context.js";
import type { ChatHistoryPage } from "../types.js";

const PAGE_SIZE = 20;

export function getChatHistoryPaginationKey(
  apiBase: string,
  pageIndex: number,
  previousPageData: unknown,
) {
  const prev = previousPageData as ChatHistoryPage | undefined;
  if (prev && prev.hasMore === false) {
    return null;
  }
  if (pageIndex === 0) return apiUrl(apiBase, `/history?limit=${PAGE_SIZE}`);

  const lastChat = prev?.chats.at(-1);
  if (!lastChat) return null;

  return apiUrl(
    apiBase,
    `/history?ending_before=${lastChat.id}&limit=${PAGE_SIZE}`,
  );
}

export function useHistory() {
  const { chatHistoryEnabled, apiBase } = useChatAgentContext();

  const {
    data: pages,
    setSize,
    isValidating,
    isLoading,
    mutate,
  } = useSWRInfinite<ChatHistoryPage>(
    chatHistoryEnabled
      ? (pageIndex: number, previousPageData: unknown) =>
          getChatHistoryPaginationKey(apiBase, pageIndex, previousPageData)
      : () => null,
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
      const response = await fetch(apiUrl(apiBase, `/${chatId}`), { method: "DELETE" });
      if (!response.ok) {
        throw new Error(`Failed to delete chat: ${response.status}`);
      }
      mutate((histories) =>
        histories?.map((h) => ({
          ...h,
          chats: h.chats.filter((c) => c.id !== chatId),
        })),
      );
    },
    [mutate, apiBase],
  );

  const renameChat = useCallback(
    async (chatId: string, title: string) => {
      const response = await fetch(apiUrl(apiBase, `/${chatId}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (!response.ok) {
        throw new Error(`Failed to rename chat: ${response.status}`);
      }
      mutate();
    },
    [mutate, apiBase],
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
