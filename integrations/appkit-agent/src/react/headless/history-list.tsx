import type { ReactNode } from "react";
import { useHistory } from "../hooks/use-history.js";
import type { Chat } from "../types.js";

export interface HistoryListRenderProps {
  chats: Chat[];
  isLoading: boolean;
  isValidating: boolean;
  hasMore: boolean;
  isEmpty: boolean;
  loadMore: () => void;
  deleteChat: (id: string) => Promise<void>;
  renameChat: (id: string, title: string) => Promise<void>;
}

export interface HistoryListProps {
  children: (props: HistoryListRenderProps) => ReactNode;
}

export function HistoryList({ children }: HistoryListProps) {
  const history = useHistory();
  return children(history);
}
