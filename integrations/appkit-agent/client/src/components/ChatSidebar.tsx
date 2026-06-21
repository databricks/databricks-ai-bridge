import { isToday, isYesterday, subMonths, subWeeks } from 'date-fns';
import { useState } from 'react';
import { motion } from 'framer-motion';
import { LoaderIcon } from 'lucide-react';
import { useChatHistory } from '@/hooks/use-chat-history';
import { useChatContext } from '@/contexts/ChatProvider';
import type { Chat } from '@/types';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupContentHeader,
  SidebarMenu,
} from '@/components/ui/sidebar';
import { ChatSidebarItem } from './ChatSidebarItem';

type GroupedChats = {
  today: Chat[];
  yesterday: Chat[];
  lastWeek: Chat[];
  lastMonth: Chat[];
  older: Chat[];
};

const groupChatsByDate = (chats: Chat[]): GroupedChats => {
  const now = new Date();
  const oneWeekAgo = subWeeks(now, 1);
  const oneMonthAgo = subMonths(now, 1);

  return chats.reduce(
    (groups, chat) => {
      const chatDate = new Date(chat.createdAt);
      if (isToday(chatDate)) groups.today.push(chat);
      else if (isYesterday(chatDate)) groups.yesterday.push(chat);
      else if (chatDate > oneWeekAgo) groups.lastWeek.push(chat);
      else if (chatDate > oneMonthAgo) groups.lastMonth.push(chat);
      else groups.older.push(chat);
      return groups;
    },
    { today: [], yesterday: [], lastWeek: [], lastMonth: [], older: [] } as GroupedChats,
  );
};

export interface ChatSidebarProps {
  activeChatId?: string;
  onSelectChat: (chatId: string) => void;
  onNewChat?: () => void;
  user?: { email: string; name?: string; preferredUsername?: string } | null;
}

/**
 * Composable sidebar for chat history — uses useChatHistory internally
 * and calls back to the consumer for navigation.
 */
export function ChatSidebar({
  activeChatId,
  onSelectChat,
  user,
}: ChatSidebarProps) {
  const { chatHistoryEnabled } = useChatContext();
  const { chats, isLoading, isValidating, hasMore, isEmpty, loadMore, deleteChat } =
    useChatHistory();

  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const handleDelete = async () => {
    if (!deleteId) return;
    await deleteChat(deleteId);
    setShowDeleteDialog(false);
    if (deleteId === activeChatId) {
      // Consumer handles navigation via callbacks
    }
  };

  if (!user) {
    return (
      <SidebarGroup>
        <SidebarGroupContent>
          <div className="flex w-full flex-row items-center justify-center gap-2 px-2 text-sm text-zinc-500">
            Login to save and revisit previous chats!
          </div>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  }

  if (isLoading) {
    return (
      <SidebarGroup>
        <SidebarGroupContentHeader>Today</SidebarGroupContentHeader>
        <SidebarGroupContent>
          <div className="flex flex-col">
            {[44, 32, 28, 64, 52].map((item) => (
              <div key={item} className="flex h-8 items-center gap-2 rounded-md px-2">
                <div
                  className="h-4 max-w-(--skeleton-width) flex-1 rounded-md bg-sidebar-accent-foreground/10"
                  style={{ '--skeleton-width': `${item}%` } as React.CSSProperties}
                />
              </div>
            ))}
          </div>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  }

  if (isEmpty) {
    return (
      <SidebarGroup>
        <SidebarGroupContent>
          <div className="flex w-full flex-row items-center justify-center gap-2 px-2 text-sm text-muted-foreground">
            {chatHistoryEnabled
              ? 'Your conversations will appear here once you start chatting!'
              : 'Chat history is disabled - conversations are not saved'}
          </div>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  }

  const grouped = groupChatsByDate(chats);
  const sections: { label: string; items: Chat[] }[] = [
    { label: 'Today', items: grouped.today },
    { label: 'Yesterday', items: grouped.yesterday },
    { label: 'Last 7 days', items: grouped.lastWeek },
    { label: 'Last 30 days', items: grouped.lastMonth },
    { label: 'Older than last month', items: grouped.older },
  ].filter((s) => s.items.length > 0);

  return (
    <>
      <SidebarGroup>
        <SidebarGroupContent>
          <SidebarMenu>
            <div className="flex flex-col gap-6">
              {sections.map((section) => (
                <div key={section.label}>
                  <SidebarGroupContentHeader>{section.label}</SidebarGroupContentHeader>
                  {section.items.map((chat) => (
                    <ChatSidebarItem
                      key={chat.id}
                      chat={chat}
                      isActive={chat.id === activeChatId}
                      onSelect={() => onSelectChat(chat.id)}
                      onDelete={(chatId) => {
                        setDeleteId(chatId);
                        setShowDeleteDialog(true);
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </SidebarMenu>

          <motion.div
            onViewportEnter={() => {
              if (!isValidating && !hasMore) return;
              loadMore();
            }}
          />

          {!hasMore ? (
            <div className="mt-8 flex w-full flex-row items-center justify-center gap-2 px-2 text-sm text-zinc-500">
              You have reached the end of your chat history.
            </div>
          ) : (
            <div className="mt-8 flex flex-row items-center gap-2 p-2 text-zinc-500 dark:text-zinc-400">
              <div className="animate-spin"><LoaderIcon /></div>
              <div>Loading Chats...</div>
            </div>
          )}
        </SidebarGroupContent>
      </SidebarGroup>

      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This action cannot be undone. This will permanently delete your
              chat and remove it from our servers.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete}>Continue</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
