import type { Chat } from '@/types';
import {
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
} from './ui/sidebar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuPortal,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import { memo } from 'react';
import { useChatVisibility } from '@/hooks/use-chat-visibility';
import { OverflowIcon, CheckIcon, ShareIcon, TrashIcon } from './icons';

const PureChatSidebarItem = ({
  chat,
  isActive,
  onSelect,
  onDelete,
}: {
  chat: Chat;
  isActive: boolean;
  onSelect: () => void;
  onDelete: (chatId: string) => void;
}) => {
  const { visibilityType, setVisibilityType } = useChatVisibility({
    chatId: chat.id,
    initialVisibilityType: chat.visibility,
  });

  return (
    <SidebarMenuItem data-testid="chat-history-item" className="mb-1">
      <SidebarMenuButton isActive={isActive} onClick={onSelect}>
        <span>{chat.title}</span>
      </SidebarMenuButton>

      <DropdownMenu modal={true}>
        <DropdownMenuTrigger asChild>
          <SidebarMenuAction
            data-testid="chat-options"
            className="mr-0.5 data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
            showOnHover={!isActive}
          >
            <OverflowIcon />
            <span className="sr-only">More</span>
          </SidebarMenuAction>
        </DropdownMenuTrigger>

        <DropdownMenuContent side="bottom" align="end">
          <DropdownMenuSub>
            <DropdownMenuSubTrigger className="cursor-pointer">
              <ShareIcon />
              <span>Share</span>
            </DropdownMenuSubTrigger>
            <DropdownMenuPortal>
              <DropdownMenuSubContent>
                <DropdownMenuItem
                  className="cursor-pointer flex-row justify-between"
                  onClick={() => setVisibilityType('private')}
                >
                  <div className="flex flex-row items-center gap-2">
                    {visibilityType === 'private' ? <CheckIcon /> : <div className="size-4" />}
                    <span>Private</span>
                  </div>
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="cursor-pointer flex-row justify-between"
                  onClick={() => setVisibilityType('public')}
                >
                  <div className="flex flex-row items-center gap-2">
                    {visibilityType === 'public' ? <CheckIcon /> : <div className="size-4" />}
                    <span>Public</span>
                  </div>
                </DropdownMenuItem>
              </DropdownMenuSubContent>
            </DropdownMenuPortal>
          </DropdownMenuSub>

          <DropdownMenuItem
            className="cursor-pointer text-destructive focus:bg-destructive/15 focus:text-destructive dark:text-red-500"
            onSelect={() => onDelete(chat.id)}
          >
            <TrashIcon />
            <span>Delete</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </SidebarMenuItem>
  );
};

export const ChatSidebarItem = memo(PureChatSidebarItem, (prev, next) => {
  if (prev.isActive !== next.isActive) return false;
  if (prev.chat.title !== next.chat.title) return false;
  if (prev.chat.visibility !== next.chat.visibility) return false;
  return true;
});
