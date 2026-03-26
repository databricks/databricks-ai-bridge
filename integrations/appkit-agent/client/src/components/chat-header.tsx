import { useNavigate } from 'react-router-dom';

import { SidebarToggle } from '@/components/sidebar-toggle';
import { Button } from '@/components/ui/button';
import { MessageSquareOff } from 'lucide-react';
import { useAppConfig } from '@/contexts/AppConfigContext';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { PlusIcon, CloudOffIcon } from './icons';
import { cn } from '../lib/utils';
import { Skeleton } from './ui/skeleton';
import type { ReactNode } from 'react';

const DOCS_URL =
  'https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app';

export interface ChatHeaderProps {
  title?: string;
  empty?: boolean;
  isLoadingTitle?: boolean;
  /** Callback when user clicks "New Chat". Falls back to react-router navigate('/') when omitted. */
  onNewChat?: () => void;
  /** Slot for extra action buttons rendered after the built-in ones */
  actions?: ReactNode;
}

export function ChatHeader({ title, empty, isLoadingTitle, onNewChat, actions }: ChatHeaderProps) {
  let navigateFallback: (() => void) | undefined;
  try {
    const navigate = useNavigate();
    navigateFallback = () => navigate('/');
  } catch {
    // react-router not available — onNewChat prop is required
  }
  const handleNewChat = onNewChat ?? navigateFallback;

  const { chatHistoryEnabled, feedbackEnabled } = useAppConfig();

  return (
    <header className={cn("sticky top-0 flex h-[60px] items-center gap-2 bg-background px-4", {
      "border-b border-border md:pb-2": !empty,
    })}>
      <div className="md:hidden">
        <SidebarToggle forceOpenIcon />
      </div>

      {(title || isLoadingTitle) &&
        <h4 className="text-[16px] font-medium truncate">
          {isLoadingTitle ?
            <Skeleton className="w-32 h-6 bg-border" /> :
            title
          }
        </h4>
      }

      <div className="ml-auto flex items-center gap-2">
        {!chatHistoryEnabled && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <a
                  href={DOCS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 rounded-lg border-border border-1 bg-muted px-2 py-1 text-foreground text-xs hover:text-foreground"
                >
                  <CloudOffIcon className="h-3 w-3" />
                  <span className="hidden sm:inline">Ephemeral</span>
                </a>
              </TooltipTrigger>
              <TooltipContent>
                <p>Chat history disabled — conversations are not saved. Click to learn more.</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
        {!feedbackEnabled && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <a
                  href={DOCS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 rounded-lg border-border border-1 bg-muted px-2 py-1 text-foreground text-xs hover:text-foreground"
                >
                  <MessageSquareOff className="h-3 w-3" />
                  <span className="hidden sm:inline">Feedback disabled</span>
                </a>
              </TooltipTrigger>
              <TooltipContent>
                <p>Feedback submission disabled. Click to learn more.</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
        {actions}
        {handleNewChat && (
          <Button
            variant="default"
            className="order-2 ml-auto h-8 px-2 md:hidden"
            onClick={handleNewChat}
          >
            <PlusIcon />
            <span>New Chat</span>
          </Button>
        )}
      </div>
    </header>
  );
}
