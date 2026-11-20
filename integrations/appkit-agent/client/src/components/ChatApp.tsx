import { BrowserRouter, Routes, Route, useNavigate, useParams, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { Toaster } from 'sonner';

const LIGHT_THEME_COLOR = 'hsl(0 0% 100%)';
const DARK_THEME_COLOR = 'hsl(240deg 10% 3.92%)';
import { ChatProvider, type ChatProviderProps, useChatContext } from '@/contexts/ChatProvider';
import { ChatPanel } from './ChatPanel';
import { ChatSidebar } from './ChatSidebar';
import { DatabricksLogo } from './DatabricksLogo';
import { DbIcon } from './ui/db-icon';
import { UserKeyIconIcon } from '../components/icons';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  useSidebar,
} from './ui/sidebar';
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip';
import { SidebarUserNav } from './sidebar-user-nav';
import { NewChatIcon, SidebarCollapseIcon, SidebarExpandIcon } from './icons';
import { cn } from '@/lib/utils';
import { Action } from './elements/actions';
import { useChatData } from '@/hooks/useChatData';
import type { LanguageModelUsage } from 'ai';
import type { Chat as ChatRecord } from '@/types';
import { generateUUID } from '@/lib/utils';
import { getChatIdFromPathname } from '@/lib/navigation';
import { useChatHistory } from '@/hooks/use-chat-history';

export interface ChatAppProps extends Omit<ChatProviderProps, 'children'> {
  /** Base path for client-side routing (default: "/chat") */
  basePath?: string;
}

/**
 * Full drop-in chat application — includes ChatProvider, BrowserRouter,
 * sidebar, routing, and the complete chat experience. Renders a standalone
 * SPA that can be mounted into any container.
 */
export function ChatApp({
  basePath = '/',
  ...providerProps
}: ChatAppProps) {
  return (
    <BrowserRouter basename={basePath}>
      <ChatProvider basePath={basePath} {...providerProps}>
        <Toaster position="top-center" />
        <ChatAppLayout />
      </ChatProvider>
    </BrowserRouter>
  );
}

function ChatAppLayout() {
  const { session, isLoading } = useChatContext();
  const isCollapsed = localStorage.getItem('sidebar:state') !== 'true';

  // Keep theme-color meta tag in sync with dark/light mode
  useEffect(() => {
    const html = document.documentElement;
    const meta = document.getElementById('theme-color-meta');
    if (!meta) return;

    const update = () => {
      meta.setAttribute(
        'content',
        html.classList.contains('dark') ? DARK_THEME_COLOR : LIGHT_THEME_COLOR,
      );
    };

    const observer = new MutationObserver(update);
    observer.observe(html, { attributes: true, attributeFilter: ['class'] });
    update();
    return () => observer.disconnect();
  }, []);

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    );
  }

  if (!session?.user) {
    return (
      <div className="flex h-screen items-center justify-center bg-secondary">
        <div className="flex flex-col items-center gap-6">
          <DatabricksLogo height={20} />
          <div className="flex w-80 flex-col items-center gap-4 rounded-md border border-border bg-background p-10 shadow-[var(--shadow-db-lg)]">
            <DbIcon icon={UserKeyIconIcon} size={32} color="muted" />
            <div className="flex flex-col items-center gap-1.5 text-center">
              <h3>Authentication Required</h3>
              <p className="text-muted-foreground">
                Please authenticate using Databricks to access this application.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <SidebarProvider defaultOpen={!isCollapsed}>
      <ChatAppSidebar />
      <SidebarInset className="h-svh overflow-hidden bg-secondary">
        <div className="flex flex-1 flex-col overflow-hidden bg-background md:my-2 md:mr-2 md:rounded-xl">
          <Routes>
            <Route index element={<NewChatView />} />
            <Route path="chat/:id" element={<ExistingChatView />} />
          </Routes>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}

function ChatAppSidebar() {
  const navigate = useNavigate();
  useLocation(); // re-render on real navigation (route changes)
  // Subscribe to the same SWR history cache that ChatSidebar uses.
  // When useChatStream mutates history (after new chat creation),
  // this triggers a re-render here too, so getChatIdFromPathname()
  // picks up the URL that softNavigateToChatId already updated.
  // SWR deduplicates — no extra network requests.
  useChatHistory();
  const activeChatId = getChatIdFromPathname();
  const { session } = useChatContext();
  const { open, openMobile, isMobile, toggleSidebar, setOpenMobile } = useSidebar();
  const effectiveOpen = open || (isMobile && openMobile);

  return (
    <Sidebar collapsible="icon" className="group-data-[side=left]:border-r-0">
      <SidebarHeader
        className={cn(
          'h-[44px] flex-row items-center gap-2 px-2 py-0',
          effectiveOpen ? 'justify-between' : 'justify-center',
        )}
      >
        {effectiveOpen && (
          <button
            onClick={() => { setOpenMobile(false); navigate('/'); }}
            className="flex items-center overflow-hidden px-1"
            type="button"
          >
            <span className="text-base font-semibold text-foreground">Chatbot</span>
          </button>
        )}
        <Action onClick={toggleSidebar} tooltip={effectiveOpen ? 'Collapse sidebar' : 'Expand sidebar'}>
          <DbIcon icon={effectiveOpen ? SidebarCollapseIcon : SidebarExpandIcon} size={16} color="muted" />
        </Action>
      </SidebarHeader>

      <div className="px-2 pt-2">
        <SidebarMenu>
          <SidebarMenuItem>
            <Tooltip>
              <TooltipTrigger asChild>
                <SidebarMenuButton
                  type="button"
                  className="h-8 p-1 md:p-2 cursor-pointer"
                  onClick={() => { setOpenMobile(false); navigate('/'); }}
                >
                  <DbIcon icon={NewChatIcon} size={16} color="default" />
                  <span className="group-data-[collapsible=icon]:hidden">New chat</span>
                </SidebarMenuButton>
              </TooltipTrigger>
              <TooltipContent side="right" style={{ display: open ? 'none' : 'block' }}>New chat</TooltipContent>
            </Tooltip>
          </SidebarMenuItem>
        </SidebarMenu>
      </div>

      <SidebarContent>
        {effectiveOpen && (
          <ChatSidebar
            activeChatId={activeChatId}
            onSelectChat={(chatId) => { setOpenMobile(false); navigate(`/chat/${chatId}`); }}
            user={session?.user}
          />
        )}
      </SidebarContent>

      <SidebarFooter>
        {session?.user && (
          <SidebarUserNav
            user={session.user}
            preferredUsername={session.user.preferredUsername ?? null}
          />
        )}
      </SidebarFooter>
    </Sidebar>
  );
}

function fromV3Usage(
  usage: ChatRecord['lastContext'] | undefined,
): LanguageModelUsage | undefined {
  if (!usage) return undefined;
  return {
    inputTokens: usage.inputTokens?.total,
    outputTokens: usage.outputTokens?.total,
    totalTokens: (usage.inputTokens?.total ?? 0) + (usage.outputTokens?.total ?? 0),
    inputTokenDetails: {
      noCacheTokens: usage.inputTokens?.noCache,
      cacheReadTokens: usage.inputTokens?.cacheRead,
      cacheWriteTokens: usage.inputTokens?.cacheWrite,
    },
    outputTokenDetails: {
      textTokens: usage.outputTokens?.text,
      reasoningTokens: usage.outputTokens?.reasoning,
    },
  };
}

function NewChatView() {
  const { session } = useChatContext();
  const [id, setId] = useState(() => generateUUID());
  const [modelId] = useState(() => localStorage.getItem('chat-model') ?? 'chat-model');
  const location = useLocation();

  useEffect(() => {
    setId(generateUUID());
  }, [location.key]);

  if (!session?.user) return null;

  return (
    <ChatPanel
      key={id}
      id={id}
      initialMessages={[]}
      model={modelId}
      initialVisibility="private"
    />
  );
}

function ExistingChatView() {
  const { id } = useParams<{ id: string }>();
  const { session } = useChatContext();
  const [modelId] = useState(() => localStorage.getItem('chat-model') ?? 'chat-model');
  const { chatData, error } = useChatData(id, !!session?.user);

  if (!session?.user) return null;

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <h1 className="mb-4 font-bold text-2xl">Error</h1>
          <p className="text-muted-foreground">{error}</p>
        </div>
      </div>
    );
  }

  if (!chatData || chatData.chat.id !== id) {
    return (
      <div className="flex h-screen justify-center flex-col">
        <div className="text-muted-foreground flex-1 flex items-center justify-center">
          Loading chat...
        </div>
      </div>
    );
  }

  const { chat, messages, feedback } = chatData;

  return (
    <ChatPanel
      key={chat.id}
      id={chat.id}
      title={chat.title}
      initialMessages={messages}
      model={modelId}
      initialVisibility={chat.visibility}
      feedback={feedback}
    />
  );
}
