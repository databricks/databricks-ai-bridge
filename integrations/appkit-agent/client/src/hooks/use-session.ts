import { useChatContext } from '@/contexts/ChatProvider';

/**
 * Public headless hook — returns the current user session from ChatProvider.
 * No UI dependency.
 */
export function useSessionData() {
  const ctx = useChatContext();
  return {
    user: ctx.session?.user ?? null,
    isLoading: ctx.isLoading,
  };
}
