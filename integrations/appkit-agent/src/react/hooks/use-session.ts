import { useChatAgentContext } from "../context.js";

export function useSession() {
  const ctx = useChatAgentContext();
  return {
    user: ctx.session?.user ?? null,
    isLoading: ctx.isLoading,
  };
}
