import type { ChatRow } from "./schema";

export interface ChatAccessResult {
  allowed: boolean;
  chat: ChatRow | null;
  reason?: "not_found" | "private_chat" | "forbidden";
}

export async function checkChatAccess(
  getChat: (id: string) => Promise<ChatRow | null>,
  chatId: string,
  userId?: string,
): Promise<ChatAccessResult> {
  const chat = await getChat(chatId);
  if (!chat) {
    return { allowed: false, chat: null, reason: "not_found" };
  }
  if (chat.visibility === "public") {
    return { allowed: true, chat };
  }
  if (chat.visibility === "private") {
    if (chat.userId !== userId) {
      return { allowed: false, chat, reason: "forbidden" };
    }
    return { allowed: true, chat };
  }
  return { allowed: true, chat };
}
