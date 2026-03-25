import { and, asc, desc, eq, gt, gte, inArray, lt, sql } from "drizzle-orm";
import { drizzle } from "drizzle-orm/node-postgres";
import type { Pool } from "pg";
import { type ChatRow, chat, type MessageRow, message, vote } from "./schema";
import type { VisibilityType } from "./types";

export type { ChatRow, MessageRow };

export function createDb(pool: Pool) {
  return drizzle(pool);
}

export async function saveChat(
  db: ReturnType<typeof createDb>,
  params: {
    id: string;
    userId: string;
    title: string;
    visibility: VisibilityType;
  },
) {
  await db.insert(chat).values({
    id: params.id,
    createdAt: new Date(),
    userId: params.userId,
    title: params.title,
    visibility: params.visibility,
  });
}

export async function getChatById(
  db: ReturnType<typeof createDb>,
  id: string,
): Promise<ChatRow | null> {
  const [row] = await db.select().from(chat).where(eq(chat.id, id));
  return row ?? null;
}

export async function deleteChatById(
  db: ReturnType<typeof createDb>,
  id: string,
) {
  await db.delete(message).where(eq(message.chatId, id));
  const [deleted] = await db.delete(chat).where(eq(chat.id, id)).returning();
  return deleted ?? null;
}

export async function getChatsByUserId(
  db: ReturnType<typeof createDb>,
  params: {
    id: string;
    limit: number;
    startingAfter: string | null;
    endingBefore: string | null;
  },
): Promise<{ chats: ChatRow[]; hasMore: boolean }> {
  const { id, limit, startingAfter, endingBefore } = params;
  const extendedLimit = limit + 1;

  if (startingAfter) {
    const [anchor] = await db
      .select()
      .from(chat)
      .where(eq(chat.id, startingAfter))
      .limit(1);
    if (!anchor) return { chats: [], hasMore: false };
    const rows = await db
      .select()
      .from(chat)
      .where(and(eq(chat.userId, id), gt(chat.createdAt, anchor.createdAt)))
      .orderBy(desc(chat.createdAt))
      .limit(extendedLimit);
    const hasMore = rows.length > limit;
    return { chats: hasMore ? rows.slice(0, limit) : rows, hasMore };
  }
  if (endingBefore) {
    const [anchor] = await db
      .select()
      .from(chat)
      .where(eq(chat.id, endingBefore))
      .limit(1);
    if (!anchor) return { chats: [], hasMore: false };
    const rows = await db
      .select()
      .from(chat)
      .where(and(eq(chat.userId, id), lt(chat.createdAt, anchor.createdAt)))
      .orderBy(desc(chat.createdAt))
      .limit(extendedLimit);
    const hasMore = rows.length > limit;
    return { chats: hasMore ? rows.slice(0, limit) : rows, hasMore };
  }

  const rows = await db
    .select()
    .from(chat)
    .where(eq(chat.userId, id))
    .orderBy(desc(chat.createdAt))
    .limit(extendedLimit);
  const hasMore = rows.length > limit;
  return {
    chats: hasMore ? rows.slice(0, limit) : rows,
    hasMore,
  };
}

export async function saveMessages(
  db: ReturnType<typeof createDb>,
  messages: Array<{
    id: string;
    chatId: string;
    role: string;
    parts: unknown;
    attachments: unknown;
    createdAt: Date;
    traceId?: string | null;
  }>,
) {
  if (messages.length === 0) return;
  await db
    .insert(message)
    .values(
      messages.map((m) => ({
        id: m.id,
        chatId: m.chatId,
        role: m.role,
        parts: m.parts,
        attachments: m.attachments ?? [],
        createdAt: m.createdAt,
        traceId: m.traceId ?? null,
      })),
    )
    .onConflictDoUpdate({
      target: message.id,
      set: {
        parts: sql`excluded.parts`,
        attachments: sql`excluded.attachments`,
        traceId: sql`excluded."traceId"`,
      },
    });
}

export async function getMessagesByChatId(
  db: ReturnType<typeof createDb>,
  chatId: string,
): Promise<MessageRow[]> {
  return db
    .select()
    .from(message)
    .where(eq(message.chatId, chatId))
    .orderBy(asc(message.createdAt));
}

export async function getMessageById(
  db: ReturnType<typeof createDb>,
  id: string,
): Promise<MessageRow | null> {
  const [row] = await db.select().from(message).where(eq(message.id, id));
  return row ?? null;
}

export async function updateChatTitleById(
  db: ReturnType<typeof createDb>,
  params: { chatId: string; title: string },
) {
  await db
    .update(chat)
    .set({ title: params.title })
    .where(eq(chat.id, params.chatId));
}

export async function updateChatVisibilityById(
  db: ReturnType<typeof createDb>,
  params: { chatId: string; visibility: VisibilityType },
) {
  await db
    .update(chat)
    .set({ visibility: params.visibility })
    .where(eq(chat.id, params.chatId));
}

export async function updateChatLastContextById(
  db: ReturnType<typeof createDb>,
  params: { chatId: string; context: Record<string, unknown> },
) {
  await db
    .update(chat)
    .set({ lastContext: params.context })
    .where(eq(chat.id, params.chatId));
}

export async function deleteMessagesAfter(
  db: ReturnType<typeof createDb>,
  params: { chatId: string; afterCreatedAt: Date },
) {
  const toDelete = await db
    .select({ id: message.id })
    .from(message)
    .where(
      and(
        eq(message.chatId, params.chatId),
        gte(message.createdAt, params.afterCreatedAt),
      ),
    );
  const ids = toDelete.map((r) => r.id);
  if (ids.length > 0) {
    await db.delete(vote).where(inArray(vote.messageId, ids));
    await db
      .delete(message)
      .where(
        and(
          eq(message.chatId, params.chatId),
          gte(message.createdAt, params.afterCreatedAt),
        ),
      );
  }
}

export async function getVotesByChatId(
  db: ReturnType<typeof createDb>,
  chatId: string,
) {
  return db.select().from(vote).where(eq(vote.chatId, chatId));
}

export async function voteMessage(
  db: ReturnType<typeof createDb>,
  params: { chatId: string; messageId: string; isUpvoted: boolean },
) {
  await db
    .insert(vote)
    .values({
      chatId: params.chatId,
      messageId: params.messageId,
      isUpvoted: params.isUpvoted,
    })
    .onConflictDoUpdate({
      target: [vote.chatId, vote.messageId],
      set: { isUpvoted: params.isUpvoted },
    });
}
