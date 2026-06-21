import {
  boolean,
  json,
  jsonb,
  pgSchema,
  primaryKey,
  text,
  timestamp,
  uuid,
  varchar,
} from "drizzle-orm/pg-core";

const schemaName = "ai_chatbot";
const customSchema = pgSchema(schemaName);

export type ChatLastContext = Record<string, unknown> | null;

export const chat = customSchema.table("Chat", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  createdAt: timestamp("createdAt").notNull(),
  title: text("title").notNull(),
  userId: text("userId").notNull(),
  visibility: varchar("visibility", {
    enum: ["public", "private"],
  })
    .notNull()
    .default("private"),
  lastContext: jsonb("lastContext").$type<ChatLastContext>(),
});

export const message = customSchema.table("Message", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  chatId: uuid("chatId")
    .notNull()
    .references(() => chat.id),
  role: varchar("role").notNull(),
  parts: json("parts").notNull(),
  attachments: json("attachments").notNull(),
  createdAt: timestamp("createdAt").notNull(),
  traceId: text("traceId"),
});

export const vote = customSchema.table(
  "Vote",
  {
    chatId: uuid("chatId")
      .notNull()
      .references(() => chat.id),
    messageId: uuid("messageId")
      .notNull()
      .references(() => message.id),
    isUpvoted: boolean("isUpvoted").notNull(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.chatId, table.messageId] }),
  }),
);

export type ChatRow = typeof chat.$inferSelect;
export type MessageRow = typeof message.$inferSelect;
export type VoteRow = typeof vote.$inferSelect;
