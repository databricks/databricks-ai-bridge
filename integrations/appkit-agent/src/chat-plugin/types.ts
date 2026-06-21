import type { Request } from "express";
import type { BasePluginConfig } from "@databricks/appkit";
import { z as zod } from "zod";

export interface ChatSession {
  user: {
    id: string;
    email?: string;
    name?: string;
    preferredUsername?: string;
  } | null;
}

export type GetSession = (
  req: Request,
) => Promise<ChatSession | null> | ChatSession | null;

/**
 * Chat backend target.
 *
 * - `string` — name of a local AppKit plugin (e.g. `'agent'`). Requests are
 *   proxied to `/api/<name>` on the same server at request time.
 * - `{ proxy: string }` — arbitrary proxy URL (e.g. `{ proxy: 'http://localhost:8000/invocations' }`).
 * - `{ endpoint: string }` — Databricks serving endpoint name
 *   (e.g. `{ endpoint: 'databricks-claude-sonnet-4-5' }`).
 *
 * When omitted, defaults to `process.env.DATABRICKS_SERVING_ENDPOINT` or `'chat-model'`.
 */
export type ChatBackend = string | { proxy: string } | { endpoint: string };

export interface ChatConfig extends BasePluginConfig {
  /** Resolve session from request. If not set, session is derived from x-forwarded-user headers. */
  getSession?: GetSession;
  /** pg Pool for persistence. When not set, runs in ephemeral mode (no history). */
  pool?: import("pg").Pool;
  /** Chat backend target. See {@link ChatBackend}. */
  backend?: ChatBackend;
  /** Enable feedback feature (thumbs up/down). Defaults to !!process.env.MLFLOW_EXPERIMENT_ID. */
  feedbackEnabled?: boolean;
  /** Auto-create the ai_chatbot schema and tables on startup. Defaults to true; set false to disable. */
  autoMigrate?: boolean;
}

const textPartSchema = zod.object({
  type: zod.enum(["text"]),
  text: zod.string().min(1),
});

const filePartSchema = zod.object({
  type: zod.enum(["file"]),
  mediaType: zod.enum(["image/jpeg", "image/png"]),
  name: zod.string().min(1),
  url: zod.string().url(),
});

const partSchema = zod.union([textPartSchema, filePartSchema]);

const previousMessageSchema = zod.object({
  id: zod.string(),
  role: zod.enum(["user", "assistant", "system"]),
  parts: zod.array(zod.any()),
});

export const postRequestBodySchema = zod.object({
  id: zod.string().min(1),
  message: zod
    .object({
      id: zod.string().min(1),
      role: zod.enum(["user"]),
      parts: zod.array(partSchema),
    })
    .optional(),
  selectedChatModel: zod.enum(["chat-model", "chat-model-reasoning"]),
  selectedVisibilityType: zod.enum(["public", "private"]),
  previousMessages: zod.array(previousMessageSchema).optional(),
});

export type PostRequestBody = zod.infer<typeof postRequestBodySchema>;

export type VisibilityType = "public" | "private";
