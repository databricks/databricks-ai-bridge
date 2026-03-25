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

export interface ChatConfig extends BasePluginConfig {
  /** Resolve session from request. If not set, session is derived from x-forwarded-user headers. */
  getSession?: GetSession;
  /** pg Pool for persistence. When not set, runs in ephemeral mode (no history). */
  pool?: import("pg").Pool;
  /** Model / endpoint id. Defaults to process.env.DATABRICKS_SERVING_ENDPOINT or "chat-model". */
  modelId?: string;
  /** Proxy URL for model requests (e.g. MLflow Agent Server). Defaults to process.env.API_PROXY. */
  apiProxy?: string;
  /** Enable feedback feature (thumbs up/down). Defaults to !!process.env.MLFLOW_EXPERIMENT_ID. */
  feedbackEnabled?: boolean;
  /** Auto-create the ai_chatbot schema and tables on startup. Defaults to false. */
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
