import {
  convertToModelMessages,
  createUIMessageStream,
  generateText,
  type LanguageModelUsage,
  pipeUIMessageStreamToResponse,
  streamText,
} from "ai";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type express from "express";
import { Plugin, toPlugin, type PluginManifest } from "@databricks/appkit";
import { createLogger } from "../logger";
import { checkChatAccess } from "./chat-acl";
import { ChatServerError } from "./errors";
import manifest from "./manifest.json";
import {
  getAssessmentId,
  getMessageMetadata,
  storeAssessmentId,
  storeMessageMeta,
} from "./message-meta";
import {
  createDb,
  deleteChatById,
  deleteMessagesAfter,
  getChatById,
  getChatsByUserId,
  getMessageById,
  getMessagesByChatId,
  getVotesByChatId,
  saveChat,
  saveMessages,
  updateChatLastContextById,
  updateChatTitleById,
  updateChatVisibilityById,
  voteMessage,
} from "./persistence";
import { getToken, getHostUrl } from "./auth";
import {
  CONTEXT_HEADER_CONVERSATION_ID,
  CONTEXT_HEADER_USER_ID,
  createChatProvider,
} from "./provider";
import { resolveSession } from "./session";
import { ensureSchema } from "./migrate";
import { drainStreamToWriter, fallbackToGenerateText } from "./stream-fallback";
import { StreamCache } from "./stream-cache";
import {
  type ChatConfig,
  type PostRequestBody,
  postRequestBodySchema,
} from "./types";

export type { ChatConfig } from "./types";
export type { ChatSession, GetSession } from "./types";
export type { ChatBackend } from "./types";

import {
  convertToUIMessages,
  generateUUID,
  truncatePreserveWords,
} from "./utils";

const logger = createLogger("chat");

type ChatMessage = NonNullable<PostRequestBody["previousMessages"]>[number];

/**
 * Convert ai's LanguageModelUsage to the nested V3 format expected by
 * the chat history's lastContext column.
 */
function toV3Usage(usage: LanguageModelUsage) {
  return {
    inputTokens: {
      total: usage.inputTokens,
      noCache: undefined,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: usage.outputTokens,
      text: undefined,
      reasoning: undefined,
    },
  };
}

export class ChatPlugin extends Plugin<ChatConfig> {
  public name = "chat" as const;
  static manifest = manifest as PluginManifest<"chat">;
  declare protected config: ChatConfig;
  private static hasLoggedMissingStaticAssets = false;

  /**
   * Resolve the path to the pre-built chat client static assets.
   * Pass this to the server plugin's `staticPath` to serve the chat UI at `/`.
   *
   * @example
   * ```ts
   * createApp({
   *   plugins: [
   *     server({ staticPath: ChatPlugin.staticAssetsPath }),
   *     chat({ ... }),
   *   ],
   * });
   * ```
   */
  static get staticAssetsPath(): string | undefined {
    const currentDir = path.dirname(fileURLToPath(import.meta.url));
    const candidates = [
      path.resolve(currentDir, "chat-client"),
      path.resolve(currentDir, "../chat-client"),
      path.resolve(process.cwd(), "dist/chat-client"),
    ];

    for (const dir of candidates) {
      if (fs.existsSync(path.join(dir, "index.html"))) {
        return dir;
      }
    }

    if (!ChatPlugin.hasLoggedMissingStaticAssets) {
      ChatPlugin.hasLoggedMissingStaticAssets = true;
      logger.error(
        "Chat static assets not found. Expected index.html in one of: %s. Try reinstalling @databricks/appkit-agent package and/or clearing your (p)npm cache.",
        candidates.join(", "),
      );
    }
    return undefined;
  }

  private provider: ReturnType<typeof createChatProvider> | null = null;
  private streamCache: StreamCache | null = null;
  private db: ReturnType<typeof createDb> | null = null;
  private getChat: (id: string) => Promise<import("./schema").ChatRow | null> =
    async () => null;

  async setup() {
    this.provider = createChatProvider();
    this.streamCache = new StreamCache();
    if (this.config.pool) {
      if (this.config.autoMigrate ?? true) {
        await ensureSchema(this.config.pool);
        logger.info("Database schema ensured (autoMigrate)");
      }
      this.db = createDb(this.config.pool);
      this.getChat = (id) => getChatById(this.db!, id);
    }
    logger.info(
      "Chat plugin initialized (persistence: %s)",
      this.config.pool ? "enabled" : "ephemeral",
    );
  }

  injectRoutes(router: express.Router) {
    const pool = this.config.pool;
    const db = pool ? createDb(pool) : null;
    const getSessionFn = this.config.getSession;
    const cache = this.streamCache!;
    const prov = this.provider!;

    const requireAuth = async (
      req: express.Request,
      res: express.Response,
      next: express.NextFunction,
    ) => {
      const session = await resolveSession(req, getSessionFn);
      if (!session?.user) {
        const err = new ChatServerError("unauthorized:chat");
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }
      (req as express.Request & { session?: typeof session }).session = session;
      next();
    };

    const requireChatAccess = (paramName: "id" | "chatId") => {
      return async (
        req: express.Request,
        res: express.Response,
        next: express.NextFunction,
      ) => {
        const session = (
          req as express.Request & {
            session?: Awaited<ReturnType<typeof resolveSession>>;
          }
        ).session;
        const id = paramName === "id" ? req.params.id : req.params.chatId;
        if (!id) {
          const err = new ChatServerError("bad_request:api");
          const { status, json } = err.toResponse();
          return res.status(status).json(json);
        }
        const { allowed, reason } = await checkChatAccess(
          this.getChat,
          id,
          session?.user?.id,
        );
        if (!allowed && reason !== "not_found") {
          const err = new ChatServerError("forbidden:chat", reason);
          const { status, json } = err.toResponse();
          return res.status(status).json(json);
        }
        next();
      };
    };

    // ── GET /config ──────────────────────────────────────────────────────
    router.get("/config", (_req, res) => {
      res.json({
        features: {
          chatHistory: !!pool,
          feedback:
            this.config.feedbackEnabled ?? !!process.env.MLFLOW_EXPERIMENT_ID,
        },
      });
    });
    this.registerEndpoint("config", `/api/${this.name}/config`);

    // ── GET /session ─────────────────────────────────────────────────────
    router.get("/session", async (req, res) => {
      const session = await resolveSession(req, getSessionFn);
      if (!session?.user) {
        return res.json({ user: null });
      }
      res.json({
        user: {
          id: session.user.id,
          email: session.user.email,
          name: session.user.name,
          preferredUsername: session.user.preferredUsername,
        },
      });
    });
    this.registerEndpoint("session", `/api/${this.name}/session`);

    // ── GET /history ─────────────────────────────────────────────────────
    router.get("/history", requireAuth, async (req, res) => {
      if (!pool) {
        return res.status(204).end();
      }
      const session = (
        req as express.Request & {
          session?: Awaited<ReturnType<typeof resolveSession>>;
        }
      ).session;
      if (!session?.user) {
        const err = new ChatServerError("unauthorized:chat");
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }
      const limit = Number.parseInt((req.query.limit as string) || "10", 10);
      const startingAfter = (req.query.starting_after as string) || null;
      const endingBefore = (req.query.ending_before as string) || null;
      if (startingAfter && endingBefore) {
        const err = new ChatServerError(
          "bad_request:api",
          "Only one of starting_after or ending_before can be provided.",
        );
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }
      try {
        const { chats, hasMore } = await getChatsByUserId(db!, {
          id: session.user.id,
          limit,
          startingAfter,
          endingBefore,
        });
        return res.json({ chats, hasMore });
      } catch (e) {
        logger.error("getChatsByUserId failed: %O", e);
        const err = new ChatServerError("bad_request:database");
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }
    });
    this.registerEndpoint("history", `/api/${this.name}/history`);

    // ── GET /messages/:id ────────────────────────────────────────────────
    router.get(
      "/messages/:id",
      requireAuth,
      requireChatAccess("id"),
      async (req, res) => {
        const chatId = req.params.id;
        if (!pool) return res.status(204).end();
        try {
          const messages = await getMessagesByChatId(db!, chatId);
          return res.json(messages);
        } catch (e) {
          logger.error("getMessagesByChatId failed: %O", e);
          return res.status(500).json({ error: "Failed to get messages" });
        }
      },
    );
    this.registerEndpoint("messages", `/api/${this.name}/messages/:id`);

    // ── DELETE /messages/:id/trailing ─────────────────────────────────────
    router.delete("/messages/:id/trailing", requireAuth, async (req, res) => {
      if (!pool) return res.status(204).end();
      const messageId = req.params.id;
      const msg = await getMessageById(db!, messageId);
      if (!msg) {
        const err = new ChatServerError("not_found:database");
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }
      const session = (
        req as express.Request & {
          session?: Awaited<ReturnType<typeof resolveSession>>;
        }
      ).session;
      const { allowed } = await checkChatAccess(
        this.getChat,
        msg.chatId,
        session?.user?.id,
      );
      if (!allowed) {
        const err = new ChatServerError("forbidden:chat");
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }
      await deleteMessagesAfter(db!, {
        chatId: msg.chatId,
        afterCreatedAt: msg.createdAt,
      });
      return res.json({ success: true });
    });
    this.registerEndpoint(
      "messagesTrailing",
      `/api/${this.name}/messages/:id/trailing`,
    );

    // ── POST /feedback (with MLflow traces API integration) ──────────────
    router.post("/feedback", requireAuth, async (req, res) => {
      try {
        const { messageId, feedbackType } = req.body;
        if (
          !messageId ||
          (feedbackType !== "thumbs_up" && feedbackType !== "thumbs_down")
        ) {
          const err = new ChatServerError("bad_request:api");
          const { status, json } = err.toResponse();
          return res.status(status).json(json);
        }
        const session = (
          req as express.Request & {
            session?: Awaited<ReturnType<typeof resolveSession>>;
          }
        ).session;
        if (!session?.user) {
          const err = new ChatServerError("unauthorized:chat");
          const { status, json } = err.toResponse();
          return res.status(status).json(json);
        }

        let traceId: string | null = null;
        let chatId: string | undefined;

        if (pool) {
          const msg = await getMessageById(db!, messageId);
          if (msg) {
            traceId = msg.traceId;
            chatId = msg.chatId;
          }
        }

        if (!chatId) {
          const metadata = getMessageMetadata(messageId);
          if (!metadata) {
            const err = new ChatServerError("not_found:database");
            const { status, json } = err.toResponse();
            return res.status(status).json(json);
          }
          traceId = metadata.traceId;
          chatId = metadata.chatId;
        }

        if (chatId) {
          const { allowed } = await checkChatAccess(
            this.getChat,
            chatId,
            session.user.id,
          );
          if (!allowed) {
            const err = new ChatServerError("forbidden:chat");
            const { status, json } = err.toResponse();
            return res.status(status).json(json);
          }
        }

        let mlflowAssessmentId: string | undefined;

        if (traceId) {
          try {
            const token = await getToken();
            const hostUrl = await getHostUrl();
            const userId = session.user.email ?? session.user.id;
            const existingAssessmentId = getAssessmentId(
              messageId,
              session.user.id,
            );

            let mlflowResponse: Response;
            if (existingAssessmentId) {
              mlflowResponse = await fetch(
                `${hostUrl}/api/3.0/mlflow/traces/${traceId}/assessments/${existingAssessmentId}`,
                {
                  method: "PATCH",
                  headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                  },
                  body: JSON.stringify({
                    assessment: {
                      trace_id: traceId,
                      assessment_name: "user_feedback",
                      feedback: { value: feedbackType === "thumbs_up" },
                    },
                    update_mask: "feedback",
                  }),
                },
              );
            } else {
              mlflowResponse = await fetch(
                `${hostUrl}/api/3.0/mlflow/traces/${traceId}/assessments`,
                {
                  method: "POST",
                  headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                  },
                  body: JSON.stringify({
                    assessment: {
                      trace_id: traceId,
                      assessment_name: "user_feedback",
                      source: { source_type: "HUMAN", source_id: userId },
                      feedback: { value: feedbackType === "thumbs_up" },
                    },
                  }),
                },
              );
            }

            if (!mlflowResponse.ok) {
              const errorText = await mlflowResponse.text();
              logger.error("MLflow feedback submission failed: %s", errorText);
              return res
                .status(mlflowResponse.status)
                .json({ error: "Failed to submit feedback" });
            }

            const mlflowResult = (await mlflowResponse.json()) as {
              assessment?: { assessment_id?: string };
            };
            mlflowAssessmentId = mlflowResult.assessment?.assessment_id;
            if (mlflowAssessmentId) {
              storeAssessmentId(messageId, session.user.id, mlflowAssessmentId);
            }
          } catch (error) {
            logger.error("MLflow feedback error: %O", error);
            const err = new ChatServerError("offline:chat");
            const { status, json } = err.toResponse();
            return res.status(status).json(json);
          }
        }

        if (pool && chatId) {
          try {
            await voteMessage(db!, {
              chatId,
              messageId,
              isUpvoted: feedbackType === "thumbs_up",
            });
          } catch (err) {
            logger.warn("DB vote save failed: %O", err);
          }
        }

        return res.json({ success: true, mlflowAssessmentId });
      } catch (error) {
        logger.error("Feedback error: %O", error);
        const err = new ChatServerError("offline:chat");
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }
    });
    this.registerEndpoint("feedback", `/api/${this.name}/feedback`);

    // ── GET /feedback/chat/:chatId ───────────────────────────────────────
    router.get(
      "/feedback/chat/:chatId",
      requireAuth,
      requireChatAccess("chatId"),
      async (req, res) => {
        if (!pool) return res.json({});
        const votes = await getVotesByChatId(db!, req.params.chatId);
        const map: Record<
          string,
          {
            messageId: string;
            feedbackType: "thumbs_up" | "thumbs_down";
            assessmentId: null;
          }
        > = {};
        for (const v of votes) {
          map[v.messageId] = {
            messageId: v.messageId,
            feedbackType: v.isUpvoted ? "thumbs_up" : "thumbs_down",
            assessmentId: null,
          };
        }
        return res.json(map);
      },
    );
    this.registerEndpoint(
      "feedbackByChat",
      `/api/${this.name}/feedback/chat/:chatId`,
    );

    // ── POST /title ──────────────────────────────────────────────────────
    router.post("/title", requireAuth, async (req, res) => {
      try {
        const { message } = req.body;
        const model = await prov.languageModel("title-model");
        const truncated =
          message?.parts?.map((p: { type: string; text?: string }) =>
            p.type === "text"
              ? { ...p, text: (p.text ?? "").slice(0, 256) }
              : p,
          ) ?? [];
        const { text: title } = await generateText({
          model,
          system: `Generate a short title (max 80 chars) for the user's first message. No quotes or colons.`,
          prompt: JSON.stringify({ ...message, parts: truncated }),
        });
        return res.json({ title: title ?? "New chat" });
      } catch (e) {
        logger.error("generate title failed: %O", e);
        return res.status(500).json({ error: "Failed to generate title" });
      }
    });
    this.registerEndpoint("title", `/api/${this.name}/title`);

    // ── PATCH /:id/visibility ────────────────────────────────────────────
    router.patch(
      "/:id/visibility",
      requireAuth,
      requireChatAccess("id"),
      async (req, res) => {
        const id = req.params.id;
        const { visibility } = req.body;
        if (!visibility || !["public", "private"].includes(visibility)) {
          return res.status(400).json({ error: "Invalid visibility type" });
        }
        if (!pool) return res.json({ success: true });
        await updateChatVisibilityById(db!, { chatId: id, visibility });
        return res.json({ success: true });
      },
    );
    this.registerEndpoint("visibility", `/api/${this.name}/:id/visibility`);

    // ── GET /:id/stream ──────────────────────────────────────────────────
    router.get("/:id/stream", requireAuth, async (req, res) => {
      const chatId = req.params.id;
      const cursor = req.headers["x-resume-stream-cursor"] as
        | string
        | undefined;
      const streamId = cache.getActiveStreamId(chatId);
      if (!streamId) {
        const err = new ChatServerError("empty:stream");
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }
      const session = (
        req as express.Request & {
          session?: Awaited<ReturnType<typeof resolveSession>>;
        }
      ).session;
      const { allowed, reason } = await checkChatAccess(
        this.getChat,
        chatId,
        session?.user?.id,
      );
      if (!allowed && reason !== "not_found") {
        const err = new ChatServerError("forbidden:chat", reason);
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }
      const stream = cache.getStream(streamId, {
        cursor: cursor ? Number.parseInt(cursor, 10) : undefined,
      });
      if (!stream) {
        const err = new ChatServerError("empty:stream");
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      stream.pipe(res);
      stream.on("error", (err) => {
        logger.error("stream resume error: %O", err);
        if (!res.headersSent) res.status(500).end();
      });
    });
    this.registerEndpoint("stream", `/api/${this.name}/:id/stream`);

    // ── GET /:id ─────────────────────────────────────────────────────────
    router.get(
      "/:id",
      requireAuth,
      requireChatAccess("id"),
      async (req, res) => {
        const id = req.params.id;
        const row = await this.getChat(id);
        if (!row) {
          const err = new ChatServerError("not_found:database");
          const { status, json } = err.toResponse();
          return res.status(status).json(json);
        }
        return res.json(row);
      },
    );
    this.registerEndpoint("getChat", `/api/${this.name}/:id`);

    // ── DELETE /:id ──────────────────────────────────────────────────────
    router.delete(
      "/:id",
      requireAuth,
      requireChatAccess("id"),
      async (req, res) => {
        const id = req.params.id;
        if (pool) await deleteChatById(db!, id);
        return res.status(200).json({});
      },
    );
    this.registerEndpoint("deleteChat", `/api/${this.name}/:id`);

    // ── POST / (main chat handler) ───────────────────────────────────────
    router.post("/", requireAuth, async (req, res) => {
      let body: PostRequestBody;
      try {
        body = postRequestBodySchema.parse(req.body);
      } catch {
        const err = new ChatServerError("bad_request:api");
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }

      try {
        const session = (
          req as express.Request & {
            session?: Awaited<ReturnType<typeof resolveSession>>;
          }
        ).session;
        if (!session?.user) {
          const err = new ChatServerError("unauthorized:chat");
          const { status, json } = err.toResponse();
          return res.status(status).json(json);
        }

        const {
          id,
          message,
          selectedChatModel,
          selectedVisibilityType,
          previousMessages = [],
        } = body;

        const { allowed, reason } = await checkChatAccess(
          this.getChat,
          id,
          session.user.id,
        );
        if (reason !== "not_found" && !allowed) {
          const err = new ChatServerError("forbidden:chat");
          const { status, json } = err.toResponse();
          return res.status(status).json(json);
        }

        const dbAvailable = !!pool;
        const messagesFromDb = dbAvailable
          ? await getMessagesByChatId(db!, id)
          : [];
        const useClientMessages =
          !dbAvailable || (!message && previousMessages.length > 0);
        const previousMessagesResolved = useClientMessages
          ? previousMessages
          : convertToUIMessages(messagesFromDb);

        let titlePromise: Promise<string | null> | undefined;

        const isNewChat = reason === "not_found";

        let uiMessages: ChatMessage[];
        if (message) {
          uiMessages = [...previousMessagesResolved, message] as ChatMessage[];
          if (dbAvailable) {
            if (isNewChat) {
              await saveChat(db!, {
                id,
                userId: session.user.id,
                title: "New chat",
                visibility: selectedVisibilityType,
              });
              titlePromise = generateTitleFromUserMessage(prov, message)
                .then(async (title) => {
                  await updateChatTitleById(db!, { chatId: id, title });
                  return title;
                })
                .catch(async (error) => {
                  logger.warn("title generation failed: %O", error);
                  const textPart = message.parts?.find(
                    (p: { type: string }) => p.type === "text",
                  ) as { text?: string } | undefined;
                  if (textPart?.text) {
                    const fallback = truncatePreserveWords(textPart.text, 128);
                    await updateChatTitleById(db!, {
                      chatId: id,
                      title: fallback,
                    });
                    return fallback;
                  }
                  return null;
                });
            }
            await saveMessages(db!, [
              {
                id: message.id,
                chatId: id,
                role: "user",
                parts: message.parts,
                attachments: [],
                createdAt: new Date(),
                traceId: null,
              },
            ]);
          }
        } else {
          uiMessages = previousMessagesResolved as ChatMessage[];

          if (dbAvailable && previousMessages.length > 0) {
            const assistantMessages = previousMessages.filter(
              (m: ChatMessage) => m.role === "assistant",
            );
            if (assistantMessages.length > 0) {
              await saveMessages(
                db!,
                assistantMessages.map((m: ChatMessage) => ({
                  chatId: id,
                  id: m.id,
                  role: m.role,
                  parts: m.parts,
                  attachments: [],
                  createdAt: new Date(),
                  traceId: null,
                })),
              );
            }

            const hasMcpDenial = previousMessages.some(
              (m: ChatMessage) =>
                Array.isArray(m.parts) &&
                m.parts.some(
                  (p: Record<string, unknown>) =>
                    p.type === "dynamic-tool" &&
                    (p.state === "output-denied" ||
                      (typeof p.approval === "object" &&
                        p.approval !== null &&
                        (p.approval as Record<string, unknown>).approved ===
                          false)),
                ),
            );

            if (hasMcpDenial) {
              res.end();
              return;
            }
          }
        }

        cache.clearActiveStream(id);
        const streamId = generateUUID();
        let finalUsage: LanguageModelUsage | undefined;
        let traceId: string | null = null;

        const { provider: activeProv, modelId: resolvedModelId } =
          this.resolveBackend(req);

        const model = await activeProv.languageModel(
          resolvedModelId ?? selectedChatModel,
        );
        const modelMessages = await convertToModelMessages(
          uiMessages as Parameters<typeof convertToModelMessages>[0],
        );

        const requestHeaders: Record<string, string> = {
          [CONTEXT_HEADER_CONVERSATION_ID]: id,
          [CONTEXT_HEADER_USER_ID]: session.user.email ?? session.user.id,
        };
        const oboToken = req.headers["x-forwarded-access-token"];
        if (typeof oboToken === "string") {
          requestHeaders["x-forwarded-access-token"] = oboToken;
        }

        const result = streamText({
          model,
          messages: modelMessages,
          providerOptions: {
            databricks: { includeTrace: true },
          },
          includeRawChunks: true,
          headers: requestHeaders,
          onChunk: ({ chunk }) => {
            if (chunk.type === "raw") {
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              const raw = chunk.rawValue as any;
              if (raw?.type === "response.output_item.done") {
                const tid = raw?.databricks_output?.trace?.info?.trace_id as
                  | string
                  | undefined;
                if (tid) traceId = tid;
              }
              if (!traceId && typeof raw?.trace_id === "string") {
                traceId = raw.trace_id;
              }
            }
          },
          onFinish: ({ usage }) => {
            finalUsage = usage;
          },
        });

        const stream = createUIMessageStream({
          originalMessages: uiMessages as Parameters<
            typeof createUIMessageStream
          >[0]["originalMessages"],
          generateId: generateUUID,
          execute: async ({ writer }) => {
            const aiStream = result.toUIMessageStream<ChatMessage>({
              sendReasoning: true,
              sendSources: true,
              sendFinish: false,
              onError: (error) => {
                const msg =
                  error instanceof Error ? error.message : String(error);
                writer.onError?.(error);
                return msg;
              },
            });

            const { failed } = await drainStreamToWriter(aiStream, writer);

            if (failed) {
              logger.info("Streaming failed, falling back to generateText");
              const fallbackResult = await fallbackToGenerateText(
                { model, messages: modelMessages, headers: requestHeaders },
                writer,
              );
              if (fallbackResult) {
                finalUsage = fallbackResult.usage;
                if (fallbackResult.traceId) traceId = fallbackResult.traceId;
              }
            }

            if (titlePromise) {
              const generatedTitle = await titlePromise;
              if (generatedTitle) {
                writer.write({
                  type: "data-title",
                  data: generatedTitle,
                } as Parameters<typeof writer.write>[0]);
              }
            }

            writer.write({ type: "data-traceId", data: traceId });
          },
          onFinish: async ({ responseMessage }) => {
            storeMessageMeta(responseMessage.id, id, traceId);

            if (dbAvailable) {
              try {
                await saveMessages(db!, [
                  {
                    id: responseMessage.id,
                    chatId: id,
                    role: responseMessage.role,
                    parts: responseMessage.parts,
                    attachments: [],
                    createdAt: new Date(),
                    traceId,
                  },
                ]);
              } catch (err) {
                logger.error("Failed to save assistant message: %O", err);
              }

              if (finalUsage) {
                try {
                  await updateChatLastContextById(db!, {
                    chatId: id,
                    context: toV3Usage(finalUsage) as unknown as Record<
                      string,
                      unknown
                    >,
                  });
                } catch (err) {
                  logger.warn("Failed to persist usage: %O", err);
                }
              }
            }

            cache.clearActiveStream(id);
          },
        });

        pipeUIMessageStreamToResponse({
          stream,
          response: res,
          consumeSseStream: ({ stream: sseStream }) => {
            cache.storeStream({ streamId, chatId: id, stream: sseStream });
          },
        });
      } catch (error) {
        logger.error("Chat handler error: %O", error);
        if (error instanceof ChatServerError) {
          const { status, json } = error.toResponse();
          return res.status(status).json(json);
        }
        const err = new ChatServerError("offline:chat");
        const { status, json } = err.toResponse();
        return res.status(status).json(json);
      }
    });
    this.registerEndpoint("chat", `/api/${this.name}`);
  }

  private resolveBackend(req: express.Request): {
    provider: ReturnType<typeof createChatProvider>;
    modelId: string | null;
  } {
    const backend = this.config.backend;
    if (!backend) {
      return { provider: this.provider!, modelId: null };
    }
    if (typeof backend === "string") {
      return {
        provider: createChatProvider({
          apiProxy: `${req.protocol}://${req.get("host")}/api/${backend}`,
        }),
        modelId: null,
      };
    }
    if ("proxy" in backend) {
      return {
        provider: createChatProvider({ apiProxy: backend.proxy }),
        modelId: null,
      };
    }
    return { provider: this.provider!, modelId: backend.endpoint };
  }

  exports() {
    return {};
  }
}

async function generateTitleFromUserMessage(
  prov: ReturnType<typeof createChatProvider>,
  message: { parts?: Array<{ type: string; text?: string }> },
): Promise<string> {
  const model = await prov.languageModel("title-model");
  const textPart = message.parts?.find((p) => p.type === "text");
  const text = (textPart && "text" in textPart ? textPart.text : "") ?? "";
  const truncated = truncatePreserveWords(text, 256);
  const { text: title } = await generateText({
    model,
    system:
      "Generate a short title (max 80 chars) for this message. No quotes or colons.",
    prompt: truncated,
  });
  return title ?? "New chat";
}

const _chat = toPlugin(ChatPlugin);

/**
 * Chat plugin factory — pass to `createApp({ plugins: [...] })`.
 *
 * Also exposes `chat.staticAssetsPath` for use with the server plugin's
 * `staticPath` option to serve the bundled chat UI at `/`.
 */
export const chat = Object.assign(_chat, {
  /**
   * Resolve the path to the pre-built chat client static assets.
   * Pass this to the server plugin's `staticPath` to serve the chat UI at `/`.
   *
   * @example
   * ```ts
   * createApp({
   *   plugins: [
   *     server({ staticPath: chat.staticAssetsPath }),
   *     chat({ ... }),
   *   ],
   * });
   * ```
   */
  get staticAssetsPath(): string | undefined {
    return ChatPlugin.staticAssetsPath;
  },
});
