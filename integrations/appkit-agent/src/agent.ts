/**
 * AgentPlugin — first-class AppKit plugin for building AI agents.
 *
 * Provides:
 *  - POST /api/agent  (standard AppKit namespaced route)
 *
 * Supports two modes:
 *  1. Bring-your-own agent via `config.agentInstance`
 *  2. Auto-build a ReAct agent from config (model, tools)
 *
 * Tools can be local (FunctionTool with an execute handler) or hosted on
 * Databricks (genie, vector_search_index, custom_mcp_server, external_mcp_server).
 * Hosted tools are resolved to managed MCP servers transparently.
 */

import type express from "express";
import { Plugin, toPlugin, type PluginManifest } from "@databricks/appkit";
import { createLogger } from "./logger";
import type { AgentInterface } from "./agent-interface";
import type { FunctionTool } from "./function-tool";
import { functionToolToStructuredTool } from "./function-tool";
import type { HostedTool } from "./hosted-tools";
import { isHostedTool, resolveHostedTools } from "./hosted-tools";
import { createInvokeHandler } from "./invoke-handler";
import manifest from "./manifest.json";
import { StandardAgent } from "./standard-agent";
import type { AgentTool, IAgentConfig } from "./types";

const logger = createLogger("agent");

const DEFAULT_SYSTEM_PROMPT =
  "You are a helpful AI assistant with access to various tools.";

type ChatDatabricksInstance = InstanceType<
  Awaited<typeof import("@databricks/langchainjs")>["ChatDatabricks"]
>;

export class AgentPlugin extends Plugin<IAgentConfig> {
  public name = "agent" as const;

  static manifest = manifest as PluginManifest<"agent">;

  protected declare config: IAgentConfig;

  private agentImpl: AgentInterface | null = null;
  private systemPrompt = DEFAULT_SYSTEM_PROMPT;
  private mcpClient: {
    getTools(): Promise<unknown[]>;
    close(): Promise<void>;
  } | null = null;

  /** Only set when building from config (not agentInstance). */
  private model: ChatDatabricksInstance | null = null;
  /** Mutable list of all tools (config + added). Only used when building from config. */
  private toolsList: AgentTool[] = [];

  async setup() {
    this.systemPrompt = this.config.systemPrompt ?? DEFAULT_SYSTEM_PROMPT;

    if (this.config.agentInstance) {
      this.agentImpl = this.config.agentInstance;
      logger.info("AgentPlugin initialized with provided agentInstance");
      return;
    }

    const modelName = this.config.model ?? process.env.DATABRICKS_MODEL;

    if (!modelName) {
      throw new Error(
        "AgentPlugin: model name is required. Set config.model or DATABRICKS_MODEL env var.",
      );
    }

    const { ChatDatabricks } = await import("@databricks/langchainjs");

    this.model = new ChatDatabricks({
      model: modelName,
      maxRetries: 3,
    });

    this.toolsList = [...(this.config.tools ?? [])];

    await this.buildStandardAgent();

    const { localTools, hostedTools } = AgentPlugin.partitionTools(
      this.toolsList,
    );
    logger.info(
      "AgentPlugin initialized: model=%s localTools=%d hostedTools=%d",
      modelName,
      localTools.length,
      hostedTools.length,
    );
  }

  /**
   * Partition the tools list into local FunctionTools and hosted tools
   * (Databricks-managed MCP services).
   */
  private static partitionTools(tools: AgentTool[]): {
    localTools: FunctionTool[];
    hostedTools: HostedTool[];
  } {
    const localTools: FunctionTool[] = [];
    const hostedTools: HostedTool[] = [];

    for (const tool of tools) {
      if (isHostedTool(tool)) {
        hostedTools.push(tool);
      } else {
        localTools.push(tool);
      }
    }

    return { localTools, hostedTools };
  }

  /**
   * Builds or rebuilds the ReAct agent from current model and toolsList.
   * FunctionTools are converted to the internal tool format; hosted tools
   * are resolved to MCP server connections.
   */
  private async buildStandardAgent(): Promise<void> {
    if (!this.model) return;

    if (this.mcpClient) {
      try {
        await this.mcpClient.close();
      } catch (err) {
        logger.warn("Error closing MCP client during rebuild: %O", err);
      }
      this.mcpClient = null;
    }

    const { localTools, hostedTools } = AgentPlugin.partitionTools(
      this.toolsList,
    );

    const tools: unknown[] = [];

    if (hostedTools.length > 0) {
      try {
        const mcpServers = await resolveHostedTools(hostedTools);
        const { buildMCPServerConfig } = await import(
          "@databricks/langchainjs"
        );
        const mcpServerConfigs = await buildMCPServerConfig(mcpServers);
        const { MultiServerMCPClient } = await import(
          "@langchain/mcp-adapters"
        );
        this.mcpClient = new MultiServerMCPClient({
          mcpServers: mcpServerConfigs,
          throwOnLoadError: false,
          prefixToolNameWithServerName: true,
        });
        const mcpTools = await this.mcpClient.getTools();
        tools.push(...mcpTools);
        logger.info(
          "Loaded %d MCP tools from %d hosted tool(s)",
          mcpTools.length,
          hostedTools.length,
        );
      } catch (err) {
        logger.warn(
          "Failed to load hosted tools — continuing without them: %O",
          err,
        );
      }
    }

    tools.push(...localTools.map(functionToolToStructuredTool));

    const { createReactAgent } = await import("@langchain/langgraph/prebuilt");
    const langGraphAgent = createReactAgent({
      llm: this.model,
      tools: tools as any,
    });

    this.agentImpl = new StandardAgent(
      langGraphAgent as any,
      this.systemPrompt,
    );
  }

  /**
   * Add tools to the agent after app creation. Only supported when the plugin
   * was initialized from config (not when using agentInstance). Rebuilds the
   * underlying agent with the new tool set.
   *
   * Accepts FunctionTool or hosted tool definitions.
   */
  async addTools(tools: AgentTool[]): Promise<void> {
    if (this.config.agentInstance) {
      throw new Error(
        "addTools() is not supported when using a custom agentInstance",
      );
    }
    if (!this.model) {
      throw new Error("AgentPlugin not initialized — call setup() first");
    }

    this.toolsList.push(...tools);
    await this.buildStandardAgent();

    logger.info(
      "Added %d tool(s); total tools=%d",
      tools.length,
      this.toolsList.length,
    );
  }

  private getAgentImpl(): AgentInterface {
    if (!this.agentImpl) {
      throw new Error("AgentPlugin not initialized — call setup() first");
    }
    return this.agentImpl;
  }

  injectRoutes(router: express.Router) {
    const handler = createInvokeHandler(() => this.getAgentImpl());
    router.post("/", handler);
    this.registerEndpoint("invoke", `/api/${this.name}`);
  }

  async abortActiveOperations() {
    await super.abortActiveOperations();
    if (this.mcpClient) {
      try {
        await this.mcpClient.close();
      } catch (err) {
        logger.warn("Error closing MCP client: %O", err);
      }
    }
  }

  exports() {
    return {
      invoke: async (
        messages: { role: string; content: string }[],
      ): Promise<string> => {
        if (!this.agentImpl) {
          throw new Error("AgentPlugin not initialized");
        }
        const lastUser = [...messages].reverse().find((m) => m.role === "user");
        const input = lastUser?.content ?? "";
        const chatHistory = messages.slice(0, -1);
        const items = await this.agentImpl.invoke({
          input,
          chat_history: chatHistory,
        });
        const msg = items.find((i) => i.type === "message") as any;
        const text = msg?.content?.[0]?.text ?? "";
        return text;
      },

      stream: async function* (
        this: AgentPlugin,
        messages: { role: string; content: string }[],
      ) {
        if (!this.agentImpl) {
          throw new Error("AgentPlugin not initialized");
        }
        const lastUser = [...messages].reverse().find((m) => m.role === "user");
        const input = lastUser?.content ?? "";
        const chatHistory = messages.slice(0, -1);
        yield* this.agentImpl.stream({
          input,
          chat_history: chatHistory,
        });
      }.bind(this),

      addTools: (tools: AgentTool[]) => this.addTools(tools),
    };
  }
}

export const agent = toPlugin(AgentPlugin);
