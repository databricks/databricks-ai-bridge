import type { BasePluginConfig } from "@databricks/appkit";
import type { AgentInterface } from "./agent-interface";
import type { FunctionTool } from "./function-tool";
import type { HostedTool } from "./hosted-tools";

/**
 * A tool that can be registered with the agent plugin.
 *
 * - `FunctionTool`: OpenResponses-aligned plain object with JSON Schema parameters and an execute handler.
 * - `HostedTool`: Databricks-hosted tool (genie, vector_search_index, custom_mcp_server, external_mcp_server).
 */
export type AgentTool = FunctionTool | HostedTool;

export interface IAgentConfig extends BasePluginConfig {
  /**
   * Pre-built agent implementing AgentInterface.
   * When provided the plugin skips internal LangGraph setup and delegates
   * directly to this instance. Use this to bring your own agent
   * implementation or a different LangChain variant.
   */
  agentInstance?: AgentInterface;

  /**
   * Databricks model serving endpoint name (e.g. "databricks-claude-sonnet-4-5").
   * Falls back to DATABRICKS_MODEL env var.
   * Ignored when `agentInstance` is provided.
   */
  model?: string;

  /** System prompt injected at the start of every conversation */
  systemPrompt?: string;

  /**
   * Tools to register with the agent. Accepts:
   * - OpenResponses-aligned `FunctionTool` objects (local tool with execute handler)
   * - Databricks hosted tools (`genie`, `vector_search_index`, `custom_mcp_server`, `external_mcp_server`)
   *
   * Ignored when `agentInstance` is provided.
   */
  tools?: AgentTool[];
}
