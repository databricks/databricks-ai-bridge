/**
 * ChatDatabricks - LangChain chat model integration for Databricks Model Serving
 *
 * Uses the Databricks AI SDK Provider internally to support multiple endpoint APIs:
 * - chat-completions: OpenAI-compatible chat completions API
 * - chat-agent: Databricks agent chat completion
 * - responses: Rich output with reasoning, citations, function calls
 */

import {
  BaseChatModel,
  BaseChatModelParams,
  BaseChatModelCallOptions,
  BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import { BaseMessage, AIMessageChunk } from "@langchain/core/messages";
import { ChatResult, ChatGenerationChunk } from "@langchain/core/outputs";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { Runnable } from "@langchain/core/runnables";
import { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { LanguageModel } from "ai";
import { generateText, streamText } from "ai";
import { createDatabricksProvider, type DatabricksProvider } from "@databricks/ai-sdk-provider";
import { Config } from "@databricks/sdk-experimental";

import {
  convertLangChainToModelMessages,
  convertGenerateTextResultToChatResult,
  convertStreamTextResultToChunks,
} from "./utils/messages.js";
import { convertToAISDKToolSet } from "./utils/tools.js";

/**
 * Endpoint API determines which Databricks API protocol to use
 */
export type EndpointAPI = "responses" | "chat-completions" | "chat-agent";

/**
 * Authentication options for Databricks
 */
export interface DatabricksAuth {
  /** Databricks workspace host URL (e.g., https://your-workspace.databricks.com) */
  host?: string;
  /** Personal access token or OAuth token */
  token?: string;
}

/**
 * Options that can be passed at call time
 */
export interface ChatDatabricksCallOptions extends BaseChatModelCallOptions {
  /** Temperature (0.0 - 2.0) */
  temperature?: number;

  /** Max tokens to generate */
  maxTokens?: number;

  /** Stop sequences */
  stop?: string[];

  /** Tools to use for this call */
  tools?: BindToolsInput[];

  /** Tool choice for this call */
  toolChoice?: "auto" | "none" | "required" | "any" | string;

  /** Extra parameters to pass to the model */
  extraParams?: Record<string, unknown>;
}

/**
 * Input parameters for ChatDatabricks constructor
 */
export interface ChatDatabricksInput extends BaseChatModelParams {
  /** Model serving endpoint name */
  endpoint: string;

  /**
   * Endpoint API - determines which API protocol to use
   * - 'responses': See https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#responses-api
   * - 'chat-completions': See https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#chat-completions-api
   * - 'chat-agent': See https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-legacy-schema
   */
  endpointAPI: EndpointAPI;

  /**
   * Authentication credentials for Databricks SDK.
   * If not provided Databricks SDK with automatically
   * attempt authentication using environment variables or CLI
   */
  auth?: DatabricksAuth;

  /** Temperature (0.0 - 2.0) */
  temperature?: number;

  /** Max tokens to generate */
  maxTokens?: number;

  /** Stop sequences */
  stop?: string[];

  /** Extra parameters to pass to the model */
  extraParams?: Record<string, unknown>;
}

/**
 * ChatDatabricks - Chat model integration for Databricks Model Serving
 *
 * Supports three endpoint APIs:
 * - chat-completions: OpenAI-compatible chat completions
 * - chat-agent: Databricks agent endpoints
 * - responses: Rich output with reasoning and citations
 *
 * @example Chat Completions
 * ```typescript
 * const model = new ChatDatabricks({
 *   endpoint: "databricks-meta-llama-3-3-70b-instruct",
 *   endpointAPI: "chat-completions",
 * });
 * const response = await model.invoke("Hello!");
 * ```
 *
 * @example Chat Agent
 * ```typescript
 * const model = new ChatDatabricks({
 *   endpoint: "my-chat-agent",
 *   endpointAPI: "chat-agent",
 * });
 * ```
 *
 * @example Responses (with reasoning)
 * ```typescript
 * const model = new ChatDatabricks({
 *   endpoint: "my-responses-agent",
 *   endpointAPI: "responses",
 * });
 * ```
 *
 * @example With explicit authentication
 * ```typescript
 * const model = new ChatDatabricks({
 *   endpoint: "databricks-meta-llama-3-3-70b-instruct",
 *   auth: {
 *     host: "https://your-workspace.databricks.com",
 *     token: "dapi...",
 *   },
 * });
 * ```
 */
export class ChatDatabricks extends BaseChatModel<ChatDatabricksCallOptions> {
  static lc_name() {
    return "ChatDatabricks";
  }

  lc_serializable = true;

  /** Model serving endpoint name */
  endpoint: string;

  /** Endpoint API */
  endpointAPI: EndpointAPI;

  /** Temperature (0.0 - 2.0) */
  temperature?: number;

  /** Max tokens to generate */
  maxTokens?: number;

  /** Stop sequences */
  stop?: string[];

  /** Extra parameters */
  extraParams?: Record<string, unknown>;

  /** Authentication credentials */
  private auth?: DatabricksAuth;

  /** Databricks AI SDK Provider */
  private provider: DatabricksProvider;

  /** AI SDK Language Model */
  private languageModel: LanguageModel;

  /** Bound tools */
  private boundTools?: BindToolsInput[];

  /** Bound tool choice */
  private boundToolChoice?: "auto" | "none" | "required" | "any" | string;

  constructor(fields: ChatDatabricksInput) {
    super(fields);

    this.endpoint = fields.endpoint;
    this.endpointAPI = fields.endpointAPI;
    this.temperature = fields.temperature;
    this.maxTokens = fields.maxTokens;
    this.stop = fields.stop;
    this.extraParams = fields.extraParams;
    this.auth = fields.auth;

    // Create Databricks AI SDK Provider
    this.provider = this.createProvider();

    // Get appropriate language model based on endpoint type
    this.languageModel = this.getLanguageModel();
  }

  /**
   * Create the Databricks AI SDK Provider with authentication
   */
  private createProvider(): DatabricksProvider {
    const config = new Config({
      host: this.auth?.host,
      token: this.auth?.token,
    });

    // Capture the global fetch to avoid recursion when passing custom fetch to the provider
    const globalFetch = globalThis.fetch;

    return createDatabricksProvider({
      baseURL: `${config.host}/serving-endpoints`,
      // Custom fetch that uses SDK authentication
      fetch: async (url, options) => {
        // Ensure config is resolved (handles async auth like OAuth)
        await config.ensureResolved();

        // Create headers and add authentication
        const headers = new Headers(options?.headers as Record<string, string>);
        await config.authenticate(headers);

        // Make the request using the global fetch (not the custom one to avoid recursion)
        const response = await globalFetch(url, {
          ...options,
          headers,
        });

        return response;
      },
    });
  }

  private getLanguageModel(): LanguageModel {
    switch (this.endpointAPI) {
      case "chat-agent":
        return this.provider.chatAgent(this.endpoint) as LanguageModel;
      case "responses":
        return this.provider.responses(this.endpoint) as LanguageModel;
      case "chat-completions":
      default:
        return this.provider.chatCompletions(this.endpoint) as LanguageModel;
    }
  }

  _llmType(): string {
    return "chat-databricks";
  }

  /**
   * Non-streaming chat completion
   */
  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const modelMessages = convertLangChainToModelMessages(messages);

    // Merge tools from bound tools and call options
    const tools = options.tools ?? this.boundTools;
    const aiSdkTools = tools ? convertToAISDKToolSet(tools) : undefined;

    // Merge tool choice from call options and bound tool choice
    const toolChoice = options.toolChoice ?? this.boundToolChoice;

    // Merge stop sequences from instance and options
    const stopSequences = options.stop ?? this.stop;

    // Merge extra params from instance and options
    const extraParams = { ...this.extraParams, ...options.extraParams };

    // Use generateText from AI SDK
    const result = await generateText({
      model: this.languageModel,
      messages: modelMessages,
      tools: aiSdkTools,
      toolChoice: toolChoice as "auto" | "none" | "required" | undefined,
      temperature: options.temperature ?? this.temperature,
      maxOutputTokens: options.maxTokens ?? this.maxTokens,
      stopSequences,
      abortSignal: options.signal,
      ...extraParams,
    });

    const chatResult = convertGenerateTextResultToChatResult(result);

    // Report to callbacks for observability (LangSmith, etc.)
    if (runManager && chatResult.generations.length > 0) {
      const generation = chatResult.generations[0];
      await runManager.handleLLMNewToken(generation.text ?? "");
    }

    return chatResult;
  }

  /**
   * Streaming chat completion
   */
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const modelMessages = convertLangChainToModelMessages(messages);

    // Merge tools from bound tools and call options
    const tools = options.tools ?? this.boundTools;
    const aiSdkTools = tools ? convertToAISDKToolSet(tools) : undefined;

    // Merge tool choice from call options and bound tool choice
    const toolChoice = options.toolChoice ?? this.boundToolChoice;

    // Merge stop sequences from instance and options
    const stopSequences = options.stop ?? this.stop;

    // Merge extra params from instance and options
    const extraParams = { ...this.extraParams, ...options.extraParams };

    // Use streamText from AI SDK for high-level streaming with normalized events
    const result = streamText({
      model: this.languageModel,
      messages: modelMessages,
      tools: aiSdkTools,
      toolChoice: toolChoice as "auto" | "none" | "required" | undefined,
      temperature: options.temperature ?? this.temperature,
      maxOutputTokens: options.maxTokens ?? this.maxTokens,
      stopSequences,
      abortSignal: options.signal,
      ...extraParams,
    });

    // Iterate over the stream and convert to LangChain chunks
    for await (const chunk of convertStreamTextResultToChunks(result)) {
      // Check for abort
      if (options.signal?.aborted) {
        throw new Error("AbortError: Request was aborted");
      }

      yield chunk;

      // Report to callbacks
      await runManager?.handleLLMNewToken(
        chunk.text ?? "",
        undefined,
        undefined,
        undefined,
        undefined,
        { chunk }
      );
    }
  }

  /**
   * Bind tools to this model for function calling
   */
  bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<ChatDatabricksCallOptions>
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, ChatDatabricksCallOptions> {
    // Create a new instance with bound tools
    const bound = new ChatDatabricks({
      endpoint: this.endpoint,
      endpointAPI: this.endpointAPI,
      auth: this.auth,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      stop: this.stop,
      extraParams: this.extraParams,
    });

    bound.boundTools = tools;
    bound.boundToolChoice = kwargs?.toolChoice;

    return bound satisfies Runnable<
      BaseLanguageModelInput,
      AIMessageChunk,
      ChatDatabricksCallOptions
    >;
  }

  /**
   * Get the identifying parameters for this model
   */
  get identifyingParams(): Record<string, unknown> {
    return {
      endpoint: this.endpoint,
      endpointAPI: this.endpointAPI,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
    };
  }

  get lc_secrets(): { [key: string]: string } {
    return {
      token: "DATABRICKS_TOKEN",
    };
  }

  get lc_aliases(): { [key: string]: string } {
    return {
      endpoint: "model",
    };
  }
}
