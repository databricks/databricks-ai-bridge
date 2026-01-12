/**
 * ChatDatabricks - LangChain chat model integration for Databricks Model Serving
 *
 * Uses the Databricks AI SDK Provider internally to support multiple endpoint types:
 * - FMAPI (Foundation Model API) - OpenAI-compatible chat completions
 * - ChatAgent - Databricks agent chat completion
 * - ResponsesAgent - Rich output with reasoning, citations, function calls
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
 * Endpoint type determines which Databricks API protocol to use
 */
export type EndpointType = "fmapi" | "chat-agent" | "responses-agent";

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
   * Endpoint type - determines which API protocol to use
   * - 'fmapi': Foundation Model API (OpenAI-compatible) - default
   * - 'chat-agent': Databricks agent chat completion
   * - 'responses-agent': Rich output with reasoning, citations
   */
  endpointType?: EndpointType;

  /**
   * Databricks SDK Config object for authentication.
   * If not provided, a default Config will be created that uses
   * environment variables and ~/.databrickscfg for authentication.
   */
  config?: Config;

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
 * Supports three endpoint types:
 * - FMAPI: OpenAI-compatible foundation models
 * - ChatAgent: Databricks agent endpoints
 * - ResponsesAgent: Rich output with reasoning and citations
 *
 * @example FMAPI (Foundation Model)
 * ```typescript
 * const model = new ChatDatabricks({
 *   endpoint: "databricks-meta-llama-3-3-70b-instruct",
 *   endpointType: "fmapi",
 * });
 * const response = await model.invoke("Hello!");
 * ```
 *
 * @example Chat Agent
 * ```typescript
 * const model = new ChatDatabricks({
 *   endpoint: "my-chat-agent",
 *   endpointType: "chat-agent",
 * });
 * ```
 *
 * @example Responses Agent (with reasoning)
 * ```typescript
 * const model = new ChatDatabricks({
 *   endpoint: "my-responses-agent",
 *   endpointType: "responses-agent",
 * });
 * ```
 *
 * @example With explicit Config
 * ```typescript
 * import { Config } from "@databricks/sdk-experimental";
 *
 * const config = new Config({
 *   host: "https://your-workspace.databricks.com",
 *   token: "dapi...",
 * });
 *
 * const model = new ChatDatabricks({
 *   endpoint: "databricks-meta-llama-3-3-70b-instruct",
 *   config,
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

  /** Endpoint type */
  endpointType: EndpointType;

  /** Temperature (0.0 - 2.0) */
  temperature?: number;

  /** Max tokens to generate */
  maxTokens?: number;

  /** Stop sequences */
  stop?: string[];

  /** Extra parameters */
  extraParams?: Record<string, unknown>;

  /** Databricks SDK Config for authentication */
  private config: Config;

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
    this.endpointType = fields.endpointType ?? "fmapi";
    this.temperature = fields.temperature;
    this.maxTokens = fields.maxTokens;
    this.stop = fields.stop;
    this.extraParams = fields.extraParams;

    // Use provided Config or create default one
    // The Config will automatically detect credentials from:
    // 1. Environment variables (DATABRICKS_HOST, DATABRICKS_TOKEN)
    // 2. ~/.databrickscfg file
    // 3. Azure CLI / Managed Identity
    // 4. Google Cloud credentials
    this.config = fields.config ?? new Config({});

    // Create Databricks AI SDK Provider with SDK-based authentication
    this.provider = this.createProvider();

    // Get appropriate language model based on endpoint type
    this.languageModel = this.getLanguageModel();
  }

  /**
   * Create the Databricks AI SDK Provider with authentication from the SDK Config
   */
  private createProvider(): DatabricksProvider {
    const config = this.config;

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
    switch (this.endpointType) {
      case "chat-agent":
        return this.provider.chatAgent(this.endpoint) as LanguageModel;
      case "responses-agent":
        return this.provider.responsesAgent(this.endpoint) as LanguageModel;
      case "fmapi":
      default:
        return this.provider.fmapi(this.endpoint) as LanguageModel;
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
      endpointType: this.endpointType,
      config: this.config,
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
      endpointType: this.endpointType,
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
