/**
 * ChatDatabricks - LangChain chat model integration for Databricks Model Serving
 */

import {
  BaseChatModel,
  BaseChatModelParams,
  BaseChatModelCallOptions,
  BindToolsInput as LangChainBindToolsInput,
} from "@langchain/core/language_models/chat_models";
import { BaseMessage, AIMessageChunk } from "@langchain/core/messages";
import { ChatResult, ChatGenerationChunk } from "@langchain/core/outputs";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { Runnable } from "@langchain/core/runnables";
import { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import { Config, ConfigOptions } from "@databricks/sdk-experimental";

import {
  createDatabricksHttpClient,
  DatabricksHttpClient,
  DatabricksHttpClientOptions,
} from "./utils/http.js";
import {
  convertMessagesToOpenAI,
  convertResponseToChatResult,
  convertChunkToGenerationChunk,
  createToolCallAccumulator,
} from "./utils/messages.js";
import { convertToOpenAITools, convertToolChoice, BindToolsInput } from "./utils/tools.js";
import type { ChatCompletionRequest, ChatCompletionTool } from "./types.js";

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

  /** Number of completions (1-5) */
  n?: number;

  /** Tools to use for this call */
  tools?: ChatCompletionTool[];

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

  /** Databricks host URL (e.g., "https://workspace.databricks.com") */
  host?: string;

  /** Personal access token */
  token?: string;

  /** Optional Config object for advanced authentication */
  config?: Config | ConfigOptions;

  /** Temperature (0.0 - 2.0) */
  temperature?: number;

  /** Max tokens to generate */
  maxTokens?: number;

  /** Stop sequences */
  stop?: string[];

  /** Number of completions (1-5) */
  n?: number;

  /** Stream usage metadata in final chunk */
  streamUsage?: boolean;

  /** Extra parameters to pass to the model */
  extraParams?: Record<string, unknown>;

  /** HTTP timeout in milliseconds */
  timeout?: number;

  /** Max retries for failed requests */
  maxRetries?: number;
}

/**
 * ChatDatabricks - Chat model integration for Databricks Model Serving
 *
 * Supports:
 * - Non-streaming and streaming chat completions
 * - Tool/function calling
 * - All Databricks authentication methods (PAT, OAuth M2M, Azure CLI/MSI, GCP)
 *
 * @example
 * ```typescript
 * import { ChatDatabricks } from "@databricks/langchain-ts";
 *
 * const model = new ChatDatabricks({
 *   endpoint: "databricks-meta-llama-3-3-70b-instruct",
 * });
 *
 * const response = await model.invoke("Hello, how are you?");
 * console.log(response.content);
 * ```
 *
 * @example Streaming
 * ```typescript
 * const stream = await model.stream("Tell me a story");
 * for await (const chunk of stream) {
 *   process.stdout.write(chunk.content);
 * }
 * ```
 *
 * @example Tool calling
 * ```typescript
 * const modelWithTools = model.bindTools([
 *   {
 *     type: "function",
 *     function: {
 *       name: "get_weather",
 *       description: "Get weather for a location",
 *       parameters: {
 *         type: "object",
 *         properties: { location: { type: "string" } },
 *         required: ["location"],
 *       },
 *     },
 *   },
 * ]);
 * ```
 */
export class ChatDatabricks extends BaseChatModel<ChatDatabricksCallOptions> {
  static lc_name() {
    return "ChatDatabricks";
  }

  lc_serializable = true;

  /** Model serving endpoint name */
  endpoint: string;

  /** Temperature (0.0 - 2.0) */
  temperature?: number;

  /** Max tokens to generate */
  maxTokens?: number;

  /** Stop sequences */
  stop?: string[];

  /** Number of completions */
  n?: number;

  /** Stream usage metadata */
  streamUsage: boolean;

  /** Extra parameters */
  extraParams?: Record<string, unknown>;

  /** HTTP client for Databricks API */
  private httpClient: DatabricksHttpClient;

  /** Bound tools */
  private boundTools?: ChatCompletionTool[];

  /** Bound tool choice */
  private boundToolChoice?: "auto" | "none" | "required" | "any" | string;

  constructor(fields: ChatDatabricksInput) {
    super(fields);

    this.endpoint = fields.endpoint;
    this.temperature = fields.temperature;
    this.maxTokens = fields.maxTokens;
    this.stop = fields.stop;
    this.n = fields.n;
    this.streamUsage = fields.streamUsage ?? true;
    this.extraParams = fields.extraParams;

    // Build config for HTTP client
    let configOptions: Config | ConfigOptions;
    if (fields.config) {
      configOptions = fields.config;
    } else {
      configOptions = {
        host: fields.host,
        token: fields.token,
      };
    }

    const httpClientOptions: DatabricksHttpClientOptions = {
      timeout: fields.timeout,
      maxRetries: fields.maxRetries,
    };

    this.httpClient = createDatabricksHttpClient(configOptions, httpClientOptions);
  }

  _llmType(): string {
    return "chat-databricks";
  }

  /**
   * Build request payload from messages and options
   */
  private buildRequest(
    messages: BaseMessage[],
    options: ChatDatabricksCallOptions,
    stream: boolean
  ): ChatCompletionRequest {
    const openAIMessages = convertMessagesToOpenAI(messages);

    // Merge tools from bound tools and call options
    const tools = options.tools || this.boundTools;
    const toolChoice = options.toolChoice || this.boundToolChoice;

    const request: ChatCompletionRequest = {
      messages: openAIMessages,
      stream,
      ...(this.temperature !== undefined && { temperature: this.temperature }),
      ...(options.temperature !== undefined && { temperature: options.temperature }),
      ...(this.maxTokens !== undefined && { max_tokens: this.maxTokens }),
      ...(options.maxTokens !== undefined && { max_tokens: options.maxTokens }),
      ...(this.n !== undefined && { n: this.n }),
      ...(options.n !== undefined && { n: options.n }),
      ...(this.stop && { stop: this.stop }),
      ...(options.stop && { stop: options.stop }),
      ...(tools && tools.length > 0 && { tools }),
      ...(toolChoice && { tool_choice: convertToolChoice(toolChoice) }),
      ...this.extraParams,
      ...options.extraParams,
    };

    return request;
  }

  /**
   * Non-streaming chat completion
   */
  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const request = this.buildRequest(messages, options, false);
    const response = await this.httpClient.chatCompletion(this.endpoint, request);
    return convertResponseToChatResult(response);
  }

  /**
   * Streaming chat completion
   */
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const request = this.buildRequest(messages, options, true);
    const toolCallAccumulator = createToolCallAccumulator();

    const stream = this.httpClient.chatCompletionStream(
      this.endpoint,
      request,
      options.signal
    );

    for await (const chunk of stream) {
      // Check for abort
      if (options.signal?.aborted) {
        throw new Error("AbortError: Request was aborted");
      }

      const generationChunk = convertChunkToGenerationChunk(chunk, toolCallAccumulator);

      yield generationChunk;

      // Report to callbacks
      await runManager?.handleLLMNewToken(generationChunk.text ?? "", undefined, undefined, undefined, undefined, {
        chunk: generationChunk,
      });
    }
  }

  /**
   * Bind tools to this model for function calling
   */
  bindTools(
    tools: (LangChainBindToolsInput | BindToolsInput)[],
    kwargs?: Partial<ChatDatabricksCallOptions>
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, ChatDatabricksCallOptions> {
    const formattedTools = convertToOpenAITools(tools as BindToolsInput[]);

    // Create a new instance with bound tools
    const bound = new ChatDatabricks({
      endpoint: this.endpoint,
      config: this.httpClient.getConfig(),
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      stop: this.stop,
      n: this.n,
      streamUsage: this.streamUsage,
      extraParams: this.extraParams,
    });

    bound.boundTools = formattedTools;
    bound.boundToolChoice = kwargs?.toolChoice;

    return bound as unknown as Runnable<
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
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      n: this.n,
    };
  }

  /**
   * Get the type of LLM
   */
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
