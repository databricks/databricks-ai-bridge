/**
 * Message conversion utilities between LangChain and AI SDK formats
 */

import { JSONValue } from "@ai-sdk/provider";
import {
  BaseMessage,
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { ChatResult, ChatGeneration, ChatGenerationChunk } from "@langchain/core/outputs";
import { type GenerateTextResult, type ToolSet, type ModelMessage, type StreamTextResult } from "ai";

type UserContent = Extract<ModelMessage, { role: "user" }>["content"];
type AssistantContent = Extract<ModelMessage, { role: "assistant" }>["content"];
type ToolContent = Extract<ModelMessage, { role: "tool" }>["content"];
/**
 * Convert LangChain messages to AI SDK ModelMessage format
 */
export function convertLangChainToModelMessages(messages: BaseMessage[]): ModelMessage[] {
  return messages.map((msg) => {
    if (SystemMessage.isInstance(msg)) {
      return {
        role: "system" as const,
        content: String(msg.content),
      };
    }

    if (HumanMessage.isInstance(msg)) {
      const content: UserContent = [];
      if (typeof msg.content === "string") {
        content.push({ type: "text", text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (typeof part === "string") {
            content.push({ type: "text", text: part });
          } else if (part.type === "text") {
            content.push({ type: "text", text: String(part.text) });
          } else if (part.type === "image_url") {
            throw new Error(
              "Image content is not yet supported in ChatDatabricks. " +
                "Please use text-only messages."
            );
          }
        }
      }
      return {
        role: "user" as const,
        content,
      };
    }

    if (AIMessage.isInstance(msg)) {
      const content: AssistantContent = [];

      // Add text content
      if (typeof msg.content === "string" && msg.content) {
        content.push({ type: "text", text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (typeof part === "string") {
            content.push({ type: "text", text: part });
          } else if (part.type === "text") {
            content.push({ type: "text", text: String(part.text) });
          }
        }
      }

      // Add tool calls
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        for (const tc of msg.tool_calls) {
          content.push({
            type: "tool-call",
            toolCallId: tc.id ?? `tool-call-${tc.name}`,
            toolName: tc.name,
            input: tc.args,
          });
        }
      }

      return {
        role: "assistant" as const,
        content,
      };
    }

    if (ToolMessage.isInstance(msg)) {
      // Convert LangChain tool message content to LanguageModelV2ToolResultOutput
      let output: ToolContent[number]["output"];
      if (typeof msg.content === "string") {
        // Try to parse as JSON, otherwise use as text
        try {
          const parsed = JSON.parse(msg.content);
          output = { type: "json", value: parsed as JSONValue };
        } catch {
          output = { type: "text", value: msg.content };
        }
      } else {
        // For complex content, treat as JSON
        output = { type: "json", value: msg.content as JSONValue };
      }

      const content: ToolContent = [
        {
          type: "tool-result",
          toolCallId: msg.tool_call_id,
          toolName: msg.name ?? "unknown",
          output,
        },
      ];
      return {
        role: "tool" as const,
        content,
      };
    }

    // Default: treat as user message
    return {
      role: "user" as const,
      content: [{ type: "text", text: String(msg.content) }],
    };
  });
}

/**
 * Convert AI SDK generateText result to LangChain ChatResult
 */
export function convertGenerateTextResultToChatResult(
  result: GenerateTextResult<ToolSet, unknown>
): ChatResult {
  const text = result.text;

  // Extract tool calls from the result
  const toolCalls =
    result.toolCalls?.map((tc) => ({
      id: tc.toolCallId,
      name: tc.toolName,
      args: tc.input as Record<string, unknown>, // AI SDK uses 'input', LangChain uses 'args'
      type: "tool_call" as const,
    })) ?? [];

  const message = new AIMessage({
    content: text,
    tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
  });

  const generation: ChatGeneration = {
    text,
    message,
    generationInfo: {
      finish_reason: result.finishReason,
    },
  };

  return {
    generations: [generation],
    llmOutput: {
      tokenUsage: result.usage
        ? {
            promptTokens: result.usage.inputTokens ?? 0,
            completionTokens: result.usage.outputTokens ?? 0,
            totalTokens: result.usage.totalTokens ?? 0,
          }
        : undefined,
    },
  };
}

/**
 * Convert AI SDK streamText result to async generator of LangChain ChatGenerationChunk
 *
 * This function handles the TextStreamPart events from streamText's fullStream,
 * which provides normalized events across all providers.
 */
export async function* convertStreamTextResultToChunks(
  result: StreamTextResult<ToolSet, unknown>
): AsyncGenerator<ChatGenerationChunk> {
  // Track accumulated tool calls for streaming
  const partialToolCalls: Map<string, { toolName: string; input: string; index: number }> = new Map();
  // Track tool call indices - each unique tool call ID gets an incrementing index
  let nextToolCallIndex = 0;

  for await (const part of result.fullStream) {
    switch (part.type) {
      case "text-delta":
        yield new ChatGenerationChunk({
          message: new AIMessageChunk({
            content: part.text,
          }),
          text: part.text,
        });
        break;

      case "tool-call":
        // Complete tool call from streamText
        yield new ChatGenerationChunk({
          message: new AIMessageChunk({
            content: "",
            tool_calls: [
              {
                id: part.toolCallId,
                name: part.toolName,
                args: part.input as Record<string, unknown>,
                type: "tool_call" as const,
              },
            ],
          }),
          text: "",
        });
        break;

      case "tool-input-start": {
        // Initialize partial tool call with assigned index
        const toolIndex = nextToolCallIndex++;
        partialToolCalls.set(part.id, {
          toolName: part.toolName,
          input: "",
          index: toolIndex,
        });
        yield new ChatGenerationChunk({
          message: new AIMessageChunk({
            content: "",
            tool_call_chunks: [
              {
                index: toolIndex,
                id: part.id,
                name: part.toolName,
                args: "",
                type: "tool_call_chunk" as const,
              },
            ],
          }),
          text: "",
        });
        break;
      }

      case "tool-input-delta": {
        // Accumulate tool call args
        const partial = partialToolCalls.get(part.id);
        if (partial) {
          partial.input += part.delta;
        }
        yield new ChatGenerationChunk({
          message: new AIMessageChunk({
            content: "",
            tool_call_chunks: [
              {
                index: partial?.index ?? 0,
                id: part.id,
                name: partial?.toolName,
                args: part.delta,
                type: "tool_call_chunk" as const,
              },
            ],
          }),
          text: "",
        });
        break;
      }

      case "finish":
        yield new ChatGenerationChunk({
          message: new AIMessageChunk({
            content: "",
            usage_metadata: part.totalUsage
              ? {
                  input_tokens: part.totalUsage.inputTokens ?? 0,
                  output_tokens: part.totalUsage.outputTokens ?? 0,
                  total_tokens: part.totalUsage.totalTokens ?? 0,
                }
              : undefined,
          }),
          text: "",
          generationInfo: {
            finish_reason: part.finishReason,
          },
        });
        break;

      case "error":
        yield new ChatGenerationChunk({
          message: new AIMessageChunk({
            content: "",
          }),
          text: "",
          generationInfo: {
            error: part.error,
          },
        });
        break;

      // Skip parts that don't produce content for LangChain
      case "start":
      case "start-step":
      case "finish-step":
      case "text-start":
      case "text-end":
      case "reasoning-start":
      case "reasoning-end":
      case "reasoning-delta":
      case "tool-input-end":
      case "tool-result":
      case "tool-error":
      case "source":
      case "file":
      case "abort":
      case "raw":
      default:
        // These don't need to be converted to LangChain chunks
        break;
    }
  }
}
