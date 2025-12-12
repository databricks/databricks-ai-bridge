/**
 * Message conversion utilities between LangChain and OpenAI formats
 */

import {
  BaseMessage,
  HumanMessage,
  AIMessage,
  SystemMessage,
  ToolMessage,
  AIMessageChunk,
} from "@langchain/core/messages";
import { ChatResult, ChatGeneration, ChatGenerationChunk } from "@langchain/core/outputs";
import type {
  ChatCompletionMessageParam,
  ChatCompletionResponse,
  ChatCompletionChunk,
  ChatCompletionMessageToolCall,
  ChatCompletionChunkChoiceDeltaToolCall,
  CompletionUsage,
} from "../types.js";

/**
 * Convert LangChain messages to OpenAI-compatible message format
 */
export function convertMessagesToOpenAI(messages: BaseMessage[]): ChatCompletionMessageParam[] {
  return messages.map((message): ChatCompletionMessageParam => {
    if (message instanceof HumanMessage || message._getType() === "human") {
      return {
        role: "user",
        content: typeof message.content === "string" ? message.content : JSON.stringify(message.content),
        ...(message.name && { name: message.name }),
      };
    }

    if (message instanceof SystemMessage || message._getType() === "system") {
      return {
        role: "system",
        content: typeof message.content === "string" ? message.content : JSON.stringify(message.content),
        ...(message.name && { name: message.name }),
      };
    }

    if (message instanceof AIMessage || message._getType() === "ai") {
      const aiMessage = message as AIMessage;
      const toolCalls = aiMessage.tool_calls?.map(
        (tc): ChatCompletionMessageToolCall => ({
          id: tc.id || "",
          type: "function",
          function: {
            name: tc.name,
            arguments: typeof tc.args === "string" ? tc.args : JSON.stringify(tc.args),
          },
        })
      );

      return {
        role: "assistant",
        content: typeof aiMessage.content === "string" ? aiMessage.content : null,
        ...(toolCalls && toolCalls.length > 0 && { tool_calls: toolCalls }),
        ...(aiMessage.name && { name: aiMessage.name }),
      };
    }

    if (message instanceof ToolMessage || message._getType() === "tool") {
      const toolMessage = message as ToolMessage;
      return {
        role: "tool",
        content: typeof toolMessage.content === "string" ? toolMessage.content : JSON.stringify(toolMessage.content),
        tool_call_id: toolMessage.tool_call_id,
      };
    }

    // Fallback: treat as user message
    return {
      role: "user",
      content: typeof message.content === "string" ? message.content : JSON.stringify(message.content),
    };
  });
}

/**
 * Convert OpenAI response to LangChain ChatResult
 */
export function convertResponseToChatResult(response: ChatCompletionResponse): ChatResult {
  const generations: ChatGeneration[] = response.choices.map((choice) => {
    const toolCalls = choice.message.tool_calls?.map((tc) => ({
      id: tc.id,
      name: tc.function.name,
      args: safeParseJSON(tc.function.arguments),
      type: "tool_call" as const,
    }));

    const message = new AIMessage({
      content: choice.message.content || "",
      tool_calls: toolCalls,
      additional_kwargs: {
        ...(choice.message.tool_calls && { tool_calls: choice.message.tool_calls }),
      },
    });

    return {
      text: choice.message.content || "",
      message,
      generationInfo: {
        finish_reason: choice.finish_reason,
        index: choice.index,
      },
    };
  });

  return {
    generations,
    llmOutput: {
      tokenUsage: response.usage
        ? {
            promptTokens: response.usage.prompt_tokens,
            completionTokens: response.usage.completion_tokens,
            totalTokens: response.usage.total_tokens,
          }
        : undefined,
      model: response.model,
      id: response.id,
    },
  };
}

/**
 * Track accumulated tool call state during streaming
 */
interface ToolCallAccumulator {
  [index: number]: {
    id: string;
    name: string;
    arguments: string;
  };
}

/**
 * Convert OpenAI streaming chunk to LangChain ChatGenerationChunk
 */
export function convertChunkToGenerationChunk(
  chunk: ChatCompletionChunk,
  toolCallAccumulator: ToolCallAccumulator
): ChatGenerationChunk {
  const choice = chunk.choices[0];

  if (!choice) {
    // Empty chunk, possibly usage-only
    return new ChatGenerationChunk({
      message: new AIMessageChunk({
        content: "",
      }),
      text: "",
      generationInfo: {
        usage: chunk.usage ? convertUsageMetadata(chunk.usage) : undefined,
      },
    });
  }

  const delta = choice.delta;

  // Accumulate tool calls
  if (delta.tool_calls) {
    for (const tc of delta.tool_calls) {
      if (!toolCallAccumulator[tc.index]) {
        toolCallAccumulator[tc.index] = {
          id: tc.id || "",
          name: tc.function?.name || "",
          arguments: tc.function?.arguments || "",
        };
      } else {
        if (tc.id) toolCallAccumulator[tc.index].id = tc.id;
        if (tc.function?.name) toolCallAccumulator[tc.index].name += tc.function.name;
        if (tc.function?.arguments) toolCallAccumulator[tc.index].arguments += tc.function.arguments;
      }
    }
  }

  // Build tool_call_chunks for streaming
  const toolCallChunks = delta.tool_calls?.map(
    (tc: ChatCompletionChunkChoiceDeltaToolCall) => ({
      index: tc.index,
      id: tc.id,
      name: tc.function?.name,
      args: tc.function?.arguments,
      type: "tool_call_chunk" as const,
    })
  );

  // On finish_reason, convert accumulated tool calls to final format
  let toolCalls;
  if (choice.finish_reason === "tool_calls") {
    toolCalls = Object.values(toolCallAccumulator).map((tc) => ({
      id: tc.id,
      name: tc.name,
      args: safeParseJSON(tc.arguments),
      type: "tool_call" as const,
    }));
  }

  const content = delta.content || "";

  // Only include usage_metadata on the final chunk (when finish_reason is set)
  const usageMetadata =
    choice.finish_reason && chunk.usage ? convertUsageMetadata(chunk.usage) : undefined;

  const message = new AIMessageChunk({
    content,
    tool_call_chunks: toolCallChunks,
    tool_calls: toolCalls,
    usage_metadata: usageMetadata,
  });

  return new ChatGenerationChunk({
    message,
    text: content,
    generationInfo: {
      finish_reason: choice.finish_reason,
      index: choice.index,
    },
  });
}

/**
 * Convert OpenAI usage to LangChain format
 */
function convertUsageMetadata(usage: CompletionUsage) {
  return {
    input_tokens: usage.prompt_tokens,
    output_tokens: usage.completion_tokens,
    total_tokens: usage.total_tokens,
  };
}

/**
 * Safely parse JSON, returning the original string if parsing fails
 */
function safeParseJSON(str: string): Record<string, unknown> {
  try {
    return JSON.parse(str);
  } catch {
    return { raw: str };
  }
}

/**
 * Create a tool call accumulator for tracking streaming tool calls
 */
export function createToolCallAccumulator(): ToolCallAccumulator {
  return {};
}
