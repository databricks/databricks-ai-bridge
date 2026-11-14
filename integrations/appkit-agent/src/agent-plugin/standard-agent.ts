/**
 * StandardAgent — LangGraph wrapper implementing AgentInterface.
 *
 * Wraps a LangGraph `createReactAgent` instance and translates its stream
 * events into Responses API SSE format. If you swap LangGraph for another
 * SDK, provide your own AgentInterface implementation instead.
 */

import { randomUUID } from "node:crypto";
import type { BaseMessage } from "@langchain/core/messages";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import type {
  AgentInterface,
  InvokeParams,
  ResponseFunctionToolCall,
  ResponseOutputItem,
  ResponseOutputMessage,
  ResponseStreamEvent,
} from "./agent-interface";

/**
 * Minimal interface for the LangGraph agent returned by createReactAgent.
 */
interface LangGraphAgent {
  invoke(input: {
    messages: BaseMessage[];
  }): Promise<{ messages: BaseMessage[] }>;
  streamEvents(
    input: { messages: BaseMessage[] },
    options: { version: "v1" | "v2" },
  ): AsyncIterable<{
    event: string;
    name: string;
    run_id: string;
    data?: any;
  }>;
}

function convertToBaseMessages(messages: any[]): BaseMessage[] {
  return messages.map((msg) => {
    if (msg instanceof HumanMessage || msg instanceof SystemMessage) {
      return msg;
    }
    const content = msg.content || "";
    switch (msg.role) {
      case "user":
        return new HumanMessage(content);
      case "assistant":
        return { role: "assistant", content } as any;
      case "system":
        return new SystemMessage(content);
      default:
        return new HumanMessage(content);
    }
  });
}

/**
 * Built-in {@link AgentInterface} implementation that wraps a LangGraph
 * `createReactAgent` and translates its stream events into Responses API
 * SSE format. Use this as the default agent unless you need a custom
 * implementation for a different LLM SDK.
 */
export class StandardAgent implements AgentInterface {
  constructor(
    private agent: LangGraphAgent,
    private systemPrompt: string,
  ) {}

  async invoke(params: InvokeParams): Promise<ResponseOutputItem[]> {
    const { input, chat_history = [] } = params;

    const messages: BaseMessage[] = [
      new SystemMessage(this.systemPrompt),
      ...convertToBaseMessages(chat_history),
      new HumanMessage(input),
    ];

    const result = await this.agent.invoke({ messages });
    const finalMessages = result.messages || [];
    const lastMessage = finalMessages[finalMessages.length - 1];
    const text =
      typeof lastMessage?.content === "string" ? lastMessage.content : "";

    const outputMessage: ResponseOutputMessage = {
      id: `msg_${randomUUID()}`,
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text, annotations: [] }],
    };

    return [outputMessage];
  }

  async *stream(params: InvokeParams): AsyncGenerator<ResponseStreamEvent> {
    const { input, chat_history = [] } = params;

    const messages: BaseMessage[] = [
      new SystemMessage(this.systemPrompt),
      ...convertToBaseMessages(chat_history),
      new HumanMessage(input),
    ];

    const toolCallIds = new Map<string, string>();
    let seqNum = 0;
    let outputIndex = 0;
    const textItemId = `msg_${randomUUID()}`;
    let textOutputIndex = -1;

    const eventStream = this.agent.streamEvents(
      { messages },
      { version: "v2" },
    );

    for await (const event of eventStream) {
      if (event.event === "on_tool_start") {
        const callId = `call_${randomUUID()}`;
        toolCallIds.set(`${event.name}_${event.run_id}`, callId);

        const fcItem: ResponseFunctionToolCall = {
          id: `fc_${randomUUID()}`,
          call_id: callId,
          name: event.name,
          arguments: JSON.stringify(event.data?.input || {}),
          type: "function_call",
          status: "completed",
        };

        const currentIndex = outputIndex++;

        yield {
          type: "response.output_item.added",
          item: fcItem,
          output_index: currentIndex,
          sequence_number: seqNum++,
        };

        yield {
          type: "response.output_item.done",
          item: fcItem,
          output_index: currentIndex,
          sequence_number: seqNum++,
        };
      }

      if (event.event === "on_tool_end") {
        const toolKey = `${event.name}_${event.run_id}`;
        const callId = toolCallIds.get(toolKey) || `call_${randomUUID()}`;
        toolCallIds.delete(toolKey);

        const outputItem = {
          id: `fco_${randomUUID()}`,
          call_id: callId,
          output: JSON.stringify(event.data?.output || ""),
          type: "function_call_output" as const,
        };

        const currentIndex = outputIndex++;

        yield {
          type: "response.output_item.added",
          item: outputItem,
          output_index: currentIndex,
          sequence_number: seqNum++,
        };

        yield {
          type: "response.output_item.done",
          item: outputItem,
          output_index: currentIndex,
          sequence_number: seqNum++,
        };
      }

      if (event.event === "on_chat_model_stream") {
        const content = event.data?.chunk?.content;
        if (content && typeof content === "string") {
          if (textOutputIndex === -1) {
            textOutputIndex = outputIndex++;

            const msgItem: ResponseOutputMessage = {
              id: textItemId,
              type: "message",
              role: "assistant",
              status: "in_progress",
              content: [],
            };
            yield {
              type: "response.output_item.added",
              item: msgItem,
              output_index: textOutputIndex,
              sequence_number: seqNum++,
            };
          }

          yield {
            type: "response.output_text.delta",
            item_id: textItemId,
            output_index: textOutputIndex,
            content_index: 0,
            delta: content,
            sequence_number: seqNum++,
          };
        }
      }
    }

    if (textOutputIndex !== -1) {
      const msgItem: ResponseOutputMessage = {
        id: textItemId,
        type: "message",
        role: "assistant",
        status: "completed",
        content: [],
      };
      yield {
        type: "response.output_item.done",
        item: msgItem,
        output_index: textOutputIndex,
        sequence_number: seqNum++,
      };
    }

    yield {
      type: "response.completed",
      sequence_number: seqNum++,
      response: {},
    };
  }
}
