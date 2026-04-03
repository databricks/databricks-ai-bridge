import { describe, expect, test } from "vitest";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StandardAgent } from "../../agent-plugin/standard-agent";
import type { ResponseStreamEvent } from "../../agent-plugin/agent-interface";

/**
 * Mock LangGraph agent with controllable invoke/streamEvents results.
 */
class MockLangGraphAgent {
  invokeResult: any;
  streamEventsData: any[];

  /** Captured messages from the last invoke/streamEvents call. */
  lastMessages: any[] = [];

  constructor(options: { invokeResult?: any; streamEvents?: any[] } = {}) {
    this.invokeResult = options.invokeResult ?? {
      messages: [{ content: "Hello from agent" }],
    };
    this.streamEventsData = options.streamEvents ?? [];
  }

  async invoke(input: any) {
    this.lastMessages = input.messages;
    return this.invokeResult;
  }

  async *streamEvents(input: any, _options: any) {
    this.lastMessages = input.messages;
    for (const event of this.streamEventsData) {
      yield event;
    }
  }
}

const SYSTEM_PROMPT = "You are a test assistant.";

describe("StandardAgent", () => {
  describe("invoke", () => {
    test("returns a completed ResponseOutputMessage", async () => {
      const mock = new MockLangGraphAgent();
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const items = await agent.invoke({ input: "hello" });

      expect(items).toHaveLength(1);
      expect(items[0].type).toBe("message");

      const msg = items[0] as any;
      expect(msg.role).toBe("assistant");
      expect(msg.status).toBe("completed");
      expect(msg.content[0].type).toBe("output_text");
      expect(msg.content[0].text).toBe("Hello from agent");
    });

    test("prepends system prompt and appends user input", async () => {
      const mock = new MockLangGraphAgent();
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      await agent.invoke({ input: "hello" });

      const messages = mock.lastMessages;
      expect(messages[0]).toBeInstanceOf(SystemMessage);
      expect(messages[0].content).toBe(SYSTEM_PROMPT);
      expect(messages[messages.length - 1]).toBeInstanceOf(HumanMessage);
      expect(messages[messages.length - 1].content).toBe("hello");
    });

    test("includes chat history as base messages", async () => {
      const mock = new MockLangGraphAgent();
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      await agent.invoke({
        input: "follow-up",
        chat_history: [
          { role: "user", content: "hi" },
          { role: "assistant", content: "hello" },
        ],
      });

      const messages = mock.lastMessages;
      // system + 2 history + user input
      expect(messages).toHaveLength(4);
      expect(messages[1]).toBeInstanceOf(HumanMessage);
      expect(messages[1].content).toBe("hi");
    });

    test("returns empty text when last message content is not a string", async () => {
      const mock = new MockLangGraphAgent({
        invokeResult: { messages: [{ content: [{ type: "tool_use" }] }] },
      });
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const items = await agent.invoke({ input: "test" });
      const msg = items[0] as any;
      expect(msg.content[0].text).toBe("");
    });

    test("returns empty text when messages array is empty", async () => {
      const mock = new MockLangGraphAgent({
        invokeResult: { messages: [] },
      });
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const items = await agent.invoke({ input: "test" });
      const msg = items[0] as any;
      expect(msg.content[0].text).toBe("");
    });
  });

  describe("stream", () => {
    test("emits function_call events for on_tool_start", async () => {
      const mock = new MockLangGraphAgent({
        streamEvents: [
          {
            event: "on_tool_start",
            name: "get_weather",
            run_id: "run-1",
            data: { input: { location: "Paris" } },
          },
        ],
      });
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const events: ResponseStreamEvent[] = [];
      for await (const e of agent.stream({ input: "weather?" })) {
        events.push(e);
      }

      const addedEvent = events.find(
        (e) => e.type === "response.output_item.added",
      );
      expect(addedEvent).toBeDefined();
      const item = (addedEvent as any).item;
      expect(item.type).toBe("function_call");
      expect(item.name).toBe("get_weather");
      expect(JSON.parse(item.arguments)).toEqual({ location: "Paris" });

      const doneEvent = events.find(
        (e) =>
          e.type === "response.output_item.done" &&
          (e as any).item.type === "function_call",
      );
      expect(doneEvent).toBeDefined();
    });

    test("emits function_call_output events for on_tool_end", async () => {
      const mock = new MockLangGraphAgent({
        streamEvents: [
          {
            event: "on_tool_start",
            name: "search",
            run_id: "run-2",
            data: { input: { q: "test" } },
          },
          {
            event: "on_tool_end",
            name: "search",
            run_id: "run-2",
            data: { output: "42" },
          },
        ],
      });
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const events: ResponseStreamEvent[] = [];
      for await (const e of agent.stream({ input: "search" })) {
        events.push(e);
      }

      const outputEvents = events.filter(
        (e) =>
          e.type === "response.output_item.added" &&
          (e as any).item.type === "function_call_output",
      );
      expect(outputEvents).toHaveLength(1);

      const output = (outputEvents[0] as any).item;
      expect(output.type).toBe("function_call_output");
      expect(JSON.parse(output.output)).toBe("42");
    });

    test("correlates tool_end call_id with tool_start", async () => {
      const mock = new MockLangGraphAgent({
        streamEvents: [
          {
            event: "on_tool_start",
            name: "calc",
            run_id: "run-3",
            data: { input: {} },
          },
          {
            event: "on_tool_end",
            name: "calc",
            run_id: "run-3",
            data: { output: "done" },
          },
        ],
      });
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const events: ResponseStreamEvent[] = [];
      for await (const e of agent.stream({ input: "calc" })) {
        events.push(e);
      }

      const fcAdded = events.find(
        (e) =>
          e.type === "response.output_item.added" &&
          (e as any).item.type === "function_call",
      );
      const fcoAdded = events.find(
        (e) =>
          e.type === "response.output_item.added" &&
          (e as any).item.type === "function_call_output",
      );

      expect((fcAdded as any).item.call_id).toBe(
        (fcoAdded as any).item.call_id,
      );
    });

    test("emits text delta events for on_chat_model_stream", async () => {
      const mock = new MockLangGraphAgent({
        streamEvents: [
          {
            event: "on_chat_model_stream",
            name: "model",
            run_id: "run-4",
            data: { chunk: { content: "Hello " } },
          },
          {
            event: "on_chat_model_stream",
            name: "model",
            run_id: "run-4",
            data: { chunk: { content: "World" } },
          },
        ],
      });
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const events: ResponseStreamEvent[] = [];
      for await (const e of agent.stream({ input: "greet" })) {
        events.push(e);
      }

      const msgAdded = events.find(
        (e) =>
          e.type === "response.output_item.added" &&
          (e as any).item.type === "message",
      );
      expect(msgAdded).toBeDefined();
      expect((msgAdded as any).item.status).toBe("in_progress");

      const deltas = events.filter(
        (e) => e.type === "response.output_text.delta",
      );
      expect(deltas).toHaveLength(2);
      expect((deltas[0] as any).delta).toBe("Hello ");
      expect((deltas[1] as any).delta).toBe("World");
    });

    test("emits response.completed as final event", async () => {
      const mock = new MockLangGraphAgent({ streamEvents: [] });
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const events: ResponseStreamEvent[] = [];
      for await (const e of agent.stream({ input: "test" })) {
        events.push(e);
      }

      const last = events[events.length - 1];
      expect(last.type).toBe("response.completed");
    });

    test("sequence numbers increment monotonically", async () => {
      const mock = new MockLangGraphAgent({
        streamEvents: [
          {
            event: "on_tool_start",
            name: "t",
            run_id: "r1",
            data: { input: {} },
          },
          {
            event: "on_tool_end",
            name: "t",
            run_id: "r1",
            data: { output: "ok" },
          },
          {
            event: "on_chat_model_stream",
            name: "m",
            run_id: "r2",
            data: { chunk: { content: "hi" } },
          },
        ],
      });
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const events: ResponseStreamEvent[] = [];
      for await (const e of agent.stream({ input: "test" })) {
        events.push(e);
      }

      const seqNums = events.map((e) => (e as any).sequence_number);
      for (let i = 1; i < seqNums.length; i++) {
        expect(seqNums[i]).toBeGreaterThan(seqNums[i - 1]);
      }
    });

    test("output_index increments across items", async () => {
      const mock = new MockLangGraphAgent({
        streamEvents: [
          {
            event: "on_tool_start",
            name: "t",
            run_id: "r1",
            data: { input: {} },
          },
          {
            event: "on_tool_end",
            name: "t",
            run_id: "r1",
            data: { output: "ok" },
          },
          {
            event: "on_chat_model_stream",
            name: "m",
            run_id: "r2",
            data: { chunk: { content: "result" } },
          },
        ],
      });
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const events: ResponseStreamEvent[] = [];
      for await (const e of agent.stream({ input: "test" })) {
        events.push(e);
      }

      const addedEvents = events.filter(
        (e) => e.type === "response.output_item.added",
      );
      const indices = addedEvents.map((e) => (e as any).output_index);
      expect(indices).toEqual([0, 1, 2]);
    });

    test("handles stream with only text (no tool calls)", async () => {
      const mock = new MockLangGraphAgent({
        streamEvents: [
          {
            event: "on_chat_model_stream",
            name: "m",
            run_id: "r1",
            data: { chunk: { content: "just text" } },
          },
        ],
      });
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const events: ResponseStreamEvent[] = [];
      for await (const e of agent.stream({ input: "test" })) {
        events.push(e);
      }

      const types = events.map((e) => e.type);
      expect(types).not.toContain(expect.stringMatching(/function_call/));
      expect(events.some((e) => e.type === "response.output_text.delta")).toBe(
        true,
      );
      expect(events.some((e) => e.type === "response.completed")).toBe(true);
    });

    test("handles stream with only tool calls (no text)", async () => {
      const mock = new MockLangGraphAgent({
        streamEvents: [
          {
            event: "on_tool_start",
            name: "t",
            run_id: "r1",
            data: { input: {} },
          },
          {
            event: "on_tool_end",
            name: "t",
            run_id: "r1",
            data: { output: "ok" },
          },
        ],
      });
      const agent = new StandardAgent(mock as any, SYSTEM_PROMPT);

      const events: ResponseStreamEvent[] = [];
      for await (const e of agent.stream({ input: "test" })) {
        events.push(e);
      }

      const types = events.map((e) => e.type);
      expect(types).not.toContain("response.output_text.delta");
      expect(types).toContain("response.completed");
      // No output_item.done for text message since no text was emitted
      const textDone = events.find(
        (e) =>
          e.type === "response.output_item.done" &&
          (e as any).item.type === "message",
      );
      expect(textDone).toBeUndefined();
    });
  });
});
