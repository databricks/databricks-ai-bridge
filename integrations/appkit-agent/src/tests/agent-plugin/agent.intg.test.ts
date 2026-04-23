/**
 * Integration test: boots the plugin with a StubAgent, mounts the route
 * handler, and drives it end-to-end through the HTTP layer — verifying
 * the full lifecycle from request parsing through SSE streaming to the
 * exports() convenience API.
 */

import { describe, expect, test, vi, beforeEach } from "vitest";
import type {
  AgentInterface,
  InvokeParams,
  ResponseStreamEvent,
  ResponseOutputItem,
  ResponseOutputItemAddedEvent,
} from "../../agent-plugin/agent-interface";
import { StubAgent } from "./stub-agent";
import {
  createMockRequest,
  createMockResponse,
  createMockRouter,
} from "./test-helpers";

vi.mock("@databricks/appkit", () => {
  class Plugin<T = unknown> {
    protected config: T;
    constructor(config: T) {
      this.config = config;
    }
    registerEndpoint(_name: string, _path: string) {}
    async abortActiveOperations() {}
  }

  function toPlugin(PluginClass: new (config: unknown) => unknown) {
    return (config: unknown) => {
      const instance = new PluginClass(config) as Record<string, unknown>;
      return { name: instance.name, instance };
    };
  }

  return { Plugin, toPlugin };
});

import { AgentPlugin } from "../../agent-plugin/agent";

function makeRes() {
  const res = createMockResponse() as Record<string, unknown>;
  const chunks: string[] = [];
  (res.write as ReturnType<typeof vi.fn>).mockImplementation(
    (chunk: string) => {
      chunks.push(chunk);
      return true;
    },
  );
  (res as Record<string, unknown>).__chunks = chunks;
  return res;
}

function parseSSE(res: Record<string, unknown>): {
  events: ResponseStreamEvent[];
  fullText: string;
} {
  const chunks = res.__chunks as string[];
  const events: ResponseStreamEvent[] = [];
  let fullText = "";

  for (const chunk of chunks) {
    for (const line of chunk.split("\n")) {
      if (line.startsWith("data: ") && line !== "data: [DONE]") {
        try {
          const data = JSON.parse(line.slice(6));
          events.push(data);
          if (data.type === "response.output_text.delta") {
            fullText += data.delta;
          }
        } catch {
          /* skip non-JSON */
        }
      }
    }
  }
  return { events, fullText };
}

describe("Agent plugin integration", () => {
  let plugin: AgentPlugin;
  let handler: (...args: unknown[]) => Promise<void>;

  beforeEach(async () => {
    plugin = new AgentPlugin({ agentInstance: new StubAgent() });
    await plugin.setup();

    const { router, getHandler } = createMockRouter();
    plugin.injectRoutes(router as never);
    handler = getHandler("POST", "/") as (...args: unknown[]) => Promise<void>;
    expect(handler).toBeDefined();
  });

  test("full streaming round-trip: request → SSE events → [DONE]", async () => {
    const req = createMockRequest({
      body: {
        input: [{ role: "user", content: "What is 2+2?" }],
        stream: true,
      },
    });
    const res = makeRes();

    await handler(req, res, vi.fn());

    expect(res.setHeader).toHaveBeenCalledWith(
      "Content-Type",
      "text/event-stream",
    );

    const { events, fullText } = parseSSE(res);

    expect(fullText).toBe("Echo: What is 2+2?");

    const types = events.map((e) => e.type);
    expect(types).toContain("response.output_item.added");
    expect(types).toContain("response.output_text.delta");
    expect(types).toContain("response.output_item.done");
    expect(types).toContain("response.completed");

    const lastChunk = (res.__chunks as string[]).at(-1);
    expect(lastChunk).toContain("[DONE]");
    expect(res.end).toHaveBeenCalled();

    const addedEvt = events.find(
      (e) => e.type === "response.output_item.added",
    ) as ResponseOutputItemAddedEvent | undefined;
    expect(addedEvt).toBeDefined();
    expect(addedEvt?.item).toMatchObject({
      type: "message",
      role: "assistant",
    });
  });

  test("non-streaming round-trip returns JSON with output items", async () => {
    const req = createMockRequest({
      body: {
        input: [{ role: "user", content: "Hello" }],
        stream: false,
      },
    });
    const res = makeRes();

    await handler(req, res, vi.fn());

    const callArgs = (res.json as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(callArgs.output).toHaveLength(1);
    expect(callArgs.output[0].type).toBe("message");
    expect(callArgs.output[0].content[0].text).toBe("Echo: Hello");
  });

  test("multi-turn conversation preserves chat history", async () => {
    const invokeSpy = vi.fn(
      async (params: InvokeParams): Promise<ResponseOutputItem[]> => {
        return [
          {
            id: "msg_1",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [
              {
                type: "output_text",
                text: `got ${params.chat_history?.length ?? 0} history items`,
                annotations: [],
              },
            ],
          },
        ];
      },
    );

    const spyAgent: AgentInterface = {
      invoke: invokeSpy,
      stream: new StubAgent().stream,
    };

    const p = new AgentPlugin({ agentInstance: spyAgent });
    await p.setup();
    const { router, getHandler } = createMockRouter();
    p.injectRoutes(router as never);
    const h = getHandler("POST", "/") as typeof handler;

    const req = createMockRequest({
      body: {
        input: [
          { role: "user", content: "First" },
          { role: "assistant", content: "Reply 1" },
          { role: "user", content: "Second" },
          { role: "assistant", content: "Reply 2" },
          { role: "user", content: "Third" },
        ],
        stream: false,
      },
    });
    const res = makeRes();

    await h(req, res, vi.fn());

    expect(invokeSpy).toHaveBeenCalledOnce();
    const [params] = invokeSpy.mock.calls[0];
    expect(params.input).toBe("Third");
    expect(params.chat_history).toHaveLength(4);

    const callArgs = (res.json as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(callArgs.output[0].content[0].text).toBe("got 4 history items");
  });

  test("exports().invoke and exports().stream agree on output", async () => {
    const exported = plugin.exports();

    const invokeResult = await exported.invoke([
      { role: "user", content: "sync test" },
    ]);

    const streamEvents: ResponseStreamEvent[] = [];
    let streamedText = "";
    for await (const event of exported.stream([
      { role: "user", content: "sync test" },
    ])) {
      streamEvents.push(event);
      if (event.type === "response.output_text.delta") {
        streamedText += event.delta;
      }
    }

    expect(invokeResult).toBe("Echo: sync test");
    expect(streamedText).toBe("Echo: sync test");
    expect(streamEvents.some((e) => e.type === "response.completed")).toBe(
      true,
    );
  });

  test("streaming error mid-flight emits error + failed events", async () => {
    let callCount = 0;
    const failingAgent: AgentInterface = {
      invoke: vi.fn(),
      async *stream(_params) {
        callCount++;
        yield {
          type: "response.output_item.added" as const,
          item: {
            id: "msg_err",
            type: "message" as const,
            role: "assistant" as const,
            status: "in_progress" as const,
            content: [],
          },
          output_index: 0,
          sequence_number: 0,
        };
        throw new Error("Simulated LLM failure");
      },
    };

    const plugin = new AgentPlugin({ agentInstance: failingAgent });
    await plugin.setup();
    const { router, getHandler } = createMockRouter();
    plugin.injectRoutes(router as never);
    const h = getHandler("POST", "/") as typeof handler;

    const req = createMockRequest({
      body: {
        input: [{ role: "user", content: "trigger error" }],
        stream: true,
      },
    });
    const res = makeRes();

    await h(req, res, vi.fn());

    const { events } = parseSSE(res);
    const errorEvt = events.find((e) => e.type === "error");
    const failedEvt = events.find((e) => e.type === "response.failed");
    expect(errorEvt).toBeDefined();
    expect(errorEvt?.type === "error" ? errorEvt.error : undefined).toContain(
      "Simulated LLM failure",
    );
    expect(failedEvt).toBeDefined();
    expect(res.end).toHaveBeenCalled();
    expect(callCount).toBe(1);
  });

  test("invalid request returns 400 without crashing the handler", async () => {
    const req = createMockRequest({ body: { garbage: true } });
    const res = makeRes();

    await handler(req, res, vi.fn());

    expect(res.status).toHaveBeenCalledWith(400);
    expect(res.json).toHaveBeenCalledWith(
      expect.objectContaining({ error: expect.any(String) }),
    );
  });
});
