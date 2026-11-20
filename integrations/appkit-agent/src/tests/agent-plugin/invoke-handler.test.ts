import { describe, expect, test, vi } from "vitest";
import { createInvokeHandler } from "../../agent-plugin/invoke-handler";
import { StubAgent } from "./stub-agent";
import { createMockRequest, createMockResponse } from "./test-helpers";

function makeReq(body: any) {
  return createMockRequest({ body }) as any;
}

function makeRes() {
  const res = createMockResponse() as any;
  const chunks: string[] = [];
  res.write.mockImplementation((chunk: string) => {
    chunks.push(chunk);
    return true;
  });
  (res as any).__chunks = chunks;
  return res;
}

function parseSSE(res: any): { events: any[]; fullOutput: string } {
  const chunks: string[] = res.__chunks;
  const events: any[] = [];
  let fullOutput = "";

  for (const chunk of chunks) {
    const lines = chunk.split("\n");
    for (const line of lines) {
      if (line.startsWith("data: ") && line !== "data: [DONE]") {
        try {
          const data = JSON.parse(line.slice(6));
          events.push(data);
          if (data.type === "response.output_text.delta") {
            fullOutput += data.delta;
          }
        } catch {}
      }
    }
  }
  return { events, fullOutput };
}

describe("createInvokeHandler", () => {
  const stubAgent = new StubAgent();
  const handler = createInvokeHandler(() => stubAgent);

  describe("streaming mode", () => {
    test("streams SSE events with correct format", async () => {
      const req = makeReq({
        input: [{ role: "user", content: "Hello" }],
        stream: true,
      });
      const res = makeRes();

      await handler(req, res, vi.fn());

      expect(res.setHeader).toHaveBeenCalledWith(
        "Content-Type",
        "text/event-stream",
      );

      const { events, fullOutput } = parseSSE(res);

      expect(fullOutput).toContain("Echo: Hello");

      const hasCompleted = events.some((e) => e.type === "response.completed");
      expect(hasCompleted).toBe(true);

      const lastChunk = res.__chunks[res.__chunks.length - 1];
      expect(lastChunk).toContain("[DONE]");

      expect(res.end).toHaveBeenCalled();
    });

    test("emits output_item.added and output_item.done events", async () => {
      const req = makeReq({
        input: [{ role: "user", content: "Test" }],
        stream: true,
      });
      const res = makeRes();

      await handler(req, res, vi.fn());

      const { events } = parseSSE(res);
      const addedEvent = events.find(
        (e) => e.type === "response.output_item.added",
      );
      const doneEvent = events.find(
        (e) => e.type === "response.output_item.done",
      );

      expect(addedEvent).toBeDefined();
      expect(addedEvent.item.type).toBe("message");
      expect(doneEvent).toBeDefined();
    });
  });

  describe("non-streaming mode", () => {
    test("returns JSON with output items", async () => {
      const req = makeReq({
        input: [{ role: "user", content: "Hello" }],
        stream: false,
      });
      const res = makeRes();

      await handler(req, res, vi.fn());

      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          output: expect.arrayContaining([
            expect.objectContaining({
              type: "message",
              content: expect.arrayContaining([
                expect.objectContaining({
                  type: "output_text",
                  text: "Echo: Hello",
                }),
              ]),
            }),
          ]),
        }),
      );
    });
  });

  describe("input parsing", () => {
    test("accepts string input", async () => {
      const req = makeReq({
        input: "Hello string",
        stream: true,
      });
      const res = makeRes();

      await handler(req, res, vi.fn());

      const { fullOutput } = parseSSE(res);
      expect(fullOutput).toContain("Echo: Hello string");
    });

    test("accepts array input with multipart content", async () => {
      const req = makeReq({
        input: [
          {
            role: "user",
            content: [
              { type: "input_text", text: "Part one" },
              { type: "input_text", text: "Part two" },
            ],
          },
        ],
        stream: true,
      });
      const res = makeRes();

      await handler(req, res, vi.fn());

      const { fullOutput } = parseSSE(res);
      expect(fullOutput).toContain("Echo: Part one\nPart two");
    });
  });

  describe("chat history", () => {
    test("passes chat history to agent", async () => {
      const spyAgent = {
        invoke: vi.fn().mockResolvedValue([
          {
            id: "msg_1",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [
              { type: "output_text", text: "response", annotations: [] },
            ],
          },
        ]),
        stream: vi.fn(),
      };
      const historyHandler = createInvokeHandler(() => spyAgent as any);

      const req = makeReq({
        input: [
          { role: "user", content: "First message" },
          { role: "assistant", content: "First reply" },
          { role: "user", content: "Second message" },
        ],
        stream: false,
      });
      const res = makeRes();

      await historyHandler(req, res, vi.fn());

      expect(spyAgent.invoke).toHaveBeenCalledWith(
        expect.objectContaining({
          input: "Second message",
          chat_history: expect.arrayContaining([
            expect.objectContaining({
              role: "user",
              content: "First message",
            }),
            expect.objectContaining({
              role: "assistant",
              content: "First reply",
            }),
          ]),
        }),
      );
    });

    test("history contains only messages before the last user message", async () => {
      const spyAgent = {
        invoke: vi.fn().mockResolvedValue([
          {
            id: "msg_1",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [
              { type: "output_text", text: "response", annotations: [] },
            ],
          },
        ]),
        stream: vi.fn(),
      };
      const historyHandler = createInvokeHandler(() => spyAgent as any);

      const req = makeReq({
        input: [
          { role: "user", content: "First message" },
          { role: "user", content: "Second message" },
          { role: "assistant", content: "Interrupted reply" },
        ],
        stream: false,
      });
      const res = makeRes();

      await historyHandler(req, res, vi.fn());

      expect(spyAgent.invoke).toHaveBeenCalledWith(
        expect.objectContaining({
          input: "Second message",
          chat_history: [
            expect.objectContaining({
              role: "user",
              content: "First message",
            }),
          ],
        }),
      );
    });

    test("handles function_call items in history", async () => {
      const spyAgent = {
        invoke: vi.fn().mockResolvedValue([
          {
            id: "msg_1",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: "done", annotations: [] }],
          },
        ]),
        stream: vi.fn(),
      };
      const historyHandler = createInvokeHandler(() => spyAgent as any);

      const req = makeReq({
        input: [
          { role: "user", content: "Look up the answer" },
          {
            type: "function_call",
            name: "search",
            arguments: '{"q":"test"}',
          },
          {
            type: "function_call_output",
            output: '"42"',
          },
          { role: "user", content: "What did you find?" },
        ],
        stream: false,
      });
      const res = makeRes();

      await historyHandler(req, res, vi.fn());

      const calledHistory = spyAgent.invoke.mock.calls[0][0].chat_history;
      expect(calledHistory).toHaveLength(3);
      expect(calledHistory[1].content).toContain("[Tool Call:");
      expect(calledHistory[2].content).toContain("[Tool Result:");
    });
  });

  describe("error handling", () => {
    test("returns 400 for missing input", async () => {
      const req = makeReq({ stream: true });
      const res = makeRes();

      await handler(req, res, vi.fn());

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({ error: "Invalid request format" }),
      );
    });

    test("returns 400 when no user message is present", async () => {
      const req = makeReq({
        input: [{ role: "assistant", content: "I am assistant" }],
        stream: true,
      });
      const res = makeRes();

      await handler(req, res, vi.fn());

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: "No user message found in input",
        }),
      );
    });

    test("returns 500 when non-streaming agent invoke throws", async () => {
      const errorAgent = {
        invoke: vi.fn().mockRejectedValue(new Error("invoke exploded")),
        stream: vi.fn(),
      };
      const errorHandler = createInvokeHandler(() => errorAgent as any);

      const req = makeReq({
        input: [{ role: "user", content: "boom" }],
        stream: false,
      });
      const res = makeRes();

      await errorHandler(req, res, vi.fn());

      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: "Internal server error",
          message: "invoke exploded",
        }),
      );
    });

    test("handles agent errors gracefully in streaming mode", async () => {
      const errorAgent = {
        invoke: vi.fn(),
        stream: (_params: any) => {
          return {
            async next() {
              throw new Error("Agent exploded");
            },
            async return() {
              return { done: true, value: undefined };
            },
            async throw(e: unknown) {
              throw e;
            },
            [Symbol.asyncIterator]() {
              return this;
            },
            [Symbol.asyncDispose]: undefined,
          } as unknown as AsyncGenerator<any>;
        },
      };
      const errorHandler = createInvokeHandler(() => errorAgent as any);

      const req = makeReq({
        input: [{ role: "user", content: "boom" }],
        stream: true,
      });
      const res = makeRes();

      await errorHandler(req, res, vi.fn());

      const { events } = parseSSE(res);
      const errorEvent = events.find((e) => e.type === "error");
      const failedEvent = events.find((e) => e.type === "response.failed");

      expect(errorEvent).toBeDefined();
      expect(errorEvent.error).toContain("Agent exploded");
      expect(failedEvent).toBeDefined();
      expect(res.end).toHaveBeenCalled();
    });
  });
});
