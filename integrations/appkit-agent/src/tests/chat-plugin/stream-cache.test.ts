import { afterEach, describe, expect, test, vi } from "vitest";
import { StreamCache } from "../../chat-plugin/stream-cache";

function makeReadableStream(chunks: string[]): ReadableStream<string> {
  return new ReadableStream<string>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(chunk);
      }
      controller.close();
    },
  });
}

describe("StreamCache", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("stores and retrieves active stream id", () => {
    const cache = new StreamCache();
    const stream = makeReadableStream(["hello"]);
    cache.storeStream({ streamId: "s1", chatId: "c1", stream });
    expect(cache.getActiveStreamId("c1")).toBe("s1");
  });

  test("returns null for unknown stream", () => {
    const cache = new StreamCache();
    expect(cache.getStream("nonexistent")).toBeNull();
  });

  test("clears active stream", () => {
    const cache = new StreamCache();
    const stream = makeReadableStream(["data"]);
    cache.storeStream({ streamId: "s1", chatId: "c1", stream });
    cache.clearActiveStream("c1");
    expect(cache.getActiveStreamId("c1")).toBeNull();
  });

  test("clears specific stream", () => {
    const cache = new StreamCache();
    const stream = makeReadableStream(["data"]);
    cache.storeStream({ streamId: "s1", chatId: "c1", stream });
    cache.clearStream("s1");
    expect(cache.getStream("s1")).toBeNull();
    expect(cache.getActiveStreamId("c1")).toBeNull();
  });

  test("getStream returns a Readable", async () => {
    const cache = new StreamCache();
    const stream = makeReadableStream(["chunk1", "chunk2"]);
    cache.storeStream({ streamId: "s1", chatId: "c1", stream });

    await new Promise((r) => setTimeout(r, 50));

    const readable = cache.getStream("s1");
    expect(readable).not.toBeNull();

    const collected: string[] = [];
    await new Promise<void>((resolve, reject) => {
      readable!.on("data", (chunk: Buffer | string) => {
        collected.push(typeof chunk === "string" ? chunk : chunk.toString());
      });
      readable!.on("end", resolve);
      readable!.on("error", reject);
    });
    expect(collected).toEqual(["chunk1", "chunk2"]);
  });
});
