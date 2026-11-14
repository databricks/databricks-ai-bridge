import { Readable } from "node:stream";

interface CachedStreamEntry {
  chatId: string;
  streamId: string;
  cache: CacheableStream<string>;
  createdAt: number;
  lastAccessedAt: number;
}

interface CacheableStream<T> {
  readonly chunks: readonly T[];
  read(opts: { cursor?: number }): AsyncIterableIterator<T>;
  close(): void;
}

function makeCacheableStream<T>(source: ReadableStream<T>): CacheableStream<T> {
  const chunks: T[] = [];
  let done = false;
  const waiters: (() => void)[] = [];

  const notify = () => {
    const current = [...waiters];
    waiters.length = 0;
    for (const resolve of current) resolve();
  };

  (async () => {
    const reader = source.getReader();
    try {
      while (true) {
        const { value, done: srcDone } = await reader.read();
        if (srcDone) break;
        chunks.push(value);
        notify();
      }
    } catch {
      // treat as early termination
    } finally {
      done = true;
      notify();
      reader.releaseLock();
    }
  })();

  return {
    get chunks() {
      return chunks as readonly T[];
    },
    async *read({ cursor }: { cursor?: number } = {}) {
      let idx = cursor ?? 0;
      while (true) {
        while (idx < chunks.length) {
          yield chunks[idx++];
        }
        if (done) return;
        await new Promise<void>((resolve) => waiters.push(resolve));
      }
    },
    close() {
      done = true;
      notify();
    },
  };
}

function cacheableToReadable(
  cache: CacheableStream<string>,
  opts: { cursor?: number } = {},
): Readable {
  const { cursor } = opts;
  let iterator: AsyncIterableIterator<string> | undefined;
  let pendingRead: Promise<IteratorResult<string>> | null = null;
  let isReading = false;

  return new Readable({
    highWaterMark: 16 * 1024,
    read() {
      if (isReading) return;
      isReading = true;
      if (!iterator) iterator = cache.read({ cursor });
      const processNext = async () => {
        try {
          for (;;) {
            if (!pendingRead) pendingRead = iterator?.next() ?? null;
            if (!pendingRead) break;
            const { value, done } = await pendingRead;
            pendingRead = null;
            if (done) {
              this.push(null);
              break;
            }
            const canContinue = this.push(value);
            if (!canContinue) break;
            pendingRead = iterator?.next() ?? null;
          }
        } catch (err) {
          this.destroy(err as Error);
        } finally {
          isReading = false;
        }
      };
      processNext();
    },
    destroy(error, callback) {
      callback(error);
    },
  });
}

export class StreamCache {
  private cache = new Map<string, CachedStreamEntry>();
  private activeStreams = new Map<string, string>();
  private readonly TTL_MS = 5 * 60 * 1000;
  private cleanupInterval: ReturnType<typeof setInterval> | null = null;

  constructor() {
    this.startCleanup();
  }

  private startCleanup() {
    if (this.cleanupInterval) return;
    this.cleanupInterval = setInterval(() => {
      const now = Date.now();
      for (const [streamId, entry] of this.cache.entries()) {
        if (now - entry.lastAccessedAt > this.TTL_MS) {
          this.activeStreams.delete(entry.chatId);
          entry.cache.close();
          this.cache.delete(streamId);
        }
      }
    }, 60 * 1000);
  }

  storeStream({
    streamId,
    chatId,
    stream,
  }: {
    streamId: string;
    chatId: string;
    stream: ReadableStream<string>;
  }) {
    this.activeStreams.set(chatId, streamId);
    const entry: CachedStreamEntry = {
      chatId,
      streamId,
      cache: makeCacheableStream(stream),
      createdAt: Date.now(),
      lastAccessedAt: Date.now(),
    };
    this.cache.set(streamId, entry);
  }

  getStream(streamId: string, opts: { cursor?: number } = {}): Readable | null {
    const entry = this.cache.get(streamId);
    if (!entry) return null;
    entry.lastAccessedAt = Date.now();
    return cacheableToReadable(entry.cache, opts);
  }

  getActiveStreamId(chatId: string): string | null {
    return this.activeStreams.get(chatId) ?? null;
  }

  clearActiveStream(chatId: string) {
    this.activeStreams.delete(chatId);
  }

  clearStream(streamId: string) {
    const entry = this.cache.get(streamId);
    if (entry) {
      entry.cache.close();
      this.cache.delete(streamId);
      this.activeStreams.delete(entry.chatId);
    }
  }
}
