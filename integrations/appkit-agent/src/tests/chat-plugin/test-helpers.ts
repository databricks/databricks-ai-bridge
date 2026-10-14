import { vi } from "vitest";

export function createMockRequest(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    params: {},
    query: {},
    body: {},
    headers: {},
    header(name: string) {
      return (this as Record<string, unknown>).headers
        ? ((this as Record<string, unknown>).headers as Record<string, string>)[
            name.toLowerCase()
          ]
        : undefined;
    },
    ...overrides,
  };
}

export function createMockResponse(): Record<string, unknown> {
  const eventListeners: Record<
    string,
    Array<(...args: unknown[]) => void>
  > = {};

  const res = {
    status: vi.fn().mockReturnThis(),
    json: vi.fn().mockReturnThis(),
    send: vi.fn().mockReturnThis(),
    sendStatus: vi.fn().mockReturnThis(),
    end: vi.fn(function (this: Record<string, unknown>) {
      this.writableEnded = true;
      if (eventListeners.close) {
        for (const handler of eventListeners.close) {
          handler();
        }
      }
      return this;
    }),
    write: vi.fn().mockReturnThis(),
    setHeader: vi.fn().mockReturnThis(),
    flushHeaders: vi.fn().mockReturnThis(),
    on: vi.fn(function (
      this: Record<string, unknown>,
      event: string,
      handler: (...args: unknown[]) => void,
    ) {
      if (!eventListeners[event]) {
        eventListeners[event] = [];
      }
      eventListeners[event].push(handler);
      return this;
    }),
    writableEnded: false,
  };
  return res;
}

export function createMockRouter(): Record<string, unknown> {
  const handlers: Record<string, unknown> = {};

  const mockRouter = {
    get: vi.fn((path: string, ...args: unknown[]) => {
      handlers[`GET:${path}`] = args[args.length - 1];
    }),
    post: vi.fn((path: string, ...args: unknown[]) => {
      handlers[`POST:${path}`] = args[args.length - 1];
    }),
    put: vi.fn((path: string, ...args: unknown[]) => {
      handlers[`PUT:${path}`] = args[args.length - 1];
    }),
    delete: vi.fn((path: string, ...args: unknown[]) => {
      handlers[`DELETE:${path}`] = args[args.length - 1];
    }),
    patch: vi.fn((path: string, ...args: unknown[]) => {
      handlers[`PATCH:${path}`] = args[args.length - 1];
    }),
  };

  return {
    router: mockRouter,
    handlers,
    getHandler: (method: string, path: string) =>
      handlers[`${method.toUpperCase()}:${path}`],
  };
}
