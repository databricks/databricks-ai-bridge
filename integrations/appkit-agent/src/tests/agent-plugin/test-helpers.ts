import { vi } from "vitest";

/**
 * Creates a mock Express request object.
 */
export function createMockRequest(overrides: any = {}) {
  return {
    params: {},
    query: {},
    body: {},
    headers: {},
    header(name: string) {
      return (this as any).headers[name.toLowerCase()];
    },
    ...overrides,
  };
}

/**
 * Creates a mock Express response object with chainable methods.
 */
export function createMockResponse(): Record<string, any> {
  const eventListeners: Record<string, Array<(...args: any[]) => void>> = {};

  const res = {
    status: vi.fn().mockReturnThis(),
    json: vi.fn().mockReturnThis(),
    send: vi.fn().mockReturnThis(),
    sendStatus: vi.fn().mockReturnThis(),
    end: vi.fn(function (this: any) {
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
      this: any,
      event: string,
      handler: (...args: any[]) => void,
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

/**
 * Creates a mock Express router with route handler capturing.
 */
export function createMockRouter(): Record<string, any> {
  const handlers: Record<string, any> = {};

  const mockRouter = {
    get: vi.fn((path: string, handler: any) => {
      handlers[`GET:${path}`] = handler;
    }),
    post: vi.fn((path: string, handler: any) => {
      handlers[`POST:${path}`] = handler;
    }),
    put: vi.fn((path: string, handler: any) => {
      handlers[`PUT:${path}`] = handler;
    }),
    delete: vi.fn((path: string, handler: any) => {
      handlers[`DELETE:${path}`] = handler;
    }),
    patch: vi.fn((path: string, handler: any) => {
      handlers[`PATCH:${path}`] = handler;
    }),
  };

  return {
    router: mockRouter,
    handlers,
    getHandler: (method: string, path: string) =>
      handlers[`${method.toUpperCase()}:${path}`],
  };
}
