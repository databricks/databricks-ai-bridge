import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ChatPlugin, chat } from "../../chat-plugin/index";
import type { ChatConfig } from "../../chat-plugin/types";
import { createMockRouter } from "./test-helpers";

vi.mock("@databricks/appkit", () => {
  class MockPlugin<T> {
    protected config: T;
    private endpoints = new Map<string, string>();

    constructor(config: T) {
      this.config = config;
    }

    registerEndpoint(name: string, path: string) {
      this.endpoints.set(name, path);
    }

    setup(): Promise<void> {
      return Promise.resolve();
    }

    injectRoutes(_router: unknown) {}
    exports() {
      return {};
    }
  }

  return {
    Plugin: MockPlugin,
    toPlugin: (Ctor: new (config: unknown) => unknown) => {
      return (config: unknown) => {
        const instance = new Ctor(config) as Record<string, unknown>;
        return {
          name: instance.name,
          instance,
        };
      };
    },
  };
});

describe("ChatPlugin", () => {
  beforeEach(() => {
    process.env.DATABRICKS_HOST = "https://test.databricks.com";
    process.env.DATABRICKS_TOKEN = "test-token";
  });

  afterEach(() => {
    delete process.env.DATABRICKS_HOST;
    delete process.env.DATABRICKS_TOKEN;
  });

  test("chat factory produces plugin with name chat", () => {
    const pluginData = chat({});
    expect(pluginData.name).toBe("chat");
  });

  test("plugin has correct manifest", () => {
    expect(ChatPlugin.manifest).toBeDefined();
    expect(ChatPlugin.manifest.name).toBe("chat");
  });

  test("plugin instance has correct name", () => {
    const config: ChatConfig = {};
    const plugin = new ChatPlugin(config);
    expect(plugin.name).toBe("chat");
  });

  describe("setup()", () => {
    test("initializes without pool (ephemeral mode)", async () => {
      const plugin = new ChatPlugin({});
      await expect(plugin.setup()).resolves.toBeUndefined();
    });
  });

  describe("injectRoutes()", () => {
    test("registers config, session, history, and chat routes", async () => {
      const plugin = new ChatPlugin({});
      await plugin.setup();

      const { router } = createMockRouter() as {
        router: Record<string, ReturnType<typeof vi.fn>>;
      };
      plugin.injectRoutes(router as never);

      expect(router.get).toHaveBeenCalledWith("/config", expect.any(Function));
      expect(router.get).toHaveBeenCalledWith("/session", expect.any(Function));
      expect(router.get).toHaveBeenCalledWith(
        "/history",
        expect.any(Function),
        expect.any(Function),
      );
      expect(router.post).toHaveBeenCalledWith(
        "/",
        expect.any(Function),
        expect.any(Function),
      );
    });
  });
});
