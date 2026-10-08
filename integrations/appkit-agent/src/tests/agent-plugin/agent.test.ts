import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { IAgentConfig } from "../../agent-plugin/types";
import { StubAgent } from "./stub-agent";
import { createMockRouter } from "./test-helpers";

// ---------------------------------------------------------------------------
// Mock @databricks/appkit — provide minimal Plugin base class + toPlugin
// ---------------------------------------------------------------------------

vi.mock("@databricks/appkit", () => {
  class Plugin<T = any> {
    protected config: T;
    constructor(config: T) {
      this.config = config;
    }
    registerEndpoint(_name: string, _path: string) {}
    async abortActiveOperations() {}
  }

  function toPlugin(PluginClass: any) {
    return (config: any) => {
      const instance = new PluginClass(config);
      return { name: instance.name, PluginClass, config };
    };
  }

  return { Plugin, toPlugin };
});

import { AgentPlugin, agent } from "../../agent-plugin/agent";

describe("AgentPlugin", () => {
  const savedEnv = { ...process.env };

  beforeEach(() => {
    delete process.env.DATABRICKS_MODEL;
  });

  afterEach(() => {
    process.env = { ...savedEnv };
  });

  test("agent factory produces correct plugin data", () => {
    const pluginData = agent({ agentInstance: new StubAgent() });
    expect(pluginData.name).toBe("agent");
  });

  test("plugin has correct manifest", () => {
    expect(AgentPlugin.manifest).toBeDefined();
    expect(AgentPlugin.manifest.name).toBe("agent");
    expect(AgentPlugin.manifest.resources.required).toHaveLength(1);
    expect(AgentPlugin.manifest.resources.required[0].type).toBe(
      "serving_endpoint",
    );
  });

  test("plugin instance has correct name", () => {
    const config: IAgentConfig = { agentInstance: new StubAgent() };
    const plugin = new AgentPlugin(config);
    expect(plugin.name).toBe("agent");
  });

  describe("setup()", () => {
    test("uses provided agentInstance", async () => {
      const stub = new StubAgent();
      const config: IAgentConfig = { agentInstance: stub };
      const plugin = new AgentPlugin(config);

      await plugin.setup();

      const exported = plugin.exports();
      const result = await exported.invoke([{ role: "user", content: "hi" }]);
      expect(result).toContain("Echo: hi");
    });

    test("throws when no model and no agentInstance", async () => {
      const config: IAgentConfig = {};
      const plugin = new AgentPlugin(config);

      await expect(plugin.setup()).rejects.toThrow("model name is required");
    });

    test("resolves model from env var when not in config", async () => {
      process.env.DATABRICKS_MODEL = "test-model";

      const config: IAgentConfig = {};
      const plugin = new AgentPlugin(config);

      // Will fail at dynamic import of @databricks/langchainjs, but should
      // get past the model name check
      try {
        await plugin.setup();
      } catch (e: any) {
        expect(e.message).not.toContain("model name is required");
      }

      delete process.env.DATABRICKS_MODEL;
    });
  });

  describe("injectRoutes()", () => {
    test("registers POST handler on router", () => {
      const stub = new StubAgent();
      const config: IAgentConfig = { agentInstance: stub };
      const plugin = new AgentPlugin(config);

      const { router } = createMockRouter();
      plugin.injectRoutes(router as any);

      expect(router.post).toHaveBeenCalledWith("/", expect.any(Function));
    });
  });

  describe("exports()", () => {
    test("returns invoke, stream, and addTools methods", async () => {
      const stub = new StubAgent();
      const config: IAgentConfig = { agentInstance: stub };
      const plugin = new AgentPlugin(config);
      await plugin.setup();

      const exported = plugin.exports();

      expect(typeof exported.invoke).toBe("function");
      expect(typeof exported.stream).toBe("function");
      expect(typeof exported.addTools).toBe("function");
    });

    test("invoke returns text from agent response", async () => {
      const stub = new StubAgent();
      const config: IAgentConfig = { agentInstance: stub };
      const plugin = new AgentPlugin(config);
      await plugin.setup();

      const result = await plugin
        .exports()
        .invoke([{ role: "user", content: "test message" }]);

      expect(result).toBe("Echo: test message");
    });

    test("stream yields ResponseStreamEvents", async () => {
      const stub = new StubAgent();
      const config: IAgentConfig = { agentInstance: stub };
      const plugin = new AgentPlugin(config);
      await plugin.setup();

      const events: any[] = [];
      for await (const event of plugin
        .exports()
        .stream([{ role: "user", content: "hello" }])) {
        events.push(event);
      }

      expect(events.length).toBeGreaterThan(0);
      const deltaEvent = events.find(
        (e) => e.type === "response.output_text.delta",
      );
      expect(deltaEvent).toBeDefined();
      expect(deltaEvent.delta).toContain("Echo: hello");

      const completedEvent = events.find(
        (e) => e.type === "response.completed",
      );
      expect(completedEvent).toBeDefined();
    });

    test("invoke history contains only messages before the last user message", async () => {
      const spyAgent = {
        invoke: vi.fn().mockResolvedValue([
          {
            id: "msg_1",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: "ok", annotations: [] }],
          },
        ]),
        stream: vi.fn(),
      };
      const config: IAgentConfig = { agentInstance: spyAgent as any };
      const plugin = new AgentPlugin(config);
      await plugin.setup();

      await plugin.exports().invoke([
        { role: "user", content: "first" },
        { role: "user", content: "second" },
        { role: "assistant", content: "interrupted" },
      ]);

      expect(spyAgent.invoke).toHaveBeenCalledWith(
        expect.objectContaining({
          input: "second",
          chat_history: [{ role: "user", content: "first" }],
        }),
      );
    });

    test("throws when not initialized", async () => {
      const config: IAgentConfig = { agentInstance: new StubAgent() };
      const plugin = new AgentPlugin(config);

      await expect(
        plugin.exports().invoke([{ role: "user", content: "hi" }]),
      ).rejects.toThrow("not initialized");
    });
  });

  describe("abortActiveOperations()", () => {
    function injectMcpClient(
      plugin: AgentPlugin,
      close: ReturnType<typeof vi.fn>,
    ) {
      Reflect.set(plugin, "mcpClient", { getTools: vi.fn(), close });
    }

    test("closes MCP client when present", async () => {
      const stub = new StubAgent();
      const config: IAgentConfig = { agentInstance: stub };
      const plugin = new AgentPlugin(config);
      await plugin.setup();

      const closeFn = vi.fn().mockResolvedValue(undefined);
      injectMcpClient(plugin, closeFn);

      await plugin.abortActiveOperations();

      expect(closeFn).toHaveBeenCalledOnce();
    });
  });

  describe("addTools()", () => {
    test("throws when using agentInstance mode", async () => {
      const stub = new StubAgent();
      const config: IAgentConfig = { agentInstance: stub };
      const plugin = new AgentPlugin(config);
      await plugin.setup();

      await expect(
        plugin.exports().addTools([
          {
            type: "function",
            name: "test",
            execute: async () => "result",
          },
        ]),
      ).rejects.toThrow("not supported when using a custom agentInstance");
    });
  });
});
