/**
 * Responses API invoke handler for the agent plugin.
 *
 * Accepts Responses API request format, parses it into InvokeParams, then
 * delegates to AgentInterface.stream() / AgentInterface.invoke(). The handler
 * is a pure pass-through — all SSE event shaping happens inside the agent.
 */

import type express from "express";
import { z } from "zod";
import type { AgentInterface } from "./agent-interface";

const responsesRequestSchema = z.object({
  input: z.union([
    z.string(),
    z.array(
      z.union([
        z.object({
          role: z.enum(["user", "assistant", "system"]),
          content: z.union([
            z.string(),
            z.array(
              z.union([
                z.object({ type: z.string(), text: z.string() }).passthrough(),
                z.object({ type: z.string() }).passthrough(),
              ]),
            ),
          ]),
        }),
        z.object({ type: z.string() }).passthrough(),
      ]),
    ),
  ]),
  stream: z.boolean().optional().default(true),
  model: z.string().optional(),
});

/**
 * Flatten a Responses API message to a plain `{ role, content }` object.
 * Handles function_call / function_call_output items and array content.
 */
function flattenHistoryItem(item: any): { role: string; content: string } {
  if (item.type === "function_call") {
    return {
      role: "assistant",
      content: `[Tool Call: ${item.name}(${item.arguments})]`,
    };
  }
  if (item.type === "function_call_output") {
    return { role: "assistant", content: `[Tool Result: ${item.output}]` };
  }

  if (Array.isArray(item.content)) {
    const textParts = item.content
      .filter(
        (p: any) =>
          p.type === "input_text" ||
          p.type === "output_text" ||
          p.type === "text",
      )
      .map((p: any) => p.text);

    const toolParts = item.content
      .filter(
        (p: any) =>
          p.type === "function_call" || p.type === "function_call_output",
      )
      .map((p: any) =>
        p.type === "function_call"
          ? `[Tool Call: ${p.name}(${JSON.stringify(p.arguments)})]`
          : `[Tool Result: ${p.output}]`,
      );

    const allParts = [...textParts, ...toolParts].filter((p) => p.length > 0);
    return { ...item, content: allParts.join("\n") };
  }

  return { role: item.role ?? "user", content: item.content ?? "" };
}

/**
 * Create an Express handler that invokes the agent via the AgentInterface
 * and streams/returns the response in Responses API format.
 */
export function createInvokeHandler(
  getAgent: () => AgentInterface,
): express.RequestHandler {
  return async (req: express.Request, res: express.Response) => {
    try {
      const parsed = responsesRequestSchema.safeParse(req.body);
      if (!parsed.success) {
        res.status(400).json({
          error: "Invalid request format",
          details: parsed.error.format(),
        });
        return;
      }

      const { stream } = parsed.data;

      const input =
        typeof parsed.data.input === "string"
          ? [{ role: "user" as const, content: parsed.data.input }]
          : parsed.data.input;

      const userMessages = input.filter((msg: any) => msg.role === "user");
      if (userMessages.length === 0) {
        res.status(400).json({ error: "No user message found in input" });
        return;
      }

      const lastUserMessage = userMessages[userMessages.length - 1];

      let userInput: string;
      if (Array.isArray(lastUserMessage.content)) {
        userInput = lastUserMessage.content
          .filter(
            (part: any) => part.type === "input_text" || part.type === "text",
          )
          .map((part: any) => part.text)
          .join("\n");
      } else {
        userInput = lastUserMessage.content as string;
      }

      const lastUserIdx = input.findLastIndex(
        (msg: any) => msg.role === "user",
      );
      const chatHistory = input.slice(0, lastUserIdx).map(flattenHistoryItem);

      const agentParams = { input: userInput, chat_history: chatHistory };
      const agent = getAgent();

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");

        try {
          for await (const event of agent.stream(agentParams)) {
            res.write(`data: ${JSON.stringify(event)}\n\n`);
          }
          res.write("data: [DONE]\n\n");
          res.end();
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : String(err);
          res.write(
            `data: ${JSON.stringify({ type: "error", error: message })}\n\n`,
          );
          res.write(`data: ${JSON.stringify({ type: "response.failed" })}\n\n`);
          res.write("data: [DONE]\n\n");
          res.end();
        }
      } else {
        const items = await agent.invoke(agentParams);
        res.json({ output: items });
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      res.status(500).json({ error: "Internal server error", message });
    }
  };
}
