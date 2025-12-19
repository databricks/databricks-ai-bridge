/**
 * Tool conversion utilities between LangChain and OpenAI formats
 */

import { StructuredToolInterface } from "@langchain/core/tools";
import { zodToJsonSchema } from "zod-to-json-schema";
import type { ZodType } from "zod";
import type {
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
} from "../types.js";

/**
 * Input types that can be bound as tools
 */
export type BindToolsInput =
  | StructuredToolInterface
  | Record<string, unknown>
  | ChatCompletionTool;

/**
 * Check if a schema is a Zod schema
 */
function isZodSchema(schema: unknown): schema is ZodType {
  return (
    typeof schema === "object" &&
    schema !== null &&
    "_def" in schema &&
    "parse" in schema &&
    typeof (schema as ZodType).parse === "function"
  );
}

/**
 * Convert LangChain tools to OpenAI-compatible tool format
 */
export function convertToOpenAITools(tools: BindToolsInput[]): ChatCompletionTool[] {
  return tools.map((tool): ChatCompletionTool => {
    // Already in OpenAI format
    if (isOpenAITool(tool)) {
      return tool;
    }

    // StructuredTool from LangChain
    if (isStructuredTool(tool)) {
      let parameters: Record<string, unknown>;

      // Check if schema is a Zod schema or already JSON schema
      if (isZodSchema(tool.schema)) {
        parameters = zodToJsonSchema(tool.schema) as Record<string, unknown>;
      } else {
        // Already a JSON schema object
        parameters = tool.schema as Record<string, unknown>;
      }

      return {
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters,
        },
      };
    }

    // Plain object with function definition
    if (isToolDefinition(tool)) {
      return {
        type: "function",
        function: {
          name: tool.name as string,
          description: tool.description as string | undefined,
          parameters: tool.parameters as Record<string, unknown> | undefined,
        },
      };
    }

    throw new Error(`Unsupported tool type: ${JSON.stringify(tool)}`);
  });
}

/**
 * Convert tool_choice parameter to OpenAI format
 */
export function convertToolChoice(
  toolChoice?: "auto" | "none" | "required" | "any" | string | { type: string; function?: { name: string } }
): ChatCompletionToolChoiceOption | undefined {
  if (toolChoice === undefined) {
    return undefined;
  }

  // String values
  if (typeof toolChoice === "string") {
    switch (toolChoice) {
      case "auto":
      case "none":
      case "required":
        return toolChoice;
      case "any":
        // "any" maps to "required" in OpenAI format
        return "required";
      default:
        // Assume it's a tool name - force that specific tool
        return {
          type: "function",
          function: { name: toolChoice },
        };
    }
  }

  // Object with type and function
  if (typeof toolChoice === "object" && toolChoice !== null) {
    if (toolChoice.type === "function" && toolChoice.function?.name) {
      return {
        type: "function",
        function: { name: toolChoice.function.name },
      };
    }
  }

  return undefined;
}

/**
 * Type guard: Check if tool is already in OpenAI format
 */
function isOpenAITool(tool: BindToolsInput): tool is ChatCompletionTool {
  return (
    typeof tool === "object" &&
    tool !== null &&
    "type" in tool &&
    tool.type === "function" &&
    "function" in tool &&
    typeof tool.function === "object" &&
    tool.function !== null &&
    "name" in tool.function
  );
}

/**
 * Type guard: Check if tool is a LangChain StructuredTool
 */
function isStructuredTool(tool: BindToolsInput): tool is StructuredToolInterface {
  return (
    typeof tool === "object" &&
    tool !== null &&
    "name" in tool &&
    "description" in tool &&
    "schema" in tool &&
    typeof (tool as StructuredToolInterface).schema !== "undefined"
  );
}

/**
 * Type guard: Check if tool is a plain object tool definition
 */
function isToolDefinition(
  tool: BindToolsInput
): tool is Record<string, unknown> & { name: string } {
  return (
    typeof tool === "object" &&
    tool !== null &&
    "name" in tool &&
    typeof tool.name === "string" &&
    !("type" in tool && tool.type === "function") &&
    !("schema" in tool)
  );
}
