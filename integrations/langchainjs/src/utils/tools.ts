/**
 * Tool conversion utilities between LangChain and AI SDK formats
 */

import {
  isStructuredTool,
  isRunnableToolLike,
  isStructuredToolParams,
} from "@langchain/core/tools";
import { isOpenAITool } from "@langchain/core/language_models/base";
import { BindToolsInput } from "@langchain/core/language_models/chat_models";
import { tool, jsonSchema, type ToolSet, TextStreamPart, TypedToolCall } from "ai";
import { DATABRICKS_TOOL_CALL_ID, DATABRICKS_TOOL_DEFINITION } from "@databricks/ai-sdk-provider";
import { ZodType } from "zod/v4";

/**
 * Convert a schema (Zod or JSON Schema) to JSON Schema format
 */
function convertSchemaToJsonSchema(schema: unknown): Record<string, unknown> {
  if (!schema) {
    return { type: "object", properties: {} };
  }
  if (schema instanceof ZodType) {
    return schema.toJSONSchema();
  }
  // Already a JSON schema object
  return schema as Record<string, unknown>;
}

/**
 * Convert LangChain tools to AI SDK ToolSet format
 *
 * Supports all BindToolsInput types:
 * - StructuredToolInterface (LangChain StructuredTool)
 * - RunnableToolLike (Runnable converted to tool via .asTool())
 * - StructuredToolParams (minimal tool definition with name, schema, extras)
 * - ToolDefinition (OpenAI format: { type: "function", function: {...} })
 * - Record<string, any> (generic object with name and parameters)
 *
 * The AI SDK expects tools as a Record<string, Tool> where each Tool
 * is created using the tool() helper function.
 */
export function convertToAISDKToolSet(tools: BindToolsInput[]): ToolSet {
  const toolSet: ToolSet = {
    // Include the Databricks tool to allow for tools executed by the model to be called
    [DATABRICKS_TOOL_CALL_ID]: DATABRICKS_TOOL_DEFINITION,
  };

  for (const toolInput of tools) {
    // OpenAI format: { type: "function", function: { name, description, parameters } }
    if (isOpenAITool(toolInput)) {
      const name = toolInput.function.name;
      const schema = toolInput.function.parameters ?? { type: "object", properties: {} };

      toolSet[name] = tool({
        description: toolInput.function.description,
        inputSchema: jsonSchema(schema as Parameters<typeof jsonSchema>[0]),
      });
      continue;
    }

    // RunnableToolLike: Runnable converted to tool via .asTool()
    // Has: name, description?, schema
    if (isRunnableToolLike(toolInput)) {
      const name = toolInput.name;
      const schema = convertSchemaToJsonSchema(toolInput.schema);

      toolSet[name] = tool({
        description: toolInput.description,
        inputSchema: jsonSchema(schema as Parameters<typeof jsonSchema>[0]),
      });
      continue;
    }

    // StructuredToolInterface: LangChain StructuredTool
    // Has: name, description, schema, returnDirect, extras?
    if (isStructuredTool(toolInput)) {
      const name = toolInput.name;
      const schema = convertSchemaToJsonSchema(toolInput.schema);

      toolSet[name] = tool({
        description: toolInput.description,
        inputSchema: jsonSchema(schema as Parameters<typeof jsonSchema>[0]),
      });
      continue;
    }

    // StructuredToolParams: Minimal tool definition
    // Has: name, schema, extras?, description?
    if (isStructuredToolParams(toolInput)) {
      const name = toolInput.name;
      const schema = convertSchemaToJsonSchema(toolInput.schema);

      toolSet[name] = tool({
        description: toolInput.description,
        inputSchema: jsonSchema(schema as Parameters<typeof jsonSchema>[0]),
      });
      continue;
    }

    // Generic Record<string, any> - fallback for plain objects
    // Try to extract name, description, and parameters/schema
    if (typeof toolInput === "object" && toolInput !== null && "name" in toolInput) {
      const name = String(toolInput.name);

      // Look for schema in various forms: parameters, schema, inputSchema
      const rawSchema = toolInput.parameters ?? toolInput.schema ?? toolInput.inputSchema;
      const schema = convertSchemaToJsonSchema(rawSchema);

      toolSet[name] = tool({
        description: toolInput.description as string | undefined,
        inputSchema: jsonSchema(schema as Parameters<typeof jsonSchema>[0]),
      });
      continue;
    }

    throw new Error(`Unsupported tool type: ${JSON.stringify(toolInput)}`);
  }

  return toolSet;
}

export const getToolNameFromAiSDKTool = (
  toolStreamPart:
    | Extract<TextStreamPart<ToolSet>, { type: "tool-call" | "tool-input-start" }>
    | TypedToolCall<ToolSet>
): string => {
  if (
    toolStreamPart.toolName === DATABRICKS_TOOL_CALL_ID &&
    toolStreamPart.providerMetadata?.databricks?.toolName
  ) {
    return toolStreamPart.providerMetadata.databricks.toolName as string;
  }
  return toolStreamPart.toolName;
};
