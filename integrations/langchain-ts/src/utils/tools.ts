/**
 * Tool conversion utilities between LangChain and AI SDK formats
 */

import {
  StructuredToolInterface,
  StructuredToolParams,
  isStructuredTool,
  isRunnableToolLike,
  isStructuredToolParams,
} from "@langchain/core/tools";
import { RunnableToolLike } from "@langchain/core/runnables";
import { isOpenAITool } from "@langchain/core/language_models/base";
import { BindToolsInput } from "@langchain/core/language_models/chat_models";
import type { ZodType } from "zod";
import { tool, jsonSchema, type ToolSet, TextStreamPart, TypedToolCall } from "ai";
import { DATABRICKS_TOOL_CALL_ID, DATABRICKS_TOOL_DEFINITION } from "@databricks/ai-sdk-provider";

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
 * Convert a schema (Zod or JSON Schema) to JSON Schema format
 */
function convertSchemaToJsonSchema(schema: unknown): Record<string, unknown> {
  if (!schema) {
    return { type: "object", properties: {} };
  }
  if (isZodSchema(schema)) {
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
    [DATABRICKS_TOOL_CALL_ID]: DATABRICKS_TOOL_DEFINITION
  };

  for (const t of tools) {
    // OpenAI format: { type: "function", function: { name, description, parameters } }
    if (isOpenAITool(t)) {
      const name = t.function.name;
      const schema = t.function.parameters ?? { type: "object", properties: {} };

      toolSet[name] = tool({
        description: t.function.description,
        inputSchema: jsonSchema(schema as Parameters<typeof jsonSchema>[0]),
      });
      continue;
    }

    // RunnableToolLike: Runnable converted to tool via .asTool()
    // Has: name, description?, schema
    if (isRunnableToolLike(t)) {
      const runnableTool = t as RunnableToolLike;
      const name = runnableTool.name;
      const schema = convertSchemaToJsonSchema(runnableTool.schema);

      toolSet[name] = tool({
        description: runnableTool.description,
        inputSchema: jsonSchema(schema as Parameters<typeof jsonSchema>[0]),
      });
      continue;
    }

    // StructuredToolInterface: LangChain StructuredTool
    // Has: name, description, schema, returnDirect, extras?
    if (isStructuredTool(t)) {
      const structuredTool = t as StructuredToolInterface;
      const name = structuredTool.name;
      const schema = convertSchemaToJsonSchema(structuredTool.schema);

      toolSet[name] = tool({
        description: structuredTool.description,
        inputSchema: jsonSchema(schema as Parameters<typeof jsonSchema>[0]),
      });
      continue;
    }

    // StructuredToolParams: Minimal tool definition
    // Has: name, schema, extras?, description?
    if (isStructuredToolParams(t)) {
      const toolParams = t as StructuredToolParams;
      const name = toolParams.name;
      const schema = convertSchemaToJsonSchema(toolParams.schema);

      toolSet[name] = tool({
        description: toolParams.description,
        inputSchema: jsonSchema(schema as Parameters<typeof jsonSchema>[0]),
      });
      continue;
    }

    // Generic Record<string, any> - fallback for plain objects
    // Try to extract name, description, and parameters/schema
    if (typeof t === "object" && t !== null && "name" in t) {
      const obj = t as Record<string, unknown>;
      const name = String(obj.name);

      // Look for schema in various forms: parameters, schema, inputSchema
      const rawSchema = obj.parameters ?? obj.schema ?? obj.inputSchema;
      const schema = convertSchemaToJsonSchema(rawSchema);

      toolSet[name] = tool({
        description: obj.description as string | undefined,
        inputSchema: jsonSchema(schema as Parameters<typeof jsonSchema>[0]),
      });
      continue;
    }

    throw new Error(`Unsupported tool type: ${JSON.stringify(t)}`);
  }

  return toolSet;
}

export const getToolNameFromAiSDKTool = (toolStreamPart: Extract<TextStreamPart<ToolSet>, { type: 'tool-call' | 'tool-input-start' }> | TypedToolCall<ToolSet>) => {
  if (toolStreamPart.toolName === DATABRICKS_TOOL_CALL_ID && toolStreamPart.providerMetadata?.databricks?.toolName) {
    return toolStreamPart.providerMetadata.databricks.toolName as string;
  }
  return toolStreamPart.toolName;
}
