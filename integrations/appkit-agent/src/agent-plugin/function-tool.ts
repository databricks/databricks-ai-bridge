/**
 * OpenResponses-aligned function tool definition and converter.
 *
 * Users define tools as plain objects matching the OpenResponses FunctionTool
 * schema (type, name, description, parameters as JSON Schema). Internally we
 * convert them to LangChain StructuredTool instances for LangGraph.
 */

import type { StructuredToolInterface } from "@langchain/core/tools";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Public type — matches OpenResponses FunctionToolParam + execute handler
// ---------------------------------------------------------------------------

export interface FunctionTool {
  type: "function";
  name: string;
  description?: string | null;
  /** JSON Schema object describing the tool's parameters. */
  parameters?: Record<string, unknown> | null;
  strict?: boolean | null;
  /** Handler invoked when the model calls this tool. */
  execute: (args: Record<string, unknown>) => Promise<string> | string;
}

// ---------------------------------------------------------------------------
// Type guard
// ---------------------------------------------------------------------------

export function isFunctionTool(t: unknown): t is FunctionTool {
  return (
    typeof t === "object" &&
    t !== null &&
    (t as any).type === "function" &&
    typeof (t as any).name === "string" &&
    typeof (t as any).execute === "function"
  );
}

// ---------------------------------------------------------------------------
// Converter: FunctionTool → LangChain StructuredToolInterface
// ---------------------------------------------------------------------------

/**
 * Converts a JSON Schema properties object to a Zod schema.
 *
 * Supports the primitive types models actually use for tool parameters:
 * string, number, integer, boolean, array, and object. Anything else falls
 * back to z.any().
 */
function jsonSchemaPropertiesToZod(
  properties: Record<string, any>,
  required: string[] = [],
): z.ZodObject<any> {
  const shape: Record<string, z.ZodTypeAny> = {};

  for (const [key, prop] of Object.entries(properties)) {
    let field: z.ZodTypeAny;

    switch (prop.type) {
      case "string":
        field = z.string();
        break;
      case "number":
        field = z.number();
        break;
      case "integer":
        field = z.number().int();
        break;
      case "boolean":
        field = z.boolean();
        break;
      case "array":
        field = z.array(z.any());
        break;
      case "object":
        if (prop.properties) {
          field = jsonSchemaPropertiesToZod(
            prop.properties,
            prop.required ?? [],
          );
        } else {
          field = z.record(z.string(), z.any());
        }
        break;
      default:
        field = z.any();
    }

    if (prop.description) {
      field = field.describe(prop.description);
    }

    if (prop.enum && Array.isArray(prop.enum) && prop.enum.length > 0) {
      field = z.enum(prop.enum as [string, ...string[]]);
      if (prop.description) {
        field = field.describe(prop.description);
      }
    }

    if (!required.includes(key)) {
      field = field.optional();
    }

    shape[key] = field;
  }

  return z.object(shape);
}

/**
 * Convert a single FunctionTool to a LangChain DynamicStructuredTool.
 */
export function functionToolToStructuredTool(
  tool: FunctionTool,
): StructuredToolInterface {
  const params = tool.parameters as Record<string, any> | null | undefined;
  const schema = params?.properties
    ? jsonSchemaPropertiesToZod(params.properties, params.required ?? [])
    : z.object({});

  return new DynamicStructuredTool({
    name: tool.name,
    description: tool.description ?? "",
    schema,
    func: async (args: Record<string, unknown>) => {
      const result = await tool.execute(args);
      return result;
    },
  });
}

/**
 * Convert an array of FunctionTool definitions to LangChain StructuredToolInterface[].
 */
export function functionToolsToStructuredTools(
  tools: FunctionTool[],
): StructuredToolInterface[] {
  return tools.map(functionToolToStructuredTool);
}
