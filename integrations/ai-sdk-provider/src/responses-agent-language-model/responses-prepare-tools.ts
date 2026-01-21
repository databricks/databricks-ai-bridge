import type { LanguageModelV2FunctionTool, LanguageModelV2ToolChoice } from '@ai-sdk/provider'
import { DATABRICKS_TOOL_CALL_ID } from '../tools'

export type ResponsesTool = {
  type: 'function'
  name: string
  description?: string
  parameters?: unknown
  strict?: boolean
}

export type ResponsesToolChoice = 'auto' | 'none' | 'required' | { type: 'function'; name: string }

/**
 * Prepare tools for the Responses API format.
 * Unlike the chat completions API, the responses API expects function tools
 * with name, description, and parameters at the top level (not nested under 'function').
 */
export function prepareResponsesTools({
  tools,
  toolChoice,
}: {
  tools?: Array<LanguageModelV2FunctionTool | { type: 'provider-defined'; id: string }>
  toolChoice?: LanguageModelV2ToolChoice
}): {
  tools?: Array<ResponsesTool>
  toolChoice?: ResponsesToolChoice
} {
  // When the tools array is empty, change it to undefined to prevent errors
  if (!tools || tools.length === 0) {
    return { tools: undefined, toolChoice: undefined }
  }

  const responsesTools: Array<ResponsesTool> = []

  for (const tool of tools) {
    if (tool.type === 'provider-defined' || tool.name === DATABRICKS_TOOL_CALL_ID) {
      // Skip provider-defined tools and Databricks-orchestrated tools
      continue
    }

    // Function tools - responses API format has properties at top level
    responsesTools.push({
      type: 'function',
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema,
    })
  }

  if (responsesTools.length === 0) {
    return { tools: undefined, toolChoice: undefined }
  }

  // Convert tool choice
  const convertedToolChoice = convertResponsesToolChoice(toolChoice)

  return {
    tools: responsesTools,
    toolChoice: convertedToolChoice,
  }
}

function convertResponsesToolChoice(
  toolChoice: LanguageModelV2ToolChoice | undefined
): ResponsesToolChoice | undefined {
  if (!toolChoice) {
    return undefined
  }

  switch (toolChoice.type) {
    case 'auto':
      return 'auto'
    case 'none':
      return 'none'
    case 'required':
      return 'required'
    case 'tool':
      return { type: 'function', name: toolChoice.toolName }
    default:
      return undefined
  }
}
