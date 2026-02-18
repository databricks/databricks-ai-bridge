import { z } from 'zod';

export const DATABRICKS_TOOL_CALL_ID = 'databricks-tool-call' as const;

export const DATABRICKS_TOOL_DEFINITION = {
  name: DATABRICKS_TOOL_CALL_ID,
  description: 'Databricks tool call',
  inputSchema: z.any(),
  outputSchema: z.any(),
};

export const MCP_APPROVAL_STATUS_KEY = '__approvalStatus__' as const;
export const MCP_APPROVAL_REQUEST_TYPE = 'mcp-approval-request' as const;
export const MCP_APPROVAL_RESPONSE_TYPE = 'mcp-approval-response' as const;

export type ApprovalStatusOutput = {
  [MCP_APPROVAL_STATUS_KEY]: boolean;
};

export function isApprovalStatusOutput(
  output: unknown,
): output is ApprovalStatusOutput {
  return (
    typeof output === 'object' &&
    output !== null &&
    MCP_APPROVAL_STATUS_KEY in output &&
    typeof (output as Record<string, unknown>)[MCP_APPROVAL_STATUS_KEY] ===
      'boolean'
  );
}

export function createApprovalStatusOutput(
  approve: boolean,
): ApprovalStatusOutput {
  return { [MCP_APPROVAL_STATUS_KEY]: approve };
}

/**
 * Extract approval status from a tool output value.
 *
 * @example
 * const status = extractApprovalStatus(output);
 * if (status === true) { // approved }
 * if (status === false) { // denied }
 * if (status === undefined) { // not an approval status output }
 */
export function extractApprovalStatus(output: unknown): boolean | undefined {
  if (isApprovalStatusOutput(output)) return output[MCP_APPROVAL_STATUS_KEY];
  return undefined;
}

/**
 * Extract approval status from a tool result's output value.
 * Handles the nested structure where output.type === 'json' and value contains the status.
 */
export function extractApprovalStatusFromToolResult(output: {
  type: string;
  value?: unknown;
}): boolean | undefined {
  if (
    output.type === 'json' &&
    output.value &&
    typeof output.value === 'object' &&
    MCP_APPROVAL_STATUS_KEY in output.value
  ) {
    const value = (output.value as Record<string, unknown>)[
      MCP_APPROVAL_STATUS_KEY
    ];
    if (typeof value === 'boolean') return value;
  }
  return undefined;
}
