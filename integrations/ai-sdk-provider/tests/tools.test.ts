import { describe, expect, it } from 'vitest'
import { DATABRICKS_TOOL_CALL_ID, DATABRICKS_TOOL_DEFINITION } from '../src/tools'

describe('tools', () => {
  it('exports DATABRICKS_TOOL_CALL_ID as expected string', () => {
    expect(DATABRICKS_TOOL_CALL_ID).toBe('databricks-tool-call')
  })

  it('DATABRICKS_TOOL_DEFINITION has correct structure', () => {
    expect(DATABRICKS_TOOL_DEFINITION.name).toBe(DATABRICKS_TOOL_CALL_ID)
    expect(DATABRICKS_TOOL_DEFINITION.description).toBe('Databricks tool call')
    expect(DATABRICKS_TOOL_DEFINITION.inputSchema).toBeDefined()
    expect(DATABRICKS_TOOL_DEFINITION.outputSchema).toBeDefined()
  })
})
