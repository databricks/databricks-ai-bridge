import { describe, expect, it } from 'vitest'
import { DATABRICKS_TOOL_DEFINITION } from '../src/tools'

describe('tools', () => {
  it('DATABRICKS_TOOL_DEFINITION has correct structure', () => {
    expect(DATABRICKS_TOOL_DEFINITION.name).toBe('databricks-tool-call')
    expect(DATABRICKS_TOOL_DEFINITION.description).toBe('Databricks tool call')
    expect(DATABRICKS_TOOL_DEFINITION.inputSchema).toBeDefined()
    expect(DATABRICKS_TOOL_DEFINITION.outputSchema).toBeDefined()
  })
})
