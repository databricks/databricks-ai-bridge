import { describe, expect, it } from 'vitest'
import {
  createDatabricksGenieAgent,
  createDatabricksGenieConversationClient,
  createDatabricksProvider,
} from '../src'

describe('genie package exports', () => {
  it('exports the existing Databricks provider alongside Genie entry points', () => {
    expect(typeof createDatabricksProvider).toBe('function')
    expect(typeof createDatabricksGenieConversationClient).toBe('function')
    expect(typeof createDatabricksGenieAgent).toBe('function')
  })
})
