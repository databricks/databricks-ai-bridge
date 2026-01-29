import { describe, expect, it } from 'vitest'
import type { LanguageModelV3StreamPart } from '@ai-sdk/provider'
import { applyDeltaBoundaryTransform } from '../src/stream-transformers/databricks-delta-boundary'
import { composeDatabricksStreamPartTransformers } from '../src/stream-transformers/compose-stream-part-transformers'

describe('applyDeltaBoundaryTransform', () => {
  it('injects text-start before first text-delta', () => {
    const parts: LanguageModelV3StreamPart[] = [{ type: 'text-delta', id: '1', delta: 'hi' }]
    const result = applyDeltaBoundaryTransform(parts, null)
    expect(result.out[0]).toEqual({ type: 'text-start', id: '1' })
    expect(result.out[1]).toEqual({ type: 'text-delta', id: '1', delta: 'hi' })
  })

  it('continues without start when same delta type and id', () => {
    const last: LanguageModelV3StreamPart = { type: 'text-delta', id: '1', delta: 'a' }
    const parts: LanguageModelV3StreamPart[] = [{ type: 'text-delta', id: '1', delta: 'b' }]
    const result = applyDeltaBoundaryTransform(parts, last)
    expect(result.out).toEqual([{ type: 'text-delta', id: '1', delta: 'b' }])
  })

  it('ends text and starts reasoning when switching types', () => {
    const last: LanguageModelV3StreamPart = { type: 'text-delta', id: '1', delta: 'a' }
    const parts: LanguageModelV3StreamPart[] = [{ type: 'reasoning-delta', id: '2', delta: 'r' }]
    const result = applyDeltaBoundaryTransform(parts, last)
    expect(result.out[0]).toEqual({ type: 'text-end', id: '1' })
    expect(result.out[1]).toEqual({ type: 'reasoning-start', id: '2' })
    expect(result.out[2]).toEqual({ type: 'reasoning-delta', id: '2', delta: 'r' })
  })

  it('passes through non-delta parts unchanged', () => {
    const parts: LanguageModelV3StreamPart[] = [{ type: 'stream-start', warnings: [] }]
    const result = applyDeltaBoundaryTransform(parts, null)
    expect(result.out).toEqual([{ type: 'stream-start', warnings: [] }])
  })
})

describe('composeDatabricksStreamPartTransformers', () => {
  it('composes single transformer', () => {
    const composed = composeDatabricksStreamPartTransformers(applyDeltaBoundaryTransform)
    const result = composed([{ type: 'text-delta', id: '1', delta: 'hi' }], null)
    expect(result.out[0]).toEqual({ type: 'text-start', id: '1' })
  })

  it('chains multiple transformers in order', () => {
    const addPrefix = (parts: LanguageModelV3StreamPart[]) => ({
      out: parts.map((p) => (p.type === 'text-delta' ? { ...p, delta: 'X' + p.delta } : p)),
    })
    const composed = composeDatabricksStreamPartTransformers(applyDeltaBoundaryTransform, addPrefix)
    const result = composed([{ type: 'text-delta', id: '1', delta: 'hi' }], null)
    expect(result.out).toContainEqual({ type: 'text-delta', id: '1', delta: 'Xhi' })
  })
})
