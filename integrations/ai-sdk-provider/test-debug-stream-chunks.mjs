#!/usr/bin/env node
/**
 * Debug script to inspect all stream chunks
 */

import { createDatabricksProvider } from './dist/index.js'
import { execSync } from 'child_process'

function getToken(profile, host) {
  const result = execSync(
    `databricks auth token --profile ${profile} --host ${host}`,
    { encoding: 'utf-8' }
  )
  const tokenData = JSON.parse(result)
  return tokenData.access_token
}

const PROFILE = 'dogfood'
const HOST = 'https://e2-dogfood.staging.cloud.databricks.com'
const ENDPOINT = 'agents_smurching-default-feb_2026_agent'

async function inspectStreamChunks() {
  console.log('🔍 Inspecting All Stream Chunks\n')

  const token = getToken(PROFILE, HOST)

  const provider = createDatabricksProvider({
    baseURL: `${HOST}/serving-endpoints`,
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })

  const model = provider.responses(ENDPOINT)

  try {
    const result = await model.doStream({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Say "hello"' }],
        },
      ],
      providerOptions: {
        databricks: {
          databricksOptions: {
            return_trace: true,
          },
        },
      },
      includeRawChunks: true,
    })

    console.log('✅ Request sent successfully\n')

    // Consume the stream and log all chunks
    const reader = result.stream.getReader()
    let chunkCount = 0

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        chunkCount++
        console.log(`\n📦 Chunk ${chunkCount}:`)
        console.log('  Type:', value.type)

        if (value.type === 'raw') {
          console.log('  Raw value:', JSON.stringify(value.rawValue, null, 2))
        } else if (value.type === 'text-delta') {
          console.log('  Delta:', JSON.stringify(value.delta))
        } else if (value.type === 'finish') {
          console.log('  Finish reason:', JSON.stringify(value.finishReason))
          console.log('  Usage:', JSON.stringify(value.usage))
        } else {
          console.log('  Full value:', JSON.stringify(value, null, 2))
        }
      }
    } finally {
      reader.releaseLock()
    }

    console.log('\n\n📥 Final Response Body:', JSON.stringify(result.response?.body, null, 2))
    console.log('Total chunks received:', chunkCount)
  } catch (error) {
    console.error('❌ ERROR:', error.message)
    console.error('\nStack trace:', error.stack)
  }
}

inspectStreamChunks().catch((error) => {
  console.error('Fatal error:', error)
  process.exit(1)
})
