#!/usr/bin/env node
/**
 * Debug script to inspect the request body being sent
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
const ENDPOINT = 'agents_smurching-default-test_chat_app_agent'

async function inspectStreamingRequest() {
  console.log('\n🔍 Inspecting Streaming Request with return_trace=true\n')

  const token = getToken(PROFILE, HOST)

  // Create a custom fetch that logs the request
  const originalFetch = globalThis.fetch
  let capturedRequest = null

  const loggingFetch = async (url, options) => {
    console.log('📤 Request URL:', url)
    console.log('📤 Request Headers:', JSON.stringify(options.headers, null, 2))
    console.log('📤 Request Body:', options.body)

    try {
      const body = JSON.parse(options.body)
      console.log('📤 Parsed Request Body:', JSON.stringify(body, null, 2))
      capturedRequest = body
    } catch (e) {
      console.log('  (Could not parse as JSON)')
    }

    return originalFetch(url, options)
  }

  const provider = createDatabricksProvider({
    baseURL: `${HOST}/serving-endpoints`,
    headers: {
      Authorization: `Bearer ${token}`,
    },
    fetch: loggingFetch,
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
    })

    console.log('\n✅ Request sent successfully')
    console.log('\n📥 Response Metadata:')
    console.log('  - Has response.body:', !!result.response?.body)
    console.log('  - Response.body:', JSON.stringify(result.response?.body, null, 2))

    // Consume the stream
    const reader = result.stream.getReader()
    let textOutput = ''
    console.log('\n📝 Streaming response:')

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        if (value.type === 'text-delta') {
          process.stdout.write(value.delta)
          textOutput += value.delta
        } else if (value.type === 'finish') {
          console.log('\n\n📊 Finish event:')
          console.log('  - finishReason:', value.finishReason)
          console.log('  - usage:', JSON.stringify(value.usage))
        }
      }
    } finally {
      reader.releaseLock()
    }

    console.log('\n\n📥 Final Response Body:', JSON.stringify(result.response?.body, null, 2))

    if (capturedRequest) {
      console.log('\n🔍 Key observations:')
      console.log('  - Has databricks_options in request?', !!capturedRequest.databricks_options)
      console.log('  - databricks_options:', JSON.stringify(capturedRequest.databricks_options, null, 2))
    }
  } catch (error) {
    console.error('❌ ERROR:', error.message)
    console.error('\nStack trace:', error.stack)
  }
}

async function inspectNonStreamingRequest() {
  console.log('\n\n🔍 Inspecting Non-Streaming Request with return_trace=true\n')

  const token = getToken(PROFILE, HOST)

  // Create a custom fetch that logs the request
  const originalFetch = globalThis.fetch
  let capturedRequest = null

  const loggingFetch = async (url, options) => {
    console.log('📤 Request URL:', url)
    console.log('📤 Request Method:', options.method)
    console.log('📤 Request Body:', options.body)

    try {
      const body = JSON.parse(options.body)
      console.log('📤 Parsed Request Body:', JSON.stringify(body, null, 2))
      capturedRequest = body
    } catch (e) {
      console.log('  (Could not parse as JSON)')
    }

    return originalFetch(url, options)
  }

  const provider = createDatabricksProvider({
    baseURL: `${HOST}/serving-endpoints`,
    headers: {
      Authorization: `Bearer ${token}`,
    },
    fetch: loggingFetch,
  })

  const model = provider.responses(ENDPOINT)

  try {
    const result = await model.doGenerate({
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
    })

    console.log('\n✅ Request sent successfully')
    console.log('\n📥 Response:')
    console.log('  - Content:', result.content)
    console.log('  - Has response.body:', !!result.response?.body)
    console.log('  - Response.body trace_id:', result.response?.body?.trace_id)
    console.log('  - Full response.body:', JSON.stringify(result.response?.body, null, 2))
  } catch (error) {
    console.error('❌ ERROR:', error.message)
    console.error('\nStack trace:', error.stack)
  }
}

async function main() {
  console.log('🚀 Debug Script - Request Inspection')
  console.log('Endpoint:', ENDPOINT)
  console.log('Host:', HOST)

  await inspectStreamingRequest()
  await inspectNonStreamingRequest()

  console.log('\n' + '='.repeat(80))
  console.log('✅ Inspection completed')
  console.log('='.repeat(80) + '\n')
}

main().catch((error) => {
  console.error('Fatal error:', error)
  process.exit(1)
})
