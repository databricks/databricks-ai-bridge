#!/usr/bin/env node
/**
 * Live test script for trace_id support with actual Databricks agent endpoint
 * This version uses the built package directly without requiring the 'ai' package
 */

import { createDatabricksProvider } from './dist/index.js'
import { execSync } from 'child_process'

// Get token from Databricks CLI
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

async function testStreaming(withTrace) {
  console.log(`\n${'='.repeat(80)}`)
  console.log(`TEST: Streaming ${withTrace ? 'WITH' : 'WITHOUT'} trace_id`)
  console.log('='.repeat(80))

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
          content: [{ type: 'text', text: 'What is 2+2? Please answer briefly.' }],
        },
      ],
      providerOptions: withTrace
        ? {
            databricks: {
              databricksOptions: {
                return_trace: true,
              },
            },
          }
        : {},
    })

    console.log('\n📝 Response:')
    let fullText = ''
    const reader = result.stream.getReader()

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        if (value.type === 'text-delta') {
          process.stdout.write(value.delta)
          fullText += value.delta
        }
      }
    } finally {
      reader.releaseLock()
    }

    console.log('\n')

    const responseBody = result.response?.body
    const traceId = responseBody?.trace_id
    const spanId = responseBody?.span_id

    console.log('📊 Result:')
    console.log('  - Response length:', fullText.length)
    console.log('  - Has response.body:', !!responseBody)
    console.log('  - trace_id:', traceId || '(not present)')
    console.log('  - span_id:', spanId || '(not present)')

    if (withTrace) {
      if (traceId) {
        console.log('✅ SUCCESS: trace_id was returned when requested')
        console.log(`   Full trace_id: ${traceId}`)
        if (spanId) {
          console.log(`   Full span_id: ${spanId}`)
        }
      } else {
        console.log('❌ FAIL: trace_id was NOT returned when requested')
        console.log('   Response body:', JSON.stringify(responseBody, null, 2))
      }
    } else {
      if (!traceId) {
        console.log('✅ SUCCESS: trace_id was not returned when not requested')
      } else {
        console.log('⚠️  UNEXPECTED: trace_id was returned even though not requested')
      }
    }
  } catch (error) {
    console.error('❌ ERROR:', error.message)
    if (error.stack) {
      console.error('\nStack trace:', error.stack)
    }
  }
}

async function testNonStreaming(withTrace) {
  console.log(`\n${'='.repeat(80)}`)
  console.log(`TEST: Non-streaming ${withTrace ? 'WITH' : 'WITHOUT'} trace_id`)
  console.log('='.repeat(80))

  const token = getToken(PROFILE, HOST)
  const provider = createDatabricksProvider({
    baseURL: `${HOST}/serving-endpoints`,
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })

  const model = provider.responses(ENDPOINT)

  try {
    const result = await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'What is 3+3? Please answer briefly.' }],
        },
      ],
      providerOptions: withTrace
        ? {
            databricks: {
              databricksOptions: {
                return_trace: true,
              },
            },
          }
        : {},
    })

    console.log('\n📝 Response:')
    const textContent = result.content.filter((c) => c.type === 'text')
    for (const content of textContent) {
      console.log(content.text)
    }

    const responseBody = result.response?.body
    const traceId = responseBody?.trace_id
    const spanId = responseBody?.span_id

    console.log('\n📊 Result:')
    console.log('  - Content parts:', result.content.length)
    console.log('  - Has response.body:', !!responseBody)
    console.log('  - trace_id:', traceId || '(not present)')
    console.log('  - span_id:', spanId || '(not present)')

    if (withTrace) {
      if (traceId) {
        console.log('✅ SUCCESS: trace_id was returned when requested')
        console.log(`   Full trace_id: ${traceId}`)
        if (spanId) {
          console.log(`   Full span_id: ${spanId}`)
        }
      } else {
        console.log('❌ FAIL: trace_id was NOT returned when requested')
        console.log('   Response body:', JSON.stringify(responseBody, null, 2))
      }
    } else {
      if (!traceId) {
        console.log('✅ SUCCESS: trace_id was not returned when not requested')
      } else {
        console.log('⚠️  UNEXPECTED: trace_id was returned even though not requested')
      }
    }
  } catch (error) {
    console.error('❌ ERROR:', error.message)
    if (error.stack) {
      console.error('\nStack trace:', error.stack)
    }
  }
}

async function main() {
  console.log('\n🚀 Testing Databricks AI SDK Provider - Trace ID Support')
  console.log('Endpoint:', ENDPOINT)
  console.log('Host:', HOST)
  console.log('Profile:', PROFILE)

  // Make sure the package is built
  console.log('\n📦 Ensuring package is built...')
  try {
    execSync('npm run build', { stdio: 'inherit' })
  } catch (error) {
    console.error('Failed to build package:', error.message)
    process.exit(1)
  }

  // Test 1: Streaming without trace_id
  await testStreaming(false)

  // Test 2: Streaming with trace_id
  await testStreaming(true)

  // Test 3: Non-streaming without trace_id
  await testNonStreaming(false)

  // Test 4: Non-streaming with trace_id
  await testNonStreaming(true)

  console.log(`\n${'='.repeat(80)}`)
  console.log('✅ All tests completed!')
  console.log('='.repeat(80) + '\n')
}

main().catch((error) => {
  console.error('Fatal error:', error)
  process.exit(1)
})
