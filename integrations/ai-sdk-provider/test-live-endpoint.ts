#!/usr/bin/env tsx
/**
 * Live test script for trace_id support with actual Databricks agent endpoint
 */

import { createDatabricksProvider } from './src/databricks-provider'
import { streamText, generateText } from 'ai'
import { execSync } from 'child_process'

// Get token from Databricks CLI
function getToken(profile: string, host: string): string {
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

async function testStreaming(withTrace: boolean) {
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

  let traceId: string | undefined
  let spanId: string | undefined
  let responseBody: any

  try {
    const result = await streamText({
      model,
      messages: [{ role: 'user', content: 'What is 2+2? Please answer briefly.' }],
      providerOptions: withTrace
        ? {
            databricks: {
              databricksOptions: {
                return_trace: true,
              },
            },
          }
        : undefined,
      onFinish: ({ response }) => {
        responseBody = response?.body
        if (responseBody) {
          traceId = responseBody.trace_id
          spanId = responseBody.span_id
        }
      },
    })

    console.log('\n📝 Response:')
    let fullText = ''
    for await (const chunk of result.textStream) {
      process.stdout.write(chunk)
      fullText += chunk
    }
    console.log('\n')

    console.log('📊 Result:')
    console.log('  - Response length:', fullText.length)
    console.log('  - Has response.body:', !!responseBody)
    console.log('  - trace_id:', traceId || '(not present)')
    console.log('  - span_id:', spanId || '(not present)')

    if (withTrace) {
      if (traceId) {
        console.log('✅ SUCCESS: trace_id was returned when requested')
      } else {
        console.log('❌ FAIL: trace_id was NOT returned when requested')
      }
    } else {
      if (!traceId) {
        console.log('✅ SUCCESS: trace_id was not returned when not requested')
      } else {
        console.log('⚠️  UNEXPECTED: trace_id was returned even though not requested')
      }
    }
  } catch (error: any) {
    console.error('❌ ERROR:', error.message)
    if (error.stack) {
      console.error('\nStack trace:', error.stack)
    }
  }
}

async function testNonStreaming(withTrace: boolean) {
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
    const result = await generateText({
      model,
      messages: [{ role: 'user', content: 'What is 3+3? Please answer briefly.' }],
      providerOptions: withTrace
        ? {
            databricks: {
              databricksOptions: {
                return_trace: true,
              },
            },
          }
        : undefined,
    })

    console.log('\n📝 Response:')
    console.log(result.text)

    const responseBody = result.response?.body as any
    const traceId = responseBody?.trace_id
    const spanId = responseBody?.span_id

    console.log('\n📊 Result:')
    console.log('  - Response length:', result.text.length)
    console.log('  - Has response.body:', !!responseBody)
    console.log('  - trace_id:', traceId || '(not present)')
    console.log('  - span_id:', spanId || '(not present)')

    if (withTrace) {
      if (traceId) {
        console.log('✅ SUCCESS: trace_id was returned when requested')
      } else {
        console.log('❌ FAIL: trace_id was NOT returned when requested')
      }
    } else {
      if (!traceId) {
        console.log('✅ SUCCESS: trace_id was not returned when not requested')
      } else {
        console.log('⚠️  UNEXPECTED: trace_id was returned even though not requested')
      }
    }
  } catch (error: any) {
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
  console.log('='.repeat(80)\n)
}

main().catch((error) => {
  console.error('Fatal error:', error)
  process.exit(1)
})
