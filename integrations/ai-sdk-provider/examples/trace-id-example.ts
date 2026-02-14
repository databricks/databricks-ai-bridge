/**
 * Example: Using trace_id for feedback collection
 *
 * This example demonstrates how to:
 * 1. Request trace IDs from Databricks Responses API
 * 2. Extract trace_id and span_id from the response
 * 3. Use them for feedback submission to MLflow
 */

import { createDatabricksProvider } from '../src/databricks-provider'
import { streamText } from 'ai'

async function example() {
  // Create provider
  const provider = createDatabricksProvider({
    baseURL: process.env.DATABRICKS_BASE_URL || 'https://your-workspace.databricks.com',
    headers: {
      Authorization: `Bearer ${process.env.DATABRICKS_TOKEN}`,
    },
  })

  // Get the model
  const model = provider.responses('your-agent-endpoint')

  // Variables to store trace information
  let traceId: string | undefined
  let spanId: string | undefined

  // Stream text with trace ID tracking enabled
  const result = await streamText({
    model,
    messages: [{ role: 'user', content: 'What is the weather like today?' }],
    providerOptions: {
      databricks: {
        databricksOptions: {
          return_trace: true, // Request trace ID from the endpoint
        },
      },
    },
    onFinish: ({ response }) => {
      // Extract trace ID and span ID from the raw response
      // The rawResponse is available as response.body
      const rawResponse = response?.body as any
      traceId = rawResponse?.trace_id
      spanId = rawResponse?.span_id

      console.log('Trace ID:', traceId)
      console.log('Span ID:', spanId)

      // You can now store these IDs with the message for later feedback submission
    },
  })

  // Stream the response
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk)
  }

  console.log('\n\n--- Trace Information ---')
  console.log('Trace ID:', traceId)
  console.log('Span ID:', spanId)

  // Later, you can submit feedback using the trace ID
  if (traceId) {
    await submitFeedback(traceId, 'thumbs_up')
  }
}

/**
 * Submit feedback to MLflow using trace ID
 */
async function submitFeedback(traceId: string, feedback: 'thumbs_up' | 'thumbs_down') {
  const databricksHost = process.env.DATABRICKS_BASE_URL || 'https://your-workspace.databricks.com'
  const token = process.env.DATABRICKS_TOKEN

  const response = await fetch(`${databricksHost}/api/2.0/mlflow/traces/assessments`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      name: 'user_feedback',
      trace_id: traceId,
      value: feedback === 'thumbs_up' ? 1 : 0,
      value_type: 'numeric',
      tags: {
        feedback_type: feedback,
      },
    }),
  })

  if (!response.ok) {
    console.error('Failed to submit feedback:', await response.text())
  } else {
    console.log('Feedback submitted successfully!')
  }
}

// Run the example
example().catch(console.error)
