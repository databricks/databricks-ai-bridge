/**
 * Agent example using createAgent from LangChain
 *
 * This example simulates a customer support agent that can look up customers,
 * retrieve their orders, and process refunds - demonstrating natural chained tool calls.
 *
 * Set environment variables before running:
 *   DATABRICKS_HOST - Your workspace URL (e.g., https://your-workspace.databricks.com)
 *   DATABRICKS_TOKEN - Your personal access token
 *
 * Run with:
 *   npm run example:agent
 */

import "dotenv/config";
import { config } from "dotenv";
config({ path: ".env.local" });

import { createAgent } from "langchain";
import { ChatDatabricks } from "../src/index.js";
import { z } from "zod";
import { DynamicStructuredTool } from "@langchain/core/tools";

// Simulated database
const customers: Record<string, { id: string; name: string; email: string; tier: string }> = {
  "alice@example.com": { id: "C001", name: "Alice Johnson", email: "alice@example.com", tier: "gold" },
  "bob@example.com": { id: "C002", name: "Bob Smith", email: "bob@example.com", tier: "standard" },
};

const orders: Record<string, Array<{ orderId: string; date: string; items: string[]; total: number; status: string }>> = {
  C001: [
    { orderId: "ORD-1001", date: "2025-01-15", items: ["Wireless Headphones", "USB-C Cable"], total: 89.99, status: "delivered" },
    { orderId: "ORD-1002", date: "2025-01-10", items: ["Laptop Stand"], total: 49.99, status: "delivered" },
  ],
  C002: [
    { orderId: "ORD-2001", date: "2025-01-14", items: ["Mechanical Keyboard"], total: 129.99, status: "delivered" },
  ],
};

// Refund policies by tier
const refundPolicies: Record<string, number> = {
  gold: 1.0,      // 100% refund
  silver: 0.9,    // 90% refund
  standard: 0.85, // 85% refund
};

// Define tools for the agent
const lookupCustomerTool = new DynamicStructuredTool({
  name: "lookup_customer",
  description: "Look up a customer by their email address. Returns customer ID, name, and membership tier.",
  schema: z.object({
    email: z.string().describe("The customer's email address"),
  }),
  func: async ({ email }) => {
    console.log(`DEBUG: lookup_customer called with email: ${email}`);
    const customer = customers[email.toLowerCase()];
    if (!customer) {
      return JSON.stringify({ error: `No customer found with email: ${email}` });
    }
    return JSON.stringify(customer);
  },
});

const getOrdersTool = new DynamicStructuredTool({
  name: "get_orders",
  description: "Get the order history for a customer. Requires the customer ID.",
  schema: z.object({
    customerId: z.string().describe("The customer ID (e.g., 'C001')"),
  }),
  func: async ({ customerId }) => {
    console.log(`DEBUG: get_orders called with customerId: ${customerId}`);
    const customerOrders = orders[customerId];
    if (!customerOrders) {
      return JSON.stringify({ error: `No orders found for customer: ${customerId}` });
    }
    return JSON.stringify(customerOrders);
  },
});

const processRefundTool = new DynamicStructuredTool({
  name: "process_refund",
  description: "Process a refund for an order. The refund amount depends on the customer's membership tier.",
  schema: z.object({
    orderId: z.string().describe("The order ID to refund"),
    customerTier: z.string().describe("The customer's membership tier (gold, silver, or standard)"),
    orderTotal: z.number().describe("The original order total"),
    reason: z.string().describe("The reason for the refund"),
  }),
  func: async ({ orderId, customerTier, orderTotal, reason }) => {
    console.log(`DEBUG: process_refund called with orderId: ${orderId}, customerTier: ${customerTier}, orderTotal: ${orderTotal}, reason: ${reason}`);
    const refundRate = refundPolicies[customerTier] ?? 0.85;
    const refundAmount = (orderTotal * refundRate).toFixed(2);
    return JSON.stringify({
      orderId,
      refundAmount: parseFloat(refundAmount),
      refundRate: `${refundRate * 100}%`,
      reason,
      status: "approved",
      message: `Refund of $${refundAmount} approved for order ${orderId}`,
    });
  },
});

async function main() {
  console.log("=== Customer Support Agent Example ===\n");

  // Initialize the model
  const model = new ChatDatabricks({
    endpoint: "databricks-meta-llama-3-3-70b-instruct",
    endpointAPI: "chat-completions",
    // endpoint: "databricks-gpt-5-2",
    // endpointAPI: "responses",
    maxTokens: 1024,
    auth: {
      host: process.env.DATABRICKS_HOST,
      token: process.env.DATABRICKS_TOKEN,
    },
  });

  // Create the agent with tools
  const agent = createAgent({
    model,
    tools: [lookupCustomerTool, getOrdersTool, processRefundTool],
  });

  console.log(`Agent created with tools: ${agent.options.tools?.map(tool => tool.name).join(", ")}\n`);

  // Run the agent - this requires chained tool calls:
  // 1. First look up the customer by email to get their ID and tier
  // 2. Then get their orders to find the most recent one
  // 3. Finally process the refund using the order details and customer tier
  const question =
    "Customer alice@example.com is requesting a refund for their most recent order because the headphones stopped working. Please process the refund.";
  console.log(`User: ${question}\n`);

  console.log("Agent response (streaming):\n");

  // Stream events from the agent - filter to only chat model events
  const stream = agent.streamEvents(
    { messages: [{ role: "user", content: question }] },
    { version: "v2" }
  );

  for await (const event of stream) {
    switch (event.event) {
      case "on_chat_model_stream":
        process.stdout.write(event.data?.chunk?.content as string);
        break;
      case "on_tool_start":
        console.log(`\n🔧 Tool call: ${event.name}(${JSON.stringify(event.data?.input)})`);
        break;
      case "on_tool_end":{
        const output = event.data?.output;
        const content = output?.content ?? JSON.stringify(output);
        console.log(`📋 Tool result [${event.name}]: ${content}\n`);
        break;
      }
    }
  }

  console.log("\nDone!");
}

main().catch(console.error);
