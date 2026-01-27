/**
 * MCP (Model Context Protocol) example demonstrating DatabricksMultiServerMCPClient
 *
 * This example shows how to:
 * - Connect to MCP servers (both Databricks and external)
 * - Load tools from MCP servers
 * - Use MCP tools with ChatDatabricks
 *
 * Prerequisites:
 * - Set DATABRICKS_HOST environment variable to your workspace URL
 * - Configure authentication (DATABRICKS_TOKEN, OAuth, or other supported method)
 *
 * Run with:
 *   npm run example:mcp
 */

import "dotenv/config";
import { config } from "dotenv";
config({ path: ".env.local" });

import {
  ChatDatabricks,
  DatabricksMultiServerMCPClient,
  DatabricksMCPServer,
} from "../src/index.js";
import { HumanMessage, ToolMessage, BaseMessage } from "@langchain/core/messages";
import { ServerInstance } from "../src/mcp/databricks_mcp_client.js";

async function main() {
  console.log("=== MCP Integration Example ===\n");

  // Example 1: Create MCP servers
  console.log("--- Creating MCP Server Configurations ---\n");

  // Databricks SQL MCP server - provides SQL query capabilities
  // Host is resolved lazily from DATABRICKS_HOST env var
  const dbsqlMCP = new DatabricksMCPServer({
    name: "dbsql-mcp",
    path: "/api/2.0/mcp/sql",
    timeout: 60, // 60 second timeout
  });

  // Example: Create server for Unity Catalog functions
  // This exposes UC functions as MCP tools
  const ucFunctionServer = DatabricksMCPServer.fromUCFunction("system", "ai", "python_exec");

  // Example: Create server for Vector Search
  // const vectorSearchServer = DatabricksMCPServer.fromVectorSearch(
  //   "my_catalog",
  //   "my_schema",
  //   "my_index"
  // );

  // Example: Create server for Genie Space
  // const genieSpaceServer = DatabricksMCPServer.fromGenieSpace(
  //   "your_genie_space_id"
  // );

  // Configure which servers to use
  // Uncomment servers as needed based on your setup
  const mcpServers: ServerInstance[] = [
    dbsqlMCP,
    ucFunctionServer,
    // vectorSearchServer,
    // genieSpaceServer,
  ];

  console.log(`Created MCP servers: ${mcpServers.map((s) => s.name).join(", ")}`);

  console.log("\n--- Creating MCP Client ---\n");

  // Create multi-server MCP client
  const mcpClient = new DatabricksMultiServerMCPClient(mcpServers, {
    throwOnLoadError: false, // Continue if some servers fail to load
    prefixToolNameWithServerName: false,
  });

  console.log(`MCP client created with servers: ${mcpClient.getServerNames().join(", ")}`);

  try {
    // Load tools from MCP servers
    console.log("\n--- Loading Tools ---\n");
    console.log("Loading tools from MCP servers...");

    const tools = await mcpClient.getTools();
    console.log(`Loaded ${tools.length} tools:`);
    for (const tool of tools) {
      console.log(`  - ${tool.name}: ${tool.description || "(no description)"}`);
    }

    if (tools.length === 0) {
      console.log("\nNo tools loaded. Make sure your MCP server is running and accessible.");
      console.log("Skipping tool usage demonstration.\n");
      return;
    }

    // Create ChatDatabricks model
    console.log("\n--- Using Tools with ChatDatabricks ---\n");

    const model = new ChatDatabricks({
      endpoint: "databricks-claude-sonnet-4-5",
      maxTokens: 1024,
    });

    // Bind MCP tools to the model
    const modelWithTools = model.bindTools(tools);
    console.log("Bound MCP tools to ChatDatabricks model\n");

    // Example conversation with tool use
    const messages: BaseMessage[] = [
      new HumanMessage("What tools do you have available? List them briefly."),
    ];

    console.log("Sending message to model...\n");

    const response = await modelWithTools.invoke(messages);

    // Helper to extract text from content (can be string or array of content blocks)
    const getTextContent = (content: BaseMessage["content"]): string => {
      if (typeof content === "string") {
        return content;
      }
      if (Array.isArray(content)) {
        return content
          .filter((block) => block.type === "text")
          .map((block) => block.text)
          .join("");
      }
      return "(no text content)";
    };

    console.log(`Response: ${getTextContent(response.content) || "(no text content)"}`);

    // Agentic loop with maxIterations
    const maxIterations = 10;
    let currentResponse = response;
    let iteration = 0;

    while (currentResponse.tool_calls && currentResponse.tool_calls.length > 0) {
      iteration++;
      if (iteration > maxIterations) {
        console.log(`\nMax iterations (${maxIterations}) reached. Stopping.`);
        break;
      }

      console.log(`\n--- Iteration ${iteration} ---`);
      console.log("Tool calls requested:");
      messages.push(currentResponse);

      for (const toolCall of currentResponse.tool_calls) {
        console.log(`  -> ${toolCall.name}(${JSON.stringify(toolCall.args)})`);

        // Find and execute the tool
        const tool = tools.find((t) => t.name === toolCall.name);
        if (tool) {
          try {
            const result = await tool.invoke(toolCall.args);
            console.log(`  <- Result: ${JSON.stringify(result).slice(0, 200)}...`);

            messages.push(
              new ToolMessage({
                content: typeof result === "string" ? result : JSON.stringify(result),
                tool_call_id: toolCall.id!,
                name: toolCall.name,
              })
            );
          } catch (error) {
            console.error(`  <- Error: ${error}`);
            messages.push(
              new ToolMessage({
                content: `Error: ${error}`,
                tool_call_id: toolCall.id!,
                name: toolCall.name,
              })
            );
          }
        }
      }

      // Get next response
      console.log("\nGetting next response...");
      currentResponse = await modelWithTools.invoke(messages);
      console.log(`Response: ${getTextContent(currentResponse.content) || "(no text content)"}`);
    }

    console.log(`\n--- Final Result (after ${iteration} iterations) ---`);
    console.log(`Final response: ${getTextContent(currentResponse.content)}`);
  } finally {
    // Always close the MCP client to clean up connections
    console.log("\n--- Cleanup ---\n");
    await mcpClient.close();
    console.log("MCP client closed");
  }

  console.log("\nDone!");
}

main().catch(console.error);
