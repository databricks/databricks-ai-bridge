# import os

# from databricks.sdk import WorkspaceClient

# from databricks_mcp import DatabricksMCPClient

# # os.environ["DATABRICKS_CONFIG_PROFILE"] = "dogfood"

# # Replace with your deployed app URL
# mcp_server_url = "https://mcp-chloe-test-6051921418418893.staging.aws.databricksapps.com/mcp"

# workspace_client = WorkspaceClient(
#     host="https://e2-dogfood.staging.cloud.databricks.com",
#     token="<token>",
# )

# print(workspace_client.current_user.me())

# mcp_client = DatabricksMCPClient(server_url=mcp_server_url, workspace_client=workspace_client)

# # List available tools
# tools = mcp_client.list_tools()
# print(f"Available tools: {tools}")

from databricks.sdk import WorkspaceClient

from databricks_mcp import DatabricksMCPClient

ws_client = WorkspaceClient(
    host="https://e2-dogfood.staging.cloud.databricks.com/",
    token="<token>",
    # client_id="<client-id>",
    # client_secret="<client-secret>",
)

mcp_client = DatabricksMCPClient(
    server_url="https://mcp-chloe-test-6051921418418893.staging.aws.databricksapps.com/mcp",
    workspace_client=ws_client,
)
# print(mcp_client.list_tools())
