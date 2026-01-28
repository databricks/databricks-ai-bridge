import "dotenv/config";
import { config } from "dotenv";
config({ path: ".env.local" });

import { Config } from "@databricks/sdk-experimental";

async function main() {
  console.log("Testing M2M OAuth...");
  console.log("DATABRICKS_HOST:", process.env.DATABRICKS_HOST);
  console.log("DATABRICKS_CLIENT_ID:", process.env.DATABRICKS_CLIENT_ID);
  console.log(
    "DATABRICKS_CLIENT_SECRET:",
    process.env.DATABRICKS_CLIENT_SECRET?.substring(0, 10) + "..."
  );

  const sdkConfig = new Config({});

  console.log("\nResolving config...");
  await sdkConfig.ensureResolved();

  console.log("Auth type:", sdkConfig.authType);
  console.log("Client ID:", sdkConfig.clientId);
  console.log("Host:", sdkConfig.host);

  console.log("\nGetting auth headers...");
  const headers = new Headers();
  await sdkConfig.authenticate(headers);

  const authHeader = headers.get("Authorization");
  console.log("Auth header:", authHeader?.substring(0, 50) + "...");
  console.log("Auth header length:", authHeader?.length);
}

main().catch(console.error);
