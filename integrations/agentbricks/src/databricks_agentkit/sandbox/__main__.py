import os

import uvicorn

uvicorn.run(
    "databricks_agentkit.sandbox.app:app",
    host="0.0.0.0",
    port=int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", "8000"))),
)
