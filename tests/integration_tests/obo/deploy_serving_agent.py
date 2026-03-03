"""
Deploy a minimal whoami agent to a Model Serving endpoint with OBO enabled.

This script logs an MLflow ChatModel that uses ModelServingUserCredentials
to return the calling user's identity, then deploys it to a serving endpoint.

Run manually or on a schedule to keep the endpoint on the latest SDK version.

Environment Variables:
    DATABRICKS_HOST           - Workspace URL
    DATABRICKS_CLIENT_ID      - Service principal client ID
    DATABRICKS_CLIENT_SECRET  - Service principal client secret
    OBO_TEST_SERVING_ENDPOINT - Target serving endpoint name
    MLFLOW_EXPERIMENT_NAME    - (optional) MLflow experiment name
"""

import os
import sys

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.models import set_model
from mlflow.models.resources import DatabricksFunction
from mlflow.pyfunc import ChatModel


class WhoAmIAgent(ChatModel):
    """Minimal agent that returns the calling user's identity via OBO."""

    def predict(self, context, messages, params):
        from databricks.sdk import WorkspaceClient

        from databricks_ai_bridge import ModelServingUserCredentials

        wc = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
        me = wc.current_user.me()
        identity = me.display_name or me.user_name or str(me.id)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": identity},
                }
            ]
        }


def main():
    endpoint_name = os.environ.get("OBO_TEST_SERVING_ENDPOINT")
    if not endpoint_name:
        print("ERROR: OBO_TEST_SERVING_ENDPOINT must be set")
        sys.exit(1)

    w = WorkspaceClient()
    print(f"Deploying whoami agent to endpoint: {endpoint_name}")
    print(f"Workspace: {w.config.host}")

    # Set up experiment
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME",
        f"/Users/{w.current_user.me().user_name}/obo-test-serving-agent",
    )
    mlflow.set_experiment(experiment_name)

    # Log model
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=WhoAmIAgent(),
            pip_requirements=[
                "databricks-ai-bridge",
                "databricks-sdk",
                "mlflow",
            ],
        )
        print(f"Logged model: {model_info.model_uri}")

    # Register in UC
    uc_model_name = f"integration_testing.databricks_ai_bridge_mcp_test.obo_whoami_agent"
    registered = mlflow.register_model(model_info.model_uri, uc_model_name)
    print(f"Registered model version: {registered.version}")

    # Deploy
    from databricks import agents

    agents.deploy(
        model_name=uc_model_name,
        model_version=registered.version,
        endpoint_name=endpoint_name,
        scale_to_zero=True,
    )
    print(f"Deployment initiated for endpoint: {endpoint_name}")


if __name__ == "__main__":
    main()
