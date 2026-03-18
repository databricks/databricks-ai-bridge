"""
Deploy the whoami OBO agent to a Model Serving endpoint.

Run manually or on a weekly schedule to keep the endpoint on the latest SDK.

Environment Variables:
    DATABRICKS_HOST           - Workspace URL
    DATABRICKS_CLIENT_ID      - Service principal client ID
    DATABRICKS_CLIENT_SECRET  - Service principal client secret
    OBO_TEST_SERVING_ENDPOINT - Target serving endpoint name (optional override)
    OBO_TEST_WAREHOUSE_ID     - SQL warehouse ID
"""

import logging
import os
import tempfile
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksSQLWarehouse

log = logging.getLogger(__name__)

# Must match the constants in whoami_serving_agent.py
LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-6"
SQL_WAREHOUSE_ID = os.environ["OBO_TEST_WAREHOUSE_ID"]

UC_CATALOG = "integration_testing"
UC_SCHEMA = "databricks_ai_bridge_mcp_test"
UC_MODEL_NAME_SHORT = "obo_test_endpoint"
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME_SHORT}"


def main():
    w = WorkspaceClient()
    log.info("Workspace: %s", w.config.host)

    mlflow.set_registry_uri("databricks-uc")

    experiment_name = f"/Users/{w.current_user.me().user_name}/obo-serving-agent-deploy"
    mlflow.set_experiment(experiment_name)

    # Copy agent file to a temp dir, injecting the warehouse ID
    agent_source = Path(__file__).parent / "model_serving_fixture" / "whoami_serving_agent.py"
    with tempfile.TemporaryDirectory() as tmp:
        agent_file = Path(tmp) / "agent.py"
        content = agent_source.read_text()
        content = content.replace(
            'SQL_WAREHOUSE_ID = ""  # Injected by deploy_serving_agent.py at log time',
            f'SQL_WAREHOUSE_ID = "{SQL_WAREHOUSE_ID}"',
        )
        agent_file.write_text(content)

        system_policy = SystemAuthPolicy(
            resources=[
                DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
                DatabricksSQLWarehouse(warehouse_id=SQL_WAREHOUSE_ID),
            ]
        )
        user_policy = UserAuthPolicy(
            api_scopes=[
                "sql",
                "model-serving",
            ]
        )

        with mlflow.start_run():
            logged_agent_info = mlflow.pyfunc.log_model(
                name="agent",
                python_model=str(agent_file),
                auth_policy=AuthPolicy(
                    system_auth_policy=system_policy,
                    user_auth_policy=user_policy,
                ),
                pip_requirements=[
                    "databricks-openai",
                    "databricks-ai-bridge",
                    "databricks-sdk",
                    "mlflow<=3.9.0",
                ],
            )
        log.info("Logged model: %s", logged_agent_info.model_uri)

    registered = mlflow.register_model(logged_agent_info.model_uri, UC_MODEL_NAME)
    log.info("Registered: %s version %s", UC_MODEL_NAME, registered.version)

    from databricks import agents

    agents.deploy(UC_MODEL_NAME, registered.version, scale_to_zero=True)
    log.info("Deployment initiated (scale_to_zero=True)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
