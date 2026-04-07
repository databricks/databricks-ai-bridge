"""
Vertex AI Agent Engine deployment helpers for Databricks-powered ADK agents.

This module provides utilities to deploy ADK agents that use Databricks tools
to Google Cloud's Vertex AI Agent Engine.
"""

from typing import Any, Optional

from google.adk.agents import Agent


def get_databricks_requirements() -> list[str]:
    """
    Get the pip requirements needed for Databricks tools in Agent Engine.

    Returns:
        List of pip package requirements.

    Example:
        ```python
        from databricks_google_adk.deployment import get_databricks_requirements

        requirements = get_databricks_requirements()
        # ['databricks-google-adk', 'databricks-sdk', ...]
        ```
    """
    return [
        "databricks-google-adk",
        "databricks-sdk",
        "databricks-vectorsearch",
        "databricks-ai-bridge",
        "databricks-mcp",
    ]


def create_agent_engine_config(
    staging_bucket: str,
    requirements: Optional[list[str]] = None,
    env_vars: Optional[dict[str, str]] = None,
    databricks_host: Optional[str] = None,
    databricks_token_secret: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Create configuration for deploying to Vertex AI Agent Engine.

    Args:
        staging_bucket: GCS bucket for staging (format: "gs://bucket-name").
        requirements: Additional pip requirements. Databricks requirements
            are automatically included.
        env_vars: Environment variables to set in the deployed agent.
        databricks_host: Databricks workspace URL. If provided, will be set
            as DATABRICKS_HOST environment variable.
        databricks_token_secret: Secret Manager secret name for Databricks token.
            Format: "projects/PROJECT/secrets/SECRET/versions/VERSION".
            If provided, will be configured for DATABRICKS_TOKEN.
        **kwargs: Additional configuration options passed to Agent Engine.

    Returns:
        Configuration dictionary for agent_engines.create().

    Example:
        ```python
        from databricks_google_adk.deployment import create_agent_engine_config

        config = create_agent_engine_config(
            staging_bucket="gs://my-bucket",
            databricks_host="https://my-workspace.databricks.com",
            databricks_token_secret="projects/my-project/secrets/databricks-token/versions/latest",
        )

        # Use with Vertex AI
        remote_agent = client.agent_engines.create(
            agent=app,
            config=config,
        )
        ```
    """
    # Combine requirements
    all_requirements = get_databricks_requirements()
    if requirements:
        all_requirements.extend(requirements)
    # Remove duplicates while preserving order
    all_requirements = list(dict.fromkeys(all_requirements))

    # Build environment variables
    all_env_vars = env_vars.copy() if env_vars else {}
    if databricks_host:
        all_env_vars["DATABRICKS_HOST"] = databricks_host

    config = {
        "requirements": all_requirements,
        "staging_bucket": staging_bucket,
        **kwargs,
    }

    if all_env_vars:
        config["env_vars"] = all_env_vars

    # Handle secret for Databricks token
    if databricks_token_secret:
        config["secrets"] = config.get("secrets", {})
        config["secrets"]["DATABRICKS_TOKEN"] = databricks_token_secret

    return config


class DatabricksAgentEngineApp:
    """
    A wrapper for deploying Databricks-powered ADK agents to Vertex AI Agent Engine.

    This class provides a simplified interface for creating and deploying
    ADK agents that use Databricks tools to Vertex AI Agent Engine.

    Example:
        ```python
        from databricks_google_adk import VectorSearchRetrieverTool, DatabricksAgentEngineApp
        from google.adk.agents import Agent

        # Create an agent with Databricks tools
        vector_search = VectorSearchRetrieverTool(index_name="catalog.schema.index")
        agent = Agent(
            name="search_agent",
            model="gemini-2.0-flash",
            tools=[vector_search.as_tool()],
        )

        # Create the deployable app
        app = DatabricksAgentEngineApp(agent=agent)

        # Deploy to Agent Engine
        import vertexai
        client = vertexai.Client(project="my-project", location="us-central1")

        remote_agent = client.agent_engines.create(
            agent=app.adk_app,
            config=app.get_deployment_config(
                staging_bucket="gs://my-bucket",
                databricks_host="https://my-workspace.databricks.com",
                databricks_token_secret="projects/my-project/secrets/db-token/versions/latest",
            ),
        )
        ```
    """

    def __init__(
        self,
        agent: Agent,
        additional_requirements: Optional[list[str]] = None,
    ):
        """
        Initialize the DatabricksAgentEngineApp.

        Args:
            agent: The ADK Agent to deploy.
            additional_requirements: Additional pip requirements beyond
                the standard Databricks packages.
        """
        self._agent = agent
        self._additional_requirements = additional_requirements or []
        self._adk_app = None

    @property
    def agent(self) -> Agent:
        """Get the underlying ADK agent."""
        return self._agent

    @property
    def adk_app(self):
        """
        Get the AdkApp for deployment.

        Returns:
            An AdkApp instance wrapping the agent.

        Note:
            Requires vertexai package: pip install google-cloud-aiplatform[agent_engines,adk]
        """
        if self._adk_app is None:
            try:
                from vertexai.agent_engines import AdkApp
            except ImportError:
                raise ImportError(
                    "vertexai package is required for Agent Engine deployment. "
                    "Install with: pip install google-cloud-aiplatform[agent_engines,adk]"
                )
            self._adk_app = AdkApp(agent=self._agent)
        return self._adk_app

    def get_deployment_config(
        self,
        staging_bucket: str,
        databricks_host: Optional[str] = None,
        databricks_token_secret: Optional[str] = None,
        env_vars: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Get the deployment configuration for Agent Engine.

        Args:
            staging_bucket: GCS bucket for staging (format: "gs://bucket-name").
            databricks_host: Databricks workspace URL.
            databricks_token_secret: Secret Manager secret for Databricks token.
            env_vars: Additional environment variables.
            **kwargs: Additional configuration options.

        Returns:
            Configuration dictionary for agent_engines.create().
        """
        return create_agent_engine_config(
            staging_bucket=staging_bucket,
            requirements=self._additional_requirements,
            env_vars=env_vars,
            databricks_host=databricks_host,
            databricks_token_secret=databricks_token_secret,
            **kwargs,
        )

    async def test_locally(
        self,
        message: str,
        user_id: str = "test-user",
    ):
        """
        Test the agent locally before deployment.

        Args:
            message: The message to send to the agent.
            user_id: User ID for the session.

        Yields:
            Events from the agent's response stream.

        Example:
            ```python
            app = DatabricksAgentEngineApp(agent=my_agent)

            async for event in app.test_locally("What documents match 'AI'?"):
                print(event)
            ```
        """
        async for event in self.adk_app.async_stream_query(
            user_id=user_id,
            message=message,
        ):
            yield event


def deploy_to_agent_engine(
    agent: Agent,
    project: str,
    location: str,
    staging_bucket: str,
    databricks_host: Optional[str] = None,
    databricks_token_secret: Optional[str] = None,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs,
):
    """
    Deploy an ADK agent with Databricks tools to Vertex AI Agent Engine.

    This is a convenience function that handles the full deployment flow.

    Args:
        agent: The ADK Agent to deploy.
        project: Google Cloud project ID.
        location: Google Cloud region (e.g., "us-central1").
        staging_bucket: GCS bucket for staging (format: "gs://bucket-name").
        databricks_host: Databricks workspace URL.
        databricks_token_secret: Secret Manager secret for Databricks token.
        display_name: Display name for the deployed agent.
        description: Description for the deployed agent.
        **kwargs: Additional configuration options.

    Returns:
        The deployed remote agent resource.

    Example:
        ```python
        from databricks_google_adk import VectorSearchRetrieverTool
        from databricks_google_adk.deployment import deploy_to_agent_engine
        from google.adk.agents import Agent

        # Create agent
        vector_search = VectorSearchRetrieverTool(index_name="catalog.schema.index")
        agent = Agent(
            name="search_agent",
            model="gemini-2.0-flash",
            tools=[vector_search.as_tool()],
        )

        # Deploy
        remote_agent = deploy_to_agent_engine(
            agent=agent,
            project="my-gcp-project",
            location="us-central1",
            staging_bucket="gs://my-staging-bucket",
            databricks_host="https://my-workspace.databricks.com",
            databricks_token_secret="projects/my-project/secrets/db-token/versions/latest",
        )

        print(f"Deployed agent: {remote_agent.resource_name}")
        ```

    Note:
        Requires vertexai package: pip install google-cloud-aiplatform[agent_engines,adk]
    """
    try:
        import vertexai
        from vertexai.agent_engines import AdkApp
    except ImportError:
        raise ImportError(
            "vertexai package is required for Agent Engine deployment. "
            "Install with: pip install google-cloud-aiplatform[agent_engines,adk]"
        )

    # Initialize Vertex AI client
    client = vertexai.Client(project=project, location=location)

    # Create the app and config
    app = DatabricksAgentEngineApp(agent=agent)
    config = app.get_deployment_config(
        staging_bucket=staging_bucket,
        databricks_host=databricks_host,
        databricks_token_secret=databricks_token_secret,
        **kwargs,
    )

    # Add display name and description if provided
    create_kwargs = {"agent": app.adk_app, "config": config}
    if display_name:
        create_kwargs["display_name"] = display_name
    if description:
        create_kwargs["description"] = description

    # Deploy
    return client.agent_engines.create(**create_kwargs)
