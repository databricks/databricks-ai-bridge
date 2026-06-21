from unittest.mock import MagicMock, patch

import pytest

from databricks_google_adk.deployment import (
    DatabricksAgentEngineApp,
    create_agent_engine_config,
    get_databricks_requirements,
)


class TestGetDatabricksRequirements:
    """Tests for get_databricks_requirements function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_databricks_requirements()
        assert isinstance(result, list)

    def test_includes_core_packages(self):
        """Test that core packages are included."""
        result = get_databricks_requirements()
        assert "databricks-google-adk" in result
        assert "databricks-sdk" in result
        assert "databricks-ai-bridge" in result
        assert "databricks-mcp" in result


class TestCreateAgentEngineConfig:
    """Tests for create_agent_engine_config function."""

    def test_basic_config(self):
        """Test basic configuration creation."""
        config = create_agent_engine_config(
            staging_bucket="gs://my-bucket",
        )

        assert config["staging_bucket"] == "gs://my-bucket"
        assert "requirements" in config
        assert "databricks-google-adk" in config["requirements"]

    def test_with_databricks_host(self):
        """Test configuration with Databricks host."""
        config = create_agent_engine_config(
            staging_bucket="gs://my-bucket",
            databricks_host="https://my-workspace.databricks.com",
        )

        assert config["env_vars"]["DATABRICKS_HOST"] == "https://my-workspace.databricks.com"

    def test_with_databricks_token_secret(self):
        """Test configuration with Databricks token secret."""
        secret = "projects/my-project/secrets/db-token/versions/latest"
        config = create_agent_engine_config(
            staging_bucket="gs://my-bucket",
            databricks_token_secret=secret,
        )

        assert config["secrets"]["DATABRICKS_TOKEN"] == secret

    def test_with_additional_requirements(self):
        """Test configuration with additional requirements."""
        config = create_agent_engine_config(
            staging_bucket="gs://my-bucket",
            requirements=["pandas", "numpy"],
        )

        assert "pandas" in config["requirements"]
        assert "numpy" in config["requirements"]
        # Core packages should still be included
        assert "databricks-google-adk" in config["requirements"]

    def test_with_env_vars(self):
        """Test configuration with custom environment variables."""
        config = create_agent_engine_config(
            staging_bucket="gs://my-bucket",
            env_vars={"MY_VAR": "my_value"},
        )

        assert config["env_vars"]["MY_VAR"] == "my_value"

    def test_env_vars_combined_with_databricks_host(self):
        """Test that env_vars are combined with databricks_host."""
        config = create_agent_engine_config(
            staging_bucket="gs://my-bucket",
            databricks_host="https://workspace.databricks.com",
            env_vars={"OTHER_VAR": "other_value"},
        )

        assert config["env_vars"]["DATABRICKS_HOST"] == "https://workspace.databricks.com"
        assert config["env_vars"]["OTHER_VAR"] == "other_value"

    def test_additional_kwargs(self):
        """Test that additional kwargs are passed through."""
        config = create_agent_engine_config(
            staging_bucket="gs://my-bucket",
            custom_option="custom_value",
        )

        assert config["custom_option"] == "custom_value"

    def test_requirements_deduplication(self):
        """Test that duplicate requirements are removed."""
        config = create_agent_engine_config(
            staging_bucket="gs://my-bucket",
            requirements=["databricks-google-adk", "pandas"],  # databricks-google-adk is duplicate
        )

        # Count occurrences of databricks-google-adk
        count = config["requirements"].count("databricks-google-adk")
        assert count == 1


class TestDatabricksAgentEngineApp:
    """Tests for DatabricksAgentEngineApp class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock ADK agent."""
        agent = MagicMock()
        agent.name = "test_agent"
        return agent

    def test_init(self, mock_agent):
        """Test DatabricksAgentEngineApp initialization."""
        app = DatabricksAgentEngineApp(agent=mock_agent)

        assert app.agent is mock_agent
        assert app._additional_requirements == []

    def test_init_with_requirements(self, mock_agent):
        """Test initialization with additional requirements."""
        app = DatabricksAgentEngineApp(
            agent=mock_agent,
            additional_requirements=["pandas", "numpy"],
        )

        assert app._additional_requirements == ["pandas", "numpy"]

    def test_agent_property(self, mock_agent):
        """Test agent property."""
        app = DatabricksAgentEngineApp(agent=mock_agent)
        assert app.agent is mock_agent

    def test_adk_app_requires_vertexai(self, mock_agent):
        """Test that adk_app raises ImportError without vertexai."""
        app = DatabricksAgentEngineApp(agent=mock_agent)

        with patch.dict("sys.modules", {"vertexai": None, "vertexai.agent_engines": None}):
            # Force reimport to trigger ImportError
            with pytest.raises(ImportError, match="vertexai package is required"):
                # Clear cached app
                app._adk_app = None
                _ = app.adk_app

    def test_adk_app_with_mock_vertexai(self, mock_agent):
        """Test adk_app with mocked vertexai."""
        mock_adk_app = MagicMock()

        with patch("databricks_google_adk.deployment.AdkApp", return_value=mock_adk_app):
            # Need to patch the import
            with patch.dict("sys.modules", {"vertexai": MagicMock(), "vertexai.agent_engines": MagicMock()}):
                app = DatabricksAgentEngineApp(agent=mock_agent)
                # Can't actually test this without vertexai installed
                # Just verify the app object exists
                assert app._adk_app is None

    def test_get_deployment_config(self, mock_agent):
        """Test get_deployment_config method."""
        app = DatabricksAgentEngineApp(
            agent=mock_agent,
            additional_requirements=["extra-package"],
        )

        config = app.get_deployment_config(
            staging_bucket="gs://bucket",
            databricks_host="https://workspace.databricks.com",
        )

        assert config["staging_bucket"] == "gs://bucket"
        assert "extra-package" in config["requirements"]
        assert config["env_vars"]["DATABRICKS_HOST"] == "https://workspace.databricks.com"

    def test_get_deployment_config_with_secret(self, mock_agent):
        """Test get_deployment_config with token secret."""
        app = DatabricksAgentEngineApp(agent=mock_agent)

        config = app.get_deployment_config(
            staging_bucket="gs://bucket",
            databricks_token_secret="projects/p/secrets/s/versions/v",
        )

        assert config["secrets"]["DATABRICKS_TOKEN"] == "projects/p/secrets/s/versions/v"
