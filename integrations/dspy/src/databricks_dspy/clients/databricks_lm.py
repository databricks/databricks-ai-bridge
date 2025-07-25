import logging
from typing import Optional

import dspy
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import PtEndpointCoreConfig, PtServedModel

logger = logging.getLogger(__name__)


class DatabricksLM(dspy.LM):
    def __init__(
        self,
        model: str,
        workspace_client: Optional[WorkspaceClient] = None,
        create_pt_endpoint: bool = False,
        pt_entity: Optional[PtServedModel] = None,
        **kwargs,
    ):
        """Subclass of `dspy.LM` for compatibility with Databricks.

        Args:
            model: The model to use. Must start with 'databricks/'.
            workspace_client: The workspace client to use. If not provided, a new one will be
                created with default credentials from the environment.
            create_pt_endpoint: Whether to create a provisioned throughput endpoint to make LM
                calls.
            pt_entity: The entity to serve, only used when `create_pt_endpoint` is True.

        Example 1: Use a Databricks model with preconfigured workspace client.

        ```python
        import dspy
        import databricks_dspy
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        lm = databricks_dspy.DatabricksLM(
            "databricks/databricks-llama-4-maverick",
            workspace_client=w,
        )
        dspy.configure(lm=lm)

        predict = dspy.Predict("q->a")
        print(predict(q="why did a chicken cross the kitchen?"))
        ```

        Example 2: Create a provisioned throughput endpoint for a Databricks model.

        ```python
        import dspy
        import databricks_dspy
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import PtServedModel

        w = WorkspaceClient()
        entity = PtServedModel(
            entity_name="system.ai.llama-4-maverick",
            entity_version="1",
            provisioned_model_units=50,
        )
        lm = databricks_dspy.DatabricksLM(
            "databricks/provisioned-llama-4-maverick",
            workspace_client=w,
            create_pt_endpoint=True,
            pt_entity=entity,
        )
        dspy.configure(lm=lm)

        predict = dspy.Predict("q->a")
        print(predict(q="why did a chicken cross the kitchen?"))
        ```
        """
        if not model.startswith("databricks/"):
            raise ValueError(
                "`model` must start with 'databricks/' when using `DatabricksLM`, "
                "e.g. dspy.LM('databricks/databricks-llama-4-maverick')"
            )

        super().__init__(model=model, **kwargs)

        if workspace_client:
            self.workspace_client = workspace_client
        else:
            self.workspace_client = WorkspaceClient()

        try:
            # If credentials are invalid, `w.current_user.me()` will throw an error.
            self.workspace_client.current_user.me()
        except Exception as e:
            raise RuntimeError(
                "Failed to validate databricks credentials, please refer to "
                "https://docs.databricks.com/aws/en/dev-tools/auth/unified-auth#default-methods-for-client-unified-authentication "  # noqa: E501
                "for how to set up the authentication."
            ) from e

        self.create_pt_endpoint = create_pt_endpoint
        self.pt_entity = pt_entity

        if create_pt_endpoint:
            self.pt_endpoint = self._create_pt_endpoint()

    def _create_pt_endpoint(self):
        # Create the provisioned throughput endpoint configuration
        config = PtEndpointCoreConfig(served_entities=[self.pt_entity])

        model_name_without_databricks_prefix = self.model[len("databricks/") :]
        # Create the provisioned throughput endpoint
        w = self.workspace_client
        try:
            return w.serving_endpoints.create_provisioned_throughput_endpoint_and_wait(
                name=model_name_without_databricks_prefix,
                config=config,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create provisioned throughput endpoint: {e}\n\n"
                "`create_pt_endpoint=True` is only supported in Databricks notebooks now."
            ) from e

    def tear_down(self):
        if not self.create_pt_endpoint:
            logger.warning("`tear_down` is an no-op when `create_pt_endpoint` is False.")
            return

        self.workspace_client.serving_endpoints.delete(self.pt_endpoint.name)

    def forward(self, **kwargs):
        return super().forward(
            headers=self.workspace_client.config.authenticate(),
            api_base=f"{self.workspace_client.config.host}/serving-endpoints",
            **kwargs,
        )
