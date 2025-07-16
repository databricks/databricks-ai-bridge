from typing import Optional

import dspy
from databricks.sdk import WorkspaceClient


class DatabricksLM(dspy.LM):
    def __init__(
        self,
        model: str,
        workspace_client: Optional[WorkspaceClient] = None,
        **kwargs,
    ):
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

    def forward(self, **kwargs):
        return super().forward(
            headers=self.workspace_client.config.authenticate(),
            api_base=f"{self.workspace_client.config.host}/serving-endpoints",
            **kwargs,
        )
