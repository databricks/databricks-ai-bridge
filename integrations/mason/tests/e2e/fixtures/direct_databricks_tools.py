"""Directly authored Databricks integrations for the Mason E2E matrix."""

import databricks_mason.integrations as mason_integrations

DATABRICKS_TOOLS: tuple[mason_integrations.Integration, ...] = (
    mason_integrations.Sandbox(
        id="sandbox",
        scopes=(
            mason_integrations.Scope.table(
                "samples.nyctaxi.trips",
                permission="read_only",
            ),
        ),
    ),
    mason_integrations.MCPService(
        id="web_search",
        service="system.ai.web_search",
    ),
    mason_integrations.UCFunction(
        id="mason_uc_marker",
        function="__UC_FUNCTION__",
    ),
)
