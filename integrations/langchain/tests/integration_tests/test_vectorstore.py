"""
This file contains the integration test for DatabricksVectorSearch class.

We run the integration tests nightly by the trusted CI/CD system defined in
a private repository, in order to securely run the tests. With this design,
integration test is not intended to be run manually by OSS contributors.
If you want to update the DatabricksVectorSearch implementation and you think
that you need to update the corresponding integration test, please contact to
the maintainers of the repository to verify the changes.
"""

import os
from datetime import timedelta

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifecycleStateV2State, TerminationTypeType

from databricks_langchain.vector_search_retriever_tool import VectorSearchRetrieverTool


@pytest.mark.timeout(3600)
def test_vectorstore():
    """
    We run the integration tests for vector store by Databricks Workflow,
    because the setup is too complex to run within a single python file.
    Thereby, this test simply triggers the workflow by calling the REST API.
    """
    test_job_id_str = os.getenv("VS_TEST_JOB_ID")
    if not test_job_id_str:
        raise RuntimeError("Please set the environment variable VS_TEST_JOB_ID")

    test_job_id = int(test_job_id_str)
    w = WorkspaceClient()

    # Check if there is any ongoing job run
    run_list = list(w.jobs.list_runs(job_id=test_job_id, active_only=True))
    no_active_run = len(run_list) == 0
    assert no_active_run, "There is an ongoing job run. Please wait for it to complete."

    # Trigger the workflow
    response = w.jobs.run_now(job_id=test_job_id)
    job_url = f"{w.config.host}/jobs/{test_job_id}/runs/{response.run_id}"
    print(f"Started the job at {job_url}")  # noqa: T201

    # Wait for the job to complete
    result = response.result(timeout=timedelta(seconds=3600))
    assert result.status.state == RunLifecycleStateV2State.TERMINATED
    assert result.status.termination_details.type == TerminationTypeType.SUCCESS


def test_vs_tool_with_workspace_client():
    # tested manually with SP creds and PAT creds
    w = WorkspaceClient()
    vs_tool = VectorSearchRetrieverTool(index_name="main.default.cities_index", workspace_client=w)
    index = vs_tool._vector_store.index
    assert index is not None
    if w.config.auth_type == "pat":
        assert index.personal_access_token is not None
    elif w.config.auth_type == "oauth-m2m":
        assert index.service_principal_client_id is not None
        assert index.service_principal_client_secret is not None
    else:
        raise ValueError(f"Unsupported auth type: {w.config.auth_type}")
