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


@pytest.mark.timeout(3600)
def test_vectorstore():
    """
    We run the integration tests for vector store by Databricks Workflow,
    because the setup is too complex to run within a single python file.
    Thereby, this test simply triggers the workflow by calling the REST API.
    """
    test_job_id = os.getenv("VS_TEST_JOB_ID")
    if not test_job_id:
        raise RuntimeError("Please set the environment variable VS_TEST_JOB_ID")

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
