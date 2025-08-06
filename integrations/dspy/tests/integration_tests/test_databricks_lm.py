import os
from datetime import timedelta

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifecycleStateV2State, TerminationTypeType


@pytest.mark.timeout(3600)
def test_databricks_lm():
    """
    This test simply triggers a predefined Databricks job to verify the functionality
    of the DatabricksLM class.
    """
    test_job_id = os.getenv("DATABRICKS_LM_TEST_JOB_ID")
    branch_name = os.getenv("BRANCH_NAME")
    fork_name = os.getenv("FORK_NAME")

    if not test_job_id:
        raise RuntimeError(
            "Please set the environment variable DATABRICKS_LM_TEST_JOB_ID",
        )

    w = WorkspaceClient()

    # Check if there is any ongoing job run
    run_list = list(w.jobs.list_runs(job_id=test_job_id, active_only=True))
    no_active_run = len(run_list) == 0
    assert no_active_run, "There is an ongoing job run. Please wait for it to complete."

    # Trigger the workflow
    response = w.jobs.run_now(
        job_id=test_job_id,
        job_parameters={
            "branch_name": branch_name or "main",
            "fork_name": fork_name or "databricks",
        },
    )
    job_url = f"{w.config.host}/jobs/{test_job_id}/runs/{response.run_id}"
    print(f"Started the job at {job_url}")  # noqa: T201

    # Wait for the job to complete
    result = response.result(timeout=timedelta(seconds=3600))
    assert result.status.state == RunLifecycleStateV2State.TERMINATED
    assert result.status.termination_details.type == TerminationTypeType.SUCCESS
