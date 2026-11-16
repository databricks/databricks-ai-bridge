"""Execution placement recorded in deployed App configuration, independent of storage."""

import os

RUNTIME_ENVIRONMENT_ENV = "AGENTBRICKS_RUNTIME_ENVIRONMENT"
SANDBOX = "sandbox"
SHARED_COMPUTE = "shared_compute"


def runtime_environment():
    value = os.getenv(RUNTIME_ENVIRONMENT_ENV, SHARED_COMPUTE)
    if value not in {SANDBOX, SHARED_COMPUTE}:
        raise ValueError(f"Unsupported runtime_environment: {value}")
    return value
