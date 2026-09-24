"""Compatibility alias for :mod:`databricks_agentbricks.langgraph.memory`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("databricks_agentbricks.langgraph.memory")
