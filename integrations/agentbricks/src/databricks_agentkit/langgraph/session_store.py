"""Compatibility alias for :mod:`databricks_agentbricks.langgraph.session_store`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("databricks_agentbricks.langgraph.session_store")
