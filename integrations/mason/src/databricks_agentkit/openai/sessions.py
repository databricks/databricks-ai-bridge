"""Compatibility alias for :mod:`databricks_mason.openai.sessions`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("databricks_mason.openai.sessions")
