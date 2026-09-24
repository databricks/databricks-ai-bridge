"""Compatibility alias for :mod:`databricks_mason.runtime.durability.store`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("databricks_mason.runtime.durability.store")
