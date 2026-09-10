"""Agent tools package.

Every module here is auto-imported, and every Anthropic Tool Runner tool it defines (via the
``@beta_tool`` decorator) is collected by ``all_tools()``. Drop a new ``*.py`` into this folder with a
``@beta_tool``-decorated function and it's picked up automatically — no wiring to edit.
"""

import importlib
import inspect
import pkgutil

from anthropic.lib.tools import BetaFunctionTool


def all_tools() -> list[BetaFunctionTool]:
    """Every Tool Runner tool defined across the modules in this package."""
    tools: list[BetaFunctionTool] = []
    for module in pkgutil.iter_modules(__path__):
        mod = importlib.import_module(f"{__name__}.{module.name}")
        for _, obj in inspect.getmembers(mod, lambda o: isinstance(o, BetaFunctionTool)):
            if obj not in tools:  # a tool imported into several modules is collected once
                tools.append(obj)
    return tools
