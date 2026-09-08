"""Auto-register every LangChain tool in this package."""

import importlib
import inspect
import pkgutil

from langchain_core.tools import BaseTool


def all_tools() -> list[BaseTool]:
    """Return every tool defined in this package."""
    tools: list[BaseTool] = []
    for module in pkgutil.iter_modules(__path__):
        imported = importlib.import_module(f"{__name__}.{module.name}")
        for _, tool in inspect.getmembers(imported, lambda value: isinstance(value, BaseTool)):
            if tool not in tools:
                tools.append(tool)
    return tools
