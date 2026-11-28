"""Click callbacks that reject store names the managed store service won't accept."""

from __future__ import annotations

from typing import Callable, Optional

import click

from databricks_agentbricks.projects.store_names import (
    memory_store_name_problem,
    session_store_name_problem,
)

_Callback = Callable[[click.Context, click.Parameter, Optional[str]], Optional[str]]


def memory_store_name(*, allow_legacy: bool = False) -> _Callback:
    """Validate a memory store name; ``allow_legacy`` also accepts an existing legacy display name."""

    def callback(ctx: click.Context, param: click.Parameter, value: Optional[str]) -> Optional[str]:
        if value is not None:
            problem = memory_store_name_problem(value, allow_legacy=allow_legacy)
            if problem:
                raise click.BadParameter(problem, ctx=ctx, param=param)
        return value

    return callback


def session_store_name(
    ctx: click.Context, param: click.Parameter, value: Optional[str]
) -> Optional[str]:
    """Validate a session store name."""
    if value is not None:
        problem = session_store_name_problem(value)
        if problem:
            raise click.BadParameter(problem, ctx=ctx, param=param)
    return value
