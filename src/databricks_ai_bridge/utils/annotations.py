# This code is copied from MLflow: https://github.com/mlflow/mlflow/blob/v2.19.0/mlflow/utils/annotations.py#L31

import inspect
import re
import types
from typing import Any, Callable, TypeVar, Union

C = TypeVar("C", bound=Callable[..., Any])


def _get_min_indent_of_docstring(docstring_str: str) -> str:
    """
    Get the minimum indentation string of a docstring, based on the assumption
    that the closing triple quote for multiline comments must be on a new line.
    Note that based on ruff rule D209, the closing triple quote for multiline
    comments must be on a new line.

    Args:
        docstring_str: string with docstring

    Returns:
        Whitespace corresponding to the indent of a docstring.
    """

    if not docstring_str or "\n" not in docstring_str:
        return ""

    return re.match(r"^\s*", docstring_str.rsplit("\n", 1)[-1]).group()


def experimental(api_or_type: Union[C, str]) -> C:
    """Decorator / decorator creator for marking APIs experimental in the docstring.

    Args:
        api_or_type: An API to mark, or an API typestring for which to generate a decorator.

    Returns:
        Decorated API (if a ``api_or_type`` is an API) or a function that decorates
        the specified API type (if ``api_or_type`` is a typestring).
    """
    if isinstance(api_or_type, str):

        def f(api: C) -> C:
            return _experimental(api=api, api_type=api_or_type)

        return f
    elif inspect.isclass(api_or_type):
        return _experimental(api=api_or_type, api_type="class")
    elif inspect.isfunction(api_or_type):
        return _experimental(api=api_or_type, api_type="function")
    elif isinstance(api_or_type, (property, types.MethodType)):
        return _experimental(api=api_or_type, api_type="property")
    else:
        return _experimental(api=api_or_type, api_type=str(type(api_or_type)))


def _experimental(api: C, api_type: str) -> C:
    indent = _get_min_indent_of_docstring(api.__doc__)
    notice = (
        indent + f".. Note:: Experimental: This {api_type} may change or "
        "be removed in a future release without warning.\n\n"
    )
    if api_type == "property":
        api.__doc__ = api.__doc__ + "\n\n" + notice if api.__doc__ else notice
    else:
        api.__doc__ = notice + api.__doc__ if api.__doc__ else notice
    return api
