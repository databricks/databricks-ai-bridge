"""Compatibility module for handling optional dependencies."""

from typing import Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable)


def mlflow_trace(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    span_type: Optional[str] = None,
) -> F:
    """Decorator that traces a function with mlflow if available.

    If mlflow is not installed, the function executes normally without tracing.

    Args:
        func: Function to trace
        name: Optional name for the span (defaults to function name)
        span_type: Optional span type (e.g., "CHAIN", "PARSER", "AGENT")

    Returns:
        Traced function if mlflow is available, otherwise the original function

    Examples:
        @mlflow_trace
        def my_function():
            pass

        @mlflow_trace(span_type="PARSER")
        def parse_data():
            pass
    """

    def decorator(f: F) -> F:
        try:
            import mlflow

            return mlflow.trace(f, name=name, span_type=span_type)  # type: ignore[return-value]
        except ImportError:
            return f

    # Support both @mlflow_trace and @mlflow_trace(...) syntax
    if func is None:
        return decorator  # type: ignore[return-value]
    else:
        return decorator(func)
