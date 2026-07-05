"""Utility functions for BioModels SSA benchmark suite."""

import functools
import logging
import signal
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when a function times out."""

    pass


def timeout(seconds: int):
    """Decorator to add timeout to a function.

    Usage:
        @timeout(30)
        def slow_function():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")

            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper

    return decorator


@contextmanager
def timeout_context(seconds: int):
    """Context manager for timeout protection.

    Usage:
        with timeout_context(30):
            slow_operation()
    """

    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_execute(
    func: Callable,
    *args,
    timeout_sec: int | None = None,
    default: Any = None,
    **kwargs,
) -> tuple[bool, Any, str | None]:
    """Safely execute a function with optional timeout.

    Args:
        func: Function to execute.
        *args: Positional arguments for func.
        timeout_sec: Optional timeout in seconds.
        default: Default value to return on error.
        **kwargs: Keyword arguments for func.

    Returns:
        Tuple of (success, result, error_msg).
    """
    try:
        if timeout_sec:
            with timeout_context(timeout_sec):
                result = func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        return True, result, None
    except TimeoutError as e:
        return False, default, f"Timeout: {e}"
    except Exception as e:
        return False, default, f"{type(e).__name__}: {e}"


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
):
    """Set up logging configuration."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
