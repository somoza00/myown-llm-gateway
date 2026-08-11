"""Structured logging bootstrap (structlog): JSON output, request-context binding,
and log-level configuration.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog
from structlog.typing import FilteringBoundLogger


def configure_logging(log_level: str = "INFO") -> None:
    """Configure stdlib logging + structlog to emit JSON to stdout at the given level."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog bound logger for the given module/component name."""
    return cast(FilteringBoundLogger, structlog.get_logger(name))


def bind_request_context(**kwargs: Any) -> None:
    """Bind key-value pairs (e.g. request_id) to all subsequent log calls in this context."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    """Clear any request-scoped context bound via `bind_request_context`."""
    structlog.contextvars.clear_contextvars()
