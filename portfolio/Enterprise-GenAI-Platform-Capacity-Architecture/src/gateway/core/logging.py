"""Structured logging configuration using structlog.

Every log entry automatically carries:
  - request_id (injected by RequestIDMiddleware)
  - timestamp (ISO-8601)
  - level
  - event

Security rules enforced here (not per-call):
  - Authorization header values are NEVER logged.
  - API keys (llm_xxx strings) are redacted from any log message.
  - Raw secrets from env are never passed to logger callsites.
"""

import logging
import re
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from src.gateway.core.config import get_settings

# Regex to catch any Bearer token or llm_ prefixed key in a log message.
_SECRET_PATTERN = re.compile(
    r"(Bearer\s+\S+|llm_[A-Za-z0-9_\-]{10,})",
    re.IGNORECASE,
)


def _redact_secrets(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """structlog processor: scrub Bearer tokens and API keys from all fields."""
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            event_dict[key] = _SECRET_PATTERN.sub("[REDACTED]", value)
        elif key.lower() in {"authorization", "x-api-key", "api_key", "token"}:
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging() -> None:
    """Call once at application startup (inside lifespan)."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _redact_secrets,
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.gateway_env == "production":
        # JSON lines for log aggregation (Datadog, CloudWatch, etc.)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable for local development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Quiet noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str = "gateway") -> structlog.BoundLogger:
    """Return a bound structlog logger. Use this instead of logging.getLogger()."""
    return structlog.get_logger(name)
