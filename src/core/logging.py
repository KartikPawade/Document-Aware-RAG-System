"""
core/logging.py
────────────────
Structured logging using structlog.
All modules use get_logger(__name__) — outputs JSON in production,
colorized text in development.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from src.core.config import get_settings


def setup_logging() -> None:
    """
    Configure structlog + stdlib logging.
    Call once at application startup (lifespan event or Prefect flow start).
    """
    settings = get_settings()
    is_production = settings.environment == "production"

    # Shared processors applied to every log event
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_production:
        # JSON output for log aggregators (Datadog, CloudWatch, Loki, etc.)
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Pretty colorized output for local development
        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    # Suppress noisy third-party loggers
    for noisy_logger in [
        "httpx", "httpcore", "openai", "langchain",
        "qdrant_client", "urllib3", "asyncio",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Root level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.DEBUG if not is_production else logging.INFO,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a named structlog logger.

    Usage:
        from src.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Event happened", key="value", count=42)
    """
    return structlog.get_logger(name)