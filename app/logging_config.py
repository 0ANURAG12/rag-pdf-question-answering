"""
Central logging configuration.

Emits structured, single-line logs so they can be piped into any log
aggregator. Use `json_logs=true` for machine-readable JSON output.
"""

import json
import logging
import sys


# Standard LogRecord attributes that must not be confused with user extras.
_STD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
)


def _extra_fields(record: logging.LogRecord) -> dict:
    """Return only the non-standard fields added via `extra={...}`."""
    return {
        key: value
        for key, value in sorted(record.__dict__.items())
        if key not in _STD_ATTRS
        and isinstance(value, (str, int, float, bool))
    }


class _OneLineFormatter(logging.Formatter):
    """Format a record as a single readable line with any extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        extras = _extra_fields(record)
        parts = [
            f"timestamp={self.formatTime(record, self.datefmt)}",
            f"level={record.levelname}",
            f"logger={record.name}",
            f"message={record.getMessage()}",
        ]
        parts.extend(f"{key}={value}" for key, value in extras.items())
        return " ".join(parts)


class _JsonFormatter(logging.Formatter):
    """Format a record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the root logger. Called once at app startup."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter() if json_logs else _OneLineFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Keep noisy third-party loggers at WARNING.
    for noisy in ("chromadb", "httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module. Use __name__ as the argument."""
    return logging.getLogger(name)
