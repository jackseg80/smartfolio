"""
JSON Log Formatter — Structured logging for file output.

Produces one JSON object per line, compatible with log aggregation tools
(ELK, Datadog, CloudWatch, etc.).

Console output stays human-readable (text format).
File output uses this formatter for machine-parseable logs.

Each log line includes:
  - ts: Unix timestamp
  - level: DEBUG/INFO/WARNING/ERROR/CRITICAL
  - logger: Logger name
  - msg: Log message
  - request_id: Per-request correlation ID (when available)
"""
from __future__ import annotations

import json
import logging
import time
import traceback


class JsonLogFormatter(logging.Formatter):
    """Structured JSON formatter for log file output."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Add request ID from ContextVar if available
        try:
            from api.middlewares.timing import request_id_var
            rid = request_id_var.get("")
            if rid:
                entry["request_id"] = rid
        except (ImportError, LookupError):
            pass

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "msg": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(entry, ensure_ascii=False, default=str)
