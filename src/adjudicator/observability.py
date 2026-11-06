"""Structured logging and per-case audit trails.

Console logs are human-readable; every case also gets an append-only JSONL audit file
recording each model call, tool call, validation result and failure.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("adjudicator")


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    # The OpenAI/httpx loggers are noisy at INFO and may log request bodies at DEBUG.
    for noisy in ("httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class AuditLog:
    """Append-only JSONL event log for one case (or one ingest job)."""

    def __init__(self, path: Path, *, scope: str) -> None:
        self._path = path
        self._scope = scope
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def scope(self) -> str:
        return self._scope

    def event(self, kind: str, **fields: object) -> None:
        record: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "scope": self._scope,
            "kind": kind,
            **fields,
        }
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def truncate(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else f"{text[:limit]}… [{len(text) - limit} chars truncated]"
