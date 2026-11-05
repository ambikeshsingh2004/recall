"""Local structured JSON logs, always on and independent of LangSmith (PRD FR-O5).

Named logs.py (not logging.py) so it never shadows the standard library module.
"""

import json
import logging
import sys
from pathlib import Path

from memory_core.config import get_settings

_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        data.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(root, "_recall_configured", False):
        return
    log_dir = Path(get_settings().log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = JsonFormatter()
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "recall.jsonl", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    root.setLevel(level)
    root._recall_configured = True  # type: ignore[attr-defined]
