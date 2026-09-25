"""Local JSONL traces of LLM and tool calls (NFR4). Traces never leave the machine."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, directory: Path | None, log_prompts: bool = True):
        self.directory = directory
        self.log_prompts = log_prompts
        self._lock = threading.Lock()
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **fields: Any) -> None:
        if self.directory is None:
            return
        now = datetime.now(timezone.utc)
        record = {"ts": now.isoformat(timespec="milliseconds"), "kind": kind, **fields}
        path = self.directory / f"{now:%Y-%m-%d}.jsonl"
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


NULL_TRACER = Tracer(None)
