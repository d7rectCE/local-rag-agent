"""Мелкие утилиты: фиксация seed и журнал запусков."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def log_run(name: str, params: dict, metrics: dict, path: str = "runs/runs.jsonl") -> None:
    """Дописать запуск в JSONL-журнал."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {"name": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "params": params, "metrics": metrics}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
