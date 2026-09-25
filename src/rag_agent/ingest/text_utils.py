"""Helpers for reading and trimming text."""

from __future__ import annotations

import re
from pathlib import Path

_ENCODINGS = ("utf-8-sig", "cp1251")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def read_text(path: Path) -> tuple[str, str]:
    """Read a text file trying UTF-8 first, then CP1251 (common for old RU files)."""
    data = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        enc = "latin-1"
        text = data.decode(enc, errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n"), enc


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def truncate_keep_tail(text: str, max_chars: int, head_share: float = 0.25) -> str:
    """Trim the middle of a long text; the tail is kept because final metrics
    usually appear at the end of a notebook output."""
    if len(text) <= max_chars:
        return text
    marker = "\n…[обрезано]…\n"
    budget = max(max_chars - len(marker), 0)
    head = int(budget * head_share)
    tail = budget - head
    return text[:head] + marker + (text[-tail:] if tail else "")
