"""Corpus traversal with include/exclude rules."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class CorpusFile:
    abs_path: Path
    rel_path: str  # POSIX, relative to the corpus root
    ext: str
    size: int
    mtime: float


@dataclass(frozen=True)
class SkippedFile:
    rel_path: str
    reason: str


class ExcludeRules:
    """Glob rules. A pattern without "/" matches any single path component
    (``__pycache__``, ``*.egg-info``, ``Аккаунты``); a pattern with "/" is
    matched against the whole relative path (``data/raw/**``)."""

    def __init__(self, patterns: Iterable[str], skip_hidden: bool = True):
        self.name_patterns: list[str] = []
        self.path_patterns: list[str] = []
        for p in patterns:
            p = p.strip().replace("\\", "/")
            if not p:
                continue
            if "/" in p.strip("/"):
                self.path_patterns.append(p.strip("/"))
            else:
                self.name_patterns.append(p.strip("/"))
        self.skip_hidden = skip_hidden

    def excludes(self, rel_path: str) -> bool:
        parts = rel_path.split("/")
        if self.skip_hidden and any(part.startswith(".") for part in parts):
            return True
        for part in parts:
            if any(fnmatch.fnmatch(part, pat) for pat in self.name_patterns):
                return True
        return any(
            fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(rel_path + "/", pat)
            for pat in self.path_patterns
        )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def iter_corpus(
    root: Path,
    include_ext: Iterable[str],
    exclude: Iterable[str] = (),
    skip_hidden: bool = True,
    max_file_mb: float = 20.0,
    protected_dirs: Iterable[Path] = (),
    skipped: list[SkippedFile] | None = None,
) -> Iterator[CorpusFile]:
    """Yield indexable files under ``root`` in a stable order.

    ``protected_dirs`` are always pruned when they lie strictly inside ``root``
    (the tool's own repository and data directory, so the agent never indexes
    itself or its indexes).
    """
    root = root.resolve()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in include_ext}
    rules = ExcludeRules(exclude, skip_hidden)
    protected = [p.resolve() for p in protected_dirs]
    protected = [p for p in protected if p != root and _is_within(p, root)]
    max_bytes = int(max_file_mb * 1024 * 1024)

    def skip(rel: str, reason: str) -> None:
        if skipped is not None:
            skipped.append(SkippedFile(rel, reason))

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        rel_dir = current.relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        kept = []
        for d in sorted(dirnames):
            rel = f"{rel_dir}/{d}" if rel_dir else d
            full = current / d
            if any(full.resolve() == p for p in protected):
                continue
            if rules.excludes(rel):
                continue
            kept.append(d)
        dirnames[:] = kept
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if rules.excludes(rel):
                continue
            path = current / name
            try:
                st = path.stat()
            except OSError as exc:
                skip(rel, f"stat failed: {exc}")
                continue
            if st.st_size > max_bytes:
                skip(rel, f"too large: {st.st_size / 2**20:.1f} MB")
                continue
            yield CorpusFile(path, rel, ext, st.st_size, st.st_mtime)
