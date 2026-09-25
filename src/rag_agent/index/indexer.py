"""Indexing pipeline: walk -> hash -> parse -> embed -> store (FR1, FR5).

Incremental at file level: unchanged files (same content hash) are skipped,
changed files are re-parsed and their vectors replaced, deleted files are
removed. Each combination of embedder and chunking settings gets its own index
directory, so ablations can switch configs without destroying other indexes.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field

from rag_agent.config import Settings
from rag_agent.index.catalog import Catalog
from rag_agent.index.embedder import Embedder
from rag_agent.index.vector_store import VectorStore
from rag_agent.ingest import CorpusFile, SkippedFile, iter_corpus, parse_file
from rag_agent.schema import ParsedFile

SCHEMA_VERSION = 2  # 2: AST chunks, symbols/calls tables, uses_var edges


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def corpus_key(root: Path) -> str:
    root = root.resolve()
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", root.name).strip("_")[:40] or "corpus"
    digest = hashlib.sha1(str(root).lower().encode("utf-8")).hexdigest()[:8]
    return f"{name}-{digest}"


def index_signature(settings: Settings) -> str:
    emb = settings.embedding
    payload = {
        "schema": SCHEMA_VERSION,
        "embedding": {"model": emb.model, "pooling": emb.pooling, "max_length": emb.max_length},
        "chunking": settings.chunking.model_dump(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class IndexProgress(BaseModel):
    state: Literal["idle", "scanning", "indexing", "done", "error", "cancelled"] = "idle"
    root: str | None = None
    files_total: int = 0
    files_done: int = 0
    current: str | None = None
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    errors: int = 0
    skipped: list[dict] = Field(default_factory=list)
    nodes_embedded: int = 0
    timings: dict[str, float] = Field(default_factory=lambda: {"parse_s": 0.0, "embed_s": 0.0, "write_s": 0.0})
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_s: float = 0.0
    message: str | None = None

    @property
    def running(self) -> bool:
        return self.state in ("scanning", "indexing")


class CorpusIndex:
    """Catalog + vector store of one indexed folder under one index signature."""

    def __init__(self, root: Path, settings: Settings, embedder: Embedder):
        self.root = root.resolve()
        self.key = corpus_key(self.root)
        self.signature = index_signature(settings)
        self.dir = settings.data_dir / "indexes" / self.key / self.signature
        self.catalog = Catalog(self.dir / "catalog.sqlite")
        self.store = VectorStore(
            self.dir / "qdrant",
            dim=embedder.dim,
            with_sparse=embedder.has_sparse,
            url=settings.storage.qdrant_url,
            # the local store directory is already unique; a short name keeps Windows paths < 260 chars
            collection=f"nodes_{self.key}_{self.signature}" if settings.storage.qdrant_url else "nodes",
        )
        self.catalog.set_meta("root", str(self.root))
        self.catalog.set_meta("signature", self.signature)
        self.catalog.set_meta("embedder", embedder.name)

    def close(self) -> None:
        self.store.close()
        self.catalog.close()


def run_indexing(
    index: CorpusIndex,
    settings: Settings,
    embedder: Embedder,
    progress: IndexProgress,
    *,
    include_ext: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    protected_dirs: Iterable[Path] = (),
    force: bool = False,
    cancel: threading.Event | None = None,
    flush_nodes: int = 256,
) -> IndexProgress:
    cfg = settings.corpus
    t_start = time.perf_counter()
    progress.state = "scanning"
    progress.root = str(index.root)
    progress.started_at = _now()

    try:
        skipped: list[SkippedFile] = []
        files = list(
            iter_corpus(
                index.root,
                include_ext=include_ext or cfg.include_ext,
                exclude=cfg.exclude if exclude is None else exclude,
                skip_hidden=cfg.skip_hidden,
                max_file_mb=cfg.max_file_mb,
                protected_dirs=protected_dirs,
                skipped=skipped,
            )
        )
        progress.files_total = len(files)
        progress.skipped = [{"path": s.rel_path, "reason": s.reason} for s in skipped]

        states = index.catalog.file_states()
        if force and states:
            index.store.delete_files(list(states))
            index.catalog.clear()
            states = {}
        present = {f.rel_path for f in files}
        deleted = [p for p in states if p not in present]
        if deleted:
            index.store.delete_files(deleted)
            index.catalog.remove_files(deleted)
        progress.deleted = len(deleted)

        progress.state = "indexing"
        buffer: list[tuple[CorpusFile, str, ParsedFile]] = []
        buffered = 0

        def flush() -> None:
            nonlocal buffered
            if not buffer:
                return
            nodes = [n for _, _, p in buffer for n in p.nodes if n.embed and n.text.strip()]
            t0 = time.perf_counter()
            enc = embedder.encode([n.embedding_text(settings.chunking.context_header) for n in nodes]) if nodes else None
            t1 = time.perf_counter()
            index.store.delete_files([cf.rel_path for cf, _, _ in buffer])
            if enc is not None:
                index.store.upsert(
                    [n.id for n in nodes],
                    enc.dense,
                    enc.sparse,
                    [{"file_path": n.file_path, "file_type": n.file_type.value, "node_type": n.node_type.value} for n in nodes],
                )
            for cf, digest, parsed in buffer:
                index.catalog.replace_file(parsed, cf.size, cf.mtime, digest)
            t2 = time.perf_counter()
            progress.timings["embed_s"] += t1 - t0
            progress.timings["write_s"] += t2 - t1
            progress.nodes_embedded += len(nodes)
            progress.files_done += len(buffer)
            buffer.clear()
            buffered = 0

        for cf in files:
            if cancel is not None and cancel.is_set():
                progress.state = "cancelled"
                break
            progress.current = cf.rel_path
            file_type = cf.ext.lstrip(".")
            try:
                digest = sha256_file(cf.abs_path)
            except OSError as exc:
                index.catalog.mark_error(cf.rel_path, file_type, cf.size, cf.mtime, "", f"read failed: {exc}")
                progress.errors += 1
                progress.files_done += 1
                continue
            old = states.get(cf.rel_path)
            if old is not None and old[0] == digest:  # unchanged; a known parse error stays recorded
                progress.unchanged += 1
                progress.files_done += 1
                continue
            if old is None:
                progress.new += 1
            else:
                progress.changed += 1
            t0 = time.perf_counter()
            try:
                parsed = parse_file(cf, settings.chunking)
            except Exception as exc:  # a broken file must not stop the run
                index.store.delete_files([cf.rel_path])
                index.catalog.mark_error(cf.rel_path, file_type, cf.size, cf.mtime, digest, f"{type(exc).__name__}: {exc}")
                progress.errors += 1
                progress.files_done += 1
                continue
            finally:
                progress.timings["parse_s"] += time.perf_counter() - t0
            buffer.append((cf, digest, parsed))
            buffered += sum(1 for n in parsed.nodes if n.embed)
            if buffered >= flush_nodes:
                flush()
        flush()
        if progress.state == "indexing":
            progress.state = "done"
    except Exception as exc:
        progress.state = "error"
        progress.message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        progress.current = None
        progress.finished_at = _now()
        progress.elapsed_s = round(time.perf_counter() - t_start, 2)
        index.catalog.set_meta("last_run", progress.model_dump_json())
    return progress
