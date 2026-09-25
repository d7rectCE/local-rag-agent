"""Relational catalog (SQLite): files, nodes and typed edges.

Э1 keeps the structural part; experiment/metric tables for text-to-SQL arrive in Э6.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from rag_agent.schema import Edge, Location, Node, ParsedFile

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    file_type TEXT,
    size INTEGER,
    mtime REAL,
    content_hash TEXT,
    status TEXT,            -- ok | error
    error TEXT,
    warnings TEXT,          -- JSON list
    n_nodes INTEGER DEFAULT 0,
    indexed_at TEXT
);
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_type TEXT,
    node_type TEXT,
    parent_id TEXT,
    title TEXT,
    text TEXT,
    context TEXT,
    location TEXT,          -- JSON
    metadata TEXT,          -- JSON
    content_hash TEXT,
    embed INTEGER
);
CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    type TEXT NOT NULL,
    file_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_file ON edges(file_path);
"""

_NODE_COLS = "id, file_path, file_type, node_type, parent_id, title, text, context, location, metadata, embed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_node(row: sqlite3.Row) -> Node:
    return Node(
        id=row["id"],
        file_path=row["file_path"],
        file_type=row["file_type"],
        node_type=row["node_type"],
        parent_id=row["parent_id"],
        title=row["title"] or "",
        text=row["text"] or "",
        context=row["context"] or "",
        location=Location.model_validate_json(row["location"]) if row["location"] else Location(),
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        embed=bool(row["embed"]),
    )


class Catalog:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()

    # --- meta -------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    # --- files ------------------------------------------------------------
    def file_states(self) -> dict[str, tuple[str | None, str]]:
        """path -> (content_hash, status)"""
        with self._lock:
            rows = self._conn.execute("SELECT path, content_hash, status FROM files").fetchall()
        return {r["path"]: (r["content_hash"], r["status"]) for r in rows}

    def replace_file(self, parsed: ParsedFile, size: int, mtime: float, content_hash: str) -> None:
        nodes, edges = parsed.nodes, parsed.edges
        with self._lock, self._conn:
            self._delete_file_rows(parsed.file_path)
            self._conn.executemany(
                f"INSERT OR REPLACE INTO nodes({_NODE_COLS}, content_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        n.id, n.file_path, n.file_type.value, n.node_type.value, n.parent_id, n.title, n.text,
                        n.context, n.location.model_dump_json(exclude_none=True), json.dumps(n.metadata, ensure_ascii=False),
                        int(n.embed), n.content_hash,
                    )
                    for n in nodes
                ],
            )
            self._conn.executemany(
                "INSERT INTO edges(src, dst, type, file_path) VALUES (?,?,?,?)",
                [(e.src, e.dst, e.type.value, parsed.file_path) for e in edges],
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO files(path, file_type, size, mtime, content_hash, status, error, warnings, n_nodes, indexed_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    parsed.file_path, parsed.file_type.value, size, mtime, content_hash, "ok", None,
                    json.dumps(parsed.warnings, ensure_ascii=False), len(nodes), _now(),
                ),
            )

    def mark_error(self, path: str, file_type: str, size: int, mtime: float, content_hash: str, error: str) -> None:
        with self._lock, self._conn:
            self._delete_file_rows(path)
            self._conn.execute(
                "INSERT OR REPLACE INTO files(path, file_type, size, mtime, content_hash, status, error, warnings, n_nodes, indexed_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (path, file_type, size, mtime, content_hash, "error", error, "[]", 0, _now()),
            )

    def remove_files(self, paths: Iterable[str]) -> None:
        with self._lock, self._conn:
            for p in paths:
                self._delete_file_rows(p)
                self._conn.execute("DELETE FROM files WHERE path=?", (p,))

    def _delete_file_rows(self, path: str) -> None:
        self._conn.execute("DELETE FROM nodes WHERE file_path=?", (path,))
        self._conn.execute("DELETE FROM edges WHERE file_path=?", (path,))

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM nodes")
            self._conn.execute("DELETE FROM edges")
            self._conn.execute("DELETE FROM files")

    # --- nodes & edges ----------------------------------------------------
    def get_nodes(self, ids: list[str]) -> dict[str, Node]:
        if not ids:
            return {}
        out: dict[str, Node] = {}
        with self._lock:
            for k in range(0, len(ids), 500):
                chunk = ids[k : k + 500]
                marks = ",".join("?" * len(chunk))
                for row in self._conn.execute(f"SELECT {_NODE_COLS} FROM nodes WHERE id IN ({marks})", chunk):
                    out[row["id"]] = _row_to_node(row)
        return out

    def get_node(self, node_id: str) -> Node | None:
        return self.get_nodes([node_id]).get(node_id)

    def file_nodes(self, path: str) -> list[Node]:
        with self._lock:
            rows = self._conn.execute(f"SELECT {_NODE_COLS} FROM nodes WHERE file_path=? ORDER BY rowid", (path,)).fetchall()
        return [_row_to_node(r) for r in rows]

    def edges_of(self, node_id: str) -> list[Edge]:
        with self._lock:
            rows = self._conn.execute("SELECT src, dst, type FROM edges WHERE src=? OR dst=?", (node_id, node_id)).fetchall()
        return [Edge(src=r["src"], dst=r["dst"], type=r["type"]) for r in rows]

    # --- reporting --------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            files = self._conn.execute("SELECT file_type, status, COUNT(*) c FROM files GROUP BY file_type, status").fetchall()
            nodes = self._conn.execute("SELECT node_type, COUNT(*) c FROM nodes GROUP BY node_type").fetchall()
            edges = self._conn.execute("SELECT type, COUNT(*) c FROM edges GROUP BY type").fetchall()
        return {
            "files": [{"file_type": r["file_type"], "status": r["status"], "count": r["c"]} for r in files],
            "nodes": {r["node_type"]: r["c"] for r in nodes},
            "edges": {r["type"]: r["c"] for r in edges},
            "n_files": sum(r["c"] for r in files),
            "n_nodes": sum(r["c"] for r in nodes),
        }

    def problems(self) -> list[dict]:
        """Per-file parse errors and warnings (FR1)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, file_type, status, error, warnings FROM files"
                " WHERE status != 'ok' OR (warnings IS NOT NULL AND warnings != '[]') ORDER BY status DESC, path"
            ).fetchall()
        return [
            {
                "path": r["path"],
                "file_type": r["file_type"],
                "status": r["status"],
                "error": r["error"],
                "warnings": json.loads(r["warnings"] or "[]"),
            }
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
