"""Relational catalog (SQLite): files, nodes and typed edges.

Structural part (Э1) plus the experiments, metrics and hyperparameters extracted
from notebooks by the LLM (Э6, `x_*` tables): they are keyed by file content and
model, so re-indexing an unchanged file keeps them. The SQL tool never queries this
database directly — see rag_agent.structured.analytics.
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
    file_path TEXT NOT NULL,
    label TEXT
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_file ON edges(file_path);
CREATE TABLE IF NOT EXISTS symbols (
    name TEXT NOT NULL,
    qualname TEXT,
    kind TEXT,
    signature TEXT,
    doc TEXT,
    file_path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    cell INTEGER
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
CREATE TABLE IF NOT EXISTS calls (
    name TEXT NOT NULL,
    full_name TEXT,
    caller TEXT,
    file_path TEXT NOT NULL,
    line INTEGER,
    cell INTEGER
);
CREATE INDEX IF NOT EXISTS idx_calls_name ON calls(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_calls_file ON calls(file_path);
CREATE TABLE IF NOT EXISTS x_state (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT,
    model TEXT,
    n_experiments INTEGER,
    n_values INTEGER,
    n_dropped INTEGER,       -- values not found in the cited cell (hallucinations) and dropped
    error TEXT,
    extracted_at TEXT
);
CREATE TABLE IF NOT EXISTS x_experiments (
    file_path TEXT NOT NULL, idx INTEGER, title TEXT, task TEXT, dataset TEXT, model TEXT, cell INTEGER
);
CREATE TABLE IF NOT EXISTS x_metrics (
    file_path TEXT NOT NULL, exp_idx INTEGER, name TEXT, value REAL, split TEXT, variant TEXT, cell INTEGER
);
CREATE TABLE IF NOT EXISTS x_hparams (
    file_path TEXT NOT NULL, exp_idx INTEGER, name TEXT, value TEXT, value_num REAL, variant TEXT, cell INTEGER
);
"""
X_TABLES = ("x_state", "x_experiments", "x_metrics", "x_hparams")

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
                "INSERT INTO edges(src, dst, type, file_path, label) VALUES (?,?,?,?,?)",
                [(e.src, e.dst, e.type.value, parsed.file_path, e.label) for e in edges],
            )
            self._conn.executemany(
                "INSERT INTO symbols(name, qualname, kind, signature, doc, file_path, line_start, line_end, cell)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (s.name, s.qualname, s.kind, s.signature, s.doc, parsed.file_path, s.line_start, s.line_end, s.cell)
                    for s in parsed.symbols
                ],
            )
            self._conn.executemany(
                "INSERT INTO calls(name, full_name, caller, file_path, line, cell) VALUES (?,?,?,?,?,?)",
                [(c.name, c.full_name, c.caller, parsed.file_path, c.line, c.cell) for c in parsed.calls],
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
                for table in X_TABLES:
                    self._conn.execute(f"DELETE FROM {table} WHERE file_path=?", (p,))

    def _delete_file_rows(self, path: str) -> None:
        for table in ("nodes", "edges", "symbols", "calls"):
            self._conn.execute(f"DELETE FROM {table} WHERE file_path=?", (path,))

    def clear(self) -> None:
        with self._lock, self._conn:
            for table in ("nodes", "edges", "symbols", "calls", "files", *X_TABLES):
                self._conn.execute(f"DELETE FROM {table}")

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
            rows = self._conn.execute(
                "SELECT src, dst, type, label FROM edges WHERE src=? OR dst=?", (node_id, node_id)
            ).fetchall()
        return [Edge(src=r["src"], dst=r["dst"], type=r["type"], label=r["label"]) for r in rows]

    # --- exact-name index (ТЗ S2, S5) ---------------------------------------
    def file_stems(self) -> set[str]:
        """Lower-cased file names without extension, e.g. ``02_logreg_baseline``."""
        with self._lock:
            rows = self._conn.execute("SELECT path FROM files").fetchall()
        return {Path(r["path"]).stem.lower() for r in rows}

    def find_symbols(self, name: str) -> list[dict]:
        """Definitions whose name matches exactly (case-insensitive)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM symbols WHERE name = ? COLLATE NOCASE ORDER BY file_path, cell, line_start", (name,)
            ).fetchall()
        return [dict(r) for r in rows]

    def find_calls(self, name: str) -> list[dict]:
        """Call sites of a name (last attribute of the callee), case-insensitive."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM calls WHERE name = ? COLLATE NOCASE ORDER BY file_path, cell, line", (name,)
            ).fetchall()
        return [dict(r) for r in rows]

    def nodes_at(self, file_path: str, line: int | None = None, cell: int | None = None,
                 line_end: int | None = None) -> list[Node]:
        """Content nodes of a file that cover a line range (``.py``) or a cell (lines relative to the cell)."""
        out = []
        for node in self.file_nodes(file_path):
            if node.node_type == "file":
                continue
            loc = node.location
            if cell is not None:
                if loc.cell != cell or node.node_type == "cell_output":
                    continue
                if line is not None and loc.line_start is not None:
                    if not (loc.line_start <= (line_end or line) and (loc.line_end or loc.line_start) >= line):
                        continue
                out.append(node)
            elif line is not None and loc.line_start is not None and loc.cell is None:
                if loc.line_start <= (line_end or line) and (loc.line_end or loc.line_start) >= line:
                    out.append(node)
        return out

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

    # --- extracted experiments (Э6) -----------------------------------------
    def extraction_states(self) -> dict[str, tuple[str | None, str | None]]:
        """path -> (content_hash, model) of the last extraction."""
        with self._lock:
            rows = self._conn.execute("SELECT file_path, content_hash, model FROM x_state").fetchall()
        return {r["file_path"]: (r["content_hash"], r["model"]) for r in rows}

    def replace_extraction(self, path: str, content_hash: str | None, model: str, experiments: list[dict],
                           n_dropped: int = 0, error: str | None = None) -> None:
        """experiments: [{title, task, dataset, model, cell, metrics: [...], hyperparameters: [...]}]"""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        n_values = sum(len(e.get("metrics", [])) + len(e.get("hyperparameters", [])) for e in experiments)
        with self._lock, self._conn:
            for table in X_TABLES:
                self._conn.execute(f"DELETE FROM {table} WHERE file_path=?", (path,))
            self._conn.execute(
                "INSERT INTO x_state VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (path, content_hash, model, len(experiments), n_values, n_dropped, error, now),
            )
            for i, e in enumerate(experiments):
                self._conn.execute(
                    "INSERT INTO x_experiments VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (path, i, e.get("title"), e.get("task"), e.get("dataset"), e.get("model"), e.get("cell")),
                )
                self._conn.executemany(
                    "INSERT INTO x_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(path, i, m["name"], m["value"], m.get("split"), m.get("variant") or "", m.get("cell"))
                     for m in e.get("metrics", [])],
                )
                self._conn.executemany(
                    "INSERT INTO x_hparams VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(path, i, h["name"], h["value"], h.get("value_num"), h.get("variant") or "", h.get("cell"))
                     for h in e.get("hyperparameters", [])],
                )

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Read rows of the internal tables (for building the analytics database)."""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
