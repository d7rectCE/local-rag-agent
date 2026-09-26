"""Builds the analytics database (schema in rag_agent.structured.schema) from the
index catalog: deterministic tables from the parsers plus the LLM-extracted
experiments. Rebuilt as a whole into a temporary file and swapped in atomically, so
a reader never sees a half-written database."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

from rag_agent.structured.schema import TABLES

ANALYTICS_FILE = "analytics.sqlite"
_ERROR = re.compile(r"(?m)^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit))\b")


def _cell(location: str | None) -> int | None:
    try:
        return json.loads(location or "{}").get("cell")
    except json.JSONDecodeError:
        return None


def build_analytics(catalog, path: Path) -> dict[str, int]:
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        for t in TABLES:
            conn.execute(t.ddl())
        q = catalog.query

        conn.executemany("INSERT INTO files VALUES (?, ?, ?, ?, ?)", [
            (r["path"], r["file_type"], r["size"], r["n_nodes"], r["status"])
            for r in q("SELECT path, file_type, size, n_nodes, status FROM files")])

        nb_nodes = q("SELECT id, file_path, node_type, title, text, location, metadata FROM nodes "
                     "WHERE file_type='ipynb' ORDER BY file_path, id")
        cells: dict[tuple[str, int], dict] = {}
        notebooks: dict[str, dict] = {}
        for r in nb_nodes:
            meta = json.loads(r["metadata"] or "{}")
            if r["node_type"] == "file":
                notebooks[r["file_path"]] = {"n_cells": meta.get("n_cells"), "out_of_order": int(bool(meta.get("out_of_order"))),
                                             "title": None}
                continue
            c = _cell(r["location"])
            if c is None:
                continue
            cell = cells.setdefault((r["file_path"], c), {"type": None, "source": "", "output": None, "error": None})
            if r["node_type"] == "cell_output":
                cell["output"] = (cell["output"] or "") + (r["text"] or "")
                if "error" in meta.get("output_types", []):
                    m = _ERROR.search(r["text"] or "")
                    cell["error"] = m.group(1) if m else "Error"
            else:
                cell["type"] = "code" if r["node_type"] == "code_cell" else "markdown"
                cell["source"] += ("\n" if cell["source"] else "") + (r["text"] or "")
        for (fp, c), cell in sorted(cells.items()):
            nb = notebooks.setdefault(fp, {"n_cells": None, "out_of_order": 0, "title": None})
            if nb["title"] is None and cell["type"] == "markdown":
                first = next((ln for ln in cell["source"].split("\n") if ln.strip()), "")
                nb["title"] = first.lstrip("#").strip() or None
        conn.executemany("INSERT INTO notebooks VALUES (?, ?, ?, ?, ?)", [
            (fp, nb["title"], nb["n_cells"], sum(1 for (f, _), cl in cells.items() if f == fp and cl["type"] == "code"),
             nb["out_of_order"]) for fp, nb in sorted(notebooks.items())])
        conn.executemany("INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (fp, c, cl["type"] or "code", cl["source"], cl["output"], int(bool(cl["error"])), cl["error"])
            for (fp, c), cl in sorted(cells.items())])

        conn.executemany("INSERT INTO functions VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (r["name"], r["qualname"], r["kind"], r["signature"], r["file_path"], r["line_start"], r["cell"])
            for r in q("SELECT name, qualname, kind, signature, file_path, line_start, cell FROM symbols")])
        conn.executemany("INSERT INTO calls VALUES (?, ?, ?, ?, ?)", [
            (r["name"], r["caller"], r["file_path"], r["line"], r["cell"])
            for r in q("SELECT name, caller, file_path, line, cell FROM calls")])
        conn.executemany("INSERT INTO relations VALUES (?, ?, ?, ?)", [
            (r["file_path"], r["src"], r["dst"], r["type"])
            for r in q("SELECT file_path, src, dst, type FROM edges WHERE type != 'contains'")])

        exp_ids: dict[tuple[str, int], int] = {}
        for r in q("SELECT * FROM x_experiments ORDER BY file_path, idx"):
            exp_ids[(r["file_path"], r["idx"])] = len(exp_ids) + 1
            conn.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (exp_ids[(r["file_path"], r["idx"])], r["file_path"], r["title"], r["task"], r["dataset"],
                          r["model"], r["cell"]))
        conn.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (exp_ids.get((r["file_path"], r["exp_idx"])), r["file_path"], r["name"], r["value"], r["split"],
             r["variant"], r["cell"]) for r in q("SELECT * FROM x_metrics")])
        conn.executemany("INSERT INTO hyperparameters VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (exp_ids.get((r["file_path"], r["exp_idx"])), r["file_path"], r["name"], r["value"], r["value_num"],
             r["variant"], r["cell"]) for r in q("SELECT * FROM x_hparams")])
        conn.commit()
        counts = {t.name: conn.execute(f"SELECT COUNT(*) FROM {t.name}").fetchone()[0] for t in TABLES}
    finally:
        conn.close()
    os.replace(tmp, path)
    return counts
