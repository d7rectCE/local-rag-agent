"""Parser for .ipynb files: every cell is a node, outputs are separate nodes
linked to the producing cell with a ``produces`` edge (ТЗ S3, text part).

Images inside outputs are only counted here; extracting them into the image
pipeline is Э5.
"""

from __future__ import annotations

import re

import nbformat

from rag_agent.config import ChunkingConfig
from rag_agent.ingest.python_parser import line_windows
from rag_agent.ingest.text_utils import read_text, strip_ansi, truncate_keep_tail
from rag_agent.ingest.walker import CorpusFile
from rag_agent.schema import Edge, EdgeType, FileType, Location, Node, NodeType, ParsedFile

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_TAG_RE = re.compile(r"<[^>]+>")
_CODE_CONTEXT_LINES = 15


def _source(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else (src or "")


def _as_text(value) -> str:
    return "".join(value) if isinstance(value, list) else str(value or "")


def _output_text(outputs) -> tuple[str, dict]:
    texts: list[str] = []
    info = {"n_images": 0, "output_types": []}
    for out in outputs:
        otype = out.get("output_type", "")
        info["output_types"].append(otype)
        if otype == "stream":
            texts.append(_as_text(out.get("text")))
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})
            if any(k.startswith("image/") for k in data):
                info["n_images"] += 1
            if "text/plain" in data:
                texts.append(_as_text(data["text/plain"]))
            elif "text/markdown" in data:
                texts.append(_as_text(data["text/markdown"]))
            elif "text/html" in data:
                texts.append(_TAG_RE.sub(" ", _as_text(data["text/html"])))
        elif otype == "error":
            texts.append(f"{out.get('ename', 'Error')}: {out.get('evalue', '')}")
    text = strip_ansi("\n".join(t.rstrip("\n") for t in texts if t and t.strip()))
    info["output_types"] = sorted(set(info["output_types"]))
    return text, info


def _first_heading(markdown: str) -> str | None:
    m = _HEADING_RE.search(markdown)
    return m.group(1).strip() if m else None


def parse_notebook(cf: CorpusFile, cfg: ChunkingConfig) -> ParsedFile:
    raw, _ = read_text(cf.abs_path)
    nb = nbformat.reads(raw, as_version=4)
    rel = cf.rel_path
    result = ParsedFile(file_path=rel, file_type=FileType.IPYNB)

    meta = nb.get("metadata", {})
    kernel = meta.get("kernelspec", {}).get("name")
    language = meta.get("language_info", {}).get("name") or meta.get("kernelspec", {}).get("language")

    file_node = Node(
        id=rel,
        file_path=rel,
        file_type=FileType.IPYNB,
        node_type=NodeType.FILE,
        title=rel,
        metadata={"n_cells": len(nb.cells), "kernel": kernel, "language": language},
        embed=False,
    )
    result.nodes.append(file_node)

    section: str | None = None
    exec_counts: list[int] = []
    for idx, cell in enumerate(nb.cells, start=1):
        ctype = cell.get("cell_type")
        src = _source(cell)
        outputs = cell.get("outputs", []) if ctype == "code" else []

        if ctype in ("markdown", "raw"):
            heading = _first_heading(src) if ctype == "markdown" else None
            if heading:
                section = heading
            if not src.strip():
                continue
            node_type = NodeType.MARKDOWN_CELL
            title = f"cell {idx} (markdown)"
            extra = {"cell_type": ctype}
        elif ctype == "code":
            node_type = NodeType.CODE_CELL
            title = f"cell {idx} (code)"
            exec_count = cell.get("execution_count")
            if exec_count is not None:
                exec_counts.append(exec_count)
            extra = {"cell_type": ctype, "execution_count": exec_count}
        else:
            result.warnings.append(f"cell {idx}: unknown cell type {ctype!r}")
            continue
        if section:
            extra["section"] = section

        cell_id = f"{rel}#cell{idx}"
        lines = src.split("\n")
        if src.strip():
            windows = [(0, len(lines))]
            if len(src) > cfg.max_chunk_chars:
                windows = line_windows(len(lines), cfg.lines_per_chunk, cfg.overlap_lines)
            for part, (start, end) in enumerate(windows, start=1):
                node_id = cell_id if len(windows) == 1 else f"{cell_id}.p{part}"
                text = "\n".join(lines[start:end])[: cfg.max_chunk_chars]
                result.nodes.append(
                    Node(
                        id=node_id,
                        file_path=rel,
                        file_type=FileType.IPYNB,
                        node_type=node_type,
                        parent_id=file_node.id,
                        title=title,
                        text=text,
                        location=Location(
                            cell=idx,
                            line_start=start + 1 if len(windows) > 1 else None,
                            line_end=end if len(windows) > 1 else None,
                        ),
                        metadata=dict(extra),
                    )
                )
                result.edges.append(Edge(src=file_node.id, dst=node_id, type=EdgeType.CONTAINS))

        if not outputs:
            continue
        out_text, out_info = _output_text(outputs)
        if not out_text and not out_info["n_images"]:
            continue
        producer = cell_id if not src.strip() or len(src) <= cfg.max_chunk_chars else f"{cell_id}.p1"
        code_tail = "\n".join(lines[-_CODE_CONTEXT_LINES:]).strip()
        out_id = f"{cell_id}/out"
        result.nodes.append(
            Node(
                id=out_id,
                file_path=rel,
                file_type=FileType.IPYNB,
                node_type=NodeType.CELL_OUTPUT,
                parent_id=producer if src.strip() else file_node.id,
                title=f"cell {idx} (output)",
                text=truncate_keep_tail(out_text, cfg.output_max_chars) if out_text else f"[{out_info['n_images']} image(s)]",
                context=f"Code:\n{code_tail}\nOutput:" if code_tail else "",
                location=Location(cell=idx),
                metadata={**out_info, **({"section": section} if section else {})},
                embed=bool(out_text),
            )
        )
        if src.strip():
            result.edges.append(Edge(src=producer, dst=out_id, type=EdgeType.PRODUCES))
        result.edges.append(Edge(src=file_node.id, dst=out_id, type=EdgeType.CONTAINS))

    file_node.metadata["out_of_order"] = exec_counts != sorted(exec_counts)
    title = next((n.metadata.get("section") for n in result.nodes if n.metadata.get("section")), None)
    if title:
        file_node.text = title
    return result
