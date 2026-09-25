"""Parser for .py files.

Э1 baseline: fixed-size line windows with overlap. AST chunking (cAST-style,
ТЗ S2) is added in Э3 behind ``chunking.python: ast`` so that H1 can compare both.
"""

from __future__ import annotations

import ast

from rag_agent.config import ChunkingConfig
from rag_agent.ingest.text_utils import read_text
from rag_agent.ingest.walker import CorpusFile
from rag_agent.schema import Edge, EdgeType, FileType, Location, Node, NodeType, ParsedFile


def _module_outline(source: str) -> tuple[dict, str | None]:
    """Cheap static facts for the catalog: docstring, imports, top-level symbols."""
    tree = ast.parse(source)
    imports: set[str] = set()
    functions: list[str] = []
    classes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    outline = {"imports": sorted(imports), "functions": functions, "classes": classes}
    return outline, ast.get_docstring(tree)


def line_windows(n_lines: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """0-based half-open windows [start, end) covering ``n_lines`` lines."""
    if n_lines <= 0:
        return []
    size = max(size, 1)
    step = max(size - max(overlap, 0), 1)
    windows = []
    start = 0
    while True:
        end = min(start + size, n_lines)
        windows.append((start, end))
        if end >= n_lines:
            break
        start += step
    return windows


def parse_python(cf: CorpusFile, cfg: ChunkingConfig) -> ParsedFile:
    source, encoding = read_text(cf.abs_path)
    rel = cf.rel_path
    result = ParsedFile(file_path=rel, file_type=FileType.PY)
    lines = source.split("\n")

    metadata: dict = {"n_lines": len(lines), "encoding": encoding}
    docstring = None
    try:
        outline, docstring = _module_outline(source)
        metadata.update(outline)
    except SyntaxError as exc:
        result.warnings.append(f"SyntaxError at line {exc.lineno}: {exc.msg}")

    file_node = Node(
        id=rel,
        file_path=rel,
        file_type=FileType.PY,
        node_type=NodeType.FILE,
        title=rel,
        text=docstring or "",
        location=Location(line_start=1, line_end=len(lines)),
        metadata=metadata,
        embed=False,
    )
    result.nodes.append(file_node)

    for start, end in line_windows(len(lines), cfg.lines_per_chunk, cfg.overlap_lines):
        chunk = "\n".join(lines[start:end])
        if not chunk.strip():
            continue
        if len(chunk) > cfg.max_chunk_chars:
            result.warnings.append(f"lines {start + 1}-{end}: chunk truncated to {cfg.max_chunk_chars} chars")
            chunk = chunk[: cfg.max_chunk_chars]
        node = Node(
            id=f"{rel}#L{start + 1}-{end}",
            file_path=rel,
            file_type=FileType.PY,
            node_type=NodeType.CODE_CHUNK,
            parent_id=file_node.id,
            title=f"lines {start + 1}-{end}",
            text=chunk,
            location=Location(line_start=start + 1, line_end=end),
        )
        result.nodes.append(node)
        result.edges.append(Edge(src=file_node.id, dst=node.id, type=EdgeType.CONTAINS))
    return result
