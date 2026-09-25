"""Parser for .py files.

Two chunkers, switched by ``chunking.python`` (hypothesis H1):

* ``lines`` — fixed windows of N lines with overlap (Э1 baseline);
* ``ast``   — structural chunking in the spirit of cAST (ТЗ [13], S2): top-level
  definitions are never cut, small neighbours are merged up to a size limit,
  a class that does not fit is split into its header and groups of methods (each
  carrying the class signature as context), an oversized function falls back to
  line windows. Decorators and comment lines right above a definition stay with it.

Symbols and call sites are extracted in both modes for the exact-name index.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from rag_agent.config import ChunkingConfig
from rag_agent.ingest.code_analysis import analyze, first_doc_line, signature, start_line
from rag_agent.ingest.text_utils import read_text
from rag_agent.ingest.walker import CorpusFile
from rag_agent.schema import Edge, EdgeType, FileType, Location, Node, NodeType, ParsedFile

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


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


@dataclass
class Piece:
    start: int  # 1-based, inclusive
    end: int
    title: str
    node_type: NodeType
    context: str = ""
    symbols: tuple[str, ...] = ()


@dataclass
class _Unit:
    start: int
    end: int
    node: ast.stmt


def _size(lines: list[str], start: int, end: int) -> int:
    """Non-whitespace characters, as in cAST: indentation does not count."""
    return sum(len("".join(line.split())) for line in lines[start - 1 : end])


def _units(body: list[ast.stmt], lines: list[str], prev_end: int) -> list[_Unit]:
    units = []
    for node in body:
        start = start_line(node)
        while start - 1 > prev_end and lines[start - 2].strip().startswith("#"):
            start -= 1  # attach the comment block right above the statement
        end = node.end_lineno or node.lineno
        units.append(_Unit(start, end, node))
        prev_end = end
    return units


def _unit_label(node: ast.stmt, owner: str | None) -> tuple[str, str | None]:
    if isinstance(node, _DEFS):
        name = f"{owner}.{node.name}" if owner else node.name
        return (f"class {name}" if isinstance(node, ast.ClassDef) else name), name
    return "module code", None


def _group_piece(group: list[_Unit], context: str, owner: str | None) -> Piece:
    labels, symbols = [], []
    for unit in group:
        label, symbol = _unit_label(unit.node, owner)
        if label not in labels:
            labels.append(label)
        if symbol:
            symbols.append(symbol)
    kinds = {type(u.node) for u in group}
    if kinds <= {ast.FunctionDef, ast.AsyncFunctionDef}:
        node_type = NodeType.FUNCTION
    elif kinds == {ast.ClassDef}:
        node_type = NodeType.CLASS
    else:
        node_type = NodeType.CODE_CHUNK
    title = ", ".join(labels)
    return Piece(group[0].start, group[-1].end, title[:150], node_type, context, tuple(symbols))


def _windows_piece(unit: _Unit, lines: list[str], cfg: ChunkingConfig, context: str, owner: str | None) -> list[Piece]:
    label, symbol = _unit_label(unit.node, owner)
    head = signature(unit.node) if isinstance(unit.node, _DEFS) else ""
    pieces = []
    windows = line_windows(unit.end - unit.start + 1, cfg.lines_per_chunk, cfg.overlap_lines)
    for k, (a, b) in enumerate(windows, start=1):
        ctx = context if k == 1 else "\n".join(p for p in (context, head) if p)
        pieces.append(
            Piece(unit.start + a, unit.start + b - 1, f"{label} (part {k}/{len(windows)})", NodeType.CODE_CHUNK, ctx,
                  (symbol,) if symbol else ())
        )
    return pieces


def _split_class(unit: _Unit, lines: list[str], cfg: ChunkingConfig, context: str, owner: str | None) -> list[Piece]:
    node: ast.ClassDef = unit.node
    name = f"{owner}.{node.name}" if owner else node.name
    first = next((i for i, n in enumerate(node.body) if isinstance(n, _DEFS)), len(node.body))
    members = _units(node.body[first:], lines, prev_end=unit.start)
    header_end = (members[0].start - 1) if members else unit.end
    pieces = [Piece(unit.start, header_end, f"class {name}", NodeType.CLASS, context, (name,))]
    doc = first_doc_line(node)
    class_ctx = signature(node) + (f": {doc}" if doc else "")
    class_ctx = "\n".join(p for p in (context, class_ctx) if p)
    pieces.extend(_pack(members, lines, cfg, class_ctx, name))
    return pieces


def _pack(units: list[_Unit], lines: list[str], cfg: ChunkingConfig, context: str = "", owner: str | None = None) -> list[Piece]:
    limit = cfg.ast_max_chars
    pieces: list[Piece] = []
    group: list[_Unit] = []
    group_size = 0
    for unit in units:
        size = _size(lines, unit.start, unit.end)
        if size > limit:
            if group:
                pieces.append(_group_piece(group, context, owner))
                group, group_size = [], 0
            if isinstance(unit.node, ast.ClassDef):
                pieces.extend(_split_class(unit, lines, cfg, context, owner))
            else:
                pieces.extend(_windows_piece(unit, lines, cfg, context, owner))
            continue
        if group and group_size + size > limit:
            pieces.append(_group_piece(group, context, owner))
            group, group_size = [], 0
        group.append(unit)
        group_size += size
    if group:
        pieces.append(_group_piece(group, context, owner))
    return pieces


def ast_pieces(tree: ast.Module, lines: list[str], cfg: ChunkingConfig) -> list[Piece]:
    return _pack(_units(tree.body, lines, prev_end=0), lines, cfg)


def window_pieces(lines: list[str], cfg: ChunkingConfig) -> list[Piece]:
    return [
        Piece(a + 1, b, f"lines {a + 1}-{b}", NodeType.CODE_CHUNK)
        for a, b in line_windows(len(lines), cfg.lines_per_chunk, cfg.overlap_lines)
    ]


def parse_python(cf: CorpusFile, cfg: ChunkingConfig) -> ParsedFile:
    source, encoding = read_text(cf.abs_path)
    rel = cf.rel_path
    result = ParsedFile(file_path=rel, file_type=FileType.PY)
    lines = source.split("\n")

    metadata: dict = {"n_lines": len(lines), "encoding": encoding}
    tree = None
    try:
        tree = ast.parse(source)
        analysis = analyze(source)
        result.symbols, result.calls = analysis.symbols, analysis.calls
        metadata["imports"] = sorted(analysis.imports)
        metadata["functions"] = [s.name for s in analysis.symbols if s.kind == "function" and "." not in s.qualname]
        metadata["classes"] = [s.name for s in analysis.symbols if s.kind == "class" and "." not in s.qualname]
    except SyntaxError as exc:
        result.warnings.append(f"SyntaxError at line {exc.lineno}: {exc.msg}")

    file_node = Node(
        id=rel,
        file_path=rel,
        file_type=FileType.PY,
        node_type=NodeType.FILE,
        title=rel,
        text=(ast.get_docstring(tree) or "") if tree is not None else "",
        location=Location(line_start=1, line_end=len(lines)),
        metadata=metadata,
        embed=False,
    )
    result.nodes.append(file_node)

    if cfg.python == "ast" and tree is not None:
        pieces = ast_pieces(tree, lines, cfg)
    else:
        if cfg.python == "ast":
            result.warnings.append("AST chunking unavailable, fell back to line windows")
        pieces = window_pieces(lines, cfg)

    for piece in pieces:
        chunk = "\n".join(lines[piece.start - 1 : piece.end])
        if not chunk.strip():
            continue
        if len(chunk) > cfg.max_chunk_chars:
            result.warnings.append(f"lines {piece.start}-{piece.end}: chunk truncated to {cfg.max_chunk_chars} chars")
            chunk = chunk[: cfg.max_chunk_chars]
        node = Node(
            id=f"{rel}#L{piece.start}-{piece.end}",
            file_path=rel,
            file_type=FileType.PY,
            node_type=piece.node_type,
            parent_id=file_node.id,
            title=piece.title,
            text=chunk,
            context=piece.context,
            location=Location(line_start=piece.start, line_end=piece.end),
            metadata={"symbols": list(piece.symbols)} if piece.symbols else {},
        )
        result.nodes.append(node)
        result.edges.append(Edge(src=file_node.id, dst=node.id, type=EdgeType.CONTAINS))
    return result
