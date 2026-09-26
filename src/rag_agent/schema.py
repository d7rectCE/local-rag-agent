"""Unified data model: every parser turns a file into nodes and typed edges (ТЗ 4.1)."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FileType(StrEnum):
    PY = "py"
    IPYNB = "ipynb"
    PDF = "pdf"
    PNG = "png"
    DOCX = "docx"
    TXT = "txt"
    CATALOG = "catalog"  # a result of an SQL query to the catalog of experiments (Э6), not a file
    WEB = "web"  # passages of a web page (Э16): file_path is the URL


class NodeType(StrEnum):
    FILE = "file"
    CODE_CHUNK = "code_chunk"
    FUNCTION = "function"
    CLASS = "class"
    MARKDOWN_CELL = "markdown_cell"
    CODE_CELL = "code_cell"
    CELL_OUTPUT = "cell_output"
    SECTION = "section"  # prose of a PDF / DOCX / TXT section (a chunk of it)
    TABLE = "table"
    FIGURE = "figure"
    IMAGE = "image"
    REVISION = "revision"  # DOCX tracked change: deleted or inserted text
    COMMENT = "comment"  # DOCX reviewer comment
    LOG_CHUNK = "log_chunk"  # window of a training / console log
    SUMMARY = "summary"  # generated digest of a file (e.g. metrics of a training log)


class EdgeType(StrEnum):
    CONTAINS = "contains"
    CALLS = "calls"
    IMPORTS = "imports"
    PRODUCES = "produces"
    ILLUSTRATES = "illustrates"
    USES_VAR = "uses_var"


class Location(BaseModel):
    """Where a node lives inside its file. Cells, pages and paragraphs are 1-based."""

    line_start: int | None = None
    line_end: int | None = None
    cell: int | None = None
    page: int | None = None
    page_end: int | None = None
    section: str | None = None  # heading path, "Title > Section"
    paragraph_start: int | None = None  # DOCX body paragraphs
    paragraph_end: int | None = None

    def describe(self) -> str:
        parts = []
        if self.section:
            parts.append(f"раздел «{self.section.split(' > ')[-1]}»")
        if self.page is not None:
            if self.page_end is not None and self.page_end != self.page:
                parts.append(f"стр. {self.page}–{self.page_end}")
            else:
                parts.append(f"стр. {self.page}")
        if self.paragraph_start is not None:
            if self.paragraph_end is not None and self.paragraph_end != self.paragraph_start:
                parts.append(f"абз. {self.paragraph_start}–{self.paragraph_end}")
            else:
                parts.append(f"абз. {self.paragraph_start}")
        if self.cell is not None:
            parts.append(f"ячейка {self.cell}")
        if self.line_start is not None:
            if self.line_end is not None and self.line_end != self.line_start:
                parts.append(f"строки {self.line_start}–{self.line_end}")
            else:
                parts.append(f"строка {self.line_start}")
        return ", ".join(parts)


class Node(BaseModel):
    id: str
    file_path: str  # POSIX path relative to the corpus root
    file_type: FileType
    node_type: NodeType
    parent_id: str | None = None
    title: str = ""
    text: str = ""
    # Extra context that helps retrieval but is not the node's own content,
    # e.g. the code that produced a notebook output.
    context: str = ""
    location: Location = Field(default_factory=Location)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embed: bool = True

    @property
    def content_hash(self) -> str:
        h = hashlib.sha256()
        for part in (self.node_type, self.title, self.context, self.text):
            h.update(part.encode("utf-8", "surrogatepass"))
            h.update(b"\x00")
        return h.hexdigest()

    def citation_label(self) -> str:
        loc = self.location.describe()
        return f"{self.file_path} ({loc})" if loc else self.file_path

    def embedding_text(self, with_header: bool = True) -> str:
        """Text for the embedder and the reranker. The header (path, title, section)
        and the context (enclosing class, producing code) are the "context header"
        of ТЗ S2; ``with_header=False`` embeds the bare fragment (H1 ablation)."""
        if not with_header:
            return self.text
        header = f"File: {self.file_path}"
        if self.title:
            header += f" | {self.title}"
        section = self.metadata.get("section")
        if section:
            header += f" | {section}"
        return "\n".join(p for p in (header, self.context, self.text) if p)


class Edge(BaseModel):
    src: str
    dst: str
    type: EdgeType
    label: str | None = None  # e.g. variable names for uses_var


class Symbol(BaseModel):
    """A definition found by static analysis (function, method, class)."""

    name: str
    qualname: str
    kind: str  # function | method | class
    signature: str = ""
    doc: str = ""
    line_start: int
    line_end: int
    cell: int | None = None  # notebook cell (1-based); lines are then relative to the cell


class CallSite(BaseModel):
    name: str  # called name: last attribute of the callee expression
    full_name: str  # callee expression as written, e.g. "np.random.default_rng"
    caller: str | None = None  # qualname of the enclosing definition
    line: int
    cell: int | None = None


class ParsedFile(BaseModel):
    file_path: str
    file_type: FileType
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    symbols: list[Symbol] = Field(default_factory=list)
    calls: list[CallSite] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
