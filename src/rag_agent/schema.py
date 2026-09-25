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


class NodeType(StrEnum):
    FILE = "file"
    CODE_CHUNK = "code_chunk"
    FUNCTION = "function"
    CLASS = "class"
    MARKDOWN_CELL = "markdown_cell"
    CODE_CELL = "code_cell"
    CELL_OUTPUT = "cell_output"
    PDF_SECTION = "pdf_section"
    TABLE = "table"
    FIGURE = "figure"
    IMAGE = "image"


class EdgeType(StrEnum):
    CONTAINS = "contains"
    CALLS = "calls"
    IMPORTS = "imports"
    PRODUCES = "produces"
    ILLUSTRATES = "illustrates"
    USES_VAR = "uses_var"


class Location(BaseModel):
    """Where a node lives inside its file. Cells and pages are 1-based."""

    line_start: int | None = None
    line_end: int | None = None
    cell: int | None = None
    page: int | None = None

    def describe(self) -> str:
        parts = []
        if self.page is not None:
            parts.append(f"стр. {self.page}")
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
        parts = []
        if with_header:
            header = f"File: {self.file_path}"
            if self.title:
                header += f" | {self.title}"
            section = self.metadata.get("section")
            if section:
                header += f" | {section}"
            parts.append(header)
        if self.context:
            parts.append(self.context)
        parts.append(self.text)
        return "\n".join(p for p in parts if p)


class Edge(BaseModel):
    src: str
    dst: str
    type: EdgeType


class ParsedFile(BaseModel):
    file_path: str
    file_type: FileType
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
