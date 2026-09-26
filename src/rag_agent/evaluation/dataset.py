"""Evaluation-set format (ТЗ 5.1, S9) and chunking-independent source matching."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_agent.schema import Node

QuestionClass = Literal["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "G"]

CLASS_NAMES = {
    "Q1": "фактологический",
    "Q2": "навигационный по коду",
    "Q3": "агрегатный",
    "Q4": "визуальный",
    "Q5": "многошаговый",
    "Q6": "неотвечаемый",
    "Q7": "по загруженному файлу",
    "Q8": "требующий рассуждения",
    "G": "общий (без файлов)",
}


class SourceRef(BaseModel):
    """A reference span: a notebook cell, a line range, a page and/or a section of
    a document, or a whole file. All given constraints must hold."""

    file: str
    cell: int | None = None
    lines: tuple[int, int] | None = None
    page: int | None = None
    section: str | None = None  # substring of the innermost heading of the node's section
    quote: str | None = None  # regex found in the fragment's text or context: independent of chunking

    @property
    def file_type(self) -> str:
        return Path(self.file).suffix.lstrip(".").lower()

    def matches(self, node: Node) -> bool:
        if node.file_path != self.file:
            return False
        loc = node.location
        if self.cell is not None and loc.cell != self.cell:
            return False
        if self.page is not None:
            if loc.page is None or not (loc.page <= self.page <= (loc.page_end or loc.page)):
                return False
        if self.section is not None:
            innermost = (loc.section or "").split(" > ")[-1]
            if self.section.lower() not in innermost.lower():
                return False
        if self.quote is not None and not re.search(self.quote, f"{node.text}\n{node.context}", re.IGNORECASE):
            return False
        if self.lines is not None:
            if loc.line_start is None or loc.cell is not None:
                return False
            end = loc.line_end if loc.line_end is not None else loc.line_start
            return loc.line_start <= self.lines[1] and end >= self.lines[0]
        return True

    def label(self) -> str:
        parts = [self.file]
        if self.cell is not None:
            parts.append(f"cell{self.cell}")
        if self.page is not None:
            parts.append(f"p{self.page}")
        if self.section is not None:
            parts.append(f"«{self.section}»")
        if self.lines is not None:
            parts.append(f"L{self.lines[0]}-{self.lines[1]}")
        if self.quote is not None:
            parts.append(f"/{self.quote}/")
        return "#".join(parts)


class EvalItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    cls: QuestionClass = Field(alias="class")
    question: str
    history: list[dict] = Field(default_factory=list)
    standalone: str | None = None
    answer: str | None = None
    must_include: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    # Э6: reference SQL over the catalog of experiments; its result is compared with the result
    # of the query the system ran (execution accuracy, as in BIRD)
    sql: str | None = None
    # Э13: the question is about this file (relative to the eval set), uploaded into the conversation
    upload: str | None = None
    notes: str | None = None

    @property
    def expected_route(self) -> str:
        return "general" if self.cls == "G" else "upload" if self.upload else "corpus"

    @property
    def expects_refusal(self) -> bool:
        return self.cls == "Q6"

    @property
    def retrieval_query(self) -> str:
        return self.standalone or self.question

    @property
    def file_types(self) -> list[str]:
        return sorted({s.file_type for s in self.sources})


class EvalSet(BaseModel):
    version: int = 1
    name: str
    corpus: str | None = None
    items: list[EvalItem]
    path: Path | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> "EvalSet":
        dup = [k for k, v in Counter(i.id for i in self.items).items() if v > 1]
        if dup:
            raise ValueError(f"duplicate item ids: {dup}")
        return self

    def corpus_root(self) -> Path | None:
        if not self.corpus:
            return None
        root = Path(self.corpus)
        if not root.is_absolute() and self.path is not None:
            root = self.path.parent / root
        return root.resolve()


def load_evalset(path: str | Path) -> EvalSet:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    es = EvalSet.model_validate(data)
    es.path = path.resolve()
    return es


def validate_evalset(es: EvalSet, catalog) -> list[str]:
    """Consistency checks against an indexed corpus: referenced files, cells and
    line ranges must exist; classes must carry the right kind of annotation."""
    problems: list[str] = []
    cache: dict[str, list[Node]] = {}
    for item in es.items:
        if item.cls in ("Q6", "G", "Q7") and item.sources:
            problems.append(f"{item.id}: class {item.cls} must not have sources")
        if item.cls not in ("Q6", "G", "Q7") and not item.sources:
            problems.append(f"{item.id}: class {item.cls} needs at least one source")
        if item.cls == "Q7" and not (item.upload and es.path and (es.path.parent / item.upload).exists()):
            problems.append(f"{item.id}: class Q7 needs an existing upload file")
        if item.cls != "Q6" and not item.answer:
            problems.append(f"{item.id}: reference answer is missing")
        for ref in item.sources:
            nodes = cache.setdefault(ref.file, catalog.file_nodes(ref.file))
            if not nodes:
                problems.append(f"{item.id}: {ref.file} is not in the index")
                continue
            if not any(ref.matches(n) for n in nodes if n.node_type != "file"):
                problems.append(f"{item.id}: {ref.label()} matches no indexed fragment")
    return problems
