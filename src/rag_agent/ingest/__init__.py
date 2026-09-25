"""Ingestion layer: file walking and per-type parsers."""

from __future__ import annotations

from typing import Callable

from rag_agent.config import ChunkingConfig, DocumentsConfig
from rag_agent.ingest.docx_parser import parse_doc, parse_docx
from rag_agent.ingest.notebook_parser import parse_notebook
from rag_agent.ingest.python_parser import parse_python
from rag_agent.ingest.txt_parser import parse_txt
from rag_agent.ingest.walker import CorpusFile, SkippedFile, iter_corpus
from rag_agent.schema import ParsedFile

Parser = Callable[[CorpusFile, ChunkingConfig], ParsedFile]

PARSERS: dict[str, Parser] = {
    ".py": parse_python,
    ".ipynb": parse_notebook,
    ".docx": parse_docx,
    ".doc": parse_doc,
    ".txt": parse_txt,
    ".md": parse_txt,
    ".log": parse_txt,
}


def parse_file(cf: CorpusFile, cfg: ChunkingConfig, documents: DocumentsConfig | None = None) -> ParsedFile:
    if cf.ext == ".pdf":  # Docling is imported only when a PDF is actually parsed
        from rag_agent.ingest.pdf_parser import parse_pdf

        return parse_pdf(cf, cfg, documents)
    try:
        parser = PARSERS[cf.ext]
    except KeyError:
        raise ValueError(f"no parser for {cf.ext}") from None
    return parser(cf, cfg)


SUPPORTED_EXTENSIONS = sorted([*PARSERS, ".pdf"])

__all__ = ["PARSERS", "SUPPORTED_EXTENSIONS", "CorpusFile", "SkippedFile", "iter_corpus", "parse_file"]
