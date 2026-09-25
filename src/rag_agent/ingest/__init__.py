"""Ingestion layer: file walking and per-type parsers."""

from __future__ import annotations

from typing import Callable

from rag_agent.config import ChunkingConfig
from rag_agent.ingest.notebook_parser import parse_notebook
from rag_agent.ingest.python_parser import parse_python
from rag_agent.ingest.walker import CorpusFile, SkippedFile, iter_corpus
from rag_agent.schema import ParsedFile

Parser = Callable[[CorpusFile, ChunkingConfig], ParsedFile]

PARSERS: dict[str, Parser] = {
    ".py": parse_python,
    ".ipynb": parse_notebook,
}


def parse_file(cf: CorpusFile, cfg: ChunkingConfig) -> ParsedFile:
    try:
        parser = PARSERS[cf.ext]
    except KeyError:
        raise ValueError(f"no parser for {cf.ext}") from None
    return parser(cf, cfg)


__all__ = ["PARSERS", "CorpusFile", "SkippedFile", "iter_corpus", "parse_file"]
