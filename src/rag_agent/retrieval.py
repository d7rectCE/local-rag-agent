"""Retrieval over a corpus index: dense, sparse or hybrid (RRF) search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rag_agent.index.embedder import Embedder
from rag_agent.index.indexer import CorpusIndex
from rag_agent.schema import Node

Mode = Literal["dense", "sparse", "hybrid"]


@dataclass
class Hit:
    node: Node
    score: float
    rank: int  # 1-based


def search(
    index: CorpusIndex,
    embedder: Embedder,
    query: str,
    top_k: int,
    mode: Mode = "dense",
    file_types: list[str] | None = None,
) -> list[Hit]:
    enc = embedder.encode([query])
    sparse = enc.sparse[0] if enc.sparse is not None else None
    found = index.store.search(enc.dense[0], sparse, mode, top_k, file_types)
    nodes = index.catalog.get_nodes([node_id for node_id, _ in found])
    hits = []
    for node_id, score in found:
        node = nodes.get(node_id)
        if node is not None:  # vector without catalog row: stale point, skip
            hits.append(Hit(node=node, score=score, rank=len(hits) + 1))
    return hits
