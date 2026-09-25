"""Retrieval pipeline (ТЗ S5):

1. first stage — dense, sparse or hybrid (dense + sparse fused with RRF inside Qdrant);
2. optional exact-name channel — definitions and call sites of identifiers that
   occur in the question and exist in the corpus, fused with the first stage by
   Reciprocal Rank Fusion (ТЗ [9]);
3. optional cross-encoder reranking of the candidate pool (ТЗ [8]) down to top_k.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rag_agent.index.embedder import Embedder
from rag_agent.index.indexer import CorpusIndex
from rag_agent.index.reranker import Reranker
from rag_agent.schema import Node

Mode = Literal["dense", "sparse", "hybrid"]

RRF_K = 60
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass
class Hit:
    node: Node
    score: float
    rank: int  # 1-based


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _code_like(ident: str) -> bool:
    """snake_case, CamelCase or containing digits — unlike ordinary words such as
    "train", "fit" or "metrics", which are also file or method names."""
    return "_" in ident or any(c.isdigit() for c in ident) or any(c.isupper() for c in ident[1:])


def corpus_mentions(index: CorpusIndex, text: str) -> list[str]:
    """Identifiers of the question that name something in the corpus: a code-like
    name of a function, class or file, or a capitalised class name. A question that
    mentions them is about the user's files, whatever its wording."""
    catalog = index.catalog
    stems = catalog.file_stems()
    found = []
    for ident in dict.fromkeys(_IDENT_RE.findall(text)):
        symbols = catalog.find_symbols(ident)
        if _code_like(ident):
            hit = ident.lower() in stems or bool(symbols)
        else:  # a plain word counts only as the exact, capitalised name of a class: "Trainer"
            hit = ident[0].isupper() and any(s["kind"] == "class" and s["name"] == ident for s in symbols)
        if hit:
            found.append(ident)
    return found


def symbol_candidates(index: CorpusIndex, query: str, limit: int) -> list[str]:
    """Node ids for identifiers of the question that are defined in the corpus:
    their definitions first, then their call sites."""
    catalog = index.catalog
    defs, calls = [], []
    for ident in dict.fromkeys(_IDENT_RE.findall(query)):
        symbols = catalog.find_symbols(ident)
        if not symbols:
            continue  # ordinary words never reach the channel
        for s in symbols:
            defs += [n.id for n in catalog.nodes_at(s["file_path"], s["line_start"], s["cell"], s["line_end"])]
        for c in catalog.find_calls(ident):
            calls += [n.id for n in catalog.nodes_at(c["file_path"], c["line"], c["cell"])]
    return list(dict.fromkeys(defs + calls))[:limit]


def search(
    index: CorpusIndex,
    embedder: Embedder,
    query: str,
    top_k: int,
    mode: Mode = "dense",
    *,
    reranker: Reranker | None = None,
    rerank_pool: int = 40,
    symbols: bool = False,
    with_header: bool = True,
    file_types: list[str] | None = None,
) -> list[Hit]:
    pool = max(top_k, rerank_pool) if reranker is not None else top_k
    enc = embedder.encode([query])
    sparse = enc.sparse[0] if enc.sparse is not None else None
    found = index.store.search(enc.dense[0], sparse, mode, pool, file_types)

    exact = symbol_candidates(index, query, pool) if symbols else []
    if reranker is None:
        ranked = rrf([[node_id for node_id, _ in found], exact])[:pool] if exact else found
    else:
        # the cross-encoder scores the union of both channels; exact matches are fused
        # back afterwards, otherwise the reranker would wash them out
        ids = list(dict.fromkeys([node_id for node_id, _ in found] + exact))
        nodes = index.catalog.get_nodes(ids)
        ids = [i for i in ids if i in nodes]
        scores = reranker.score(query, [nodes[i].embedding_text(with_header) for i in ids]) if ids else []
        ranked = sorted(zip(ids, (float(s) for s in scores)), key=lambda x: -x[1])
        if exact:
            ranked = rrf([[i for i, _ in ranked], exact])

    nodes = index.catalog.get_nodes([node_id for node_id, _ in ranked[:top_k]])
    hits = [(nodes[node_id], score) for node_id, score in ranked if node_id in nodes]  # skip stale points
    return [Hit(node=node, score=float(score), rank=i) for i, (node, score) in enumerate(hits[:top_k], start=1)]
