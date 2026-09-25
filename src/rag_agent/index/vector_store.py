"""Vector index on Qdrant: named dense vector + optional sparse vector per node.

Embedded (local, file-based) mode by default; a Qdrant server is used when
``storage.qdrant_url`` is set.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from qdrant_client import QdrantClient, models

_POINT_NS = uuid.UUID("6f1c1f5e-2b0e-4c55-9d53-8f2a3c1b7e10")


def point_id(node_id: str) -> str:
    return str(uuid.uuid5(_POINT_NS, node_id))


def _sparse(vec: dict[int, float]) -> models.SparseVector:
    return models.SparseVector(indices=list(vec.keys()), values=list(vec.values()))


class VectorStore:
    def __init__(self, path: Path, dim: int, with_sparse: bool, url: str | None = None, collection: str = "nodes"):
        self.collection = collection
        self.dim = dim
        self.with_sparse = with_sparse
        self._lock = threading.Lock()
        if url:
            self.client = QdrantClient(url=url)
        else:
            path.mkdir(parents=True, exist_ok=True)
            try:
                self.client = QdrantClient(path=str(path))
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Qdrant storage {path} is locked by another process "
                    "(is the API server already running?)"
                ) from exc
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            size = vectors["dense"].size if isinstance(vectors, dict) else vectors.size
            if size != self.dim:
                raise ValueError(f"index dimension {size} != embedder dimension {self.dim}; reindex required")
            return
        self.client.create_collection(
            self.collection,
            vectors_config={"dense": models.VectorParams(size=self.dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()} if self.with_sparse else None,
        )

    def upsert(
        self,
        node_ids: list[str],
        dense: np.ndarray,
        sparse: list[dict[int, float]] | None,
        payloads: list[dict],
        batch: int = 256,
    ) -> None:
        points = []
        for i, node_id in enumerate(node_ids):
            vector: dict = {"dense": dense[i].tolist()}
            if self.with_sparse and sparse is not None and sparse[i]:
                vector["sparse"] = _sparse(sparse[i])
            points.append(models.PointStruct(id=point_id(node_id), vector=vector, payload={"node_id": node_id, **payloads[i]}))
        with self._lock:
            for k in range(0, len(points), batch):
                self.client.upsert(self.collection, points=points[k : k + batch], wait=True)

    def delete_files(self, file_paths: list[str]) -> None:
        if not file_paths:
            return
        selector = models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="file_path", match=models.MatchAny(any=file_paths))])
        )
        with self._lock:
            self.client.delete(self.collection, points_selector=selector, wait=True)

    def search(
        self,
        dense: np.ndarray | None,
        sparse: dict[int, float] | None,
        mode: Literal["dense", "sparse", "hybrid"],
        limit: int,
        file_types: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        if mode != "dense" and not self.with_sparse:
            raise ValueError(f"retrieval mode {mode!r} needs sparse vectors (use BGE-M3)")
        flt = None
        if file_types:
            flt = models.Filter(must=[models.FieldCondition(key="file_type", match=models.MatchAny(any=file_types))])
        kwargs: dict = {"limit": limit, "query_filter": flt, "with_payload": ["node_id"]}
        if mode == "dense":
            kwargs.update(query=dense.tolist(), using="dense")
        elif mode == "sparse":
            kwargs.update(query=_sparse(sparse or {}), using="sparse")
        else:
            pool = max(limit * 5, 50)
            kwargs.update(
                prefetch=[
                    models.Prefetch(query=dense.tolist(), using="dense", limit=pool, filter=flt),
                    models.Prefetch(query=_sparse(sparse or {}), using="sparse", limit=pool, filter=flt),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
            )
        with self._lock:
            res = self.client.query_points(self.collection, **kwargs)
        return [(p.payload["node_id"], float(p.score)) for p in res.points]

    def count(self) -> int:
        with self._lock:
            return self.client.count(self.collection, exact=True).count

    def close(self) -> None:
        with self._lock:
            self.client.close()
