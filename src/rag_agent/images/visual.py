"""The visual index (ТЗ S4, H4): ColQwen2 [ColPali, 20] multi-vector embeddings of the images of the corpus
and of the pages of its PDFs, searched by late interaction (MaxSim), as a channel next to the text index.

Items are stored per index folder in ``visual/``: one float16 matrix per item (a few hundred 128-d vectors)
and a manifest that maps items to image nodes or to PDF pages. A page hit stands for the text nodes of that
page. The model (~4.5 GB in bf16) is loaded only to build the index or to encode a query; it does not fit
next to a 27B LLM on a 24 GB GPU, so the channel is off by default and used in the H4 comparison."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from rag_agent.images.sources import big_enough, image_bytes, pdf_page

log = logging.getLogger(__name__)

_MODELS: dict[str, tuple] = {}
_LOCK = threading.Lock()


def _load(settings):
    import torch
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

    name = settings.images.visual_model
    with _LOCK:
        if name not in _MODELS:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = ColQwen2ForRetrieval.from_pretrained(name, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                                         local_files_only=settings.embedding.local_files_only).to(dev).eval()
            processor = ColQwen2Processor.from_pretrained(name, local_files_only=settings.embedding.local_files_only)
            _MODELS[name] = (model, processor, dev)
    return _MODELS[name]


def unload_visual() -> None:
    import torch

    with _LOCK:
        _MODELS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class VisualIndex:
    def __init__(self, index, settings):
        self.index, self.settings = index, settings
        self.dir = index.dir / "visual"
        self.manifest_path = self.dir / "manifest.json"

    def manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"items": {}}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    @property
    def available(self) -> bool:
        return self.manifest_path.exists()

    def _items(self) -> list[tuple[str, dict]]:
        """(key, meta) of every image node and PDF page, with the bytes read lazily."""
        out = []
        for node in self.index.catalog.image_nodes():
            if node.metadata.get("image_sha256"):
                out.append((f"node:{node.id}", {"node_id": node.id, "file_path": node.file_path,
                                                "sha": node.metadata["image_sha256"]}))
        if self.settings.images.visual_pages:
            for r in self.index.catalog.query("SELECT path FROM files WHERE file_type = 'pdf' AND status = 'ok'"):
                pages = self.index.catalog.query("SELECT DISTINCT json_extract(location, '$.page') AS p FROM nodes "
                                                 "WHERE file_path = ? AND p IS NOT NULL", (r["path"],))
                for p in pages:
                    out.append((f"page:{r['path']}#{p['p']}", {"file_path": r["path"], "page": int(p["p"])}))
        return out

    def build(self, progress=None, batch: int = 4) -> dict:
        import torch
        from PIL import Image

        t0 = time.perf_counter()
        self.dir.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest()
        items = self._items()
        keep = {k for k, _ in items}
        manifest["items"] = {k: v for k, v in manifest["items"].items() if k in keep}
        todo = []
        for key, meta in items:
            old = manifest["items"].get(key)
            if old and old.get("sha") == meta.get("sha", old.get("sha")) and (self.dir / old["file"]).exists():
                continue
            todo.append((key, meta))
        if todo:
            model, processor, dev = _load(self.settings)
        done = 0
        for k in range(0, len(todo), batch):
            chunk, images = [], []
            for key, meta in todo[k:k + batch]:
                if "node_id" in meta:
                    node = self.index.catalog.get_node(meta["node_id"])
                    data = image_bytes(self.index.root, node) if node else None
                else:
                    data = pdf_page(self.index.root / meta["file_path"], meta["page"])
                if not data or not big_enough(data):
                    continue
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.thumbnail((1024, 1024))
                chunk.append((key, meta))
                images.append(img)
            if not images:
                continue
            if progress is not None:
                progress.current = f"визуальный индекс {k + len(images)}/{len(todo)}"
            with torch.no_grad():
                inputs = processor(images=images, return_tensors="pt").to(dev)
                emb = model(**inputs).embeddings
            mask = inputs.get("attention_mask")
            for i, (key, meta) in enumerate(chunk):
                vecs = emb[i][mask[i].bool()] if mask is not None else emb[i]
                fname = hashlib.sha1(key.encode()).hexdigest()[:16] + ".npy"
                np.save(self.dir / fname, vecs.float().cpu().numpy().astype(np.float16))
                manifest["items"][key] = {**meta, "file": fname}
                done += 1
        manifest["model"] = self.settings.images.visual_model
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"items": len(manifest["items"]), "embedded": done, "elapsed_s": round(time.perf_counter() - t0, 1)}

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        """Items by MaxSim of the query tokens over the item's vectors, best first."""
        import torch

        manifest = self.manifest()
        if not manifest["items"]:
            return []
        model, processor, dev = _load(self.settings)
        with torch.no_grad():
            inputs = processor(text=[query], return_tensors="pt").to(dev)
            q = model(**inputs).embeddings[0].float()
        mask = inputs.get("attention_mask")
        if mask is not None:
            q = q[mask[0].bool()]
        scores = []
        for meta in manifest["items"].values():
            d = torch.from_numpy(np.load(self.dir / meta["file"]).astype(np.float32)).to(q.device)
            scores.append((meta, float((q @ d.T).max(dim=1).values.sum())))
        return sorted(scores, key=lambda x: -x[1])[:top_k]
