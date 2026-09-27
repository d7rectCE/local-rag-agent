"""The image stage of indexing (ТЗ S4): after the files are parsed and embedded, every image node gets a
type (the classifier), OCR text (screenshots, scans) and a VLM description by type; the text replaces
the node's placeholder and is embedded. Results are cached by the image's content hash and the models,
so a re-index describes only new images. The optional visual index (ColQwen2) is built last, after the
VLM has been unloaded: both do not fit next to each other in 24 GB."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

from rag_agent.images.classifier import ImageClassifier
from rag_agent.images.describe import describe, node_text, ocr_text
from rag_agent.images.sources import big_enough, image_bytes
from rag_agent.schema import Node

log = logging.getLogger(__name__)

PROMPT_VERSION = 1  # bump when the description prompts change: cached descriptions are redone


def image_source(node: Node) -> str | None:
    if node.metadata.get("image_source"):
        return node.metadata["image_source"]
    if node.metadata.get("image_rel"):
        return "docx"
    if node.file_type == "pdf" and node.location.page:
        return "pdf"
    return None


def process_images(index, settings, vlm, embedder, progress=None, cancel: threading.Event | None = None,
                   classifier: ImageClassifier | None = None) -> dict:
    cfg = settings.images
    classifier = classifier or ImageClassifier(settings.data_dir / "models" / "image_classifier")
    signature = f"{vlm.name}|{classifier.version if classifier._load() else 'vlm'}|p{PROMPT_VERSION}"
    cache_dir = settings.data_dir / "image_cache"
    stats = {"images": 0, "described": 0, "cached": 0, "skipped": 0, "failed": 0, "types": {}, "signature": signature}
    todo = []
    for node in index.catalog.image_nodes():
        if image_source(node) is None:
            continue
        stats["images"] += 1
        if node.metadata.get("described") != signature:
            todo.append(node)
    t0 = time.perf_counter()
    batch: list[Node] = []
    for k, node in enumerate(todo[: cfg.max_images], start=1):
        if cancel is not None and cancel.is_set():
            break
        if progress is not None:
            progress.current = f"изображения {k}/{len(todo)}: {node.file_path}"
        data = image_bytes(index.root, node)
        if not data or not big_enough(data):
            stats["skipped"] += 1
            node.metadata["described"] = signature
            index.catalog.update_node(node)
            continue
        sha = hashlib.sha256(data).hexdigest()
        cache_file = cache_dir / sha[:2] / f"{sha}-{hashlib.sha1(signature.encode()).hexdigest()[:10]}.json"
        if cache_file.exists():
            rec = json.loads(cache_file.read_text(encoding="utf-8"))
            stats["cached"] += 1
        else:
            pred = classifier.predict(data) if cfg.classify else None
            kind, prob = pred if pred else (None, None)
            ocr = ocr_text(data, settings.documents.ocr_langs, settings.documents.models_dir) \
                if cfg.ocr and kind in ("screenshot", "scan") else ""
            context = "\n".join(x for x in (node.title if not node.title.startswith(("figure", "cell ")) else "",
                                            node.context) if x)
            description = describe(data, kind, vlm, where=node.citation_label(), context=context, ocr=ocr)
            if not description:
                stats["failed"] += 1
                continue
            rec = {"type": kind, "p": prob, "description": description, "ocr": ocr, "sha256": sha}
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            stats["described"] += 1
        caption = node.text if node.node_type == "figure" else ""  # a PDF / DOCX caption stays in front
        node.text = node_text(rec["type"], rec["p"], rec["description"], rec["ocr"], caption)
        node.metadata.update(image_type=rec["type"], image_p=rec["p"], image_sha256=sha, described=signature)
        node.embed = True
        stats["types"][rec["type"] or "?"] = stats["types"].get(rec["type"] or "?", 0) + 1
        index.catalog.update_node(node)
        batch.append(node)
        if len(batch) >= 16:
            _embed(index, settings, embedder, batch)
            batch = []
    _embed(index, settings, embedder, batch)
    stats["elapsed_s"] = round(time.perf_counter() - t0, 1)
    if cfg.visual_index and not (cancel is not None and cancel.is_set()):
        from rag_agent.images.visual import VisualIndex

        vlm.unload()  # the VLM and ColQwen2 do not fit in memory together
        stats["visual"] = VisualIndex(index, settings).build(progress)
    return stats


def _embed(index, settings, embedder, nodes: list[Node]) -> None:
    nodes = [n for n in nodes if n.text.strip()]
    if not nodes:
        return
    enc = embedder.encode([n.embedding_text(settings.chunking.context_header) for n in nodes])
    index.store.upsert([n.id for n in nodes], enc.dense, enc.sparse,
                       [{"file_path": n.file_path, "file_type": n.file_type.value, "node_type": n.node_type.value}
                        for n in nodes])
