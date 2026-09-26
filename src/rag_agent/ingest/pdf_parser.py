"""Parser for .pdf via Docling (ТЗ S1, P1).

Docling detects the page layout (a model trained on DocLayNet), restores the
reading order of multi-column pages and the structure of tables (TableFormer).
On top of that:

* sections come from section headers; prose is chunked by section at paragraph
  boundaries (structural chunking, ТЗ [11]) with page ranges for citations;
* tables become Markdown ``table`` nodes with their captions;
* pictures become ``figure`` nodes (caption, page, bounding box) linked to the
  section they illustrate — describing the image itself is Э5;
* OCR runs only when some page has no text layer (``documents.ocr: auto``): the
  text layer is checked with pdfium before conversion, which saves the OCR cost
  for born-digital PDFs.

Docling and its models are loaded lazily and only from ``documents.models_dir``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from rag_agent.config import DEFAULT_LAYOUT_MODEL, ChunkingConfig, DocumentsConfig
from rag_agent.ingest.walker import CorpusFile
from rag_agent.schema import Edge, EdgeType, FileType, Location, Node, NodeType, ParsedFile

log = logging.getLogger(__name__)
_CONVERTERS: dict[tuple, object] = {}
_LOCK = threading.Lock()
_SKIP_LABELS = {"page_header", "page_footer", "caption", "document_index"}


def pages_without_text(path: Path, min_chars: int) -> tuple[int, list[int]]:
    """(number of pages, 1-based pages whose text layer is (almost) empty)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        empty = []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            if len(textpage.get_text_range().strip()) < min_chars:
                empty.append(i + 1)
            textpage.close()
            page.close()
        return len(pdf), empty
    finally:
        pdf.close()


def models_dir(cfg: DocumentsConfig) -> Path:
    return Path(cfg.models_dir).expanduser()


def pdf_backend(cfg: DocumentsConfig) -> str:
    """docling-parse loads its glyph and font resources with narrow-char file APIs, which
    fail when the package lives under a non-ASCII path (e.g. a Windows profile with a Cyrillic user name)."""
    if cfg.backend != "auto":
        return cfg.backend
    import importlib.util

    spec = importlib.util.find_spec("docling_parse")
    location = str(Path(spec.origin).parent) if spec and spec.origin else ""
    return "docling_parse" if location.isascii() else "pypdfium"


def _converter(cfg: DocumentsConfig, ocr: bool):
    key = (ocr, cfg.models_dir, cfg.device, tuple(cfg.ocr_langs), cfg.table_mode, pdf_backend(cfg), cfg.layout_model)
    with _LOCK:
        if key in _CONVERTERS:
            return _CONVERTERS[key]
        from rag_agent.index.embedder import enforce_hf_offline

        enforce_hf_offline()  # Docling models come from models_dir only
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption

        artifacts = models_dir(cfg)
        if not artifacts.exists():
            raise RuntimeError(
                f"Docling models not found in {artifacts}; download them first: docling-tools models download "
                "layout tableformer easyocr --easyocr-lang ru --easyocr-lang en"
            )
        opts = PdfPipelineOptions(
            artifacts_path=str(artifacts),
            do_ocr=ocr,
            do_table_structure=True,
            enable_remote_services=False,
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice(cfg.device)),
        )
        opts.table_structure_options.mode = TableFormerMode(cfg.table_mode)
        if cfg.layout_model != DEFAULT_LAYOUT_MODEL:
            from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
            from docling.datamodel.stage_model_specs import ObjectDetectionModelSpec

            folder = artifacts / cfg.layout_model.replace("/", "--")
            if not folder.exists():
                raise RuntimeError(f"layout model {cfg.layout_model} not found in {folder}; install it with "
                                   "scripts/train_layout_detector.py export")
            opts.layout_options = LayoutObjectDetectionOptions(
                model_spec=ObjectDetectionModelSpec(name=cfg.layout_model, repo_id=cfg.layout_model))
        if ocr:
            opts.ocr_options = EasyOcrOptions(lang=list(cfg.ocr_langs), download_enabled=False,
                                              use_gpu=cfg.device != "cpu")
        fmt = PdfFormatOption(pipeline_options=opts)
        if pdf_backend(cfg) == "pypdfium":
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

            fmt = PdfFormatOption(pipeline_options=opts, backend=PyPdfiumDocumentBackend)
        converter = DocumentConverter(format_options={InputFormat.PDF: fmt})
        _CONVERTERS[key] = converter
        log.info("docling converter ready (ocr=%s, backend=%s, models=%s)", ocr, pdf_backend(cfg), artifacts)
        return converter


def _page(item) -> int | None:
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


def _bbox(item) -> list[float] | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    b = prov[0].bbox
    return [round(b.l, 1), round(b.t, 1), round(b.r, 1), round(b.b, 1)]


def parse_pdf(cf: CorpusFile, cfg: ChunkingConfig, documents: DocumentsConfig | None = None) -> ParsedFile:
    documents = documents or DocumentsConfig()
    rel = cf.rel_path
    result = ParsedFile(file_path=rel, file_type=FileType.PDF)
    n_pages, scanned = pages_without_text(cf.abs_path, documents.min_text_chars)
    ocr = documents.ocr == "always" or (documents.ocr == "auto" and bool(scanned))
    doc = _converter(documents, ocr).convert(str(cf.abs_path)).document

    file_node = Node(
        id=rel, file_path=rel, file_type=FileType.PDF, node_type=NodeType.FILE, title=rel,
        location=Location(page=1, page_end=n_pages),
        metadata={"n_pages": n_pages, "ocr": ocr, "pages_without_text": scanned}, embed=False,
    )
    result.nodes.append(file_node)
    if scanned and not ocr:
        result.warnings.append(f"pages without a text layer, OCR disabled: {scanned}")

    stack: dict[int, str] = {}
    section_path: str | None = None
    buffer: list[tuple[str, int | None]] = []
    counters = {"chunk": 0, "table": 0, "figure": 0}
    pending_figures: list[Node] = []
    page_headers: set[str] = set()

    def add(node: Node) -> None:
        result.nodes.append(node)
        result.edges.append(Edge(src=file_node.id, dst=node.id, type=EdgeType.CONTAINS))

    def flush() -> None:
        if not buffer:
            return
        pages = [p for _, p in buffer if p is not None]
        counters["chunk"] += 1
        node = Node(
            id=f"{rel}#s{counters['chunk']}", file_path=rel, file_type=FileType.PDF, node_type=NodeType.SECTION,
            parent_id=file_node.id, title=(section_path or "").split(" > ")[-1] or "text",
            text="\n".join(t for t, _ in buffer),
            location=Location(section=section_path, page=min(pages) if pages else None,
                              page_end=max(pages) if pages else None),
            metadata={"section": section_path} if section_path else {},
        )
        add(node)
        for fig in pending_figures:
            result.edges.append(Edge(src=fig.id, dst=node.id, type=EdgeType.ILLUSTRATES))
        pending_figures.clear()
        buffer.clear()

    for item, _level in doc.iterate_items():
        label = str(getattr(item, "label", "")).split(".")[-1].lower()
        page = _page(item)
        if label in ("page_header", "page_footer"):
            if getattr(item, "text", "").strip():
                page_headers.add(item.text.strip())
            continue
        if label in _SKIP_LABELS:
            continue
        if label in ("title", "section_header"):
            text = item.text.strip()
            if not text:
                continue
            flush()
            level = 0 if label == "title" else int(getattr(item, "level", 1) or 1)
            stack = {lvl: t for lvl, t in stack.items() if lvl < level}
            stack[level] = text
            section_path = " > ".join(stack[lvl] for lvl in sorted(stack))
            buffer.append((text, page))
            continue
        if label == "table":
            counters["table"] += 1
            caption = item.caption_text(doc) if hasattr(item, "caption_text") else ""
            add(Node(
                id=f"{rel}#table{counters['table']}", file_path=rel, file_type=FileType.PDF, node_type=NodeType.TABLE,
                parent_id=file_node.id, title=caption or f"table {counters['table']}",
                text=item.export_to_markdown(doc=doc)[: cfg.max_chunk_chars], context=caption,
                location=Location(section=section_path, page=page),
                metadata={"bbox": _bbox(item), **({"section": section_path} if section_path else {})},
            ))
            continue
        if label in ("picture", "chart"):
            counters["figure"] += 1
            caption = item.caption_text(doc) if hasattr(item, "caption_text") else ""
            fig = Node(
                id=f"{rel}#fig{counters['figure']}", file_path=rel, file_type=FileType.PDF, node_type=NodeType.FIGURE,
                parent_id=file_node.id, title=caption or f"figure {counters['figure']}", text=caption,
                location=Location(section=section_path, page=page),
                metadata={"bbox": _bbox(item), **({"section": section_path} if section_path else {})},
                embed=bool(caption),
            )
            add(fig)
            pending_figures.append(fig)
            continue
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        if label == "list_item":
            text = f"- {text}"
        if buffer and sum(len(t) for t, _ in buffer) + len(text) > cfg.text_max_chars:
            flush()
        buffer.append((text, page))
    flush()
    if pending_figures and result.nodes:
        sections = [n for n in result.nodes if n.node_type == NodeType.SECTION]
        for fig in pending_figures:
            if sections:
                result.edges.append(Edge(src=fig.id, dst=sections[-1].id, type=EdgeType.ILLUSTRATES))

    title = next((n.title for n in result.nodes if n.node_type == NodeType.SECTION and n.title != "text"), None)
    file_node.text = stack.get(0) or title or ""
    file_node.metadata.update({"n_tables": counters["table"], "n_figures": counters["figure"],
                               "page_headers": sorted(page_headers)[:5]})
    return result
