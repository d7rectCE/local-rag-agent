"""Parser for .docx (and .doc via LibreOffice): declarative reading of the XML (ТЗ ч.2, S13).

* heading styles (and outline levels) give the section hierarchy; prose is chunked
  by section at paragraph boundaries;
* text is read with tracked changes accepted: inserted runs count, deleted runs do
  not; each change becomes a separate ``revision`` node, reviewer comments become
  ``comment`` nodes — available on request, never mixed into the current text;
* tables become Markdown ``table`` nodes; embedded images become ``figure``
  nodes (with the caption, if any) linked to their section — the image pipeline is Э5;
* document properties, headers and footers go to the file node's metadata.

``chunking.docx: plain`` is the H10 baseline: python-docx paragraph text and table
cells concatenated and cut into fixed windows.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from rag_agent.config import ChunkingConfig
from rag_agent.ingest.walker import CorpusFile
from rag_agent.schema import Edge, EdgeType, FileType, Location, Node, NodeType, ParsedFile

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"
R_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
A_BLIP = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"

_HEADING_STYLE_RE = re.compile(r"^(heading|заголовок)\s*(\d)$", re.IGNORECASE)
_TITLE_STYLES = {"title", "название", "заголовок"}
_CAPTION_STYLES = {"caption", "название объекта"}
_CAPTION_TEXT_RE = re.compile(r"^\s*(рис\.?|рисунок|figure|fig\.|таблица|table)\s*\d*", re.IGNORECASE)
MAX_UNZIPPED_MB = 300  # zip-bomb guard: DOCX is a ZIP archive (ТЗ ч.2, P16)


@dataclass
class _Paragraph:
    index: int  # 1-based among non-empty body paragraphs
    text: str
    deleted: str
    inserted: str
    style: str
    level: int | None  # heading level, 0 for the title
    change_meta: dict
    images: list[str] = field(default_factory=list)  # relationship ids
    comment_ids: list[str] = field(default_factory=list)


def _walk_text(el, out: dict, in_del: bool = False, in_ins: bool = False) -> None:
    for child in el:
        tag = child.tag
        if tag == W + "del":
            out["meta"].setdefault("del", (child.get(W + "author"), child.get(W + "date")))
            _walk_text(child, out, True, in_ins)
        elif tag == W + "ins":
            out["meta"].setdefault("ins", (child.get(W + "author"), child.get(W + "date")))
            _walk_text(child, out, in_del, True)
        elif tag == W + "t":
            if not in_del:
                out["text"].append(child.text or "")
                if in_ins:
                    out["inserted"].append(child.text or "")
        elif tag == W + "delText":
            out["deleted"].append(child.text or "")
        elif tag == W + "tab" and not in_del:
            out["text"].append("\t")
        elif tag in (W + "br", W + "cr") and not in_del:
            out["text"].append("\n")
        elif tag == A_BLIP and child.get(R_EMBED):
            out["images"].append(child.get(R_EMBED))
        elif tag == W + "commentRangeStart":
            out["comments"].append(child.get(W + "id"))
        else:
            _walk_text(child, out, in_del, in_ins)


def _read_paragraph(p_el) -> dict:
    out = {"text": [], "deleted": [], "inserted": [], "images": [], "comments": [], "meta": {}}
    _walk_text(p_el, out)
    return {k: ("".join(v).strip() if isinstance(v, list) and k in ("text", "deleted", "inserted") else v)
            for k, v in out.items()}


def _style_names(document) -> dict[str, str]:
    return {s.style_id: (s.name or "") for s in document.styles if getattr(s, "style_id", None)}


def _heading_level(p_el, style_name: str) -> int | None:
    m = _HEADING_STYLE_RE.match(style_name.strip())
    if m:
        return int(m.group(2))
    if style_name.strip().lower() in _TITLE_STYLES:
        return 0
    lvl = p_el.find(f"{W}pPr/{W}outlineLvl")
    if lvl is not None and lvl.get(W + "val", "").isdigit() and int(lvl.get(W + "val")) < 9:
        return int(lvl.get(W + "val")) + 1
    return None


def _table_rows(tbl_el) -> list[list[str]]:
    rows = []
    for tr in tbl_el.findall(f"{W}tr"):
        cells = []
        for tc in tr.findall(f"{W}tc"):
            parts = [_read_paragraph(p)["text"] for p in tc.iter(f"{W}p")]  # nested tables flattened
            cells.append(" ".join(p for p in parts if p).replace("|", "\\|").replace("\n", " "))
        rows.append(cells)
    return rows


def rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _comments(zf: zipfile.ZipFile) -> dict[str, dict]:
    try:
        root = etree.fromstring(zf.read("word/comments.xml"))
    except KeyError:
        return {}
    out = {}
    for c in root.iter(f"{W}comment"):
        text = "\n".join(_read_paragraph(p)["text"] for p in c.iter(f"{W}p")).strip()
        out[c.get(W + "id")] = {"author": c.get(W + "author") or "", "date": c.get(W + "date") or "", "text": text}
    return out


def _check_zip(path: Path) -> zipfile.ZipFile:
    zf = zipfile.ZipFile(path)
    total = sum(i.file_size for i in zf.infolist())
    if total > MAX_UNZIPPED_MB * 2**20 or len(zf.infolist()) > 10_000:
        raise ValueError(f"suspicious archive: {total / 2**20:.0f} MB unpacked, {len(zf.infolist())} entries")
    return zf


def _properties(document) -> dict:
    cp = document.core_properties
    props = {
        "title": cp.title, "author": cp.author, "last_modified_by": cp.last_modified_by,
        "created": cp.created.isoformat() if cp.created else None,
        "modified": cp.modified.isoformat() if cp.modified else None,
        "revision": cp.revision,
    }
    headers, footers = set(), set()
    for section in document.sections:
        headers.update(p.text.strip() for p in section.header.paragraphs if p.text.strip())
        footers.update(p.text.strip() for p in section.footer.paragraphs if p.text.strip())
    props["header"] = " | ".join(sorted(headers)) or None
    props["footer"] = " | ".join(sorted(footers)) or None
    return {k: v for k, v in props.items() if v not in (None, "")}


def _windows(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    spans, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # prefer to cut at whitespace
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > 0 else end
        spans.append((start, end))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return spans


def _parse_plain(document, rel: str, result: ParsedFile, file_node: Node, cfg: ChunkingConfig) -> None:
    """H10 baseline: flat text, fixed windows, no structure."""
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        parts += [" ".join(c.text for c in row.cells) for row in table.rows]
    text = "\n".join(parts)
    for k, (a, b) in enumerate(_windows(text, cfg.text_max_chars, 150), start=1):
        node = Node(id=f"{rel}#chunk{k}", file_path=rel, file_type=FileType.DOCX, node_type=NodeType.SECTION,
                    parent_id=file_node.id, title=f"chunk {k}", text=text[a:b].strip())
        result.nodes.append(node)
        result.edges.append(Edge(src=file_node.id, dst=node.id, type=EdgeType.CONTAINS))


def parse_docx(cf: CorpusFile, cfg: ChunkingConfig, source: Path | None = None) -> ParsedFile:
    import docx

    path = source or cf.abs_path
    rel = cf.rel_path
    result = ParsedFile(file_path=rel, file_type=FileType.DOCX)
    zf = _check_zip(path)
    document = docx.Document(str(path))

    file_node = Node(id=rel, file_path=rel, file_type=FileType.DOCX, node_type=NodeType.FILE, title=rel,
                     metadata=_properties(document), embed=False)
    result.nodes.append(file_node)
    if cfg.docx == "plain":
        _parse_plain(document, rel, result, file_node, cfg)
        return result

    styles = _style_names(document)
    comments = _comments(zf)
    stack: dict[int, str] = {}
    section_path: str | None = None
    buffer: list[_Paragraph] = []
    counters = {"chunk": 0, "table": 0, "figure": 0, "rev": 0}
    last_section_node: dict[str, str] = {}
    pending_figures: list[Node] = []
    para_index = 0
    last_caption = ""  # a caption not yet claimed by a picture: tables usually have theirs above

    def add(node: Node, parent: str | None = None) -> None:
        result.nodes.append(node)
        result.edges.append(Edge(src=parent or file_node.id, dst=node.id, type=EdgeType.CONTAINS))

    def flush(final: bool = False) -> None:
        if not buffer:
            return
        if not final and all(p.level is not None for p in buffer):
            return  # headings without a body travel on into the next chunk as its context
        text = "\n".join(p.text for p in buffer)
        counters["chunk"] += 1
        node = Node(
            id=f"{rel}#s{counters['chunk']}",
            file_path=rel, file_type=FileType.DOCX, node_type=NodeType.SECTION, parent_id=file_node.id,
            title=(section_path or "").split(" > ")[-1] or "text",
            text=text,
            location=Location(section=section_path, paragraph_start=buffer[0].index, paragraph_end=buffer[-1].index),
            metadata={"section": section_path} if section_path else {},
        )
        add(node)
        if section_path:
            last_section_node[section_path] = node.id
        for fig in pending_figures:
            result.edges.append(Edge(src=fig.id, dst=node.id, type=EdgeType.ILLUSTRATES))
        pending_figures.clear()
        buffer.clear()

    body = document.element.body
    for el in body.iterchildren():
        if el.tag == W + "p":
            info = _read_paragraph(el)
            style_id = el.find(f"{W}pPr/{W}pStyle")
            style = styles.get(style_id.get(W + "val"), "") if style_id is not None else ""
            level = _heading_level(el, style)
            if not info["text"] and not info["deleted"] and not info["images"]:
                continue
            para_index += 1
            par = _Paragraph(para_index, info["text"], info["deleted"], info["inserted"], style, level,
                             info["meta"], info["images"], info["comments"])
            if level is not None and par.text:
                flush()
                stack = {lvl: t for lvl, t in stack.items() if lvl < level}
                stack[level] = par.text
                section_path = " > ".join(stack[lvl] for lvl in sorted(stack))
            is_caption = style.strip().lower() in _CAPTION_STYLES or bool(_CAPTION_TEXT_RE.match(par.text))
            if is_caption and pending_figures and not pending_figures[-1].text:
                pending_figures[-1].text = par.text  # caption right after the picture
                pending_figures[-1].embed = True
            elif is_caption:
                last_caption = par.text
            for rid in par.images:
                counters["figure"] += 1
                fig = Node(
                    id=f"{rel}#fig{counters['figure']}", file_path=rel, file_type=FileType.DOCX,
                    node_type=NodeType.FIGURE, parent_id=file_node.id, title=f"figure {counters['figure']}",
                    text="", location=Location(section=section_path, paragraph_start=par.index),
                    metadata={"image_rel": rid, **({"section": section_path} if section_path else {})}, embed=False,
                )
                add(fig)
                pending_figures.append(fig)
            if par.deleted or par.inserted:
                counters["rev"] += 1
                who = par.change_meta.get("del") or par.change_meta.get("ins") or (None, None)
                lines = []
                if par.deleted:
                    lines.append(f"Удалено: «{par.deleted}»")
                if par.inserted:
                    lines.append(f"Вставлено: «{par.inserted}»")
                add(Node(
                    id=f"{rel}#rev{counters['rev']}", file_path=rel, file_type=FileType.DOCX,
                    node_type=NodeType.REVISION, parent_id=file_node.id, title="tracked change",
                    text="\n".join(lines), context=f"Правка в абзаце. Текущий текст: {par.text}" if par.text else "",
                    location=Location(section=section_path, paragraph_start=par.index),
                    metadata={"author": who[0], "date": who[1], **({"section": section_path} if section_path else {})},
                ))
            for cid in par.comment_ids:
                c = comments.get(cid)
                if not c or not c["text"]:
                    continue
                add(Node(
                    id=f"{rel}#comment{cid}", file_path=rel, file_type=FileType.DOCX, node_type=NodeType.COMMENT,
                    parent_id=file_node.id, title="comment",
                    text=f"Комментарий ({c['author']}): {c['text']}", context=f"К тексту: {par.text[:300]}",
                    location=Location(section=section_path, paragraph_start=par.index),
                    metadata={"author": c["author"], "date": c["date"], **({"section": section_path} if section_path else {})},
                ))
            if par.text and not is_caption:
                last_caption = ""
                if buffer and sum(len(p.text) for p in buffer) + len(par.text) > cfg.text_max_chars:
                    flush()
                buffer.append(par)
        elif el.tag == W + "tbl":
            rows = _table_rows(el)
            if not rows:
                continue
            counters["table"] += 1
            caption, last_caption = last_caption, ""
            add(Node(
                id=f"{rel}#table{counters['table']}", file_path=rel, file_type=FileType.DOCX, node_type=NodeType.TABLE,
                parent_id=file_node.id, title=caption or f"table {counters['table']}",
                text=rows_to_markdown(rows), context=caption,
                location=Location(section=section_path, paragraph_start=para_index or None),
                metadata={"n_rows": len(rows), **({"section": section_path} if section_path else {})},
            ))
    flush(final=True)
    for fig in pending_figures:  # pictures after the last paragraph
        if section_path in last_section_node:
            result.edges.append(Edge(src=fig.id, dst=last_section_node[section_path], type=EdgeType.ILLUSTRATES))
    file_node.metadata.update({"n_paragraphs": para_index, "n_tables": counters["table"],
                               "n_figures": counters["figure"], "n_revisions": counters["rev"],
                               "n_comments": sum(1 for n in result.nodes if n.node_type == NodeType.COMMENT)})
    if not file_node.metadata.get("title") and stack:
        file_node.metadata["title"] = stack[min(stack)]
    file_node.text = file_node.metadata.get("title", "")
    return result


def find_soffice() -> str | None:
    found = shutil.which("soffice") or shutil.which("soffice.exe")
    if found:
        return found
    for candidate in (r"C:\Program Files\LibreOffice\program\soffice.exe",
                      r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"):
        if Path(candidate).exists():
            return candidate
    return None


def parse_doc(cf: CorpusFile, cfg: ChunkingConfig) -> ParsedFile:
    """Legacy .doc: convert to .docx with LibreOffice (headless), then parse as usual."""
    soffice = find_soffice()
    if soffice is None:
        raise RuntimeError("для .doc нужен LibreOffice (soffice): установите его или сохраните файл как .docx")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([soffice, "--headless", "--convert-to", "docx", "--outdir", tmp, str(cf.abs_path)],
                       check=True, capture_output=True, timeout=180)
        converted = Path(tmp) / (cf.abs_path.stem + ".docx")
        parsed = parse_docx(cf, cfg, source=converted)
    parsed.nodes[0].metadata["converted_from"] = "doc"
    return parsed
