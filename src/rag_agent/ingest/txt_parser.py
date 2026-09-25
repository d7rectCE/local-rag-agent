"""Parser for plain-text files: .txt, .md, .log (ТЗ ч.2, S14).

1. Encoding is detected from byte statistics (charset-normalizer) and stored;
   undecodable files are reported, never silently mangled.
2. A heuristic classifier picks the content type: markdown, log, table, code, prose.
3. Segmentation by type:
   * markdown — by headings, then by paragraphs;
   * prose    — by paragraphs; a text without paragraph breaks is split at lexical
                cohesion minima between sentence blocks (TextTiling, ТЗ ч.2 [3]);
   * log      — windows of lines; metrics per epoch/step are extracted into a
                ``summary`` node (best and last values) and the file metadata (Э6 catalog);
   * table    — delimited export as Markdown tables with the header repeated;
   * code     — line windows.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter

from rag_agent.config import ChunkingConfig
from rag_agent.ingest.docx_parser import rows_to_markdown
from rag_agent.ingest.python_parser import line_windows
from rag_agent.ingest.walker import CorpusFile
from rag_agent.schema import Edge, EdgeType, FileType, Location, Node, NodeType, ParsedFile

_STEP_RE = re.compile(r"\b(epoch|step|iter(?:ation)?)\b\s*[=:#]?\s*(\d+)", re.IGNORECASE)
_METRIC_RE = re.compile(r"\b([A-Za-z][\w/.\-]*)\s*[=:]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_TIME_RE = re.compile(r"^\s*\[?(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\d{2}:\d{2}:\d{2})")
_MD_RE = re.compile(r"^\s{0,3}(#{1,6}\s|[-*+]\s|\d+\.\s|```|\|.*\|)")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_CODE_RE = re.compile(r"^\s*(def |class |import |from \S+ import |return\b|if __name__|@\w+|for .+ in .+:)")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_LOWER_IS_BETTER = ("loss", "error", "err", "rmse", "mae", "mse", "perplexity", "ppl")


def decode(data: bytes) -> tuple[str, str]:
    """(text, encoding); raises ValueError when no encoding fits."""
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass
    from charset_normalizer import from_bytes

    best = from_bytes(data).best()
    if best is None:
        raise ValueError("не удалось определить кодировку")
    return str(best), best.encoding


def classify(text: str) -> str:
    lines = [ln for ln in text.split("\n") if ln.strip()][:500]
    if not lines:
        return "prose"
    n = len(lines)
    log_share = sum(1 for ln in lines if _TIME_RE.match(ln) or (_STEP_RE.search(ln) and _METRIC_RE.search(ln))) / n
    if log_share >= 0.3:
        return "log"
    for delim in ("\t", ";", "|", ","):
        counts = [ln.count(delim) for ln in lines[:200]]
        common, freq = Counter(counts).most_common(1)[0]
        if common >= 1 and freq / len(counts) >= 0.8 and n >= 3:
            return "table"
    if any(_HEADING_RE.match(ln) for ln in lines) or sum(1 for ln in lines if _MD_RE.match(ln)) / n >= 0.2:
        return "markdown"
    if sum(1 for ln in lines if _CODE_RE.match(ln)) / n >= 0.25:
        return "code"
    return "prose"


# --------------------------------------------------------------------------- segmentation


def _paragraphs(lines: list[str], offset: int = 0) -> list[tuple[int, int, str]]:
    """(first line, last line, text) of blank-line separated paragraphs; lines 1-based."""
    out, start, buf = [], None, []
    for i, line in enumerate(lines, start=1 + offset):
        if line.strip():
            if start is None:
                start = i
            buf.append(line)
        elif buf:
            out.append((start, i - 1, "\n".join(buf)))
            start, buf = None, []
    if buf:
        out.append((start, start + len(buf) - 1, "\n".join(buf)))
    return out


def _pack(paragraphs: list[tuple[int, int, str]], max_chars: int) -> list[tuple[int, int, str]]:
    chunks, cur = [], []
    for p in paragraphs:
        if cur and sum(len(c[2]) for c in cur) + len(p[2]) > max_chars:
            chunks.append((cur[0][0], cur[-1][1], "\n\n".join(c[2] for c in cur)))
            cur = []
        cur.append(p)
    if cur:
        chunks.append((cur[0][0], cur[-1][1], "\n\n".join(c[2] for c in cur)))
    return chunks


def texttiling(text: str, block: int = 3) -> list[str]:
    """Split a long unbroken text at lexical cohesion minima between blocks of
    sentences (simplified TextTiling): gaps whose cosine similarity is below
    mean - std/2 of all gaps become boundaries."""
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]
    if len(sentences) < 2 * block + 1:
        return [text]
    bags = [Counter(w.lower() for w in _WORD_RE.findall(s)) for s in sentences]

    def cos(a: Counter, b: Counter) -> float:
        num = sum(a[k] * b[k] for k in a if k in b)
        den = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
        return num / den if den else 0.0

    gaps = []
    for g in range(block, len(sentences) - block + 1):
        left = sum(bags[g - block:g], Counter())
        right = sum(bags[g:g + block], Counter())
        gaps.append((g, cos(left, right)))
    values = [v for _, v in gaps]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    cuts = [g for k, (g, v) in enumerate(gaps)
            if v < mean - std / 2 and (k == 0 or v <= gaps[k - 1][1]) and (k == len(gaps) - 1 or v <= gaps[k + 1][1])]
    pieces, prev = [], 0
    for g in cuts:
        if g - prev >= block:
            pieces.append(" ".join(sentences[prev:g]))
            prev = g
    pieces.append(" ".join(sentences[prev:]))
    return pieces


def _markdown_sections(lines: list[str], max_chars: int) -> list[tuple[int, int, str, str | None]]:
    stack: dict[int, str] = {}
    out: list[tuple[int, int, str, str | None]] = []
    seg_start, seg_lines, path = 1, [], None

    def emit():
        for a, b, t in _pack(_paragraphs(seg_lines, seg_start - 1), max_chars):
            out.append((a, b, t, path))

    for i, line in enumerate(lines, start=1):
        m = _HEADING_RE.match(line)
        if m:
            emit()
            level = len(m.group(1))
            stack = {lvl: t for lvl, t in stack.items() if lvl < level}
            stack[level] = m.group(2).strip()
            path = " > ".join(stack[lvl] for lvl in sorted(stack))
            seg_start, seg_lines = i, [line]
        else:
            seg_lines.append(line)
    emit()
    return out


def _log_metrics(lines: list[str]) -> dict[str, list[tuple[int | None, float]]]:
    series: dict[str, list[tuple[int | None, float]]] = {}
    for line in lines:
        step = _STEP_RE.search(line)
        step_no = int(step.group(2)) if step else None
        for name, value in _METRIC_RE.findall(line):
            if step and name.lower() == step.group(1).lower():
                continue
            series.setdefault(name, []).append((step_no, float(value)))
    return {k: v for k, v in series.items() if len(v) >= 2}


def _metric_summary(series: dict[str, list[tuple[int | None, float]]], unit: str) -> tuple[str, dict]:
    lines, meta = [], {}
    for name, points in series.items():
        lower = any(tag in name.lower() for tag in _LOWER_IS_BETTER)
        best_step, best = (min if lower else max)(points, key=lambda p: p[1])
        last_step, last = points[-1]
        where = f" ({unit} {best_step})" if best_step is not None else ""
        lines.append(f"{name}: {'минимум' if lower else 'максимум'} {best:g}{where}, последнее значение {last:g}")
        meta[name] = {"best": best, "best_step": best_step, "last": last, "last_step": last_step, "n": len(points),
                      "direction": "min" if lower else "max"}
    return "\n".join(lines), meta


def parse_txt(cf: CorpusFile, cfg: ChunkingConfig) -> ParsedFile:
    raw = cf.abs_path.read_bytes()
    text, encoding = decode(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    rel = cf.rel_path
    result = ParsedFile(file_path=rel, file_type=FileType.TXT)
    lines = text.split("\n")
    kind = classify(text)
    file_node = Node(id=rel, file_path=rel, file_type=FileType.TXT, node_type=NodeType.FILE, title=rel,
                     metadata={"encoding": encoding, "content_type": kind, "n_lines": len(lines)}, embed=False)
    result.nodes.append(file_node)

    def add(node_id: str, node_type: NodeType, title: str, body: str, a: int | None, b: int | None,
            section: str | None = None, **meta) -> None:
        node = Node(id=node_id, file_path=rel, file_type=FileType.TXT, node_type=node_type, parent_id=file_node.id,
                    title=title, text=body[: cfg.max_chunk_chars],
                    location=Location(line_start=a, line_end=b, section=section),
                    metadata={**({"section": section} if section else {}), **meta})
        result.nodes.append(node)
        result.edges.append(Edge(src=file_node.id, dst=node.id, type=EdgeType.CONTAINS))

    max_chars = cfg.text_max_chars
    if kind == "markdown":
        for k, (a, b, body, path) in enumerate(_markdown_sections(lines, max_chars), start=1):
            add(f"{rel}#L{a}-{b}", NodeType.SECTION, (path or "").split(" > ")[-1] or f"lines {a}-{b}", body, a, b, path)
    elif kind == "log":
        series = _log_metrics(lines)
        if series:
            unit = next((m.group(1).lower() for ln in lines if (m := _STEP_RE.search(ln))), "шаг")
            summary, meta = _metric_summary(series, unit)
            file_node.metadata["metrics"] = meta
            add(f"{rel}#summary", NodeType.SUMMARY, "metrics summary",
                f"Метрики из лога ({len(lines)} строк):\n{summary}", 1, len(lines), metrics=list(meta))
        for a, b in line_windows(len(lines), cfg.lines_per_chunk, cfg.overlap_lines):
            body = "\n".join(lines[a:b])
            if body.strip():
                add(f"{rel}#L{a + 1}-{b}", NodeType.LOG_CHUNK, f"lines {a + 1}-{b}", body, a + 1, b)
    elif kind == "table":
        sample = "\n".join(lines[:50])
        delim = csv.Sniffer().sniff(sample, delimiters="\t;|,").delimiter
        rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if any(c.strip() for c in r)]
        header, body_rows = rows[0], rows[1:]
        file_node.metadata.update({"columns": header, "n_rows": len(body_rows), "delimiter": delim})
        step = 40
        for k in range(0, max(len(body_rows), 1), step):
            part = body_rows[k:k + step]
            add(f"{rel}#rows{k + 1}-{k + len(part)}", NodeType.TABLE, f"rows {k + 1}-{k + len(part)}",
                rows_to_markdown([header] + part), k + 2, k + 1 + len(part),
                columns=header)
    elif kind == "code":
        for a, b in line_windows(len(lines), cfg.lines_per_chunk, cfg.overlap_lines):
            body = "\n".join(lines[a:b])
            if body.strip():
                add(f"{rel}#L{a + 1}-{b}", NodeType.CODE_CHUNK, f"lines {a + 1}-{b}", body, a + 1, b)
    else:  # prose
        paragraphs = _paragraphs(lines)
        pieces: list[tuple[int, int, str]] = []
        for a, b, body in paragraphs:
            if len(body) > 3 * max_chars:  # one long unbroken text: topic segmentation
                tiles = texttiling(body)
                pieces += [(a, b, t) for t in tiles]
            else:
                pieces.append((a, b, body))
        for k, (a, b, body) in enumerate(_pack(pieces, max_chars), start=1):
            add(f"{rel}#p{k}", NodeType.SECTION, f"lines {a}-{b}", body, a, b)
    return result
