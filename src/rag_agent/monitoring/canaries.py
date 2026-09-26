"""Canary questions for monitoring (ТЗ S12 части 1, Э9).

Canaries are synthetic questions with known answers, generated from the catalog by
templates — no LLM, deterministic, and separate from the evaluation sets. Each has
a segment ("component x segment" series of the monitoring fleet) and a set of
target locations; a retrieved fragment answers the canary if it covers one of them.

Segments:
    py:def:<pkg>     "Где определена функция X?"      target: the definition's lines
    py:call:<pkg>    "Где вызывается X?"              target: any call site in .py files
    ipynb:call       "В каких ноутбуках используется X?"  target: a notebook cell calling X
    ipynb:section    "О чём раздел «H» ноутбука N?"   target: the heading cell and the next two
    doc              "Что написано в разделе «S»?"    target: fragments of that section (PDF, DOCX)
    text             quoted phrase of a note or log    target: the fragment it comes from (TXT, MD, LOG)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

from rag_agent.schema import Node


@dataclass(frozen=True)
class Target:
    file: str
    lines: tuple[int, int] | None = None
    cell: int | None = None
    section: str | None = None
    node_id: str | None = None

    def covers(self, node: Node) -> bool:
        if node.file_path != self.file:
            return False
        if self.node_id is not None:
            return node.id == self.node_id
        loc = node.location
        if self.cell is not None:
            return loc.cell == self.cell
        if self.section is not None:
            return bool(loc.section) and self.section in loc.section
        if self.lines is not None and loc.line_start is not None:
            a, b = self.lines
            return loc.line_start <= b and (loc.line_end or loc.line_start) >= a
        return self.lines is None


@dataclass
class Canary:
    id: str
    segment: str
    question: str
    targets: list[Target] = field(default_factory=list)

    def answered_by(self, node: Node) -> bool:
        return any(t.covers(node) for t in self.targets)


def _pkg(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else "root"


def build_canaries(catalog, min_per_segment: int = 4) -> list[Canary]:
    q = catalog.query
    out: list[Canary] = []
    symbols = q("SELECT name, qualname, kind, file_path, line_start, line_end, cell FROM symbols")
    defined = {r["name"] for r in symbols if not r["name"].startswith("_")}

    for r in symbols:
        if r["name"].startswith("_") or not r["file_path"].endswith(".py") or r["line_start"] is None:
            continue
        what = {"class": f"Где определён класс {r['name']}?", "method": f"Где определён метод {r['qualname']}?"}.get(
            r["kind"], f"Где определена функция {r['name']}?")
        out.append(Canary(f"def:{r['file_path']}:{r['qualname']}", f"py:def:{_pkg(r['file_path'])}", what,
                          [Target(r["file_path"], (r["line_start"], r["line_end"] or r["line_start"]))]))

    sites: dict[tuple[str, str], list[Target]] = defaultdict(list)
    nb_sites: dict[str, list[Target]] = defaultdict(list)
    for r in q("SELECT name, file_path, line, cell FROM calls"):
        if r["name"] not in defined:
            continue
        if r["file_path"].endswith(".ipynb") and r["cell"] is not None:
            nb_sites[r["name"]].append(Target(r["file_path"], cell=r["cell"]))
        elif r["file_path"].endswith(".py") and r["line"] is not None:
            sites[(r["name"], _pkg(r["file_path"]))].append(Target(r["file_path"], (r["line"], r["line"])))
    for (name, pkg), targets in sorted(sites.items()):
        out.append(Canary(f"call:{pkg}:{name}", f"py:call:{pkg}", f"Где вызывается {name}?", targets))
    for name, targets in sorted(nb_sites.items()):
        out.append(Canary(f"nbcall:{name}", "ipynb:call", f"В каких ноутбуках используется {name}?", targets))

    for r in q("SELECT id, file_path, node_type, text, location FROM nodes WHERE embed = 1 ORDER BY file_path, id"):
        loc = json.loads(r["location"] or "{}")
        text = r["text"] or ""
        if r["node_type"] == "markdown_cell" and loc.get("cell"):
            heading = next((ln.lstrip("#").strip() for ln in text.split("\n") if ln.startswith("#")), "")
            if len(heading) >= 4:
                stem = r["file_path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
                c = loc["cell"]
                out.append(Canary(f"nbsec:{r['id']}", "ipynb:section", f"О чём раздел «{heading}» ноутбука {stem}?",
                                  [Target(r["file_path"], cell=k) for k in (c, c + 1, c + 2)]))
        elif r["file_path"].endswith((".pdf", ".docx")) and loc.get("section"):
            title = loc["section"].split(" > ")[-1]
            cid = f"sec:{r['file_path']}:{title}"
            if len(title) >= 4 and all(c.id != cid for c in out):
                out.append(Canary(cid, "doc", f"Что написано в разделе «{title}»?",
                                  [Target(r["file_path"], section=title)]))
        elif r["file_path"].endswith((".txt", ".log", ".md")):
            words = re.findall(r"[\w.,%=-]+", text)
            if len(words) >= 12:
                phrase = " ".join(words[4:14])
                out.append(Canary(f"text:{r['id']}", "text", f"Где в моих файлах написано «{phrase}»?",
                                  [Target(r["file_path"], node_id=r["id"])]))

    counts = defaultdict(int)
    for c in out:
        counts[c.segment] += 1
    return [c for c in out if counts[c.segment] >= min_per_segment]
