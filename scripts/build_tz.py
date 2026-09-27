"""Builds docs/TZ.md from the two local spec documents (the .docx files stay local and are not committed):
the literature analysis of both parts first, then the specification formed from it, one reference list at
the end. Part 1 keeps its reference numbers; new references of part 2 continue the list, references present
in both parts get the part 1 number, and the citations in the part 2 text are renumbered. Like-named
subsections of the two parts are merged, and tables with the same header are joined (query classes,
hypotheses, requirements, stages, risks).

    python scripts/build_tz.py TZ_multimodal_agent.docx TZ_part2_agent_extensions.docx docs/TZ.md docs/tz_citation_map.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn

W = qn


# --------------------------------------------------------------------------- docx -> blocks


def numbering_formats(d) -> dict[str, str]:
    """numId -> numFmt of level 0 (decimal / bullet ...)."""
    try:
        numbering = d.part.numbering_part.element
    except (KeyError, NotImplementedError, AttributeError):
        return {}
    abstract = {}
    for an in numbering.findall(W("w:abstractNum")):
        lvl0 = an.find(W("w:lvl"))
        fmt = lvl0.find(W("w:numFmt")).get(W("w:val")) if lvl0 is not None and lvl0.find(W("w:numFmt")) is not None else "bullet"
        abstract[an.get(W("w:abstractNumId"))] = fmt
    out = {}
    for num in numbering.findall(W("w:num")):
        aid = num.find(W("w:abstractNumId")).get(W("w:val"))
        out[num.get(W("w:numId"))] = abstract.get(aid, "bullet")
    return out


def run_md(p_el) -> str:
    """Paragraph text with **bold** and *italic* runs (adjacent runs of the same format are merged)."""
    parts: list[tuple[str, bool, bool]] = []
    for r in p_el.iter(W("w:r")):
        text = "".join((t.text or "") if t.tag == W("w:t") else ("\t" if t.tag == W("w:tab") else "\n")
                       for t in r if t.tag in (W("w:t"), W("w:tab"), W("w:br")))
        if not text:
            continue
        rpr = r.find(W("w:rPr"))

        def on(tag):
            if rpr is None or rpr.find(W(tag)) is None:
                return False
            v = rpr.find(W(tag)).get(W("w:val"))
            return v not in ("0", "false")

        b, i = on("w:b"), on("w:i")
        if parts and parts[-1][1] == b and parts[-1][2] == i:
            parts[-1] = (parts[-1][0] + text, b, i)
        else:
            parts.append((text, b, i))
    out = []
    for text, b, i in parts:
        lead = len(text) - len(text.lstrip())
        trail = len(text) - len(text.rstrip())
        core = text.strip()
        if core and (b or i):
            mark = "**" if b and not i else ("*" if i and not b else "***")
            core = f"{mark}{core}{mark}"
        out.append(text[:lead] + core + (text[len(text) - trail:] if trail else ""))
    return "".join(out)


def cell_md(tc) -> str:
    paras = [run_md(p).strip() for p in tc.findall(W("w:p"))]
    return "<br>".join(p for p in paras if p).replace("|", "\\|")


def blocks_of(path: str) -> list[dict]:
    d = docx.Document(path)
    fmts = numbering_formats(d)
    blocks = []
    for el in d.element.body.iterchildren():
        tag = el.tag.split("}")[1]
        if tag == "p":
            ppr = el.find(W("w:pPr"))
            style = ""
            if ppr is not None and ppr.find(W("w:pStyle")) is not None:
                style = ppr.find(W("w:pStyle")).get(W("w:val"))
            text = run_md(el).strip()
            plain = "".join(t.text or "" for t in el.iter(W("w:t"))).strip()
            if not plain:
                continue
            if style.startswith("Heading"):
                blocks.append({"kind": "h", "level": int(style[-1]), "text": plain})
            elif ppr is not None and ppr.find(W("w:numPr")) is not None:
                num_id = ppr.find(W("w:numPr")).find(W("w:numId"))
                fmt = fmts.get(num_id.get(W("w:val")) if num_id is not None else "", "bullet")
                blocks.append({"kind": "li", "ordered": fmt not in ("bullet", "none"), "text": text})
            else:
                blocks.append({"kind": "p", "text": text, "plain": plain})
        elif tag == "tbl":
            rows = [[cell_md(tc) for tc in tr.findall(W("w:tc"))] for tr in el.findall(W("w:tr"))]
            blocks.append({"kind": "table", "rows": rows})
    return blocks


def split_sections(blocks: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Blocks before the first Heading1, and Heading1 title -> {"blocks": [...], "subs": [(title, blocks)]}."""
    pre, sections, cur, sub = [], {}, None, None
    for b in blocks:
        if b["kind"] == "h" and b["level"] == 1:
            cur = {"blocks": [], "subs": []}
            sections[b["text"]] = cur
            sub = None
        elif cur is None:
            pre.append(b)
        elif b["kind"] == "h" and b["level"] == 2:
            sub = (b["text"], [])
            cur["subs"].append(sub)
        elif sub is not None:
            sub[1].append(b)
        else:
            cur["blocks"].append(b)
    return pre, sections


# --------------------------------------------------------------------------- references

REF_RE = re.compile(r"^\[(\d+)\]\s*(.+)$", re.S)
CIT_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")


def refs_of(section: dict) -> tuple[dict[int, str], list[str]]:
    refs, tools = {}, []
    for b in section["blocks"]:
        m = REF_RE.match(b.get("plain", "") or "")
        if b["kind"] == "p" and m:
            refs[int(m.group(1))] = m.group(2).strip()
    for title, blocks in section["subs"]:
        if title.startswith("Программные"):
            tools += [b["text"] for b in blocks if b["kind"] == "li"]
    return refs, tools


def title_key(ref: str) -> str:
    """Normalised title: the text between the authors and '//'."""
    body = ref.split("//")[0]
    body = re.sub(r"^.*?(?:et al\.|\.\s(?=[A-Z][a-z]))", "", body, count=1)
    return re.sub(r"\W+", " ", body).strip().lower()[:80]


def expand(group: str) -> list[int]:
    out = []
    for part in re.split(r"\s*,\s*", group):
        if re.search(r"[–-]", part):
            a, b = (int(x) for x in re.split(r"\s*[–-]\s*", part))
            out += list(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def compress(nums: list[int]) -> str:
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ", ".join(out)


def remap_text(text: str, mapping: dict[int, int]) -> str:
    return CIT_RE.sub(lambda m: "[" + compress([mapping[n] for n in expand(m.group(1))]) + "]", text)


def remap_block(b: dict, mapping: dict[int, int]) -> dict:
    b = dict(b)
    if b["kind"] in ("p", "li", "h"):
        b["text"] = remap_text(b["text"], mapping)
    elif b["kind"] == "table":
        b["rows"] = [[remap_text(c, mapping) for c in row] for row in b["rows"]]
    return b


# --------------------------------------------------------------------------- markdown


def md(blocks: list[dict]) -> str:
    out, prev = [], None
    for b in blocks:
        if b["kind"] == "li":
            if prev != "li":
                out.append("")
            out.append(("1. " if b["ordered"] else "- ") + b["text"])
        elif b["kind"] == "p":
            out += ["", b["text"]]
        elif b["kind"] == "table":
            rows = b["rows"]
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            header = [re.sub(r"^\*+|\*+$", "", c) for c in rows[0]]  # a header row is bold in Markdown anyway
            out += ["", "| " + " | ".join(header) + " |", "|" + "---|" * width]
            out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        elif b["kind"] == "h":
            out += ["", "#" * (b["level"] + 1) + " " + b["text"]]
        prev = b["kind"]
    return "\n".join(out).strip() + "\n"


def strip_number(title: str) -> str:
    return re.sub(r"^\d+(\.\d+)*\.\s*", "", title).strip()


def main(p1: str, p2: str, out: str, mapping_out: str) -> None:
    pre1, s1 = split_sections(blocks_of(p1))
    pre2, s2 = split_sections(blocks_of(p2))
    lit1 = next(k for k in s1 if "Список литературы" in k)
    lit2 = next(k for k in s2 if "Список литературы" in k)
    refs1, tools1 = refs_of(s1[lit1])
    refs2, tools2 = refs_of(s2[lit2])
    keys1 = {title_key(r): n for n, r in refs1.items()}
    mapping, merged = {}, dict(refs1)
    nxt = max(refs1) + 1
    for n in sorted(refs2):
        k = title_key(refs2[n])
        if k in keys1:
            mapping[n] = keys1[k]
        else:
            mapping[n] = nxt
            merged[nxt] = refs2[n]
            nxt += 1
    tools = list(dict.fromkeys(tools1 + [t for t in tools2 if not any(t.split(" — ")[0] == x.split(" — ")[0]
                                                                         for x in tools1)]))
    s2 = {k: {"blocks": [remap_block(b, mapping) for b in v["blocks"]],
              "subs": [(remap_text(t, mapping), [remap_block(b, mapping) for b in bl]) for t, bl in v["subs"]]}
          for k, v in s2.items()}

    def sec(sections, prefix):
        return next(v for k, v in sections.items() if strip_number(k).startswith(prefix))

    doc = ["# Мультимодальный агент над корпусом исследовательских файлов: анализ литературы и техническое задание", "",
           "Документ объединяет две части технического задания магистерского проекта: основу — вопросно-ответную "
           "систему с цитированием над корпусом `.py`, `.ipynb`, `.pdf`, `.png` (этапы Э1–Э10) — и рабочего "
           "ассистента с новыми форматами, рассуждением, загрузками, исполнением кода и интернетом (этапы Э11–Э17). "
           "Сначала — анализ литературы по обеим частям (раздел 1). Из его выводов сформировано техническое "
           "задание (раздел 2): проблемы кейса, решения, методология оценки, требования, этапы и риски. Ссылки "
           "вида [n] ведут в общий список литературы (раздел 3). Состояние этапов и результаты проверки гипотез — "
           "в [README](../README.md) и [отчётах](../reports).", ""]

    # 1. literature
    doc += ["## 1. Анализ литературы", ""]
    r1, r2 = sec(s1, "Обзор литературы"), sec(s2, "Обзор литературы")
    k = 0
    if r1["blocks"]:
        doc.append(md(r1["blocks"]))
    if r2["blocks"]:
        doc.append(md(r2["blocks"]))
    summary = []
    for part, rev in ((1, r1), (2, r2)):
        for title, blocks in rev["subs"]:
            if title.lower().startswith(tuple(f"{i}.{j}. сводная" for i in (2,) for j in range(1, 20))) or "Сводная таблица" in title:
                summary.append(blocks)
                continue
            k += 1
            doc += [f"### 1.{k}. {strip_number(title)}", "", md(blocks)]
    k += 1
    doc += [f"### 1.{k}. Сводная таблица ключевых работ", ""]
    tables = [b for blocks in summary for b in blocks if b["kind"] == "table"]
    if tables:
        rows = tables[0]["rows"] + [r for t in tables[1:] for r in t["rows"][1:]]
        doc.append(md([{"kind": "table", "rows": rows}]))
    doc += [md([b for blocks in summary for b in blocks if b["kind"] != "table"])] if any(
        b["kind"] != "table" for blocks in summary for b in blocks) else []

    # 2. specification: like-named subsections of both parts merged, tables with equal headers joined
    doc += ["## 2. Техническое задание", ""]
    synonyms = {"новые классы запросов": "типология пользовательских запросов",
                "изменения архитектуры": "целевая архитектура",
                "расширение эталонного набора": "состав эталонного набора"}
    after = {"новые возможности": "характеристика корпуса"}

    def norm(t: str) -> str:
        return strip_number(t).lower()

    def split_at_table(blocks):
        idx = [i for i, b in enumerate(blocks) if b["kind"] == "table"]
        if len(idx) != 1:
            return None
        i = idx[0]
        return blocks[:i], blocks[i], blocks[i + 1:]

    def merge_blocks(x, y):
        sx, sy = split_at_table(x), split_at_table(y)
        if sx and sy and [c.lower() for c in sx[1]["rows"][0]] == [c.lower() for c in sy[1]["rows"][0]]:
            table = {"kind": "table", "rows": sx[1]["rows"] + sy[1]["rows"][1:]}
            return sx[0] + sy[0] + [table] + sx[2] + sy[2]
        return x + y

    def merged_section(a, b):
        if a is None:
            return b["blocks"], [list(t) for t in b["subs"]]
        top = merge_blocks(a["blocks"], b["blocks"])
        slots = [[t, list(bl)] for t, bl in a["subs"]]
        for t, bl in b["subs"]:
            key = synonyms.get(norm(t), norm(t))
            hit = next((sl for sl in slots if norm(sl[0]) == key), None)
            if hit is not None:
                hit[1] = merge_blocks(hit[1], bl)
            elif norm(t) in after:
                pos = next((i for i, sl in enumerate(slots) if norm(sl[0]) == after[norm(t)]), len(slots) - 1)
                slots.insert(pos + 1, [t, list(bl)])
            else:
                slots.append([t, list(bl)])
        return top, slots

    def md_shift(blocks, shift):
        return md([dict(b, level=b["level"] + shift) if b["kind"] == "h" else b for b in blocks])

    n = 0
    for title in ("Аннотация", "Постановка проблемы", "Существующие проблемы", "Теоретические решения",
                  "Методология оценки", "Требования", "Этапы работ", "Риски"):
        a = sec(s1, title) if any(strip_number(k).startswith(title) for k in s1) else None
        b = sec(s2, title)
        name = next(strip_number(k) for k in (list(s1) + list(s2)) if strip_number(k).startswith(title))
        if title == "Аннотация":
            name = "Аннотация и допущения"
        n += 1
        top, slots = merged_section(a, b)
        doc.extend([f"### 2.{n}. {name}", ""])
        if top:
            doc.append(md_shift(top, 0))
        for m, (t, bl) in enumerate(slots, start=1):
            doc.extend([f"#### 2.{n}.{m}. {strip_number(t)}", "", md_shift(bl, 1)])

    # 3. references
    doc += ["## 3. Список литературы", ""]
    doc += [f"[{i}] {merged[i]}  " for i in sorted(merged)]
    doc += ["", "### Программные инструменты", "", "Инструменты не являются научными публикациями и приведены "
            "отдельно от списка литературы.", ""]
    doc += [f"- {t}" for t in tools]
    text = "\n".join(doc).replace("\n\n\n", "\n\n")
    # sentences about the layout of the separate files do not hold in the merged document
    text = re.sub(r"\nДокумент состоит из пяти частей\.[^\n]*\n", "\n", text)
    text = text.replace("Документ продолжает техническое задание «Мультимодальный агент над гетерогенным корпусом "
                        "исследовательских файлов» (далее — Часть 1).", "Часть 2 продолжает Часть 1.")
    text = re.sub(r"Структура документа повторяет Часть 1:[^.]*\. (Нумерация сквозная:[^\n]*?\.) Ссылки в квадратных "
                  r"скобках ведут на список литературы этой части\.", r"\1", text)
    Path(out).write_text(re.sub(r"\n{3,}", "\n\n", text).strip() + "\n", encoding="utf-8")
    Path(mapping_out).write_text(json.dumps({"part2_to_merged": mapping, "n_refs": len(merged)}, indent=1),
                                 encoding="utf-8")
    print(f"{len(refs1)} + {len(refs2)} references -> {len(merged)}; shared: "
          f"{sorted((a, b) for a, b in mapping.items() if b <= max(refs1))}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
