"""Policy layer and data provenance (ТЗ ч.2 S20, S21; Э14).

Every tool call of the agent passes through ``Policy.check`` before it runs, and
every observation is labelled with its trust level:

- trusted   — the user's own request;
- private   — the corpus and the catalog;
- untrusted — uploads, web pages, output of code that processed untrusted data.

Security comes from the architecture, not from the model spotting an attack
(ТЗ ч.2 2.x, CaMeL [41], design patterns [42]):

Rule 1  a web query must come from the user's question: private entities of the
        corpus (file names, functions, classes) in it need the user's confirmation,
        and so does any distinctive token (identifier, code, number) that entered the
        context from private data but is not in the question — a data-flow check in
        the spirit of CaMeL: private content cannot reach an external query unseen;
Rule 2  once untrusted data is in the context, any action with an external effect
        needs confirmation ("plan, then execute");
Rule 3  a page is fetched only by a URL from search results or from the user;
        a URL the model composed (e.g. with data in its query string) is blocked;
Rule 4  answers never load external images and never make links active by
        themselves (``defang_markdown``), so rendering cannot send data out.

Writes (apply_changes) and package installs always need confirmation. Tool sets
depend on the mode (S21): reading tools always, web and code only when enabled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal
from urllib.parse import urlsplit


class Trust(StrEnum):
    TRUSTED = "trusted"
    PRIVATE = "private"
    UNTRUSTED = "untrusted"


Effect = Literal["none", "external", "write"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    effect: Effect  # none: reads only; external: data leaves the machine; write: changes the user's files
    output: Trust | Literal["inherit"]  # trust of what the tool returns; inherit: the lowest trust of its inputs
    mode: Literal["read", "web", "code"] = "read"  # which toggle enables it (S21)


TOOL_SPECS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("search", "none", Trust.PRIVATE),
    ToolSpec("exact_search", "none", Trust.PRIVATE),
    ToolSpec("sql_query", "none", Trust.PRIVATE),
    ToolSpec("read_file", "none", Trust.PRIVATE),
    ToolSpec("list_dir", "none", Trust.PRIVATE),
    ToolSpec("read_upload", "none", Trust.UNTRUSTED),
    ToolSpec("web_search", "external", Trust.UNTRUSTED, mode="web"),
    ToolSpec("fetch_page", "external", Trust.UNTRUSTED, mode="web"),
    ToolSpec("run_code", "none", "inherit", mode="code"),  # the sandbox has no network (Э15)
    ToolSpec("apply_changes", "write", Trust.TRUSTED, mode="code"),
    ToolSpec("install_package", "external", Trust.UNTRUSTED, mode="code"),
]}
CONTROL = {"answer", "refuse"}
ALWAYS_CONFIRM = {"apply_changes", "install_package"}


@dataclass
class Decision:
    action: Literal["allow", "confirm", "deny"]
    rule: str = ""
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.action == "allow"


@dataclass
class Provenance:
    """What has entered the agent's context in this request, with trust labels."""

    labels: list[tuple[str, Trust]] = field(default_factory=lambda: [("question", Trust.TRUSTED)])
    known_urls: set[str] = field(default_factory=set)  # from search results or from the user's message
    question_words: set[str] = field(default_factory=set)  # every word the user typed (may go out)
    private_tokens: set[str] = field(default_factory=set)  # distinctive tokens of private observations

    @classmethod
    def for_question(cls, question: str) -> Provenance:
        return cls(question_words=words(question), known_urls=urls_in(question))

    @property
    def tainted(self) -> bool:
        return any(t == Trust.UNTRUSTED for _, t in self.labels)

    def lowest(self) -> Trust:
        order = [Trust.UNTRUSTED, Trust.PRIVATE, Trust.TRUSTED]
        return next(t for t in order if any(lbl == t for _, lbl in self.labels))

    def add(self, source: str, trust: Trust) -> None:
        self.labels.append((source, trust))


_URL = re.compile(r"https?://[^\s<>()\"'`\]]+")
_WORD = re.compile(r"[\w][\w.\-]*[\w]|\w")


def words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "")}


def distinctive_tokens(text: str) -> set[str]:
    """Tokens that identify private content: codes and identifiers (digits with letters, underscores,
    inner capitals, hyphenated codes), long numbers; ordinary words are left out."""
    out = set()
    for w in _WORD.findall(text or ""):
        has_digit, has_alpha = any(c.isdigit() for c in w), any(c.isalpha() for c in w)
        if len(w) >= 5 and has_alpha and (has_digit or "_" in w or (w.isupper() and len(w) >= 6)
                                          or re.search(r"[a-z][A-Z]", w) or re.search(r"[A-Za-z]-[A-Z0-9]", w)):
            out.add(w.lower())
        elif not has_alpha and len(re.sub(r"\D", "", w)) >= 4 and not (w.isdigit() and 1900 <= int(w) <= 2100):
            out.add(w.lower())  # 0.9913, 4417-12; years are ordinary words of a web query
    return out


def urls_in(text: str) -> set[str]:
    return {u.rstrip(".,;:!?") for u in _URL.findall(text or "")}


def _norm_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{parts.path.rstrip('/')}" + (f"?{parts.query}" if parts.query else "")


class Policy:
    def __init__(self, private_entities: set[str] | None = None, *, web: bool = False, code: bool = False,
                 enabled: bool = True):
        self.private = {e for e in (private_entities or set()) if len(e) >= 4}
        self.modes = {"read"} | ({"web"} if web else set()) | ({"code"} if code else set())
        self.enabled = enabled  # False only for the H16 baseline: every enabled tool call is allowed

    @classmethod
    def for_catalog(cls, catalog, **modes) -> Policy:
        """Private entities of the corpus: file stems and defined names that look like identifiers."""
        names: set[str] = set()
        if catalog is not None:
            # "features", "metrics", "trainer" are plain words: web queries about them are not a leak;
            # "hparam_search", "08_numpy_mlp", "report-march" name this archive
            names |= {s for s in catalog.file_stems()
                      if len(s) >= 6 and not s.startswith("__") and re.search(r"[_\-\d]", s)}
            for r in catalog.query("SELECT DISTINCT name FROM symbols"):
                n = r["name"]
                if len(n) >= 6 and ("_" in n or re.search(r"[a-z][A-Z]", n)):  # compute_f1, trainModel
                    names.add(n)
        return cls(names, **modes)

    def tools(self) -> list[str]:
        """Tools available in the current modes (S21): fewer tools, fewer wrong choices."""
        return [n for n, s in TOOL_SPECS.items() if s.mode in self.modes]

    def private_in(self, text: str) -> list[str]:
        low = (text or "").lower()
        return sorted(e for e in self.private if re.search(rf"(?<![\w]){re.escape(e.lower())}(?![\w])", low))

    def check(self, tool: str, args: dict, prov: Provenance) -> Decision:
        if tool in CONTROL:
            return Decision("allow")
        spec = TOOL_SPECS.get(tool)
        if spec is None:
            return Decision("deny", "tools", f"неизвестный инструмент {tool}")
        if spec.mode not in self.modes:
            return Decision("deny", "S21", f"инструмент {tool} выключен в текущем режиме")
        if not self.enabled:
            return Decision("allow", "off")
        if tool == "web_search":
            query = str(args.get("query", ""))
            found = self.private_in(query)
            if found:
                return Decision("confirm", "rule 1", "в поисковом запросе есть приватные имена из корпуса: " + ", ".join(found))
            leaked = sorted((distinctive_tokens(query) & prov.private_tokens) - prov.question_words)
            if leaked:
                return Decision("confirm", "rule 1", "в запрос попали данные из ваших файлов, которых нет в вопросе: "
                                + ", ".join(leaked[:5]))
        if tool == "fetch_page":
            url = str(args.get("url", ""))
            if _norm_url(url) not in {_norm_url(u) for u in prov.known_urls}:
                return Decision("deny", "rule 3", "адрес не из результатов поиска и не от пользователя")
            return Decision("allow")  # a page from the results or from the user needs no confirmation (table 5)
        if tool in ALWAYS_CONFIRM:
            return Decision("confirm", "FR14", "изменение файлов или окружения только с подтверждения")
        if spec.effect != "none" and prov.tainted:
            return Decision("confirm", "rule 2", "в контексте есть недоверенные данные, а действие имеет внешний эффект")
        return Decision("allow")

    def observe(self, tool: str, prov: Provenance, output: str = "") -> Trust:
        """Label what the tool returned; URLs from search results become fetchable."""
        spec = TOOL_SPECS.get(tool)
        trust = prov.lowest() if spec is None or spec.output == "inherit" else spec.output
        prov.add(tool, trust)
        if tool == "web_search":
            prov.known_urls |= urls_in(output)
        if trust == Trust.PRIVATE:  # data flow: private tokens must not reach external queries unnoticed
            prov.private_tokens |= distinctive_tokens(output)
        return trust


# --------------------------------------------------------------------------- rule 4: rendering

_IMAGE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_REF_DEF = re.compile(r"(?m)^\s*\[[^\]]+\]:\s*\S+.*$")
_BARE = re.compile(r"(?<![`\w])(https?://[^\s<>()`]+)")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)


def defang_markdown(text: str) -> str:
    """Rule 4: images become text, links lose their activity (the URL stays visible as code),
    so rendering an answer makes no request. Citations like [1] are untouched."""
    if not text:
        return text
    text = _IMAGE.sub(lambda m: f"[изображение: {m.group(1) or 'без подписи'}] `{m.group(2)}`", text)
    text = _LINK.sub(lambda m: f"{m.group(1)} (`{m.group(2)}`)", text)
    text = _REF_DEF.sub(lambda m: f"`{m.group(0).strip()}`", text)
    # bare URLs outside code spans (URLs inside code are already inert)
    parts = _CODE.split(text)
    return "".join(p if i % 2 else _BARE.sub(lambda m: f"`{m.group(1)}`", p) for i, p in enumerate(parts))
