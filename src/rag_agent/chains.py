"""Chains of steps in one message: "[IMG] мерседес в горах -> [VID] анимируй, как она едет -> [TEXT] расскажи,
что сделал". A tag says what a step is ([TEXT], [CODE], [IMG], [VID], or their Russian names); "->" (or →)
before a tag starts the next step. The canvas of the UI builds the same list of steps as a graph.

Each step runs as the chat would run it, with what the earlier steps made: a picture of the chain is the
source of the next [IMG] (an edit) and the first frame of the next [VID]; every result is a turn of the
history, so [TEXT] and [CODE] know what was done."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

KINDS = {"text": "text", "текст": "text", "code": "code", "код": "code", "img": "img", "image": "img",
         "картинка": "img", "изображение": "img", "фото": "img", "vid": "vid", "video": "vid", "видео": "vid"}
TAG = re.compile(r"\[\s*(" + "|".join(KINDS) + r")\s*\]", re.IGNORECASE)
# an arrow starts a new step only before a tag: "f -> int" inside a task for the code agent stays whole
ARROW = re.compile(r"\s*(?:->|→|=>)\s*(?=\[\s*(?:" + "|".join(KINDS) + r")\s*\])", re.IGNORECASE)
DEFAULT_TEXT = {"text": "Расскажи, что сделано.", "code": "Выполни задачу.", "img": "Нарисуй картинку.",
                "vid": "Оживи картинку."}
LABELS = {"text": "текст", "code": "код", "img": "картинка", "vid": "видео"}


@dataclass
class ChainStep:
    kind: str  # text | code | img | vid
    text: str
    id: str = ""
    inputs: list[str] = field(default_factory=list)  # the steps whose results this one takes (the links)
    options: dict = field(default_factory=dict)  # the canvas: model, size (pictures), seconds (videos)


def parse_chain(message: str) -> list[ChainStep] | None:
    """The steps of a message with tags; None for a plain message (the chat routes it as usual)."""
    if not TAG.search(message or ""):
        return None
    steps = []
    for part in ARROW.split(message.strip()):
        part = part.strip()
        if not part:
            continue
        m = TAG.search(part)
        kind = KINDS[m.group(1).lower()] if m else "text"
        text = TAG.sub("", part, count=1).strip(" :—-") if m else part
        n = len(steps) + 1
        steps.append(ChainStep(kind=kind, text=text or DEFAULT_TEXT[kind], id=str(n),
                               inputs=[str(n - 1)] if n > 1 else []))
    return steps or None


MAX_NODES = 24
OPTION_KEYS = {"model", "size", "seconds"}


class GraphError(ValueError):
    pass


def graph_steps(nodes: list[dict]) -> list[ChainStep]:
    """The canvas: nodes {id, kind, text, inputs, options} in an order that runs every node after the nodes
    linked into it (a cycle is refused)."""
    if not nodes:
        raise GraphError("холст пуст")
    if len(nodes) > MAX_NODES:
        raise GraphError(f"на холсте больше {MAX_NODES} узлов")
    steps: dict[str, ChainStep] = {}
    for node in nodes:
        nid, kind = str(node.get("id") or "")[:40], KINDS.get(str(node.get("kind") or "").lower())
        if not nid or nid in steps:
            raise GraphError("у каждого узла должен быть свой id")
        if kind is None:
            raise GraphError(f"неизвестный вид узла {node.get('kind')!r}")
        text = str(node.get("text") or "").strip()[:4000] or DEFAULT_TEXT[kind]
        options = {k: v for k, v in (node.get("options") or {}).items() if k in OPTION_KEYS and v not in (None, "")}
        steps[nid] = ChainStep(kind=kind, text=text, id=nid, inputs=[str(i) for i in node.get("inputs") or []],
                               options=options)
    for s in steps.values():
        unknown = [i for i in s.inputs if i not in steps]
        if unknown:
            raise GraphError(f"связь с несуществующим узлом {unknown[0]}")
    order, done = [], set()
    pending = list(steps.values())
    while pending:  # Kahn: the nodes whose inputs are done, in the canvas's order
        ready = [s for s in pending if all(i in done for i in s.inputs)]
        if not ready:
            raise GraphError("на холсте цикл: узлы ссылаются друг на друга по кругу")
        for s in ready:
            order.append(s)
            done.add(s.id)
        pending = [s for s in pending if s.id not in done]
    return order


def describe(steps: list[ChainStep]) -> str:
    return " → ".join(LABELS[s.kind] for s in steps)
