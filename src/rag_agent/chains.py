"""Chains of steps in one message: "[IMG] мерседес в горах -> [VID] анимируй, как она едет -> [TEXT] расскажи,
что сделал". A tag says what a step is ([TEXT], [CODE], [IMG], [VID], or their Russian names); "->" (or →)
before a tag starts the next step. The canvas of the UI builds the same list of steps as a graph.

Each step runs as the chat would run it, with what the earlier steps made: a picture of the chain is the
source of the next [IMG] (an edit) and the first frame of the next [VID]; every result is a turn of the
history, so [TEXT] and [CODE] know what was done."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
        steps.append(ChainStep(kind=kind, text=text or DEFAULT_TEXT[kind]))
    return steps or None


def describe(steps: list[ChainStep]) -> str:
    return " → ".join(LABELS[s.kind] for s in steps)
