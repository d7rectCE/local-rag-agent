"""Query router (ТЗ S7): decides whether a question needs the user's files and
rewrites follow-up questions into standalone ones using the dialogue history.

Routes:
    corpus  — about the user's files (code, experiments, results) -> grounded RAG
    general — general knowledge, new code, conversation -> answered by the LLM directly

Stage 7 adds agentic routes (SQL, multi-step) on top of the same interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from rag_agent.llm import BaseLLM, LLMError

Route = Literal["corpus", "general"]
RouteChoice = Literal["auto", "corpus", "general"]

ROUTER_PROMPT = """Ты — маршрутизатор запросов ассистента, у которого есть доступ к личному архиву исследовательских файлов пользователя: код на Python, Jupyter-ноутбуки с экспериментами и результатами, PDF, изображения.

Определи, нужен ли для ответа поиск по файлам пользователя:
- "corpus" — вопрос о содержимом файлов пользователя: его код, функции, эксперименты, результаты, метрики, гиперпараметры, датасеты, графики, заметки («где у меня…», «какой … я использовал», «в каком ноутбуке…», «покажи/найди…»), а также вопросы, ответ на которые может зависеть от его файлов.
- "general" — вопрос, для которого файлы пользователя не нужны: определения и теория, общие советы, написание нового кода с нуля, перевод, приветствие, разговор о возможностях ассистента.
Если сомневаешься, выбирай "corpus".

Кроме того, перепиши последний вопрос пользователя в самостоятельный вопрос: раскрой местоимения и отсылки к предыдущим репликам. Если истории нет или вопрос уже самостоятельный, верни его без изменений. Не отвечай на вопрос.

Верни JSON: {"route": "corpus" | "general", "standalone_question": "..."}"""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["corpus", "general"]},
        "standalone_question": {"type": "string"},
    },
    "required": ["route", "standalone_question"],
    "additionalProperties": False,
}

HISTORY_TURNS = 6
HISTORY_CHARS = 1500


@dataclass
class RouteDecision:
    route: Route
    standalone_question: str
    latency_s: float = 0.0
    fallback: bool = False  # router failed and the default route was used


def trim_history(history: list[dict] | None) -> list[dict]:
    """Last few user/assistant turns, each truncated."""
    turns = [
        {"role": m["role"], "content": str(m.get("content", ""))[:HISTORY_CHARS]}
        for m in (history or [])
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ]
    return turns[-HISTORY_TURNS:]


def route_question(question: str, history: list[dict] | None, llm: BaseLLM) -> RouteDecision:
    turns = trim_history(history)
    dialogue = "\n".join(f"{'Пользователь' if t['role'] == 'user' else 'Ассистент'}: {t['content']}" for t in turns)
    user = (f"История диалога:\n{dialogue}\n\n" if dialogue else "История диалога: пусто\n\n") + f"Последний вопрос: {question}"
    t0 = time.perf_counter()
    try:
        resp = llm.chat(
            [{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": user}],
            json_schema=ROUTER_SCHEMA,
            max_tokens=300,
            purpose="route",
        )
        data = resp.json()
        route = data.get("route")
        standalone = str(data.get("standalone_question") or "").strip() or question
        if route not in ("corpus", "general"):
            raise LLMError(f"unknown route {route!r}")
        return RouteDecision(route, standalone, time.perf_counter() - t0)
    except LLMError:
        return RouteDecision("corpus", question, time.perf_counter() - t0, fallback=True)
