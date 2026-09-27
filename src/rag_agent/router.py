"""Query router (ТЗ S7): decides whether a question needs the user's files and
rewrites follow-up questions into standalone ones using the dialogue history.

Routes:
    corpus  — about the user's files (code, experiments, results) -> grounded RAG
    general — general knowledge, new code, conversation -> answered by the LLM directly

Stage 7 adds agentic routes (SQL, multi-step) on top of the same interface.

In the chat a separate check decides whether the request needs running code (the task then
goes to the code agent). It is not a field of the router's answer: adding it to the router
prompt shifted the 9B router's other decisions on the eval sets (route accuracy 104 -> 100
of 115, twice the web false positives), so the evaluated router stays as it was.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Literal

from rag_agent.llm import BaseLLM, LLMError

Route = Literal["corpus", "general"]
RouteChoice = Literal["auto", "corpus", "general"]
ReasoningLevel = Literal["none", "light", "deep"]

ROUTER_PROMPT = """Ты — маршрутизатор запросов ассистента, у которого есть доступ к личному архиву исследовательских файлов пользователя: код на Python, Jupyter-ноутбуки с экспериментами и результатами, PDF, изображения.

Определи, нужен ли для ответа поиск по файлам пользователя:
- "corpus" — вопрос о содержимом файлов пользователя: его код, функции, эксперименты, результаты, метрики, гиперпараметры, датасеты, графики, заметки («где у меня…», «какой … я использовал», «в каком ноутбуке…», «покажи/найди…»), а также вопросы, ответ на которые может зависеть от его файлов.
- "general" — вопрос, для которого файлы пользователя не нужны: определения и теория, общие советы, написание нового кода с нуля, перевод, приветствие, разговор о возможностях ассистента.
Если сомневаешься, выбирай "corpus".

Кроме того, перепиши последний вопрос пользователя в самостоятельный вопрос: раскрой местоимения и отсылки к предыдущим репликам. Если истории нет или вопрос уже самостоятельный, верни его без изменений. Не отвечай на вопрос.

Оцени также, сколько рассуждений нужно перед ответом (reasoning):
- "none" — найти и пересказать факт: значение, метрику, параметр, файл, место в коде, что делает функция, определение; разговор. Вопросы «какой», «где», «в каком», «сколько», «что делает», «покажи» — обычно "none", даже если ответ состоит из нескольких чисел.
- "light" — явно просят сравнить или сопоставить несколько фактов, посчитать, перечислить шаги: «сравни», «чем отличается», «совпадает ли», «согласуется ли».
- "deep" — нужно сделать вывод, которого нет в файлах готовым: объяснить причину («почему», «чем вызвано», «из-за чего»), оценить корректность или значимость («корректно ли», «можно ли утверждать», «не случаен ли»), предложить, как проверить или исправить, спланировать эксперимент, разрешить противоречие.
Примеры (не из архива пользователя): «Какой оптимизатор в эксперименте A?» — "none"; «Сравни конфиги A и B» — "light"; «Почему метрика упала после смены препроцессинга?» — "deep"; «Не подогнан ли порог под валидацию?» — "deep".

Отметь, агрегатный ли вопрос (aggregate): true — нужно собрать или сравнить значения по многим экспериментам, запускам, файлам или функциям: лучший или худший по метрике, максимум, минимум, среднее, «сколько», «все», «список», «в каких», сортировка; false — вопрос об одном конкретном месте или значении.

Отметь, нужен ли интернет (web): true — ответ зависит от свежих или внешних сведений, которых не может быть в личном архиве: «последняя версия», «сейчас», «в этом году», «недавно», новости, релизы и изменения библиотек, факты о чужих статьях и моделях; false — ответ есть в файлах пользователя или в общих знаниях.

Верни JSON: {"route": "corpus" | "general", "standalone_question": "...", "reasoning": "none" | "light" | "deep", "aggregate": true | false, "web": true | false}"""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["corpus", "general"]},
        "standalone_question": {"type": "string"},
        "reasoning": {"type": "string", "enum": ["none", "light", "deep"]},
        "aggregate": {"type": "boolean"},
        "web": {"type": "boolean"},
    },
    "required": ["route", "standalone_question", "reasoning", "aggregate", "web"],
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
    reasoning: ReasoningLevel = "none"  # difficulty estimate (ТЗ ч.2 S15): sets the reasoning budget
    aggregate: bool = False  # over many experiments / files: the catalog is queried with SQL (Э6)
    web: bool = False  # needs fresh or external information (Э16)

    @property
    def needs_reasoning(self) -> bool:
        return self.reasoning != "none"


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
        level = data.get("reasoning")
        return RouteDecision(route, standalone, time.perf_counter() - t0,
                             reasoning=level if level in ("none", "light", "deep") else "none",
                             aggregate=bool(data.get("aggregate", False)), web=bool(data.get("web", False)))
    except LLMError:
        return RouteDecision("corpus", question, time.perf_counter() - t0, fallback=True)


# --- does the request need running code (the chat's hand-off to the code agent, Э15) ---------------

# cheap pre-filter: only requests with an action word reach the model check
CODE_HINT = re.compile(
    r"запус|выполни|прогон|построй|нарисуй|сохрани|создай|исправь|почини|посчитай|пересчитай|вычисли|собери|"
    r"сгенерируй|экспортир|конвертир|преобразуй|скачай|установи|протестируй|попробуй|"
    r"\b(run|execute|plot|fix|save|compute|create|generate|convert|download|install)\b",
    re.IGNORECASE)

CODE_PROMPT = """Реши, нужно ли для выполнения просьбы пользователя запустить код над его файлами или данными.

true — просят сделать действие: запустить скрипт, ноутбук или тесты; исправить ошибку, из-за которой падает запуск; построить график и сохранить картинку; посчитать что-то по файлам или логам и сохранить результат; создать, изменить или преобразовать файлы в папке; скачать данные.
false — достаточно ответа текстом: вопрос о содержимом файлов или о том, что в них есть («покажи график…», «какой результат…», «почему упало и как исправить?»); объяснение; просьба написать код прямо в ответе («напиши функцию…»).

Примеры: «Построй график loss по логу обучения» — true; «Запусти 02_baseline.ipynb и проверь, что он выполняется» — true; «Исправь ошибку в train.py» — true; «Напиши функцию бинарного поиска» — false; «Покажи график ROC-AUC из ноутбука» — false; «Почему упала ячейка и как это исправить?» — false.

Верни JSON: {"code": true | false}"""

CODE_SCHEMA = {"type": "object", "properties": {"code": {"type": "boolean"}}, "required": ["code"],
               "additionalProperties": False}


@dataclass
class CodeDecision:
    code: bool
    latency_s: float = 0.0
    checked: bool = False  # the model was asked (the pre-filter matched)


def needs_code(question: str, llm: BaseLLM) -> CodeDecision:
    """``question`` is the standalone question (follow-ups like "try again" are already rewritten)."""
    if not CODE_HINT.search(question or ""):
        return CodeDecision(False)
    t0 = time.perf_counter()
    try:
        data = llm.chat([{"role": "system", "content": CODE_PROMPT}, {"role": "user", "content": question}],
                        json_schema=CODE_SCHEMA, max_tokens=20, purpose="route_code").json()
        return CodeDecision(bool(data.get("code", False)), time.perf_counter() - t0, True)
    except LLMError:
        return CodeDecision(False, time.perf_counter() - t0, True)
