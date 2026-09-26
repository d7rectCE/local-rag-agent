"""ReAct agent over the user's archive (ТЗ S7, Э7).

The router sends only aggregate and multi-step questions here; simple ones are
answered by direct retrieval. The agent gathers evidence with a small fixed set of
tools, each call generated under a JSON schema (constrained decoding: no malformed
calls), and stops with ``answer`` or ``refuse``. The final answer is written by the
same grounded generator as direct retrieval, from the gathered evidence, so every
claim still cites a fragment.

CRAG [Yan et al. 2024]: the fragments a search returns are scored by the
cross-encoder (the reranker, relevance in [0, 1]). A low score is reported to the
agent with the advice to rephrase or switch tools; if nothing relevant has been
found when the loop ends, the agent refuses instead of answering from noise.

Limits: number of steps, generated tokens of tool calls, wall-clock time. Every
step (thought, call, observation, relevance) is kept in the answer's trace.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Literal

from rag_agent.config import AgentConfig
from rag_agent.generation import Answer, TraceStep, generate_answer
from rag_agent.llm import LLMError
from rag_agent.retrieval import Hit
from rag_agent.schema import Node

if TYPE_CHECKING:
    from rag_agent.engine import Engine

log = logging.getLogger(__name__)

Verdict = Literal["correct", "ambiguous", "incorrect", "unknown"]

TOOLS = {
    "search": 'search {"query": "...", "file_type": ""} — семантический поиск фрагментов (file_type: py, ipynb, pdf, docx, txt, md, log или пусто)',
    "exact_search": 'exact_search {"name": "..."} — где определена функция или класс и где вызывается',
    "sql_query": 'sql_query {"question": "..."} — агрегатный вопрос к каталогу экспериментов: лучший, худший, сколько, все, среднее, по всем ноутбукам; вернёт таблицу',
    "read_file": 'read_file {"path": "...", "cell": N} — фрагменты файла; для ноутбука можно указать ячейку, для PDF — страницу через "page"',
    "list_dir": 'list_dir {"path": "..."} — файлы архива в папке (пустой path — корень)',
    "answer": "answer {} — улик достаточно, перейти к ответу",
    "refuse": 'refuse {"reason": "..."} — в архиве этого нет (только после нескольких разных попыток поиска)',
}

AGENT_PROMPT = """Ты — исследовательский агент по личному архиву пользователя: код на Python, Jupyter-ноутбуки, документы. Твоя задача — собрать в архиве доказательства для ответа на вопрос, вызывая инструменты по одному за шаг. Сам ответ напишет отдельный шаг по собранным фрагментам.

Инструменты:
{tools}

Правила:
- за шаг — один вызов; не повторяй вызов с теми же аргументами;
- если результат нерелевантен, переформулируй запрос или возьми другой инструмент;{sql_rule}
- как только улик достаточно, выбирай answer; не больше {max_steps} шагов.

Верни JSON: {{"thought": "коротко: что знаешь и что делаешь дальше", "action": "...", "args": {{...}}}}"""

REWRITE_PROMPT = """Найденные по запросу фрагменты архива нерелевантны. Перепиши поисковый запрос иначе: другими словами, с терминами, которые вероятно встречаются в коде, ноутбуках или документах (имена функций, метрик, датасетов), на русском или английском. Верни JSON: {"query": "..."}"""


def _schema(actions: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "action": {"type": "string", "enum": actions},
            "args": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}, "file_type": {"type": "string"}, "name": {"type": "string"},
                    "question": {"type": "string"}, "path": {"type": "string"}, "cell": {"type": "integer"},
                    "page": {"type": "integer"}, "reason": {"type": "string"},
                },
            },
        },
        "required": ["thought", "action", "args"],
    }


def rewrite_query(llm, question: str) -> str:
    """CRAG corrective action: the question rephrased as a search query."""
    try:
        data = llm.chat([{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": question}],
                        json_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                        max_tokens=120, purpose="rewrite").json()
        return str(data.get("query") or "").strip() or question
    except LLMError:
        return question


class RelevanceEvaluator:
    """CRAG-style evaluator: the cross-encoder's relevance of fragments to the question."""

    def __init__(self, engine: Engine, cfg: AgentConfig):
        self.engine, self.cfg = engine, cfg

    @property
    def available(self) -> bool:
        return self.cfg.crag

    def scores(self, query: str, nodes: list[Node]) -> list[float]:
        if not nodes or not self.available:
            return [1.0] * len(nodes)
        with_header = self.engine.settings.chunking.context_header
        return [float(s) for s in self.engine.reranker.score(query, [n.embedding_text(with_header) for n in nodes])]

    def verdict(self, best: float | None) -> Verdict:
        if not self.available or best is None:
            return "unknown"
        if best >= self.cfg.crag_upper:
            return "correct"
        return "incorrect" if best < self.cfg.crag_lower else "ambiguous"


class Agent:
    def __init__(self, engine: Engine, index, *, top_k: int, mode: str, sql: bool, reasoning_budget: int | None):
        self.engine, self.index = engine, index
        self.cfg: AgentConfig = engine.settings.agent
        self.top_k, self.mode, self.reasoning_budget = top_k, mode, reasoning_budget
        self.tools = [t for t in TOOLS if sql or t != "sql_query"]
        self.evaluator = RelevanceEvaluator(engine, self.cfg)
        self.evidence: dict[str, tuple[Node, float]] = {}  # node id -> (node, relevance to the question)
        self.steps: list[TraceStep] = []
        self.generated = 0

    # --- tools -----------------------------------------------------------------
    def _snippet(self, node: Node) -> str:
        text = " ".join((node.text or "").split())
        return text[: self.cfg.observation_chars] + ("…" if len(text) > self.cfg.observation_chars else "")

    def _add(self, question: str, nodes: list[Node]) -> list[float]:
        fresh = [n for n in nodes if n.id not in self.evidence]
        for n, s in zip(fresh, self.evaluator.scores(question, fresh)):
            self.evidence[n.id] = (n, s)
        return [self.evidence[n.id][1] for n in nodes]

    def _list_fragments(self, nodes: list[Node], scores: list[float]) -> str:
        return "\n".join(f"- [{n.id}] {n.file_path} — {n.location.describe() or n.title} (релевантность {s:.2f}): {self._snippet(n)}"
                         for n, s in zip(nodes, scores))

    def tool(self, question: str, action: str, args: dict) -> tuple[str, dict]:
        """Runs one tool; returns the observation for the agent and details for the trace."""
        if action == "search":
            query = str(args.get("query") or question)
            ft = str(args.get("file_type") or "").lstrip(".").lower() or None
            hits = self.engine.search(query, self.top_k, self.mode, file_types=[ft] if ft else None)
            nodes = [h.node for h in hits]
            scores = self._add(question, nodes)
            best = max(scores, default=None)
            verdict = self.evaluator.verdict(best)
            advice = {"incorrect": "\nОценка релевантности: низкая — переформулируй запрос или используй другой инструмент.",
                      "ambiguous": "\nОценка релевантности: средняя — возможно, стоит уточнить поиск."}.get(verdict, "")
            obs = (self._list_fragments(nodes, scores) or "ничего не найдено") + advice
            return obs, {"query": query, "file_type": ft, "hits": [n.id for n in nodes], "best": best, "verdict": verdict}
        if action == "exact_search":
            found = self.engine.lookup_symbol(str(args.get("name") or ""))
            nodes = []
            for d in found["definitions"][:3]:
                nodes += self.index.catalog.nodes_at(d["file_path"], line=d.get("line_start"), cell=d.get("cell"))[:1]
            for c in found["calls"][:5]:
                nodes += self.index.catalog.nodes_at(c["file_path"], line=c.get("line"), cell=c.get("cell"))[:1]
            nodes = list({n.id: n for n in nodes}.values())
            scores = self._add(question, nodes)
            obs = (f"определений {len(found['definitions'])}, вызовов {len(found['calls'])}\n"
                   + self._list_fragments(nodes, scores)) if nodes else "имя не найдено в коде"
            return obs, {"name": found["name"], "definitions": len(found["definitions"]), "calls": len(found["calls"]),
                         "hits": [n.id for n in nodes]}
        if action == "sql_query":
            res = self.engine.query_catalog(str(args.get("question") or question))
            detail = {"sql": res.sql, "rows": len(res.rows), "attempts": res.attempts, "error": res.error}
            if not (res.ok and res.rows):
                return f"запрос к каталогу не дал результата: {res.error or 'пусто'}", detail
            hits = self.engine._sql_hits(res, self.index)
            self.evidence[hits[0].node.id] = (hits[0].node, 1.0)  # the query result answers the question by construction
            self._add(question, [h.node for h in hits[1:]])
            return f"SQL: {res.sql}\n{res.markdown(max_rows=20)}", detail
        if action == "read_file":
            path = str(args.get("path") or "")
            nodes = [n for n in self.index.catalog.file_nodes(path) if n.node_type != "file"]
            if args.get("cell") is not None:
                nodes = [n for n in nodes if n.location.cell == args["cell"]]
            if args.get("page") is not None:
                nodes = [n for n in nodes if n.location.page is not None and n.location.page <= args["page"]
                         <= (n.location.page_end or n.location.page)]
            nodes = nodes[:6]
            if not nodes:
                return f"нет такого файла или фрагмента: {path}", {"path": path, "hits": []}
            scores = self._add(question, nodes)
            return self._list_fragments(nodes, scores), {"path": path, "hits": [n.id for n in nodes]}
        if action == "list_dir":
            prefix = str(args.get("path") or "").strip("/").replace("\\", "/")
            rows = self.index.catalog.query("SELECT path, file_type FROM files WHERE path LIKE ? ORDER BY path LIMIT 60",
                                            (f"{prefix}/%" if prefix else "%",))
            return "\n".join(f"- {r['path']} ({r['file_type']})" for r in rows) or "пусто", {"path": prefix, "files": len(rows)}
        return f"неизвестный инструмент {action}", {}

    # --- loop -------------------------------------------------------------------
    def run(self, question: str, *, aggregate: bool = False) -> Answer:
        t0 = time.perf_counter()
        sql_rule = ("\n- для вопросов «лучший / сколько / все / среднее» по многим экспериментам начинай с sql_query;"
                    if "sql_query" in self.tools else "")
        system = AGENT_PROMPT.format(tools="\n".join(f"- {TOOLS[t]}" for t in self.tools), max_steps=self.cfg.max_steps,
                                     sql_rule=sql_rule)
        hint = "\nПодсказка маршрутизатора: вопрос агрегатный, начни с sql_query." if aggregate and "sql_query" in self.tools else ""
        # step 0 without the model: a search by the question itself, so the agent starts from context
        t1 = time.perf_counter()
        seed_args = {"query": question}
        seed, extra = self.tool(question, "search", seed_args)
        self.steps.append(TraceStep(name="agent", duration_s=round(time.perf_counter() - t1, 3), detail={
            "step": 0, "thought": "начальный поиск по вопросу", "action": "search", "args": seed_args, **extra,
            "observation": seed[:1500]}))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"Вопрос: {question}{hint}\n\nРезультат начального поиска:\n{seed}"}]
        schema = _schema(self.tools)
        seen_calls: set[str] = {json.dumps(["search", seed_args], sort_keys=True, ensure_ascii=False)}
        refused, stop = None, "answer"
        low_searches = int(extra.get("verdict") == "incorrect")
        for step in range(1, self.cfg.max_steps + 1):
            if time.perf_counter() - t0 > self.cfg.timeout_s:
                stop = "timeout"
                break
            if self.generated > self.cfg.max_generated_tokens:
                stop = "token budget"
                break
            t1 = time.perf_counter()
            try:
                resp = self.engine.llm.chat(messages, json_schema=schema, max_tokens=400, purpose="agent")
                call = resp.json()
            except LLMError as exc:
                self.steps.append(TraceStep(name="agent", duration_s=round(time.perf_counter() - t1, 3),
                                            detail={"step": step, "error": str(exc)}))
                stop = "llm error"
                break
            self.generated += int(resp.usage.get("completion_tokens", 0))
            action, args = call.get("action"), call.get("args") or {}
            key = json.dumps([action, args], sort_keys=True, ensure_ascii=False)
            detail = {"step": step, "thought": call.get("thought", ""), "action": action, "args": args}
            if action in ("answer", "refuse"):
                if action == "refuse":
                    refused = str(args.get("reason") or "в архиве этого нет")
                self.steps.append(TraceStep(name="agent", duration_s=round(time.perf_counter() - t1, 3), detail=detail))
                stop = action
                break
            if key in seen_calls:
                observation, extra = "этот вызов уже был — используй его результат или сделай другой", {"repeat": True}
            else:
                seen_calls.add(key)
                observation, extra = self.tool(question, action, args)
            if extra.get("verdict") == "incorrect":
                low_searches += 1
                if low_searches > self.cfg.crag_retries and not self._relevant():
                    observation += "\nНесколько попыток поиска не нашли релевантного. Если в архиве этого нет — refuse."
            detail.update(extra)
            detail["observation"] = observation[:1500]
            self.steps.append(TraceStep(name="agent", duration_s=round(time.perf_counter() - t1, 3), detail=detail))
            messages.append({"role": "assistant", "content": json.dumps(call, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"Результат {action}:\n{observation}"})
        else:
            stop = "max steps"
        self.steps.append(TraceStep(name="agent_stop", duration_s=0.0, detail={
            "reason": stop, "steps": len([s for s in self.steps if s.name == "agent"]), "evidence": len(self.evidence),
            "generated_tokens": self.generated, "best_relevance": self._best()}))
        return self._finish(question, refused)

    def _best(self) -> float | None:
        return max((s for _, s in self.evidence.values()), default=None)

    def _relevant(self) -> bool:
        return self.evaluator.verdict(self._best()) in ("correct", "ambiguous", "unknown")

    def _finish(self, question: str, refused: str | None) -> Answer:
        # CRAG: an answer only from evidence that is relevant enough; otherwise an honest refusal
        if refused is not None or not self.evidence or not self._relevant():
            reason = refused or ("релевантных фрагментов не найдено" if self.evidence else "ничего не найдено")
            return Answer(question=question, answer=f"В архиве не нашлось ответа на этот вопрос ({reason}).",
                          answerable=False, grounded=True, trace=list(self.steps), model=self.engine.llm.name)
        ranked = sorted(self.evidence.values(), key=lambda ns: -ns[1])
        if self.evaluator.available:  # drop fragments the evaluator judged irrelevant
            ranked = [ns for ns in ranked if ns[1] >= self.cfg.crag_lower] or ranked[:1]
        hits = [Hit(node=n, score=s, rank=i) for i, (n, s) in enumerate(ranked[: self.cfg.evidence], start=1)]
        answer = generate_answer(question, hits, self.engine.llm, self.engine.settings.generation.max_source_chars,
                                 reasoning_budget=self.reasoning_budget)
        answer.trace[:0] = self.steps
        return answer
