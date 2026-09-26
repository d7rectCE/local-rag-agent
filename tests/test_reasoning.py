"""Э12: reasoning mode (off / on / auto) and budget forcing."""

import json
from pathlib import Path

import httpx

from rag_agent.config import LLMConfig
from rag_agent.engine import Engine
from rag_agent.llm import OllamaLLM
from tests.conftest import FakeLLM


def engine_with(settings, fake_embedder, corpus: Path, route: dict | None = None) -> Engine:
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings, route=route))
    eng.index_folder(corpus)
    return eng


def test_reasoning_off_on(settings, fake_embedder, corpus: Path):
    eng = engine_with(settings, fake_embedder, corpus)
    off = eng.ask("learning rate", route="corpus", reasoning="off")
    assert off.reasoning is None and off.reasoning_level == "none" and eng.llm.kinds[-1] == "answer"
    on = eng.ask("learning rate", route="corpus", reasoning="on")
    assert on.reasoning and on.reasoning_tokens == 3 and eng.llm.kinds[-1] == "answer+reasoning"
    assert on.reasoning_level == "deep" and eng.llm.budgets == [settings.reasoning.budget_tokens["deep"]]
    eng.close()


def test_auto_budget_follows_the_router_estimate(settings, fake_embedder, corpus: Path):
    budgets = settings.reasoning.budget_tokens
    eng = engine_with(settings, fake_embedder, corpus)
    for level, question in [("deep", "Почему lr 0.05?"), ("light", "Сравни lr в двух ноутбуках"), ("none", "Какой lr?")]:
        eng.llm.route = {"route": "corpus", "standalone_question": question, "reasoning": level}
        ans = eng.ask(question, reasoning="auto")
        assert ans.reasoning_level == level and ans.trace[0].detail["reasoning"] == level
        assert (ans.reasoning is None) == (level == "none")
    assert eng.llm.budgets == [budgets["deep"], budgets["light"]]
    eng.close()


def test_on_mode_keeps_a_lighter_router_estimate(settings, fake_embedder, corpus: Path):
    route = {"route": "corpus", "standalone_question": "Сравни", "reasoning": "light"}
    eng = engine_with(settings, fake_embedder, corpus, route=route)
    assert eng.ask("Сравни", reasoning="on").reasoning_level == "light"
    eng.llm.route = {**route, "reasoning": "none"}  # "on" still reasons when the router says it is not needed
    assert eng.ask("Сравни", reasoning="on").reasoning_level == "deep"
    eng.close()


def test_general_route_can_reason(settings, fake_embedder, corpus: Path):
    route = {"route": "general", "standalone_question": "Сравни L1 и L2", "reasoning": "deep"}
    eng = engine_with(settings, fake_embedder, corpus, route=route)
    ans = eng.ask("Сравни L1 и L2", reasoning="auto")
    assert ans.route == "general" and ans.reasoning and eng.llm.kinds[-1] == "general+reasoning"
    eng.close()


def test_ungrounded_fast_answer_escalates_to_reasoning(settings, fake_embedder, corpus: Path):
    route = {"route": "corpus", "standalone_question": "Какой lr?", "reasoning": "none"}
    eng = engine_with(settings, fake_embedder, corpus, route=route)
    eng.llm.reply = {"answerable": True, "answer": "Learning rate был 0.05.", "general": ""}  # no citation
    ans = eng.ask("Какой lr?", reasoning="auto")
    assert eng.llm.kinds == ["route", "answer", "answer+reasoning"]
    assert ans.reasoning_level == "deep" and "escalate" in [s.name for s in ans.trace]
    settings.reasoning.escalate = False
    eng.llm.kinds.clear()
    eng.ask("Какой lr?", reasoning="auto")
    assert eng.llm.kinds == ["route", "answer"]
    eng.close()


def ollama_stream(chunks: list[dict]) -> bytes:
    return b"\n".join(json.dumps(c).encode() for c in chunks)


def test_budget_forcing_cuts_reasoning_and_forces_the_answer():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if body["stream"]:  # endless thinking: never reaches the answer on its own
            return httpx.Response(200, content=ollama_stream([{"message": {"thinking": f"t{i} "}} for i in range(50)]))
        return httpx.Response(200, json={"message": {"content": '{"answer": "391"}'}, "prompt_eval_count": 9, "eval_count": 5})

    llm = OllamaLLM(LLMConfig())
    llm._http = httpx.Client(transport=httpx.MockTransport(handler))
    resp = llm.chat_reasoning([{"role": "user", "content": "17*23?"}], budget_tokens=10,
                              json_schema={"type": "object"})
    assert resp.content == '{"answer": "391"}' and resp.thinking_truncated
    assert resp.usage["thinking_tokens"] == 10 and resp.thinking.startswith("t0 t1")
    forced = calls[1]
    assert forced["think"] is False and forced["stream"] is False
    assert "<черновик>" in forced["messages"][-2]["content"] and "Больше не рассуждай" in forced["messages"][-1]["content"]


def test_reasoning_within_budget_is_not_forced():
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [{"message": {"thinking": "short "}}, {"message": {"content": '{"answer": "ok"}'}},
                  {"done": True, "done_reason": "stop", "prompt_eval_count": 4, "eval_count": 7}]
        return httpx.Response(200, content=ollama_stream(chunks))

    llm = OllamaLLM(LLMConfig())
    llm._http = httpx.Client(transport=httpx.MockTransport(handler))
    resp = llm.chat_reasoning([{"role": "user", "content": "q"}], budget_tokens=100)
    assert resp.content == '{"answer": "ok"}' and not resp.thinking_truncated and resp.usage["thinking_tokens"] == 1
