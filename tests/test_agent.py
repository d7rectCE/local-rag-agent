"""Э7: ReAct agent with tools, the CRAG relevance check, limits."""

from pathlib import Path

from rag_agent.engine import Engine
from tests.conftest import FakeLLM, FakeReranker

ROUTE = {"route": "corpus", "standalone_question": "Какой lr в эксперименте A?", "reasoning": "light", "aggregate": False}


def agent_engine(settings, fake_embedder, corpus: Path, reranker=None, *, crag=True, mode="auto", **llm) -> Engine:
    settings.agent.mode, settings.agent.crag = mode, crag
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings, route=llm.pop("route", ROUTE), **llm),
                 reranker=reranker or FakeReranker())
    eng.index_folder(corpus)
    return eng


def steps(ans, name="agent"):
    return [s.detail for s in ans.trace if s.name == name]


def test_simple_questions_skip_the_agent(settings, fake_embedder, corpus: Path):
    eng = agent_engine(settings, fake_embedder, corpus, route={**ROUTE, "reasoning": "none"})
    ans = eng.ask("Какой lr в эксперименте A?")
    assert "agent" not in eng.llm.kinds and not steps(ans)
    eng.close()


def test_agent_gathers_evidence_then_answers(settings, fake_embedder, corpus: Path):
    script = [{"thought": "ищу", "action": "search", "args": {"query": "lr learning rate", "file_type": "ipynb"}},
              {"thought": "хватает", "action": "answer", "args": {}}]
    eng = agent_engine(settings, fake_embedder, corpus, agent=script)
    ans = eng.ask("Какой lr в эксперименте A?")
    assert eng.llm.kinds == ["route", "agent", "agent", "answer"]
    seed, first = steps(ans)[:2]
    assert seed["step"] == 0 and seed["args"]["query"] == "Какой lr в эксперименте A?"
    assert first["action"] == "search" and first["verdict"] == "correct" and first["file_type"] == "ipynb"
    assert all(h.startswith("notebooks/") for h in first["hits"])
    assert steps(ans, "agent_stop")[0]["reason"] == "answer"
    assert ans.answerable and ans.sources and all(s.score >= settings.agent.crag_lower for s in ans.sources)
    eng.close()


def test_agent_refuses_when_nothing_relevant(settings, fake_embedder, corpus: Path):
    script = [{"thought": "", "action": "search", "args": {"query": "видеокарта"}},
              {"thought": "", "action": "search", "args": {"query": "GPU"}},
              {"thought": "", "action": "answer", "args": {}}]
    eng = agent_engine(settings, fake_embedder, corpus, reranker=FakeReranker(relevant=False), agent=script)
    ans = eng.ask("Какая видеокарта использовалась?")
    assert "answer" not in eng.llm.kinds and not ans.answerable  # CRAG: no answer from noise
    second = steps(ans)[1]
    assert second["verdict"] == "incorrect" and "refuse" in second["observation"]
    eng.close()


def test_agent_limits_and_repeated_calls(settings, fake_embedder, corpus: Path):
    same = {"thought": "", "action": "search", "args": {"query": "lr"}}
    eng = agent_engine(settings, fake_embedder, corpus, agent=[same] * 5)
    settings.agent.max_steps = 3
    ans = eng.ask("Какой lr в эксперименте A?")
    assert eng.llm.kinds.count("agent") == 3 and steps(ans, "agent_stop")[0]["reason"] == "max steps"
    assert steps(ans)[2].get("repeat") is True
    assert ans.answerable  # the evidence of the first search is still used
    eng.close()


def test_agent_read_list_and_exact_tools(settings, fake_embedder, corpus: Path):
    script = [{"thought": "", "action": "list_dir", "args": {"path": "notebooks"}},
              {"thought": "", "action": "read_file", "args": {"path": "notebooks/exp_a.ipynb", "cell": 2}},
              {"thought": "", "action": "exact_search", "args": {"name": "compute_f1"}},
              {"thought": "", "action": "answer", "args": {}}]
    eng = agent_engine(settings, fake_embedder, corpus, agent=script)
    ans = eng.ask("Какой lr в эксперименте A?")
    listed, read, exact = steps(ans)[1:4]
    assert listed["files"] == 1 and "exp_a.ipynb" in listed["observation"]
    assert read["hits"] == ["notebooks/exp_a.ipynb#cell2", "notebooks/exp_a.ipynb#cell2/out"]
    assert exact["definitions"] == 1 and exact["hits"][0].startswith("pkg/metrics.py")
    eng.close()


def test_agent_sql_tool(settings, fake_embedder, corpus: Path):
    settings.catalog.extract = True
    route = {**ROUTE, "aggregate": True}
    script = [{"thought": "", "action": "sql_query", "args": {"question": "лучший ROC-AUC"}},
              {"thought": "", "action": "answer", "args": {}}]
    sql = [{"sql": "SELECT path, cell, value FROM metrics WHERE name = 'roc_auc'", "reason": ""}]
    eng = agent_engine(settings, fake_embedder, corpus, route=route, agent=script, sql=sql)
    ans = eng.ask("В каком ноутбуке лучший ROC-AUC?")
    assert steps(ans)[1]["rows"] == 1 and ans.sources[0].file_type == "catalog"
    agent_calls = lambda: [c for c in eng.llm.calls if "исследовательский агент" in c[0]["content"]]  # noqa: E731
    assert "sql_query" in agent_calls()[0][0]["content"] and "начни с sql_query" in agent_calls()[0][1]["content"]
    settings.catalog.sql = False  # H5 switch: the tool is not offered
    eng.llm.agent = [{"thought": "", "action": "answer", "args": {}}]
    eng.ask("В каком ноутбуке лучший ROC-AUC?")
    assert "sql_query" not in agent_calls()[-1][0]["content"]
    eng.close()


def test_direct_path_crag_rewrites_then_refuses(settings, fake_embedder, corpus: Path):
    route = {**ROUTE, "reasoning": "none"}
    eng = agent_engine(settings, fake_embedder, corpus, reranker=FakeReranker(relevant=False), route=route)
    ans = eng.ask("Какая видеокарта?")
    assert eng.llm.kinds == ["route", "rewrite"] and not ans.answerable
    crag = steps(ans, "crag")[0]
    assert crag["verdict"] == "incorrect" and crag["rewrites"][0]["query"] == "learning rate lr"
    settings.agent.crag = False  # H6 ablation switch: answer from whatever was found
    eng.llm.kinds.clear()
    assert eng.ask("Какая видеокарта?").answerable and eng.llm.kinds == ["route", "answer"]
    eng.close()


def test_agent_metrics_in_evaluation(settings, fake_embedder, corpus: Path):
    from rag_agent.evaluation.ablation import AblationRun, AblationSpec, render_ablation
    from rag_agent.evaluation.runner import run_eval, run_judge, summarize
    from tests.test_evaluation import small_evalset

    eng = agent_engine(settings, fake_embedder, corpus, mode="always", route=None)  # standalone = the question
    es = small_evalset()
    es.items = [i for i in es.items if i.id in ("a", "b", "c")]
    results = run_eval(eng, es, retrieval_k=5, top_k=3)
    run_judge(results, es, FakeLLM(settings))
    by_id = {r.id: r for r in results}
    assert by_id["a"].agent_used and by_id["a"].agent_stop == "answer" and by_id["a"].crag_verdict == "correct"
    assert by_id["c"].crag_refused  # "Какая видеокарта?": nothing relevant -> refusal by the evaluator
    summary = summarize(results, es, n_boot=50)
    agent = summary["agent"]
    assert agent["used"] == 1.0 and agent["crag_refusals_q6"] == 1 and agent["latency_agent"]["p95"] is not None
    rows = [{"run": AblationRun(name="agent"), "summary": summary}]
    report = render_ablation(AblationSpec(name="h6", evalset="x", runs=[rows[0]["run"]]), es, rows, [by_id])
    assert "## Агент и проверка релевантности (H6, NFR2)" in report
    eng.close()
