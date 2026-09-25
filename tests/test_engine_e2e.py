"""End-to-end over a synthetic corpus with fake models (runs in CI without GPU or private data)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_agent.api import create_app
from rag_agent.engine import Engine
from tests.conftest import FakeLLM


@pytest.fixture
def engine(settings, fake_embedder):
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    yield eng
    eng.close()


def test_index_and_ask(engine: Engine, corpus: Path):
    progress = engine.index_folder(corpus)
    assert progress.state == "done"
    assert progress.files_total == 2 and progress.new == 2 and progress.errors == 0

    stats = engine.index.catalog.stats()
    assert stats["n_files"] == 2
    assert stats["nodes"]["cell_output"] == 3
    assert engine.index.store.count() > 0

    ans = engine.ask("какой learning rate в эксперименте A?", top_k=4)
    assert ans.answerable and ans.grounded
    assert ans.citations[0].n == 1
    assert ans.sources[0].cited
    assert [s.name for s in ans.trace] == ["route", "retrieve", "generate"]
    assert ans.route == "corpus" and ans.standalone_question is None
    prompt = engine.llm.calls[-1][1]["content"]
    assert '<source id="1"' in prompt and "Вопрос:" in prompt


def test_sparse_and_hybrid_modes(engine: Engine, corpus: Path):
    engine.index_folder(corpus)
    for mode in ("sparse", "hybrid"):
        ans = engine.ask("compute_f1 precision recall", top_k=3, mode=mode)
        assert ans.sources, mode
        assert ans.sources[0].file_path == "pkg/metrics.py", mode


def test_incremental_reindex(engine: Engine, corpus: Path):
    engine.index_folder(corpus)
    (corpus / "pkg" / "metrics.py").write_text("def new_metric():\n    return 1\n", encoding="utf-8")
    (corpus / "notebooks" / "exp_a.ipynb").unlink()
    (corpus / "pkg" / "extra.py").write_text("X = 1\n", encoding="utf-8")
    (corpus / "pkg" / "broken.ipynb").write_text("{not json", encoding="utf-8")

    progress = engine.index_folder(corpus)
    assert (progress.new, progress.changed, progress.unchanged, progress.deleted, progress.errors) == (2, 1, 0, 1, 1)
    problems = engine.index.catalog.problems()
    assert problems[0]["path"] == "pkg/broken.ipynb" and problems[0]["status"] == "error"

    progress = engine.index_folder(corpus)
    assert progress.unchanged == 3 and progress.new == 0 and progress.changed == 0
    # vectors of deleted / replaced files are gone
    hits = engine.ask("roc_auc 0.9925", top_k=10).sources
    assert all(s.file_path != "notebooks/exp_a.ipynb" for s in hits)


def test_refusal_when_model_says_unanswerable(settings, fake_embedder, corpus: Path):
    llm = FakeLLM(settings, reply={"answerable": False, "answer": "В фрагментах нет данных о GPU."})
    eng = Engine(settings, embedder=fake_embedder, llm=llm)
    eng.index_folder(corpus)
    ans = eng.ask("на какой GPU обучали?")
    assert not ans.answerable and ans.grounded and not ans.citations
    eng.close()


def test_api_roundtrip(settings, fake_embedder, corpus: Path):
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    with TestClient(create_app(eng)) as client:
        # nothing indexed yet: general answer in auto mode, 404 when the user's files are required
        assert client.post("/ask", json={"question": "x"}).json()["route"] == "general"
        assert client.post("/ask", json={"question": "x", "route": "corpus"}).status_code == 404
        r = client.post("/index", json={"root": str(corpus)})
        assert r.status_code == 202
        eng._thread.join(timeout=30)
        assert client.get("/index/progress").json()["state"] == "done"
        body = client.post("/ask", json={"question": "learning rate", "top_k": 3}).json()
        assert body["citations"] and body["sources"]
        assert client.get("/status").json()["corpus"]["stats"]["n_files"] == 2
        view = client.get("/file", params={"path": "pkg/metrics.py"}).json()
        assert view[0]["node_type"] == "file"
        assert client.post("/index", json={"root": str(corpus / "missing")}).status_code == 404


def test_general_route_answers_without_retrieval(settings, fake_embedder, corpus: Path):
    llm = FakeLLM(settings, route={"route": "general", "standalone_question": "Что такое ROC-AUC?"},
                  general="ROC-AUC — площадь под ROC-кривой.")
    eng = Engine(settings, embedder=fake_embedder, llm=llm)
    eng.index_folder(corpus)
    ans = eng.ask("Что такое ROC-AUC?")
    assert ans.route == "general" and ans.answer.startswith("ROC-AUC")
    assert not ans.sources and not ans.citations and ans.grounded is None
    assert llm.kinds == ["route", "general"]
    eng.close()


def test_forced_route_skips_router_without_history(engine: Engine, corpus: Path):
    engine.index_folder(corpus)
    engine.ask("Что такое F1?", route="general")
    engine.ask("learning rate", route="corpus")
    assert engine.llm.kinds == ["general", "answer"]


def test_follow_up_is_rewritten_with_history(settings, fake_embedder, corpus: Path):
    llm = FakeLLM(settings, route={"route": "corpus", "standalone_question": "Где реализована функция compute_f1?"})
    eng = Engine(settings, embedder=fake_embedder, llm=llm)
    eng.index_folder(corpus)
    history = [{"role": "user", "content": "Есть ли у меня функция compute_f1?"},
               {"role": "assistant", "content": "Да [1]."}]
    ans = eng.ask("А где она?", history=history)
    assert ans.standalone_question == "Где реализована функция compute_f1?"
    assert ans.sources[0].file_path == "pkg/metrics.py"  # retrieval used the rewritten question
    assert "Есть ли у меня функция compute_f1?" in llm.calls[0][-1]["content"]  # router saw the history
    eng.close()


def test_router_failure_falls_back_to_corpus(settings, fake_embedder, corpus: Path):
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings, route="not json"))
    eng.index_folder(corpus)
    ans = eng.ask("learning rate")
    assert ans.route == "corpus" and ans.trace[0].detail["fallback"] is True
    eng.close()


def test_without_index_general_answer_with_notice(settings, fake_embedder):
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    ans = eng.ask("Какой learning rate я использовал?")
    assert ans.route == "general" and ans.notice and "не проиндексирована" in ans.notice
    eng.close()


def test_router_general_is_overridden_when_question_names_corpus_code(settings, fake_embedder, corpus: Path):
    llm = FakeLLM(settings, route={"route": "general", "standalone_question": "What does compute_f1 return?"})
    eng = Engine(settings, embedder=fake_embedder, llm=llm)
    eng.index_folder(corpus)
    ans = eng.ask("What does compute_f1 return?")
    assert ans.route == "corpus"
    assert ans.trace[0].detail["override"] == "corpus" and ans.trace[0].detail["corpus_mentions"] == ["compute_f1"]
    # a file name counts as well; ordinary words do not
    llm.route = {"route": "general", "standalone_question": "Explain metrics in exp_a"}
    assert eng.ask("Explain metrics in exp_a").route == "corpus"
    llm.route = {"route": "general", "standalone_question": "What is recall?"}
    assert eng.ask("What is recall?").route == "general"
    # ordinary words that happen to be method or file names do not count; a class name does
    llm.route = {"route": "general", "standalone_question": "How do I fit a model with good metrics?"}
    assert eng.ask("How do I fit a model with good metrics?").route == "general"
    llm.route = {"route": "general", "standalone_question": "What does Trainer do?"}
    assert eng.ask("What does Trainer do?").route == "corpus"
    eng.close()
