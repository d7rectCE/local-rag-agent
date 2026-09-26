"""Э6: catalog of experiments (LLM extraction with grounding), analytics database, SQL tool."""

import sqlite3
from pathlib import Path

import pytest

from rag_agent.engine import Engine
from rag_agent.structured.analytics import ANALYTICS_FILE
from rag_agent.structured.extract import normalize_metric, value_in
from rag_agent.structured.sql import SQLValidationError, run_readonly, validate_sql
from tests.conftest import FakeLLM


def catalog_engine(settings, fake_embedder, corpus: Path, **llm) -> Engine:
    settings.catalog.extract = True
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings, **llm))
    eng.index_folder(corpus)
    return eng


def test_value_matching_and_names():
    assert value_in("roc_auc 0.9925", 0.9925) and value_in("0.97777", 0.9778) and value_in("97.78 %", 0.9778)
    assert not value_in("roc_auc 0.9925", 0.99) and not value_in("lr = 0.05", 0.5)
    assert value_in("learning_rate_init=3e-3", 0.003)
    assert normalize_metric("ROC-AUC", "validation") == ("roc_auc", "val")
    assert normalize_metric("val_acc", "") == ("accuracy", "val")
    assert normalize_metric("F1 macro", "Test") == ("f1", "test")


def test_extraction_is_grounded_and_cached(settings, fake_embedder, corpus: Path):
    eng = catalog_engine(settings, fake_embedder, corpus)
    assert eng.progress.state == "done" and eng.progress.catalog["notebooks"] == 1
    [exp] = eng.experiments()
    assert (exp["dataset"], exp["task"], exp["model"]) == ("breast_cancer", "classification", "HistGradientBoosting")
    # the printed ROC-AUC stays; the accuracy printed nowhere and max_depth absent from the code are dropped
    assert [(m["name"], m["value"], m["split"], m["cell"]) for m in exp["metrics"]] == [("roc_auc", 0.9925, "val", 2)]
    assert [(h["name"], h["value"], h["cell"]) for h in exp["hyperparameters"]] == [("lr", "0.05", 2)]
    assert eng.catalog_summary()["dropped"] == 2

    calls = len(eng.llm.kinds)
    eng.index_folder(corpus)  # unchanged notebook: no new extraction call
    assert eng.llm.kinds.count("extract") == 1 and len(eng.llm.kinds) == calls
    eng.close()


def test_analytics_database(settings, fake_embedder, corpus: Path):
    eng = catalog_engine(settings, fake_embedder, corpus)
    db = eng.index.dir / ANALYTICS_FILE
    q = lambda sql: run_readonly(db, sql, 5)[1]  # noqa: E731
    assert q("SELECT path, title, n_cells FROM notebooks") == [["notebooks/exp_a.ipynb", "Experiment A", 5]]
    assert q("SELECT cell, error_name FROM cells WHERE has_error = 1") == [[4, "ZeroDivisionError"]]
    assert {r[0] for r in q("SELECT name FROM functions")} >= {"compute_f1", "Trainer", "fit"}
    assert q("SELECT e.dataset, m.name, m.value FROM metrics m JOIN experiments e ON e.id = m.experiment_id") == [
        ["breast_cancer", "roc_auc", 0.9925]]
    assert q("SELECT name, value_num FROM hyperparameters") == [["lr", 0.05]]
    eng.close()


@pytest.mark.parametrize("sql", [
    "DELETE FROM files", "DROP TABLE files", "SELECT 1; DELETE FROM files", "PRAGMA table_info(files)",
    "INSERT INTO files VALUES (1, 2, 3, 4, 5)", "SELECT * FROM x_state", "ATTACH DATABASE 'x.db' AS x",
])
def test_validation_rejects_anything_but_select(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql, max_rows=10)


def test_limit_is_forced_and_connection_is_read_only(settings, fake_embedder, corpus: Path):
    eng = catalog_engine(settings, fake_embedder, corpus)
    db = eng.index.dir / ANALYTICS_FILE
    final = validate_sql("SELECT name FROM functions ORDER BY name", max_rows=2)
    assert final.endswith("LIMIT 3")
    assert len(run_readonly(db, final, 5)[1]) == 3
    # even a statement that slipped past validation cannot write: read-only file + authorizer
    for sql in ("DELETE FROM files", "CREATE TABLE t (x)", "PRAGMA writable_schema = 1"):
        with pytest.raises(sqlite3.DatabaseError):
            run_readonly(db, sql, 5)
    with pytest.raises(sqlite3.OperationalError, match="таймаут"):
        run_readonly(db, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT MAX(x) FROM c", 0.05)
    eng.close()


def test_sql_tool_corrects_itself(settings, fake_embedder, corpus: Path):
    sql = [
        {"sql": "SELECT * FROM runs", "reason": ""},  # unknown table
        {"sql": "SELECT path, cell, value FROM metrics WHERE name = 'accuracy'", "reason": ""},  # empty
        {"sql": "SELECT path, cell, value FROM metrics WHERE name = 'roc_auc'", "reason": ""},
    ]
    eng = catalog_engine(settings, fake_embedder, corpus, sql=sql)
    res = eng.query_catalog("Какой лучший ROC-AUC?")
    assert res.ok and res.attempts == 3 and res.rows == [["notebooks/exp_a.ipynb", 2, 0.9925]]
    assert "metrics.name = 'roc_auc'" in res.hints and "metrics" in res.tables and "experiments" in res.tables
    assert res.provenance() == [("notebooks/exp_a.ipynb", 2)]
    eng.close()


def test_aggregate_question_uses_the_catalog(settings, fake_embedder, corpus: Path):
    route = {"route": "corpus", "standalone_question": "В каком ноутбуке лучший ROC-AUC?", "reasoning": "none",
             "aggregate": True}
    sql = [{"sql": "SELECT path, cell, value FROM metrics WHERE name = 'roc_auc' ORDER BY value DESC", "reason": ""}]
    eng = catalog_engine(settings, fake_embedder, corpus, route=route, sql=sql)
    ans = eng.ask("В каком ноутбуке лучший ROC-AUC?")
    assert [s.name for s in ans.trace][:3] == ["route", "retrieve", "sql"]
    assert ans.sources[0].file_path.startswith("каталог") and "0.9925" in ans.sources[0].text
    assert ans.sources[1].node_id == "notebooks/exp_a.ipynb#cell2/out"  # the cell the value comes from
    settings.catalog.sql = False  # H5 ablation switch
    eng.llm.kinds.clear()
    eng.ask("В каком ноутбуке лучший ROC-AUC?")
    assert "sql" not in eng.llm.kinds
    eng.close()


def test_execution_accuracy_and_h5_section(settings, fake_embedder, corpus: Path):
    from rag_agent.evaluation.ablation import AblationRun, AblationSpec, render_ablation
    from rag_agent.evaluation.dataset import EvalSet
    from rag_agent.evaluation.runner import run_eval, summarize

    route = {"route": "corpus", "standalone_question": "q", "reasoning": "none", "aggregate": True}
    sql = [{"sql": "SELECT path, cell, value FROM metrics WHERE name = 'roc_auc'", "reason": ""}]
    eng = catalog_engine(settings, fake_embedder, corpus, route=route, sql=sql)
    src = [{"file": "notebooks/exp_a.ipynb", "cell": 2}]
    es = EvalSet.model_validate({"name": "t", "items": [
        {"id": "a", "class": "Q3", "question": "Лучший ROC-AUC?", "answer": "0.9925", "sources": src,
         "sql": "SELECT value FROM metrics WHERE name = 'roc_auc'"},  # extra path/cell columns are fine
        {"id": "b", "class": "Q3", "question": "Сколько ноутбуков?", "answer": "1", "sources": src,
         "sql": "SELECT COUNT(*) FROM notebooks"},
    ]})
    results = run_eval(eng, es, retrieval_k=5, top_k=3)
    assert [(r.sql_used, r.sql_ex) for r in results] == [(True, True), (True, False)]
    summary = summarize(results, es, n_boot=50)
    assert summary["sql"]["used_q3"] == 1.0 and summary["sql"]["execution_accuracy"]["mean"] == 0.5
    rows = [{"run": AblationRun(name=n), "summary": summary} for n in ("sql", "same")]
    per_item = [{r.id: r for r in results}] * 2
    report = render_ablation(AblationSpec(name="h5", evalset="x", runs=[r["run"] for r in rows]), es, rows, per_item)
    assert "## SQL по каталогу экспериментов (H5)" in report and "0.500 (n=2)" in report
    eng.close()


def test_sql_sources_are_numbered_once(settings, fake_embedder, corpus: Path):
    route = {"route": "corpus", "standalone_question": "q", "reasoning": "none", "aggregate": True}
    sql = [{"sql": "SELECT path, cell, value FROM metrics", "reason": ""}]
    eng = catalog_engine(settings, fake_embedder, corpus, route=route, sql=sql)
    ans = eng.ask("Лучший ROC-AUC?", top_k=3)
    assert [s.n for s in ans.sources] == list(range(1, len(ans.sources) + 1))
    assert len({s.node_id for s in ans.sources}) == len(ans.sources)
    eng.close()
