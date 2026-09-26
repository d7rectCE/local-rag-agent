from pathlib import Path

import pytest

from rag_agent.config import REPO_ROOT
from rag_agent.engine import Engine
from rag_agent.evaluation.dataset import EvalSet, SourceRef, load_evalset, validate_evalset
from rag_agent.evaluation.metrics import bootstrap_ci, precision_recall, retrieval_metrics
from rag_agent.evaluation.report import render_report
from rag_agent.evaluation.runner import run_eval, run_judge, summarize
from rag_agent.schema import FileType, Location, Node, NodeType
from tests.conftest import FakeLLM


def node(file: str, cell=None, lines=None, node_id=None) -> Node:
    loc = Location(cell=cell, line_start=lines[0] if lines else None, line_end=lines[1] if lines else None)
    return Node(id=node_id or f"{file}#{cell}{lines}", file_path=file, file_type=FileType.PY,
                node_type=NodeType.CODE_CHUNK, location=loc)


def test_source_ref_matching_is_chunking_independent():
    ref = SourceRef(file="m.py", lines=(13, 27))
    assert ref.matches(node("m.py", lines=(1, 60)))  # whole-file line window
    assert ref.matches(node("m.py", lines=(20, 40)))  # partial overlap
    assert not ref.matches(node("m.py", lines=(28, 40)))
    assert not ref.matches(node("other.py", lines=(1, 60)))
    cell = SourceRef(file="nb.ipynb", cell=5)
    assert cell.matches(node("nb.ipynb", cell=5)) and not cell.matches(node("nb.ipynb", cell=6))


def test_retrieval_metrics_do_not_double_count_overlapping_chunks():
    refs = [SourceRef(file="a.py", lines=(10, 20)), SourceRef(file="b.ipynb", cell=2)]
    retrieved = [
        node("x.py", lines=(1, 5)),
        node("a.py", lines=(1, 15)),
        node("a.py", lines=(12, 30)),  # same span again: no extra gain
        node("b.ipynb", cell=2),
    ]
    m = retrieval_metrics(retrieved, refs)
    assert m["recall@1"] == 0 and m["recall@3"] == 0.5 and m["recall@5"] == 1.0
    assert m["mrr"] == 0.5
    assert 0 < m["ndcg@10"] < 1
    perfect = retrieval_metrics([node("a.py", lines=(10, 20)), node("b.ipynb", cell=2)], refs)
    assert perfect["ndcg@10"] == pytest.approx(1.0)


def test_precision_recall_and_bootstrap():
    pr = precision_recall([True, True, False, False], [True, False, True, False])
    assert pr["precision"] == 0.5 and pr["recall"] == 0.5
    lo, hi = bootstrap_ci([0, 1, 1, 1, 0, 1], n_boot=200)
    assert 0 <= lo <= hi <= 1


def test_demo_evalset_is_well_formed():
    es = load_evalset(REPO_ROOT / "evalsets" / "demo_v1.yaml")
    assert es.corpus_root() == (REPO_ROOT / "demo_corpus").resolve()
    classes = {i.cls for i in es.items}
    assert classes == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "G"}
    assert all(not i.sources for i in es.items if i.cls in ("Q6", "G"))


def small_evalset() -> EvalSet:
    return EvalSet.model_validate({
        "name": "t",
        "items": [
            {"id": "a", "class": "Q1", "question": "compute_f1 precision recall", "answer": "в pkg/metrics.py",
             "must_include": ["0[.]05"], "sources": [{"file": "pkg/metrics.py", "lines": [5, 8]}]},
            {"id": "b", "class": "Q1", "question": "roc_auc", "answer": "0.9925",
             "sources": [{"file": "notebooks/exp_a.ipynb", "cell": 2}]},
            {"id": "c", "class": "Q6", "question": "Какая видеокарта?"},
            {"id": "d", "class": "G", "question": "Что такое F1?", "answer": "среднее гармоническое"},
            {"id": "bad", "class": "Q1", "question": "x", "answer": "y",
             "sources": [{"file": "notebooks/exp_a.ipynb", "cell": 99}]},
        ],
    })


def test_validate_and_run_eval_end_to_end(settings, fake_embedder, corpus: Path):
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    eng.index_folder(corpus)
    es = small_evalset()
    problems = validate_evalset(es, eng.index.catalog)
    assert problems == ["bad: notebooks/exp_a.ipynb#cell99 matches no indexed fragment"]
    es.items = [i for i in es.items if i.id != "bad"]

    results = run_eval(eng, es, retrieval_k=10, top_k=3)
    by_id = {r.id: r for r in results}
    assert all(r.error is None for r in results)
    assert by_id["a"].retrieval["recall@10"] == 1.0 and by_id["a"].must_include_ok is True
    assert by_id["c"].retrieval is None  # Q6: no reference spans
    assert by_id["d"].route == "corpus" and by_id["d"].route_ok is False  # fake router always says corpus

    run_judge(results, es, FakeLLM(settings))
    assert by_id["a"].judge.correctness == "correct"
    summary = summarize(results, es, n_boot=50)
    assert summary["routing"]["accuracy"]["mean"] == 0.75
    assert summary["refusals"]["false_refusals"] == 0
    assert "ipynb" in summary["retrieval_by_file_type"]
    report = render_report(summary, results, es, {
        "corpus": str(corpus), "index_signature": "x", "embedder": "fake", "chunking": {"python": "lines"},
        "retrieval": {"mode": "dense", "top_k": 3, "eval_k": 10}, "route": "auto", "llm": "fake", "judge": "fake",
        "git": None, "date": "today",
    })
    assert "## Поиск" in report and "маршрут corpus вместо general" in report
    eng.close()


def test_reasoning_accounting_and_h11_section(settings, fake_embedder, corpus: Path):
    from rag_agent.evaluation.ablation import AblationRun, AblationSpec, render_ablation

    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    eng.index_folder(corpus)
    es = small_evalset()
    es.items = [i for i in es.items if i.id != "bad"]
    rows, per_item = [], []
    for mode in ("on", "off"):
        results = run_eval(eng, es, retrieval_k=10, top_k=3, reasoning=mode)
        run_judge(results, es, FakeLLM(settings))
        summary = summarize(results, es, n_boot=50)
        rows.append({"run": AblationRun(name=mode), "summary": summary})
        per_item.append({r.id: r for r in results})
    on, off = rows[0]["summary"]["reasoning"], rows[1]["summary"]["reasoning"]
    assert on["reasoning_share"] == 1.0 and on["levels"] == {"deep": 4} and on["reasoning_tokens"] == 3
    assert off["reasoning_share"] == 0.0 and off["by_class"]["Q1"]["n"] == 2
    assert per_item[0]["a"].generated_tokens > per_item[1]["a"].generated_tokens
    report = render_ablation(AblationSpec(name="h11", evalset="x", runs=[r["run"] for r in rows]), es, rows, per_item)
    assert "## Рассуждения: точность против затрат" in report and "не давшие прироста" in report
    assert "- 1. on: 1.00" in report  # the fake judge scores both runs the same: every extra token is wasted
    eng.close()


def test_ablation_report_without_reference_fragments(settings, fake_embedder, corpus: Path):
    """Upload (Q7) and web (Q10) sets have no corpus fragments to find: the report skips retrieval tables."""
    from rag_agent.evaluation.ablation import AblationRun, AblationSpec, render_ablation

    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    eng.index_folder(corpus)
    es = small_evalset()
    es.items = [i for i in es.items if i.id == "d"]  # a general question: no sources
    results = run_eval(eng, es, retrieval_k=5, top_k=3)
    summary = summarize(results, es, n_boot=50)
    assert summary["retrieval"]["recall@5"]["mean"] is None
    rows = [{"run": AblationRun(name=n), "summary": summary} for n in ("a", "b")]
    report = render_ablation(AblationSpec(name="x", evalset="x", runs=[r["run"] for r in rows]), es, rows,
                             [{r.id: r for r in results}] * 2)
    assert "метрики поиска не считаются" in report and "Recall@5 [95% CI]" not in report
    eng.close()
