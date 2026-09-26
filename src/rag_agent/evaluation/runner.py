"""Evaluation run: retrieval is measured as a component (on the raw or standalone
question, top-10), answers end-to-end through the router; the judge runs as a
separate pass so the generator and judge models are not swapped per question."""

from __future__ import annotations

import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Callable

from pydantic import BaseModel, Field

from rag_agent.config import REPO_ROOT
from rag_agent.engine import Engine
from rag_agent.evaluation.dataset import EvalItem, EvalSet
from rag_agent.evaluation.judge import Verdict, judge_answer
from rag_agent.evaluation.metrics import (
    KS,
    citation_precision,
    first_match_ranks,
    mean_ci,
    must_include_ok,
    percentiles,
    precision_recall,
    retrieval_metrics,
)
from rag_agent.llm import BaseLLM

RETRIEVAL_METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "hit@5", "mrr", "ndcg@10"]


class ItemResult(BaseModel):
    id: str
    cls: str
    question: str
    expected_route: str
    file_types: list[str] = Field(default_factory=list)
    # retrieval component
    retrieved: list[str] = Field(default_factory=list)
    ref_ranks: list[int | None] = Field(default_factory=list)  # first rank of each reference span
    retrieval: dict[str, float] | None = None
    retrieval_latency_s: float | None = None
    # end-to-end answer
    route: str | None = None
    route_ok: bool | None = None
    standalone_question: str | None = None
    answer: str | None = None
    general: str = ""
    answerable: bool | None = None
    refused: bool | None = None
    grounded: bool | None = None
    citations: list[str] = Field(default_factory=list)
    cited_sources: list[tuple[int, str, str]] = Field(default_factory=list)
    must_include_ok: bool | None = None
    citation_precision_ref: float | None = None
    latency_s: float | None = None
    tokens: int = 0
    generated_tokens: int = 0  # completion tokens including reasoning: the cost of the answer
    # reasoning mode (ТЗ ч.2 S15)
    reasoning_level: str | None = None  # none | light | deep
    reasoning_tokens: int = 0
    reasoning_truncated: bool = False
    escalated: bool = False
    # SQL over the catalog (Э6)
    sql_used: bool = False
    sql_pred: str | None = None
    sql_error: str | None = None
    sql_ex: bool | None = None  # execution accuracy against item.sql
    # agent (Э7)
    agent_used: bool = False
    agent_steps: int = 0
    agent_stop: str | None = None
    crag_verdict: str | None = None  # of the direct path's check, or of the agent's evidence
    crag_refused: bool = False
    # uploads (Э13)
    upload: str | None = None
    upload_route: str | None = None
    # web (Э16)
    web_supported: int = 0
    web_unsupported: int = 0
    judge: Verdict | None = None
    error: str | None = None


def git_revision() -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        return rev.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "") if rev.returncode == 0 else None
    except OSError:
        return None


def run_config(engine: Engine, es: EvalSet, *, top_k: int, mode: str, route: str, judge: str | None, retrieval_k: int) -> dict:
    s = engine.settings
    idx = engine.index
    try:  # keep reports free of local absolute paths when the corpus lives in the repo
        corpus = idx.root.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        corpus = str(idx.root)
    return {
        "evalset": {"name": es.name, "version": es.version, "items": len(es.items)},
        "corpus": corpus,
        "index_signature": idx.signature,
        "embedder": s.embedding.model,
        "chunking": s.chunking.model_dump(),
        "retrieval": {**s.retrieval.model_dump(), "mode": mode, "top_k": top_k, "eval_k": retrieval_k},
        "route": route,
        "llm": engine.llm.name,
        "judge": judge,
        "git": git_revision(),
        "date": datetime.now().isoformat(timespec="seconds"),
    }


EVAL_SESSION = "eval"


def _upload(engine: Engine, es_dir, rel: str):
    """Each eval upload is parsed and indexed once per engine (the session scope of a run)."""
    cache = engine.__dict__.setdefault("_eval_uploads", {})
    if rel not in cache:
        path = es_dir / rel
        cache[rel] = engine.upload(EVAL_SESSION, path.name, path.read_bytes())
    return cache[rel]


def evaluate_item(engine: Engine, item: EvalItem, *, retrieval_k: int, top_k: int, mode: str, route: str, generate: bool,
                  reasoning: str | None = None, es_dir=None) -> ItemResult:
    r = ItemResult(
        id=item.id, cls=item.cls, question=item.question, expected_route=item.expected_route, file_types=item.file_types
    )
    try:
        if item.expected_route == "corpus":
            t0 = time.perf_counter()
            hits = engine.search(item.retrieval_query, retrieval_k, mode)
            r.retrieval_latency_s = round(time.perf_counter() - t0, 4)
            nodes = [h.node for h in hits]
            r.retrieved = [n.id for n in nodes]
            if item.sources:
                first, _ = first_match_ranks(nodes, item.sources)
                r.ref_ranks = [first.get(j) for j in range(len(item.sources))]
                r.retrieval = retrieval_metrics(nodes, item.sources)
        if not generate:
            return r
        uploads = [_upload(engine, es_dir, item.upload).id] if item.upload else None
        ans = engine.ask(item.question, history=item.history, top_k=top_k, mode=mode, route=route, reasoning=reasoning,
                         uploads=uploads, session=EVAL_SESSION if uploads else None)
        if item.upload:
            r.upload = item.upload
            r.upload_route = next((s.detail.get("route") for s in ans.trace if s.name == "upload"), None)
        r.route = ans.route
        r.route_ok = ans.route == item.expected_route
        r.standalone_question = ans.standalone_question
        r.answer = ans.answer
        r.general = ans.general
        r.answerable = ans.answerable
        r.refused = ans.route == "corpus" and not ans.answerable
        r.grounded = ans.grounded
        r.citations = [c.node_id for c in ans.citations]
        r.cited_sources = [(s.n, f"{s.file_path} ({s.location})", s.text) for s in ans.sources if s.cited]
        r.must_include_ok = must_include_ok(ans.answer, item.must_include)
        if item.sources and r.citations:
            cited = engine.index.catalog.get_nodes(r.citations)
            r.citation_precision_ref = citation_precision([cited[i] for i in r.citations if i in cited], item.sources)
        r.latency_s = ans.latency_s
        r.tokens = sum(int(s.detail.get("prompt_tokens", 0)) + int(s.detail.get("completion_tokens", 0)) for s in ans.trace)
        r.generated_tokens = sum(int(s.detail.get("completion_tokens", 0)) for s in ans.trace)
        r.reasoning_level = ans.reasoning_level
        r.reasoning_tokens = ans.reasoning_tokens
        r.reasoning_truncated = ans.reasoning_truncated
        r.escalated = any(s.name == "escalate" for s in ans.trace)
        support = next((s for s in ans.trace if s.name == "web_support"), None)
        if support is not None:
            r.web_supported, r.web_unsupported = support.detail.get("supported", 0), support.detail.get("unsupported", 0)
        stop = next((s for s in ans.trace if s.name == "agent_stop"), None)
        if stop is not None:
            r.agent_used, r.agent_steps, r.agent_stop = True, stop.detail.get("steps", 0), stop.detail.get("reason")
            best = stop.detail.get("best_relevance")
            r.crag_verdict = None if best is None else engine_verdict(engine, best)
        crag = next((s for s in ans.trace if s.name == "crag"), None)
        if crag is not None:
            r.crag_verdict = crag.detail.get("verdict")
        r.crag_refused = not ans.answerable and r.crag_verdict == "incorrect"
        step = next((s for s in ans.trace if s.name == "sql" or (s.name == "agent" and s.detail.get("action") == "sql_query")), None)
        if step is not None:
            r.sql_used, r.sql_pred, r.sql_error = True, step.detail.get("sql"), step.detail.get("error")
        if item.sql:
            r.sql_ex = execution_match(engine, item.sql, r.sql_pred if r.sql_error is None else None)
    except Exception as exc:  # one failing question must not abort the run
        r.error = f"{type(exc).__name__}: {exc}"
    return r


def engine_verdict(engine: Engine, best: float) -> str:
    cfg = engine.settings.agent
    if not cfg.crag:
        return "unknown"
    return "correct" if best >= cfg.crag_upper else "incorrect" if best < cfg.crag_lower else "ambiguous"


def _cells(row) -> list:
    return [round(v, 4) if isinstance(v, float) else (v.strip().lower() if isinstance(v, str) else v) for v in row]


def execution_match(engine: Engine, gold_sql: str, pred_sql: str | None) -> bool:
    """Execution accuracy, relaxed for extra columns: the predicted query returns as many rows
    as the reference one and every reference row is contained in a predicted row (the system
    adds path/cell columns for citations). Floats compare to 4 decimals, strings case-insensitively."""
    if not pred_sql:
        return False
    from rag_agent.structured.analytics import ANALYTICS_FILE
    from rag_agent.structured.sql import SQLValidationError, run_readonly, validate_sql

    db, cfg = engine.index.dir / ANALYTICS_FILE, engine.settings.catalog
    try:
        _, gold = run_readonly(db, validate_sql(gold_sql, 10_000), cfg.timeout_s)
        _, pred = run_readonly(db, validate_sql(pred_sql, 10_000), cfg.timeout_s)
    except (SQLValidationError, Exception):
        return False
    if len(gold) != len(pred):
        return False
    pool = [_cells(p) for p in pred]
    for g in map(_cells, gold):
        hit = next((i for i, p in enumerate(pool) if all(p.count(v) >= g.count(v) for v in g)), None)
        if hit is None:
            return False
        pool.pop(hit)
    return True


def run_eval(
    engine: Engine,
    es: EvalSet,
    *,
    retrieval_k: int = 10,
    top_k: int | None = None,
    mode: str | None = None,
    route: str = "auto",
    reasoning: str | None = None,
    generate: bool = True,
    limit: int | None = None,
    on_item: Callable[[ItemResult], None] | None = None,
) -> list[ItemResult]:
    top_k = top_k or engine.settings.retrieval.top_k
    mode = mode or engine.settings.retrieval.mode
    engine.search("warm-up", 1, mode)  # load the embedder before anything is timed
    if generate:
        engine.llm.chat([{"role": "user", "content": "ok"}], max_tokens=1, purpose="warmup")
    results = []
    for item in es.items[:limit] if limit else es.items:
        r = evaluate_item(engine, item, retrieval_k=retrieval_k, top_k=top_k, mode=mode, route=route, generate=generate,
                          reasoning=reasoning, es_dir=es.path.parent if es.path else None)
        results.append(r)
        if on_item:
            on_item(r)
    return results


def run_judge(
    results: list[ItemResult], es: EvalSet, judge: BaseLLM, on_item: Callable[[ItemResult], None] | None = None
) -> None:
    items = {i.id: i for i in es.items}
    for r in results:
        if r.answer is None or r.error:
            continue
        item = items[r.id]
        reference = item.answer if item.cls != "Q6" else None
        r.judge = judge_answer(judge, item.retrieval_query, reference, r.answer, r.general, r.cited_sources)
        if on_item:
            on_item(r)


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(results: list[ItemResult], es: EvalSet, n_boot: int = 1000) -> dict:
    items = {i.id: i for i in es.items}
    ok = [r for r in results if not r.error]
    summary: dict = {"n_items": len(results), "n_errors": len(results) - len(ok)}

    # --- retrieval (component) ---
    with_src = [r for r in ok if r.retrieval]
    summary["retrieval"] = {m: mean_ci([r.retrieval[m] for r in with_src], n_boot) for m in RETRIEVAL_METRICS}
    by_class = defaultdict(list)
    for r in with_src:
        by_class[r.cls].append(r)
    summary["retrieval_by_class"] = {
        c: {"n": len(rs), **{m: _mean(r.retrieval[m] for r in rs) for m in ("recall@5", "mrr", "ndcg@10")}}
        for c, rs in sorted(by_class.items())
    }
    spans = defaultdict(list)  # micro-average over reference spans by file type
    for r in with_src:
        for ref, rank in zip(items[r.id].sources, r.ref_ranks):
            spans[ref.file_type].append(rank)
    summary["retrieval_latency"] = percentiles([r.retrieval_latency_s for r in with_src])
    summary["retrieval_by_file_type"] = {
        ft: {"spans": len(ranks), **{f"recall@{k}": sum(1 for x in ranks if x is not None and x <= k) / len(ranks) for k in KS}}
        for ft, ranks in sorted(spans.items())
    }

    answered = [r for r in ok if r.answer is not None]
    if not answered:
        return summary

    # --- routing ---
    routed = [r for r in answered if not r.upload]  # questions with an upload do not go through the router's choice
    confusion = Counter((r.expected_route, r.route) for r in routed)
    summary["routing"] = {
        "accuracy": mean_ci([float(r.route_ok) for r in routed], n_boot) if routed else None,
        "confusion": {f"{e}->{g}": n for (e, g), n in sorted(confusion.items())},
    }

    # --- answers ---
    corpus_items = [r for r in answered if r.expected_route == "corpus"]
    summary["answers"] = {
        "must_include": mean_ci([float(r.must_include_ok) for r in answered if r.must_include_ok is not None], n_boot),
        "citation_precision_ref": mean_ci([r.citation_precision_ref for r in answered], n_boot),
        "grounded_rate": _mean(float(bool(r.grounded)) for r in corpus_items if r.route == "corpus" and r.answerable),
    }
    judged = [r for r in answered if r.judge is not None and r.judge.error is None]
    if judged:
        corpus_judged = [r for r in judged if r.expected_route == "corpus"]
        cited_total = sum(len(r.cited_sources) for r in corpus_judged)
        summary["judge"] = {
            "correctness": mean_ci([r.judge.score for r in corpus_judged], n_boot),
            "correctness_general": mean_ci([r.judge.score for r in judged if r.expected_route == "general"], n_boot),
            "faithfulness": mean_ci([float(r.judge.faithful) for r in corpus_judged if r.route == "corpus"], n_boot),
            "relevance": mean_ci([float(r.judge.relevant) for r in judged], n_boot),
            "citation_precision": (
                sum(len(r.judge.supported_citations) for r in corpus_judged) / cited_total if cited_total else None
            ),
            "n_judge_errors": sum(1 for r in answered if r.judge is not None and r.judge.error),
        }
        per_class = defaultdict(list)
        for r in judged:
            per_class[r.cls].append(r.judge.score)
        summary["judge_by_class"] = {c: {"n": len(v), "correctness": _mean(v)} for c, v in sorted(per_class.items())}

    # --- refusals (Q6) among questions routed to the corpus ---
    refusal_pool = [r for r in corpus_items if r.refused is not None]
    summary["refusals"] = precision_recall([bool(r.refused) for r in refusal_pool], [r.cls == "Q6" for r in refusal_pool])
    summary["refusals"]["false_refusals"] = sum(1 for r in refusal_pool if r.refused and r.cls != "Q6")

    # --- latency and cost ---
    summary["latency"] = {
        "all": percentiles([r.latency_s for r in answered]),
        "corpus": percentiles([r.latency_s for r in answered if r.route == "corpus"]),
        "general": percentiles([r.latency_s for r in answered if r.route == "general"]),
        "mean_tokens": _mean(r.tokens for r in answered),
    }

    # --- SQL over the catalog (Э6, H5) ---
    aggregate = [r for r in answered if r.cls == "Q3"]
    others = [r for r in answered if r.cls != "Q3" and r.expected_route == "corpus"]
    gold = [r for r in answered if r.sql_ex is not None]
    summary["sql"] = {
        "used_q3": _mean(float(r.sql_used) for r in aggregate),
        "used_other": _mean(float(r.sql_used) for r in others),
        "errors": sum(1 for r in answered if r.sql_used and r.sql_error),
        "n_gold": len(gold),
        "execution_accuracy": mean_ci([float(r.sql_ex) for r in gold], n_boot) if gold else None,
    }

    # --- uploads (Э13, H12): accuracy and cost by file ---
    by_upload: dict[str, list[ItemResult]] = defaultdict(list)
    for r in answered:
        if r.upload:
            by_upload[r.upload].append(r)
    summary["uploads"] = {
        f: {"n": len(rs), "must_include": _mean(float(r.must_include_ok) for r in rs if r.must_include_ok is not None),
            "correctness": _mean(r.judge.score for r in rs if r.judge is not None and r.judge.error is None),
            "tokens": _mean(r.tokens for r in rs), "latency": percentiles([r.latency_s for r in rs]),
            "routes": dict(Counter(r.upload_route for r in rs))}
        for f, rs in sorted(by_upload.items())
    }

    # --- web (Э16, H15): how often questions go to the web, and support of web answers ---
    web_answers = [r for r in answered if r.route == "web"]
    sentences = sum(r.web_supported + r.web_unsupported for r in web_answers)
    summary["web"] = {
        "used_q10": _mean(float(r.route == "web") for r in answered if r.cls == "Q10"),
        "used_other": _mean(float(r.route == "web") for r in answered if r.cls != "Q10"),
        "unsupported_share": sum(r.web_unsupported for r in web_answers) / sentences if sentences else None,
        "latency_web": percentiles([r.latency_s for r in web_answers]),
    }

    # --- agent and CRAG (Э7, H6, NFR2) ---
    via_agent = [r for r in answered if r.agent_used]
    direct = [r for r in answered if r.route == "corpus" and not r.agent_used]
    judged_corpus = [r for r in answered if r.expected_route == "corpus" and r.judge is not None and r.judge.error is None]
    summary["agent"] = {
        "used": _mean(float(r.agent_used) for r in answered if r.route == "corpus"),
        "used_by_class": {c: _mean(float(r.agent_used) for r in rs) for c, rs in sorted(by_cls_all(answered).items())},
        "mean_steps": _mean(r.agent_steps for r in via_agent),
        "stops": dict(Counter(r.agent_stop for r in via_agent)),
        "crag_refusals": sum(r.crag_refused for r in answered),
        "crag_refusals_q6": sum(r.crag_refused for r in answered if r.cls == "Q6"),
        # H6: share of wrong answers (judge: incorrect) among questions that have an answer in the files
        "wrong_rate": _mean(float(r.judge.correctness == "incorrect") for r in judged_corpus if r.cls != "Q6"),
        "latency_direct": percentiles([r.latency_s for r in direct]),
        "latency_agent": percentiles([r.latency_s for r in via_agent]),
        "latency_q1": percentiles([r.latency_s for r in answered if r.cls == "Q1"]),
    }

    # --- reasoning (ТЗ ч.2 S15, H11): accuracy against generated tokens and latency, per class ---
    by_cls = defaultdict(list)
    for r in answered:
        by_cls[r.cls].append(r)
    reasoned = [r for r in answered if r.reasoning_level not in (None, "none")]
    summary["reasoning"] = {
        **_cost_row(answered),
        "levels": dict(Counter(r.reasoning_level or "none" for r in answered)),
        "truncated_share": _mean(float(r.reasoning_truncated) for r in reasoned),
        "escalated": sum(r.escalated for r in answered),
        "latency_reasoning": percentiles([r.latency_s for r in reasoned]),
        "by_class": {c: _cost_row(rs) for c, rs in sorted(by_cls.items())},
    }
    return summary


def by_cls_all(rs: list[ItemResult]) -> dict[str, list[ItemResult]]:
    out: dict[str, list[ItemResult]] = defaultdict(list)
    for r in rs:
        out[r.cls].append(r)
    return out


def _cost_row(rs: list[ItemResult]) -> dict:
    judged = [r for r in rs if r.judge is not None and r.judge.error is None]
    return {
        "n": len(rs),
        "reasoning_share": _mean(float(r.reasoning_level not in (None, "none")) for r in rs),
        "reasoning_tokens": _mean(r.reasoning_tokens for r in rs),
        "generated_tokens": _mean(r.generated_tokens for r in rs),
        "correctness": _mean(r.judge.score for r in judged),
        "must_include": _mean(float(r.must_include_ok) for r in rs if r.must_include_ok is not None),
        "latency": percentiles([r.latency_s for r in rs]),
    }
