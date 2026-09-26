"""Command-line interface: ``rag index | ask | status | serve | ui | demo``."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from rag_agent.config import REPO_ROOT, load_settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Local RAG agent over a research archive.")

DEMO_CORPUS = REPO_ROOT / "demo_corpus"
UI_SCRIPT = Path(__file__).with_name("ui.py")


def _engine():
    from rag_agent.engine import Engine

    return Engine()


@app.command()
def index(
    root: Path = typer.Argument(..., help="Folder to index"),
    exclude: Optional[list[str]] = typer.Option(None, "--exclude", "-x", help="Extra exclude pattern (repeatable)"),
    ext: Optional[list[str]] = typer.Option(None, "--ext", help="File extension to include (repeatable)"),
    force: bool = typer.Option(False, help="Rebuild the index from scratch"),
) -> None:
    """Index a folder (incrementally)."""
    engine = _engine()
    prefs = engine.corpus_prefs(root)
    excl = prefs["exclude"] + list(exclude) if exclude else None  # extra patterns are remembered for the folder
    progress = engine.index_folder(root, include_ext=ext or None, exclude=excl, force=force, background=True)
    last = None
    while engine.indexing_alive:
        line = f"{progress.state}: {progress.files_done}/{progress.files_total} {progress.current or ''}"
        if line != last:
            typer.echo(line[:150])
            last = line
        time.sleep(0.5)
    typer.echo(
        f"{progress.state} in {progress.elapsed_s}s: new {progress.new}, changed {progress.changed}, "
        f"unchanged {progress.unchanged}, deleted {progress.deleted}, errors {progress.errors}, "
        f"fragments embedded {progress.nodes_embedded}"
    )
    for p in engine.index.catalog.problems():
        typer.echo(f"  [{p['status']}] {p['path']}: {p['error'] or '; '.join(p['warnings'])}")
    engine.close()
    if progress.state != "done":
        raise typer.Exit(1)


@app.command()
def ask(
    question: str,
    root: Optional[Path] = typer.Option(None, help="Corpus folder (default: last used)"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    mode: Optional[str] = typer.Option(None, help="dense | sparse | hybrid"),
    route: str = typer.Option("auto", help="auto | corpus (my files) | general (general knowledge)"),
    reasoning: Optional[str] = typer.Option(None, help="off | on | auto (default: config)"),
    as_json: bool = typer.Option(False, "--json", help="Print the full answer object"),
) -> None:
    """Ask a question: about the indexed folder or a general one (routed automatically)."""
    engine = _engine()
    if root:
        engine.open_corpus(root)
    ans = engine.ask(question, top_k=top_k, mode=mode, route=route, reasoning=reasoning)
    if as_json:
        typer.echo(ans.model_dump_json(indent=2))
    else:
        if ans.notice:
            typer.echo(f"({ans.notice})\n")
        typer.echo(ans.answer)
        if ans.general:
            typer.echo(f"\nИз общих знаний:\n{ans.general}")
        typer.echo("")
        for c in ans.citations:
            typer.echo(f"[{c.n}] {c.file_path} — {c.location}")
        if ans.reasoning:
            cut = ", обрезано по бюджету" if ans.reasoning_truncated else ""
            typer.echo(f"\n(рассуждение: {ans.reasoning_tokens} токенов{cut})")
        typer.echo(f"\n({ans.latency_s:.1f}s, route={ans.route}, answerable={ans.answerable}, grounded={ans.grounded})")
    engine.close()


@app.command()
def sql(question: str, root: Optional[Path] = typer.Option(None, help="Corpus folder (default: last used)")) -> None:
    """Answer an aggregate question with SQL over the catalog of experiments: shows the query and the rows."""
    engine = _engine()
    if root:
        engine.open_corpus(root)
    res = engine.query_catalog(question)
    engine.close()
    typer.echo(f"SQL ({res.attempts} попыт.): {res.sql}\n")
    typer.echo(res.markdown() if res.ok else f"Ошибка: {res.error}")
    typer.echo(f"\n(таблицы: {', '.join(res.tables)}; подсказки: {'; '.join(res.hints) or '—'}; {res.latency_s:.1f}s)")


@app.command()
def experiments(root: Optional[Path] = typer.Option(None, help="Corpus folder (default: last used)")) -> None:
    """Experiments extracted into the catalog, with their metric values and source cells."""
    engine = _engine()
    if root:
        engine.open_corpus(root)
    summary, items = engine.catalog_summary(), engine.experiments()
    engine.close()
    typer.echo(f"ноутбуков {summary['notebooks']}, экспериментов {summary['experiments']}, значений {summary['values']}, "
               f"отброшено непроверенных {summary['dropped']}\n")
    for e in items:
        typer.echo(f"{e['file_path']} — {e['title']} [{e['task']}; {e['dataset']}; {e['model']}]")
        for m in e["metrics"]:
            variant = f" ({m['variant']})" if m["variant"] else ""
            typer.echo(f"    {m['name']}@{m['split']}{variant} = {m['value']:g}   ячейка {m['cell']}")
        for h in e["hyperparameters"]:
            variant = f" ({h['variant']})" if h["variant"] else ""
            typer.echo(f"    {h['name']} = {h['value']}{variant}   ячейка {h['cell']}")


@app.command()
def symbols(name: str, root: Optional[Path] = typer.Option(None, help="Corpus folder (default: last used)")) -> None:
    """Where a function or class is defined and where it is called (static index, no LLM)."""
    engine = _engine()
    if root:
        engine.open_corpus(root)
    found = engine.lookup_symbol(name)
    engine.close()

    def where(row: dict, line_key: str) -> str:
        cell = f", ячейка {row['cell']}" if row.get("cell") else ""
        return f"{row['file_path']}{cell}, строка {row[line_key]}"

    typer.echo(f"Определения {name}:" if found["definitions"] else f"Определений {name} не найдено")
    for d in found["definitions"]:
        typer.echo(f"  {where(d, 'line_start')}: {d['signature']}" + (f" — {d['doc']}" if d["doc"] else ""))
    typer.echo(f"Вызовы ({len(found['calls'])}):")
    for c in found["calls"]:
        typer.echo(f"  {where(c, 'line')}" + (f" в {c['caller']}" if c["caller"] else ""))


@app.command()
def status() -> None:
    """Show the active corpus, index statistics and models."""
    engine = _engine()
    typer.echo(json.dumps(engine.status(), ensure_ascii=False, indent=2, default=str))
    engine.close()


@app.command()
def serve(host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Run the HTTP API."""
    import uvicorn

    from rag_agent.api import create_app

    settings = load_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=host or settings.api.host, port=port or settings.api.port)


DEFAULT_EVALSET = REPO_ROOT / "evalsets" / "demo_v4.yaml"


@app.command("eval")
def eval_cmd(
    evalset: Path = typer.Argument(DEFAULT_EVALSET, help="Evaluation set (YAML)"),
    root: Optional[Path] = typer.Option(None, help="Corpus folder (default: the one named in the eval set)"),
    name: str = typer.Option("", help="Run name for the output folder"),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Score answers with the LLM judge"),
    retrieval_only: bool = typer.Option(False, help="Only retrieval metrics, no generation"),
    mode: Optional[str] = typer.Option(None, help="dense | sparse | hybrid"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    route: str = typer.Option("auto", help="auto | corpus | general"),
    limit: Optional[int] = typer.Option(None, help="Evaluate only the first N questions"),
) -> None:
    """Evaluate on a reference question set; writes results and a Markdown report."""
    from datetime import datetime

    from rag_agent.engine import Engine
    from rag_agent.evaluation.dataset import load_evalset, validate_evalset
    from rag_agent.evaluation.report import render_report
    from rag_agent.evaluation.runner import run_config, run_eval, run_judge, summarize
    from rag_agent.llm import make_llm

    settings = load_settings()
    ev = settings.evaluation
    es = load_evalset(evalset)
    corpus = root or es.corpus_root()
    if corpus is None:
        raise typer.BadParameter("the eval set names no corpus; pass --root")
    engine = Engine(settings)
    progress = engine.index_folder(corpus)  # incremental: makes sure the index matches the files
    if progress.state != "done":
        raise typer.Exit(1)
    problems = validate_evalset(es, engine.index.catalog)
    if problems:
        typer.echo("Eval set does not match the index:\n  " + "\n  ".join(problems))
        raise typer.Exit(1)

    mode = mode or settings.retrieval.mode
    top_k = top_k or settings.retrieval.top_k
    judge_name = None if retrieval_only or not judge else ev.judge_model

    def show(r) -> None:
        parts = [f"{r.id:6s}"]
        if r.retrieval:
            parts.append(f"R@5={r.retrieval['recall@5']:.2f}")
        if r.route:
            parts.append(f"route={r.route}{'' if r.route_ok else '(!)'}")
        if r.refused:
            parts.append("refused")
        if r.must_include_ok is not None:
            parts.append(f"must={'ok' if r.must_include_ok else 'FAIL'}")
        if r.latency_s is not None:
            parts.append(f"{r.latency_s:.1f}s")
        if r.error:
            parts.append(f"ERROR {r.error}")
        typer.echo("  ".join(parts))

    results = run_eval(
        engine, es, retrieval_k=ev.retrieval_k, top_k=top_k, mode=mode, route=route,
        generate=not retrieval_only, limit=limit, on_item=show,
    )
    if judge_name:
        typer.echo(f"Judging with {judge_name}…")
        # generator and judge never share GPU memory: a model that does not fit may be put on another device
        engine.llm.unload()
        judge_llm = make_llm(settings.llm.model_copy(update={"model": judge_name, "think": ev.judge_think}), engine.tracer)
        run_judge(results, es, judge_llm, on_item=lambda r: typer.echo(f"{r.id:6s}  {r.judge.correctness}"))
        judge_llm.unload()
        judge_llm.close()

    summary = summarize(results, es, n_boot=ev.bootstrap)
    config = run_config(engine, es, top_k=top_k, mode=mode, route=route, judge=judge_name, retrieval_k=ev.retrieval_k)
    config["name"] = name or None
    run_name = "-".join(p for p in (datetime.now().strftime("%Y%m%d-%H%M%S"), name or es.name, mode) if p)
    out_dir = REPO_ROOT / ev.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    (out_dir / "report.md").write_text(render_report(summary, results, es, config), encoding="utf-8")
    engine.close()

    ret = summary["retrieval"]
    typer.echo(
        f"\nRecall@5 {ret['recall@5']['mean']:.3f}  MRR {ret['mrr']['mean']:.3f}  nDCG@10 {ret['ndcg@10']['mean']:.3f}"
    )
    if "judge" in summary:
        typer.echo(f"Correctness (judge) {summary['judge']['correctness']['mean']:.3f}")
    typer.echo(f"Report: {out_dir / 'report.md'}")


@app.command()
def ablate(spec: Path = typer.Argument(..., help="Ablation spec (YAML)")) -> None:
    """Run several configurations on one eval set and write a comparison table."""
    from rag_agent.evaluation.ablation import load_spec, run_ablation

    settings = load_settings()
    rows, out_dir = run_ablation(load_spec(spec), settings, REPO_ROOT / "runs" / "ablations", log=typer.echo)
    typer.echo(f"Report: {out_dir / 'ablation.md'}")


@app.command("eval-validate")
def eval_validate(evalset: Path = typer.Argument(DEFAULT_EVALSET), root: Optional[Path] = None) -> None:
    """Check that every reference in the eval set points to an indexed fragment."""
    from rag_agent.engine import Engine
    from rag_agent.evaluation.dataset import CLASS_NAMES, load_evalset, validate_evalset

    es = load_evalset(evalset)
    engine = Engine()
    engine.index_folder(root or es.corpus_root())
    problems = validate_evalset(es, engine.index.catalog)
    engine.close()
    counts = {c: n for c in CLASS_NAMES if (n := sum(1 for i in es.items if i.cls == c))}
    typer.echo(f"{es.name}: {len(es.items)} questions {counts}")
    typer.echo("OK" if not problems else "Problems:\n  " + "\n  ".join(problems))
    if problems:
        raise typer.Exit(1)


def _ui_command(port: int) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run", str(UI_SCRIPT),
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]


@app.command()
def ui(port: int = 8501) -> None:
    """Run the Streamlit UI (expects the API to be running)."""
    raise typer.Exit(subprocess.call(_ui_command(port)))


@app.command()
def demo(port: int = 8501, rebuild_corpus: bool = typer.Option(False, help="Regenerate demo_corpus/")) -> None:
    """One-command demo: build the demo corpus if needed, start the API, index, open the UI."""
    import httpx

    if rebuild_corpus or not DEMO_CORPUS.exists():
        typer.echo("Building demo corpus…")
        subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "build_demo_corpus.py")])

    settings = load_settings()
    base = f"http://{settings.api.host}:{settings.api.port}"
    api_proc = subprocess.Popen([sys.executable, "-m", "rag_agent.cli", "serve"])
    try:
        with httpx.Client(base_url=base, timeout=600) as client:
            for _ in range(120):
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if api_proc.poll() is not None:
                    raise typer.Exit(1)
                time.sleep(1)
            client.post("/index", json={"root": str(DEMO_CORPUS)}).raise_for_status()
            while True:
                prog = client.get("/index/progress").json()
                typer.echo(f"\r{prog['state']}: {prog['files_done']}/{prog['files_total']}", nl=False)
                if prog["state"] not in ("scanning", "indexing"):
                    typer.echo("")
                    break
                time.sleep(1)
        subprocess.call(_ui_command(port))
    finally:
        api_proc.terminate()


if __name__ == "__main__":
    app()
