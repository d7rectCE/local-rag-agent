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
    excl = prefs["exclude"] + list(exclude or [])
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
    as_json: bool = typer.Option(False, "--json", help="Print the full answer object"),
) -> None:
    """Ask a question about the indexed folder."""
    engine = _engine()
    if root:
        engine.open_corpus(root)
    ans = engine.ask(question, top_k=top_k, mode=mode)
    if as_json:
        typer.echo(ans.model_dump_json(indent=2))
    else:
        typer.echo(ans.answer)
        typer.echo("")
        for c in ans.citations:
            typer.echo(f"[{c.n}] {c.file_path} — {c.location}")
        typer.echo(f"\n({ans.latency_s:.1f}s, answerable={ans.answerable}, grounded={ans.grounded})")
    engine.close()


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
