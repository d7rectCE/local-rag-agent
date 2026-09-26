"""Evaluation of the code agent on tasks with automatic checks (ТЗ ч.2: class Q9; H13, H14).

A task is a statement, the starting state of the working copy (the corpus with
``patch`` edits applied — this is how bugs are planted) and a hidden pytest check
that runs in the sandbox after the agent is done. Metrics: share of solved tasks,
failed runs (iterations), steps, share of changed files left unparseable (H14),
latency. ``rag ablate-code`` compares configurations (e.g. one attempt vs the fix
loop for H13, ACI vs whole-file overwrite for H14) with paired bootstrap.

``validate`` checks the task set itself: every check must fail on the starting
state, and for a planted bug it must pass on the original corpus.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml
from pydantic import BaseModel, Field

from rag_agent.code.agent import CodeAgent
from rag_agent.code.sandbox import DockerSandbox
from rag_agent.code.workspace import Workspace
from rag_agent.config import Settings, _deep_merge
from rag_agent.evaluation.metrics import mean_ci, paired_bootstrap, percentiles


class Patch(BaseModel):
    file: str
    old: str
    new: str


class CodeTask(BaseModel):
    id: str
    task: str
    patch: list[Patch] = Field(default_factory=list)
    check: str
    notes: str | None = None


class CodeTaskSet(BaseModel):
    version: int = 1
    name: str
    corpus: str
    tasks: list[CodeTask]
    path: Path | None = None

    def corpus_root(self) -> Path:
        root = Path(self.corpus)
        return (self.path.parent / root).resolve() if not root.is_absolute() and self.path else root


def load_code_tasks(path: str | Path) -> CodeTaskSet:
    path = Path(path)
    ts = CodeTaskSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    ts.path = path.resolve()
    return ts


def prepare_source(corpus: Path, task: CodeTask, dest: Path) -> Path:
    """The starting state: a copy of the corpus with the task's patches applied."""
    shutil.copytree(corpus, dest, ignore=shutil.ignore_patterns(".git", "__pycache__", ".ipynb_checkpoints"))
    for p in task.patch:
        f = dest / p.file
        text = f.read_text(encoding="utf-8")
        if text.count(p.old) != 1:
            raise ValueError(f"{task.id}: patch for {p.file} matches {text.count(p.old)} times")
        f.write_text(text.replace(p.old, p.new, 1), encoding="utf-8")
    return dest


CHECK_HEADER = "import os, sys\nsys.path.insert(0, os.getcwd())\n"


def run_check(sandbox, work: Path, task: CodeTask) -> tuple[bool, str]:
    check = work / ".agent" / f"check_{task.id.replace('-', '_')}.py"
    check.parent.mkdir(exist_ok=True)
    check.write_text(CHECK_HEADER + task.check, encoding="utf-8")
    run = sandbox.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", ".", f".agent/{check.name}"],
                      work, timeout_s=600)
    return run.ok, (run.stdout[-2000:] + run.stderr[-1000:])


class TaskResult(BaseModel):
    id: str
    solved: bool
    status: str
    pipeline: str
    steps: int
    runs: int
    failed_runs: int
    changed: int
    broken: int
    latency_s: float
    check_output: str = ""
    summary: str = ""
    error: str | None = None


def run_code_eval(settings: Settings, ts: CodeTaskSet, llm, sandbox=None, catalog_rows: list[dict] | None = None,
                  log: Callable[[str], None] = print, work_root: Path | None = None) -> list[TaskResult]:
    sandbox = sandbox or DockerSandbox(settings.code)
    results = []
    with tempfile.TemporaryDirectory(prefix="rag-code-eval-", dir=work_root) as tmp:
        for k, task in enumerate(ts.tasks, start=1):
            src = prepare_source(ts.corpus_root(), task, Path(tmp) / f"src-{task.id}")
            ws = Workspace.create(Path(tmp) / f"ws-{task.id}", src, max_mb=settings.code.workspace_max_mb)
            try:
                res = CodeAgent(settings, llm, sandbox, ws, task.task, task.id, catalog_rows=catalog_rows).run()
                solved, out = run_check(sandbox, ws.root, task)
                r = TaskResult(id=task.id, solved=solved, status=res.status, pipeline=res.pipeline, steps=len(res.steps),
                               runs=res.runs, failed_runs=res.failed_runs, changed=len(res.changed),
                               broken=len(res.broken_files), latency_s=res.latency_s, check_output=out,
                               summary=res.summary[:500])
            except Exception as exc:  # one task must not stop the run
                r = TaskResult(id=task.id, solved=False, status="error", pipeline="", steps=0, runs=0, failed_runs=0,
                               changed=0, broken=0, latency_s=0.0, error=f"{type(exc).__name__}: {exc}")
            results.append(r)
            log(f"  [{k}/{len(ts.tasks)}] {task.id}: {'решена' if r.solved else 'не решена'} ({r.status}, "
                f"шагов {r.steps}, неудачных запусков {r.failed_runs}, {r.latency_s:.0f} с)")
    return results


def summarize_code(results: list[TaskResult], n_boot: int = 1000) -> dict:
    changed = sum(r.changed for r in results)
    return {
        "n": len(results),
        "solved": mean_ci([float(r.solved) for r in results], n_boot),
        "mean_steps": sum(r.steps for r in results) / max(1, len(results)),
        "mean_failed_runs": sum(r.failed_runs for r in results) / max(1, len(results)),
        "broken_files_share": sum(r.broken for r in results) / changed if changed else 0.0,
        "tasks_with_broken_files": sum(1 for r in results if r.broken),
        "errors": sum(1 for r in results if r.status == "error"),
        "latency": percentiles([r.latency_s for r in results]),
    }


def validate_tasks(ts: CodeTaskSet, sandbox, log: Callable[[str], None] = print) -> list[str]:
    """Every check fails on the starting state; a planted bug's check passes on the original corpus."""
    problems = []
    with tempfile.TemporaryDirectory(prefix="rag-code-validate-") as tmp:
        for task in ts.tasks:
            start = prepare_source(ts.corpus_root(), task, Path(tmp) / f"start-{task.id}")
            ok, out = run_check(sandbox, start, task)
            if ok:
                problems.append(f"{task.id}: check passes before any work")
            if task.patch:
                orig = Path(tmp) / f"orig-{task.id}"
                shutil.copytree(ts.corpus_root(), orig, ignore=shutil.ignore_patterns(".git", "__pycache__"))
                ok, out = run_check(sandbox, orig, task)
                if not ok:
                    problems.append(f"{task.id}: check fails on the original corpus: {out[-400:]}")
            log(f"  {task.id}: ok" if not any(p.startswith(task.id) for p in problems) else f"  {task.id}: ПРОБЛЕМА")
    return problems


# --------------------------------------------------------------------------- ablation


class CodeRun(BaseModel):
    name: str
    overrides: dict = Field(default_factory=dict)


class CodeAblationSpec(BaseModel):
    name: str
    tasks: str
    runs: list[CodeRun]
    path: Path | None = None


def run_code_ablation(spec_path: str | Path, base: Settings, out_root: Path, make_llm, catalog_rows=None,
                      log: Callable[[str], None] = print) -> Path:
    spec_path = Path(spec_path)
    spec = CodeAblationSpec.model_validate(yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    ts = load_code_tasks((spec_path.parent / spec.tasks).resolve())
    out_dir = out_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{spec.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, run in enumerate(spec.runs, start=1):
        settings = Settings.model_validate(_deep_merge(base.model_dump(), json.loads(json.dumps(run.overrides))))
        log(f"[{i}/{len(spec.runs)}] {run.name}")
        results = run_code_eval(settings, ts, make_llm(settings), catalog_rows=catalog_rows, log=log)
        summary = summarize_code(results, settings.evaluation.bootstrap)
        rows.append((run, results, summary))
        (out_dir / f"{i:02d}-results.jsonl").write_text("\n".join(r.model_dump_json() for r in results) + "\n",
                                                         encoding="utf-8")
    ref = {r.id: float(r.solved) for r in rows[0][1]}
    lines = [f"# Код-агент: {spec.name}", "",
             f"Набор `{ts.name}`: задач {len(ts.tasks)}. Решена — прошла скрытая проверка в песочнице. Δ — парный "
             "бутстреп по задачам относительно первой конфигурации. Испорченные — доля изменённых файлов, которые не "
             "разбираются после правок (H14).", "",
             "| # | Конфигурация | Решено [95% CI] | Δ, p | Шагов | Неудачных запусков | Испорченные файлы | p50 / p95, с |",
             "|---|---|---|---|---|---|---|---|"]
    for k, (run, results, s) in enumerate(rows, start=1):
        delta = "—"
        if k > 1:
            cmp = paired_bootstrap([float(r.solved) for r in results], [ref[r.id] for r in results])
            if cmp["ci"]:
                delta = f"{cmp['diff']:+.2f} [{cmp['ci'][0]:+.2f}; {cmp['ci'][1]:+.2f}], p={cmp['p']:.3f}"
        sol = s["solved"]
        lines.append(f"| {k} | {run.name} | {sol['mean']:.2f} [{sol['ci'][0]:.2f}; {sol['ci'][1]:.2f}] | {delta} "
                     f"| {s['mean_steps']:.1f} | {s['mean_failed_runs']:.2f} | {s['broken_files_share']:.2f} "
                     f"({s['tasks_with_broken_files']} задач) | {s['latency']['p50']:.0f} / {s['latency']['p95']:.0f} |")
    lines += ["", "По задачам (решена / неудачных запусков):", "",
              "| Задача | " + " | ".join(f"{k}" for k in range(1, len(rows) + 1)) + " |", "|---|" + "---|" * len(rows)]
    for t in ts.tasks:
        cells = []
        for _, results, _ in rows:
            r = next(x for x in results if x.id == t.id)
            cells.append(f"{'✓' if r.solved else '✗'} {r.failed_runs}")
        lines.append(f"| {t.id} | " + " | ".join(cells) + " |")
    lines += ["", "Переопределения:", ""] + [f"{k}. **{run.name}** — `{json.dumps(run.overrides, ensure_ascii=False)}`"
                                          for k, (run, _, _) in enumerate(rows, start=1)]
    (out_dir / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "ablation.json").write_text(json.dumps([{"name": run.name, "overrides": run.overrides, "summary": s}
                                                       for run, _, s in rows], ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return out_dir
