"""Code agent (ТЗ ч.2 S17, Э15): plan -> edit in the working copy -> run in the sandbox ->
read the errors -> fix, at most ``code.max_iterations`` failed runs.

Actions are JSON calls under a schema. Code is still the main action (CodeAct): the
agent runs Python snippets, scripts, tests and notebooks and gets stdout, stderr and
the traceback back. Files are changed through the ACI of SWE-agent (numbered views,
search, fragment replacement with a syntax check); the H14 baseline swaps it for a
plain whole-file overwrite. Lessons from failed runs stay in the system prompt for
the rest of the task (Reflexion). Typical tasks go through fixed pipelines first
(Agentless): metrics from notebooks, plots from logs, a fix from a traceback.

Nothing leaves the working copy: the result is a diff, and ``apply_changes`` copies
it into the user's folder only after an explicit confirmation (policy: always).
"""

from __future__ import annotations

import json
import re
import shlex
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from rag_agent.code.sandbox import DockerSandbox, RunResult, SandboxError
from rag_agent.code.workspace import TEXT_EXTS, Workspace, WorkspaceError
from rag_agent.config import Settings
from rag_agent.llm import BaseLLM, LLMError

TOOLS_ACI = {
    "list_files": "list_files {} — список файлов рабочей копии",
    "view_file": 'view_file {"path": "...", "start": 1} — окно файла из 80 строк с номерами строк',
    "search": 'search {"pattern": "регулярное выражение", "glob": "*.py"} — поиск по рабочей копии',
    "edit_file": 'edit_file {"path": "...", "old": "точный фрагмент", "new": "замена"} — заменить фрагмент, который '
                 "встречается ровно один раз; правка с синтаксической ошибкой отклоняется и не записывается",
    "create_file": 'create_file {"path": "...", "content": "..."} — создать файл (с проверкой синтаксиса)',
    "run_code": 'run_code {"code": "..."} — выполнить фрагмент Python в песочнице (текущая папка — рабочая копия)',
    "run": 'run {"command": "..."} — запустить: "python путь.py [аргументы]", "pytest [аргументы]" или '
           '"notebook путь.ipynb" (выполнить ноутбук и сохранить выводы)',
    "finish": 'finish {"summary": "что сделано и как проверено"}',
}
TOOLS_OVERWRITE = {  # H14 baseline: no syntax check, the whole file is rewritten
    "list_files": TOOLS_ACI["list_files"],
    "view_file": TOOLS_ACI["view_file"],
    "write_file": 'write_file {"path": "...", "content": "..."} — записать файл целиком (создать или перезаписать)',
    "run_code": TOOLS_ACI["run_code"],
    "run": TOOLS_ACI["run"],
    "finish": TOOLS_ACI["finish"],
}

CODE_PROMPT = """Ты — код-агент. Решаешь задачу пользователя в рабочей копии его проекта. Рабочая копия под git: все изменения покажут пользователю как diff и перенесут в его папку только после подтверждения. Код исполняется в песочнице без сети: Python 3.12, numpy, pandas, scikit-learn, matplotlib, pytest, nbclient. Текущая папка — рабочая копия (/work) с текстовыми файлами проекта пользователя.

Инструменты (один вызов за шаг):
{tools}

Правила:
- сначала посмотри нужные файлы, потом правь; правки — минимальные;
- после правки проверь результат запуском (скрипт, тесты или ноутбук); если упало — прочитай traceback и исправь;
- неудачных запусков — не больше {max_iterations};
- графики сохраняй в файлы через plt.savefig, не вызывай plt.show;
- когда задача решена и проверена запуском — finish с кратким отчётом.{lessons}

Верни JSON: {{"thought": "коротко: что знаешь и что делаешь", "action": "...", "args": {{...}}}}"""

PIPELINE_PROMPT = """Определи тип задачи для код-агента:
- collect_metrics — собрать метрики экспериментов из ноутбуков в таблицу (CSV);
- plot_logs — только построить график (картинку) по логу обучения (target — путь к логу, если назван); если нужна таблица или CSV — это free;
- fix_traceback — исправить ошибку при запуске скрипта или ноутбука (target — команда или путь, если названы);
- free — всё остальное.
Верни JSON: {"pipeline": "...", "target": "..."}"""

PIPELINE_SCHEMA = {
    "type": "object",
    "properties": {"pipeline": {"type": "string", "enum": ["collect_metrics", "plot_logs", "fix_traceback", "free"]},
                   "target": {"type": "string"}},
    "required": ["pipeline", "target"],
}

PLOT_TEMPLATE = '''"""График метрик по эпохам из лога обучения (сгенерировано конвейером plot_logs)."""
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

log, out = {log!r}, {out!r}
rows = []
for line in open(log, encoding="utf-8", errors="replace"):
    pairs = dict(re.findall(r"([A-Za-z_][\\w]*)=([-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?)", line))
    if "epoch" in pairs:
        rows.append({{k: float(v) for k, v in pairs.items()}})
if not rows:
    sys.exit(f"в {{log}} нет строк с epoch=…")
df = pd.DataFrame(rows).drop_duplicates("epoch").set_index("epoch").sort_index()
cols = [c for c in df.columns if df[c].nunique() > 1]
fig, axes = plt.subplots(len(cols), 1, figsize=(7, 2.4 * len(cols)), sharex=True, squeeze=False)
for ax, col in zip(axes[:, 0], cols):
    ax.plot(df.index, df[col], marker=".")
    ax.set_ylabel(col)
    ax.grid(alpha=0.3)
axes[-1, 0].set_xlabel("epoch")
fig.suptitle(log)
fig.tight_layout()
fig.savefig(out, dpi=110)
print(f"{{len(df)}} эпох, метрики: {{', '.join(cols)}} -> {{out}}")
'''


def _schema(actions: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "action": {"type": "string", "enum": actions},
            "args": {"type": "object", "properties": {
                "path": {"type": "string"}, "start": {"type": "integer"}, "pattern": {"type": "string"},
                "glob": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"},
                "content": {"type": "string"}, "code": {"type": "string"}, "command": {"type": "string"},
                "summary": {"type": "string"}}},
        },
        "required": ["thought", "action", "args"],
    }


class CodeResult(BaseModel):
    task_id: str
    task: str
    status: str = "running"  # done | failed | limit | error
    pipeline: str = "free"
    summary: str = ""
    diff: str = ""
    changed: list[list[str]] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)  # new images, tables, files
    steps: list[dict] = Field(default_factory=list)
    failed_runs: int = 0
    runs: int = 0
    lessons: list[str] = Field(default_factory=list)
    broken_files: list[str] = Field(default_factory=list)
    checkpoints: list[dict] = Field(default_factory=list)
    applied: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
    created_at: str = ""


def parse_command(command: str) -> list[str]:
    """Only scripts, tests and notebooks run as commands; everything else goes through run_code."""
    parts = shlex.split(command.strip().removeprefix("!"), posix=True)
    if not parts:
        raise WorkspaceError("пустая команда")
    head, rest = parts[0], parts[1:]
    if head in ("python", "python3") and rest and rest[0] not in ("-c",):
        return ["python", *rest]
    if head in ("pytest", "py.test") or (head in ("python", "python3") and rest[:2] == ["-m", "pytest"]):
        return ["python", "-m", "pytest", *(rest[2:] if head.startswith("python") else rest)]
    if head in ("notebook", "jupyter") and rest:
        nb = rest[-1]
        return ["python", "-c", NOTEBOOK_RUNNER, nb]
    raise WorkspaceError("разрешены: python <файл.py>, pytest [...], notebook <файл.ipynb>; для остального — run_code")


NOTEBOOK_RUNNER = (
    "import os, sys, nbformat, nbclient\n"
    "path = sys.argv[1]\n"
    "nb = nbformat.read(path, as_version=4)\n"
    "client = nbclient.NotebookClient(nb, timeout=300, kernel_name='python3',"
    " resources={'metadata': {'path': os.path.dirname(os.path.abspath(path))}})\n"
    "try:\n"
    "    client.execute()\n"
    "finally:\n"
    "    nbformat.write(nb, path)\n"
    "print('notebook executed:', path)\n"
)


def _error_line(run: RunResult) -> str:
    lines = [ln for ln in (run.stderr or run.stdout).strip().splitlines() if ln.strip()]
    err = next((ln for ln in reversed(lines) if re.match(r"^\w*(Error|Exception)\b", ln.strip())), lines[-1] if lines else "")
    return ("таймаут" if run.timed_out else err.strip())[:300]


class CodeAgent:
    def __init__(self, settings: Settings, llm: BaseLLM, sandbox: DockerSandbox, ws: Workspace, task: str,
                 task_id: str, corpus: Path | None = None, catalog_rows: list[dict] | None = None):
        self.settings, self.cfg, self.llm, self.sandbox, self.ws = settings, settings.code, llm, sandbox, ws
        self.corpus = corpus if settings.code.mount_corpus else None
        self.catalog_rows = catalog_rows or []
        self.tools = TOOLS_ACI if self.cfg.aci else TOOLS_OVERWRITE
        self.result = CodeResult(task_id=task_id, task=task, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._files_before = set(ws.files())

    # --- running --------------------------------------------------------------------
    def _run(self, command: list[str]) -> RunResult:
        run = self.sandbox.run(command, self.ws.root, self.corpus)
        self.result.runs += 1
        if not run.ok:
            self.result.failed_runs += 1
            lesson = f"запуск {' '.join(command[:3])[:80]} упал: {_error_line(run)}"
            self.result.lessons.append(lesson)
        return run

    def tool(self, action: str, args: dict, step: int) -> str:
        if action == "list_files":
            files = self.ws.files()
            return "\n".join(files[:300]) + (f"\n(ещё {len(files) - 300})" if len(files) > 300 else "")
        if action == "view_file":
            return self.ws.view(str(args.get("path", "")), int(args.get("start") or 1))
        if action == "search":
            return self.ws.search(str(args.get("pattern", "")), str(args.get("glob") or "*"))
        if action == "edit_file":
            out = self.ws.edit(str(args.get("path", "")), str(args.get("old", "")), str(args.get("new", "")))
            self.ws.commit(f"шаг {step}: правка {args.get('path')}")
            return out
        if action in ("create_file", "write_file"):
            out = self.ws.create_file(str(args.get("path", "")), str(args.get("content", "")), check=action == "create_file")
            self.ws.commit(f"шаг {step}: {'создан' if action == 'create_file' else 'записан'} {args.get('path')}")
            return out
        if action == "run_code":
            snippet = self.ws.root / ".agent" / f"snippet_{step}.py"
            snippet.parent.mkdir(exist_ok=True)
            snippet.write_text(str(args.get("code", "")), encoding="utf-8")
            run = self._run(["python", f".agent/{snippet.name}"])
            self.ws.commit(f"шаг {step}: выполнен фрагмент кода")
            return run.report()
        if action == "run":
            run = self._run(parse_command(str(args.get("command", ""))))
            self.ws.commit(f"шаг {step}: запуск {str(args.get('command'))[:60]}")
            return run.report()
        raise WorkspaceError(f"неизвестный инструмент {action}")

    # --- pipelines (Agentless) ----------------------------------------------------------
    def choose_pipeline(self) -> tuple[str, str]:
        try:
            data = self.llm.chat([{"role": "system", "content": PIPELINE_PROMPT},
                                  {"role": "user", "content": self.result.task}],
                                 json_schema=PIPELINE_SCHEMA, max_tokens=120, purpose="code_pipeline").json()
            pipeline, target = str(data.get("pipeline") or "free"), str(data.get("target") or "")
        except LLMError:
            return "free", ""
        task = self.result.task.lower()
        if pipeline == "plot_logs" and not re.search(r"график|графи|plot|png|картин|диаграм|визуализ", task):
            pipeline = "free"  # a table or CSV from a log is not a plot: the template would draw the wrong artifact
        if pipeline == "collect_metrics" and (re.search(r"[\w./-]+\.(?:log|txt|json|out)\b", task)
                                              or not re.search(r"ноутбук|notebook|ipynb|эксперимент", task)):
            pipeline = "free"  # the template reads the catalog of the notebooks, not a log or another file
        return pipeline, target

    def pipeline_collect_metrics(self) -> str | None:
        """Metrics already extracted (and grounded) in the catalog -> CSV; checked by reading it back."""
        if not self.catalog_rows:
            return None
        import csv
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["notebook", "dataset", "model", "metric", "split", "variant", "value", "cell"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(self.catalog_rows)
        named = re.search(r"[\w./-]+\.csv\b", self.result.task)  # the path the task asks for, if any
        out = named.group(0).lstrip("./") if named else "reports/metrics_from_notebooks.csv"
        self.ws.create_file(out, buf.getvalue(), check=False)
        self.ws.commit("конвейер collect_metrics: метрики из каталога")
        run = self._run(["python", "-c", f"import pandas as pd; df = pd.read_csv({out!r}); "
                                         "print(df.shape); print(df.groupby('notebook').size())"])
        return f"{out}: {len(self.catalog_rows)} строк\n{run.report()}" if run.ok else None

    def pipeline_plot_logs(self, target: str) -> str | None:
        logs = [target] if target and (self.ws.root / target).is_file() else [f for f in self.ws.files() if f.endswith(".log")]
        if not logs:
            return None
        outputs = []
        named = re.search(r"[\w./\\-]+\.png", self.result.task)  # the task may name the output file
        for log in logs[:3]:
            stem = Path(log).stem
            script = f"scripts/plot_{stem}.py"
            png = named.group(0).replace("\\", "/").lstrip("/") if named and len(logs) == 1 else f"reports/{stem}.png"
            (self.ws.root / Path(png).parent).mkdir(parents=True, exist_ok=True)
            self.ws.create_file(script, PLOT_TEMPLATE.format(log=log, out=png))
            run = self._run(["python", script])
            self.ws.commit(f"конвейер plot_logs: {log}")
            if not run.ok:
                return None
            outputs.append(run.report(limit=600))
        return "\n".join(outputs)

    def seed_traceback(self, target: str) -> str | None:
        """fix_traceback: reproduce the failure and localise it before the agent starts editing."""
        if not target:
            return None
        try:
            command = parse_command(target if " " in target or target.startswith(("python", "pytest")) else
                                    ("notebook " + target if target.endswith(".ipynb") else "python " + target))
        except WorkspaceError:
            return None
        run = self._run(command)
        if run.ok:
            return f"команда {target} выполняется без ошибок:\n{run.report(limit=1500)}"
        frames = re.findall(r'File "(?:/work/)?([^"]+)", line (\d+)', run.stderr)
        local = [(f, int(n)) for f, n in frames if not f.startswith("/") and (self.ws.root / f).is_file()]
        view = ""
        if local:
            f, n = local[-1]
            view = "\n\n" + self.ws.view(f, max(1, n - 15), 30)
        return f"воспроизвёл ошибку: {target}\n{run.report(limit=2500)}{view}"

    # --- the loop ------------------------------------------------------------------------
    def run(self) -> CodeResult:
        t0 = time.perf_counter()
        res = self.result
        try:
            res.pipeline, target = self.choose_pipeline()
            done = None
            if res.pipeline == "collect_metrics":
                done = self.pipeline_collect_metrics()
            elif res.pipeline == "plot_logs":
                done = self.pipeline_plot_logs(target)
            if done is not None:
                res.status, res.summary = "done", f"Конвейер {res.pipeline}.\n{done}"
            else:
                seed = self.seed_traceback(target) if res.pipeline == "fix_traceback" else None
                self.loop(seed)
        except SandboxError as exc:
            res.status, res.summary = "error", str(exc)
        res.diff = self.ws.diff()
        res.changed = [list(c) for c in self.ws.changed()]
        res.artifacts = [f for f in self.ws.files() if f not in self._files_before and not f.startswith(".agent/")
                         and Path(f).suffix.lower() in (".png", ".svg", ".csv", ".tsv", ".json", ".html", ".txt", ".md")]
        res.broken_files = self.ws.broken_files()
        res.checkpoints = self.ws.checkpoints()
        res.latency_s = round(time.perf_counter() - t0, 1)
        return res

    def loop(self, seed: str | None) -> None:
        res, cfg = self.result, self.cfg
        schema = _schema(list(self.tools))
        first = f"Задача: {res.task}" + (f"\n\nПодготовка:\n{seed}" if seed else "") + \
                f"\n\nФайлы рабочей копии (первые):\n" + "\n".join(self.ws.files()[:80])
        history: list[dict] = []
        for step in range(1, cfg.max_steps + 1):
            if res.failed_runs >= cfg.max_iterations:
                res.status = "failed"
                res.summary = f"Не удалось за {cfg.max_iterations} неудачных запусков: {res.lessons[-1] if res.lessons else ''}"
                return
            lessons = ("\n\nВыводы из прошлых попыток в этой задаче:\n" + "\n".join(f"- {l}" for l in res.lessons[-6:])
                       if res.lessons else "")
            system = CODE_PROMPT.format(tools="\n".join(f"- {d}" for d in self.tools.values()),
                                        max_iterations=cfg.max_iterations, lessons=lessons)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": first}, *history[-14:]]
            t1 = time.perf_counter()
            try:
                call = self.llm.chat(messages, json_schema=schema, max_tokens=2048, purpose="code_agent").json()
            except LLMError as exc:
                res.status, res.summary = "error", f"LLM: {exc}"
                return
            action, args = str(call.get("action")), call.get("args") or {}
            step_log = {"step": step, "thought": call.get("thought", ""), "action": action,
                        "args": {k: (v if len(str(v)) < 400 else str(v)[:400] + "…") for k, v in args.items()}}
            if action == "finish":
                res.status, res.summary = "done", str(args.get("summary") or call.get("thought") or "")
                res.steps.append({**step_log, "duration_s": round(time.perf_counter() - t1, 2)})
                return
            try:
                observation = self.tool(action, args, step)
            except (WorkspaceError, ValueError) as exc:
                observation = f"ошибка: {exc}"
            step_log.update(observation=observation[:3000], duration_s=round(time.perf_counter() - t1, 2))
            res.steps.append(step_log)
            history += [{"role": "assistant", "content": json.dumps(call, ensure_ascii=False)},
                        {"role": "user", "content": f"Результат {action}:\n{observation[:6000]}"}]
        res.status = "limit"
        res.summary = f"Достигнут лимит шагов ({cfg.max_steps})."


def new_task_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def catalog_metric_rows(engine) -> list[dict]:
    """The catalog's metric values as CSV rows for the collect_metrics pipeline."""
    try:
        rows = []
        for e in engine.experiments():
            for m in e["metrics"]:
                rows.append({"notebook": e["file_path"], "dataset": e["dataset"], "model": e["model"], "metric": m["name"],
                             "split": m["split"], "variant": m["variant"], "value": m["value"], "cell": m["cell"]})
        return rows
    except Exception:
        return []


__all__ = ["CodeAgent", "CodeResult", "TEXT_EXTS", "catalog_metric_rows", "new_task_id", "parse_command"]
