"""Code agent (ТЗ ч.2 S17, Э15): plan -> edit in the working copy -> run in the sandbox ->
read the errors -> fix, at most ``code.max_iterations`` failed runs.

Actions are JSON calls under a schema. Code is still the main action (CodeAct): the
agent runs Python snippets, scripts, tests and notebooks and gets stdout, stderr and
the traceback back. Files are changed through the ACI of SWE-agent (numbered views,
search, fragment replacement with a syntax check); the H14 baseline swaps it for a
plain whole-file overwrite. Lessons from failed runs are restated after every new
failure (Reflexion); the prompt only grows, so the model server reuses its cache. Typical tasks go through fixed pipelines first
(Agentless): metrics from notebooks, plots from logs, a fix from a traceback.

Nothing leaves the working copy: the result is a diff, and ``apply_changes`` copies
it into the user's folder only after an explicit confirmation (policy: always).

In the chat the task comes rewritten by the router into a standalone one, together with
the recent dialogue and the results of earlier code tasks of the same dialogue, so "try
again" or "now plot it" continue the previous work instead of starting blind.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from rag_agent.code.sandbox import PKGS, DockerSandbox, RunResult, SandboxError
from rag_agent.code.workspace import DATA_EXTS, TEXT_EXTS, Workspace, WorkspaceError, check_syntax
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

CODE_PROMPT = """Ты — код-агент. Решаешь задачу пользователя в рабочей копии его проекта. Рабочая копия под git: все изменения покажут пользователю как diff и перенесут в его папку только после подтверждения. Код исполняется в песочнице {network}: Python 3.12, numpy, pandas, scikit-learn, matplotlib, pytest, nbclient. Текущая папка — рабочая копия (/work) с текстовыми файлами проекта пользователя.

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
- project — написать новую программу из нескольких модулей: сервис, приложение, утилиту, бота, API по данным пользователя (target — файл данных, если назван);
- free — всё остальное (правка, скрипт из одного файла, расчёт, график, таблица).
Верни JSON: {"pipeline": "...", "target": "..."}"""

PIPELINE_SCHEMA = {
    "type": "object",
    "properties": {"pipeline": {"type": "string", "enum": ["collect_metrics", "plot_logs", "fix_traceback", "project", "free"]},
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


# --- a new program of several modules (the "project" pipeline) ------------------------------------------
# Plan -> write the files one by one (each sees the interfaces of the ones before) -> check (compile, tests,
# the plan's own check command) -> the usual fix loop on what fails. The program is a package in its own
# folder at the root of the user's folder, so after "Применить" it runs from there: python -m <name> ...

PROJECT_PLAN_PROMPT = """Ты — архитектор. Пользователь просит написать программу (сервис, приложение, утилиту) по его данным. Спроектируй её как Python-пакет в отдельной папке в корне папки пользователя.
Правила:
- ООП: классы с одной ответственностью — модели данных (dataclass), доступ к данным (репозиторий), бизнес-логика (сервисы), интерфейс (командная строка argparse и, если просят сервис, HTTP API); аннотации типов;
- доступны только стандартная библиотека и numpy, pandas, scikit-learn, matplotlib, pytest{net}; веб-сервис без внешних пакетов — на http.server и json;
- данные пользователя программа только читает (SQLite — в режиме только чтения); пути к ним по умолчанию — относительно корня папки пользователя, с переопределением аргументом;
- файлы (пути относительно папки проекта): __init__.py, __main__.py (точка входа: python -m {{name}} <команда>), модули по слоям, tests/test_*.py (pytest; на временной копии данных или своей маленькой базе во tmp_path), README.md; requirements.txt — только если нужны пакеты вне списка выше;
- 4–12 файлов, каждый до ~300 строк;
- run — как запустить из корня папки пользователя (python -m {{name}} ...); check — команда проверки, которая сама завершается (например, отчёт или --self-test; не запуск сервера), тоже из корня папки.
name — имя пакета (латиница, snake_case). summary и architecture — по-русски.
Верни JSON: {{"name": "...", "title": "...", "summary": "...", "architecture": "какие классы и слои, кто за что отвечает", "files": [{{"path": "...", "purpose": "...", "api": "классы, методы, функции с сигнатурами"}}], "run": "...", "check": "...", "requirements": []}}"""

PROJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "title": {"type": "string"}, "summary": {"type": "string"},
        "architecture": {"type": "string"},
        "files": {"type": "array", "items": {"type": "object", "properties": {
            "path": {"type": "string"}, "purpose": {"type": "string"}, "api": {"type": "string"}},
            "required": ["path", "purpose", "api"]}},
        "run": {"type": "string"}, "check": {"type": "string"},
        "requirements": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "title", "summary", "architecture", "files", "run", "check", "requirements"],
}

PROJECT_FILE_PROMPT = """Ты пишешь один файл Python-проекта по утверждённому плану. Код законченный и рабочий, без заглушек и TODO:
- ООП там, где у сущности есть состояние и поведение: классы с одной ответственностью, dataclass для моделей, аннотации типов, docstring у классов и публичных методов;
- импорты внутри проекта — абсолютные от пакета: from {name}.<модуль> import <Класс>; используй интерфейсы уже написанных файлов как есть;
- только стандартная библиотека и numpy, pandas, scikit-learn, matplotlib, pytest{net};
- данные пользователя только читай (sqlite3.connect(f"file:{{Path(path).as_posix()}}?mode=ro", uri=True)); путь по умолчанию — Path(__file__).resolve().parents[1] / "<путь от корня папки пользователя>", с переопределением аргументом;
- тесты (pytest) не трогают данные пользователя: своя маленькая база или копия во tmp_path; запуск — python -m pytest {name}/tests из корня папки;
- README.md — по-русски: что делает программа, структура (модули и классы), как запустить (команды из корня папки), примеры, как запустить тесты.
Верни только содержимое файла, без пояснений."""

PROJECT_FIX = """Создан проект {root}/ по плану ниже, но проверка не прошла. Найди причину по выводу, исправь файлы проекта (view_file, edit_file) и запусти проверки снова — сначала упавшую. Не запускай сервер без конца: для проверки — тесты и команда проверки из плана. Когда все проверки проходят — finish.

Проверки (из корня рабочей копии): {checks}

Вывод:
{report}

План: {plan}"""

# reads the user's data for the plan: tables and columns, row counts, a few rows (argv: the files)
SURVEY_SCRIPT = """import sqlite3, sys
from pathlib import Path

for rel in sys.argv[1:]:
    p = Path(rel)
    print(f"== {rel} ({p.stat().st_size // 1024} КБ)")
    try:
        if p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            for name, sql in con.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'view') "
                                         "AND name NOT LIKE 'sqlite_%'"):
                n = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                print(f"{sql};  -- {n} строк")
                cur = con.execute(f'SELECT * FROM "{name}" LIMIT 3')
                cols = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    print("   ", dict(zip(cols, row)))
            con.close()
        else:
            import pandas as pd
            read = {".csv": pd.read_csv, ".tsv": lambda f: pd.read_csv(f, sep="\\t"), ".json": pd.read_json,
                    ".parquet": pd.read_parquet, ".feather": pd.read_feather, ".xlsx": pd.read_excel,
                    ".xls": pd.read_excel}[p.suffix.lower()]
            df = read(p)
            print(f"{len(df)} строк; колонки: {dict(df.dtypes.astype(str))}")
            print(df.head(3).to_string(max_colwidth=60))
    except Exception as exc:
        print(f"   не прочитан: {type(exc).__name__}: {exc}")
"""


def _unwrap(text: str, suffix: str) -> str:
    """The file's content from a model reply: without an outer ``` fence; for code, the longest code block."""
    t = text.strip()
    if t.startswith("```") and t.endswith("```") and "\n" in t:
        return t[t.index("\n") + 1:-3].rstrip() + "\n"
    if suffix == ".py":
        blocks = re.findall(r"```[\w.+-]*\n(.*?)```", t, re.DOTALL)
        if blocks:
            return max(blocks, key=len).rstrip() + "\n"
    return t + "\n"


def interfaces(ws: Workspace, root: str, limit: int = 6000) -> str:
    """Classes, public methods and functions of the project's modules, for the files written after them."""
    out = []
    for rel in ws.files():
        if not (rel.startswith(root + "/") and rel.endswith(".py")) or "/tests/" in rel:
            continue
        try:
            tree = ast.parse((ws.root / rel).read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        lines = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                lines.append(f"class {node.name}({bases}):" if bases else f"class {node.name}:")
                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        lines.append(f"    {ast.unparse(item)}")
                    elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                            not item.name.startswith("_") or item.name == "__init__"):
                        ret = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                        lines.append(f"    def {item.name}({ast.unparse(item.args)}){ret}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                lines.append(f"def {node.name}({ast.unparse(node.args)}){ret}")
        if lines:
            out.append(f"# {rel}\n" + "\n".join(lines))
    return "\n\n".join(out)[:limit]


NETWORK_OFF = "без сети"
NETWORK_ON = ("с доступом в интернет (пользователь разрешил его для этой задачи): можно скачивать данные; "
              'недостающий пакет ставь командой run "pip install пакет"')


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
    network: bool = False  # the runs had network (the user's confirmed toggle)
    project: dict | None = None  # the "project" pipeline: root, title, summary, architecture, files, run, check


def parse_command(command: str, network: bool = False) -> list[str]:
    """Only scripts, tests and notebooks run as commands; everything else goes through run_code.
    With network, ``pip install`` puts packages into the working copy (never into the image)."""
    parts = shlex.split(command.strip().removeprefix("!"), posix=True)
    if not parts:
        raise WorkspaceError("пустая команда")
    head, rest = parts[0], parts[1:]
    if head in ("python", "python3") and rest[:2] == ["-m", "pip"]:
        head, rest = "pip", rest[2:]
    if head in ("pip", "pip3"):
        if not network:
            raise WorkspaceError("установка пакетов недоступна: в песочнице нет сети; обойдись установленными "
                                 "(numpy, pandas, scikit-learn, matplotlib, scipy)")
        pkgs = [a for a in rest[1:] if not a.startswith("-")] if rest[:1] == ["install"] else []
        if not pkgs:
            raise WorkspaceError("разрешено только pip install <пакеты>")
        return ["python", "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check", "--target", PKGS,
                *pkgs]
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
                 task_id: str, corpus: Path | None = None, catalog_rows: list[dict] | None = None,
                 context: str = "", network: bool = False):
        self.settings, self.cfg, self.llm, self.sandbox, self.ws = settings, settings.code, llm, sandbox, ws
        self.corpus = corpus if settings.code.mount_corpus else None
        self.catalog_rows = catalog_rows or []
        self.tools = TOOLS_ACI if self.cfg.aci else TOOLS_OVERWRITE
        self.context = context.strip()  # the recent dialogue and earlier code tasks of the dialogue
        self.network = network
        self.result = CodeResult(task_id=task_id, task=task, network=network,
                                 created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._files_before = set(ws.files())

    # --- running --------------------------------------------------------------------
    def _run(self, command: list[str]) -> RunResult:
        run = self.sandbox.run(command, self.ws.root, self.corpus, network=self.network)
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
            run = self._run(parse_command(str(args.get("command", "")), self.network))
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
        if pipeline == "project" and not re.search(
                r"сервис|приложени|программ|проект|бот\b|api|cli|утилит|модул|пакет|библиотек|ооп|класс|архитектур|"
                r"service|app\b|application|package", task):
            pipeline = "free"  # a one-file script or an edit is not worth a planned package
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

    # --- a new program (project) --------------------------------------------------------
    def survey_data(self, target: str = "") -> str:
        """The user's data as the program will see it: tables, columns, row counts, sample rows."""
        task = self.result.task.lower()
        files = [f for f in self.ws.files() if not f.startswith(".agent/")]
        data = [f for f in files if Path(f).suffix.lower() in DATA_EXTS | {".csv", ".tsv", ".json"}]
        named = [f for f in data if f == target or Path(f).name.lower() in task
                 or (len(Path(f).stem) >= 4 and Path(f).stem.lower() in task)]
        dbs = [f for f in data if Path(f).suffix.lower() in DATA_EXTS]
        tables = [f for f in data if Path(f).suffix.lower() in (".csv", ".tsv")]
        chosen = list(dict.fromkeys(named + dbs + tables))[:8]
        if not chosen:
            return "файлов данных (баз, таблиц) в папке нет"
        script = self.ws.root / ".agent" / "survey.py"
        script.parent.mkdir(exist_ok=True)
        script.write_text(SURVEY_SCRIPT, encoding="utf-8")
        run = self._run(["python", ".agent/survey.py", *chosen])
        return run.stdout[-6000:] if run.ok else run.report(limit=3000)

    def plan_project(self, survey: str) -> dict | None:
        net = "; сеть есть — можно и другие пакеты, указав их в requirements.txt" if self.network else ""
        user = (f"Задача: {self.result.task}\n\n" + (f"Контекст из диалога:\n{self.context}\n\n" if self.context else "")
                + f"Данные пользователя (обзор):\n{survey}\n\nФайлы папки (первые):\n" + "\n".join(self.ws.files()[:80]))
        try:
            plan = self.llm.chat([{"role": "system", "content": PROJECT_PLAN_PROMPT.format(net=net)},
                                  {"role": "user", "content": user}],
                                 json_schema=PROJECT_SCHEMA, max_tokens=6000, purpose="code_project_plan").json()
        except LLMError:
            return None
        name = re.sub(r"[^a-z0-9_]+", "_", str(plan.get("name") or "").lower()).strip("_")
        name = name if name and name[0].isalpha() else f"app_{name}".rstrip("_")
        root, n = name, 2
        while (self.ws.root / root).exists():  # never write over a folder of the user
            root, n = f"{name}_{n}", n + 1
        files, seen = [], set()
        for spec in plan.get("files") or []:
            path = str(spec.get("path") or "").replace("\\", "/").strip().lstrip("./")
            path = path.removeprefix(f"{name}/").removeprefix(f"{root}/")
            if not path or ".." in path.split("/") or path in seen:
                continue
            seen.add(path)
            files.append({"path": path, "purpose": str(spec.get("purpose") or ""), "api": str(spec.get("api") or "")})
        for path, purpose in (("__init__.py", "пакет"), ("__main__.py", "точка входа: python -m " + root)):
            if path not in seen:
                files.insert(0, {"path": path, "purpose": purpose, "api": ""})
        if not any(f["path"].startswith("tests/") for f in files):
            files.append({"path": f"tests/test_{root}.py", "purpose": "тесты pytest основных классов", "api": ""})
        if "README.md" not in seen:
            files.append({"path": "README.md", "purpose": "описание, структура, как запустить", "api": ""})
        # the order of writing: code first (each file sees the ones before), then tests, then the README
        files.sort(key=lambda f: (f["path"] == "README.md", f["path"].startswith("tests/"), f["path"] == "__main__.py"))
        def fix(cmd) -> str:  # the commands name the package; a taken folder name got a suffix
            return re.sub(rf"\b{re.escape(name)}\b", root, str(cmd or "")) if root != name else str(cmd or "")

        return {"root": root, "title": str(plan.get("title") or root), "summary": str(plan.get("summary") or ""),
                "architecture": str(plan.get("architecture") or ""), "files": files, "run": fix(plan.get("run")),
                "check": fix(plan.get("check")), "requirements": [str(r) for r in plan.get("requirements") or []]}

    def write_project_file(self, plan: dict, spec: dict, survey: str) -> str:
        root, rel = plan["root"], f"{plan['root']}/{spec['path']}"
        net = "; сеть есть — другие пакеты перечисли в requirements.txt" if self.network else ""
        brief = {k: plan[k] for k in ("root", "title", "summary", "architecture", "run", "check")}
        brief["files"] = [f"{f['path']} — {f['purpose']}" + (f" ({f['api']})" if f["api"] else "") for f in plan["files"]]
        user = (f"Задача пользователя: {self.result.task}\n\nПроект: папка {root}/ (пакет Python) в корне папки "
                f"пользователя.\nПлан: {json.dumps(brief, ensure_ascii=False)}\n\nДанные пользователя (обзор):\n"
                f"{survey[:5000]}\n\nУже написанные файлы (интерфейсы):\n{interfaces(self.ws, root) or '—'}\n\n"
                f"Файл: {rel}\nНазначение: {spec['purpose']}\nИнтерфейс: {spec['api'] or '—'}")
        messages = [{"role": "system", "content": PROJECT_FILE_PROMPT.format(name=root, net=net)},
                    {"role": "user", "content": user}]
        content = ""
        for _ in range(2):  # a file that does not parse gets one more try with the error
            try:
                content = _unwrap(self.llm.chat(messages, max_tokens=12000, purpose="code_project_file").content,
                                  Path(rel).suffix.lower())
            except LLMError as exc:
                return f"{rel}: не написан ({exc})"
            error = check_syntax(rel, content)
            if not error:
                break
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"Файл не разбирается: {error}. Верни исправленный файл целиком."}]
        out = self.ws.create_file(rel, content, check=False)  # a broken file is left to the fix loop
        self.result.steps.append({"step": len(self.result.steps) + 1, "thought": spec["purpose"], "action": "create_file",
                                  "args": {"path": rel}, "observation": out})
        return out

    def verify_project(self, plan: dict) -> tuple[bool, str, list[str]]:
        """Compile, the tests, the plan's check command; stops at the first failure (it is fixed first)."""
        root = plan["root"]
        checks = [f"python -m compileall -q {root}"]
        if any(f.startswith(f"{root}/tests/") for f in self.ws.files()):
            checks.append(f"pytest -q {root}/tests")
        if plan["check"]:
            checks.append(plan["check"])
        reports, ok = [], True
        for cmd in checks:
            try:
                command = parse_command(cmd, self.network)
            except WorkspaceError as exc:
                reports.append(f"$ {cmd}\nпропущено: {exc}")
                continue
            run = self._run(command)
            reports.append(f"$ {cmd}\n{run.report(limit=2500)}")
            if not run.ok:
                ok = False
                break
        self.ws.commit(f"проект {root}: проверка")
        return ok, "\n\n".join(reports), checks

    def build_project(self, target: str) -> None:
        res, cfg = self.result, self.cfg
        survey = self.survey_data(target)
        plan = self.plan_project(survey)
        if plan is None:  # no plan from the model: the usual loop takes the task as it is
            res.pipeline = "free"
            self.loop(None)
            return
        res.project = {k: plan[k] for k in ("root", "title", "summary", "architecture", "run", "check", "requirements")}
        res.project["files"] = [f"{plan['root']}/{f['path']}" for f in plan["files"]]
        for spec in plan["files"]:
            self.write_project_file(plan, spec, survey)
        self.ws.commit(f"проект {plan['root']}: файлы по плану")
        ok, report, checks = self.verify_project(plan)
        notes = ""
        if not ok:
            brief = json.dumps({k: plan[k] for k in ("title", "architecture", "run", "check")} |
                               {"files": res.project["files"]}, ensure_ascii=False)
            self.loop(PROJECT_FIX.format(root=plan["root"], checks="; ".join(checks), report=report[-5000:], plan=brief),
                      max_steps=cfg.project_max_steps, max_iterations=cfg.project_max_iterations)
            notes = res.summary
            if res.status == "done":  # the agent says it is fixed: the checks say whether it is
                ok, report, _ = self.verify_project(plan)
                if not ok:
                    res.status = "failed"
        if ok:
            res.status = "done"
        res.summary = (f"Проект {plan['root']}/ — {plan['title']}. {plan['summary']}\n"
                       + (f"\nИсправления: {notes}\n" if notes else "")
                       + f"\nПроверка{' пройдена' if ok else ' не пройдена'}:\n{report[-3000:]}")

    # --- the loop ------------------------------------------------------------------------
    def run(self) -> CodeResult:
        t0 = time.perf_counter()
        res = self.result
        try:
            res.pipeline, target = self.choose_pipeline()
            done = None
            if res.pipeline == "project":
                self.build_project(target)
                done = False
            elif res.pipeline == "collect_metrics":
                done = self.pipeline_collect_metrics()
            elif res.pipeline == "plot_logs":
                done = self.pipeline_plot_logs(target)
            if done is False:
                pass  # the project pipeline has set the status and the summary
            elif done is not None:
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

    def loop(self, seed: str | None, max_steps: int | None = None, max_iterations: int | None = None) -> None:
        """The ReAct loop. The prompt only grows (system, task, then every step appended), so the model server
        reuses the cached prefix and reads only the new step: ~3 s a step instead of re-reading ~9k tokens
        (~60-90 s for a 27B model). Old outputs are cut once the history is too long (one cache miss), and the
        lessons of failed runs come with the failure, not in the system prompt (which would change the prefix)."""
        res, cfg = self.result, self.cfg
        max_steps, max_iterations = max_steps or cfg.max_steps, max_iterations or cfg.max_iterations
        schema = _schema(list(self.tools))
        system = CODE_PROMPT.format(tools="\n".join(f"- {d}" for d in self.tools.values()),
                                    max_iterations=max_iterations, lessons="",
                                    network=NETWORK_ON if self.network else NETWORK_OFF)
        first = f"Задача: {res.task}" + (f"\n\nКонтекст из диалога с пользователем (для справки):\n{self.context}"
                                          if self.context else "") + (f"\n\nПодготовка:\n{seed}" if seed else "") + \
                f"\n\nФайлы рабочей копии (первые):\n" + "\n".join(self.ws.files()[:80])
        history: list[dict] = []
        seen: dict[tuple[str, int], int] = {}  # file fragments shown (path, start) -> the step
        idle = 0  # steps in a row that only read
        for step in range(1, max_steps + 1):
            if res.failed_runs >= max_iterations:
                res.status = "failed"
                res.summary = f"Не удалось за {max_iterations} неудачных запусков: {res.lessons[-1] if res.lessons else ''}"
                return
            if sum(len(m["content"]) for m in history) > HISTORY_BUDGET:
                history, seen = compact_history(history), {}
            messages = [{"role": "system", "content": system}, {"role": "user", "content": first}, *history]
            t1 = time.perf_counter()
            try:
                call = self.llm.chat(messages, json_schema=schema, max_tokens=8192, purpose="code_agent").json()
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
            path = str(args.get("path") or "").replace("\\", "/").lstrip("./")
            try:
                start = int(args.get("start") or 1)
            except (TypeError, ValueError):
                start = 1
            view = (path, start) if action == "view_file" else None
            failures = res.failed_runs
            if view in seen:  # a model that has forgotten reads the same files in a circle
                observation = (f"этот фрагмент уже показан на шаге {seen[view]} — он выше в истории. Не перечитывай: "
                               "исправь файл или запусти упавшую проверку.")
            else:
                try:
                    observation = self.tool(action, args, step)
                except (WorkspaceError, ValueError) as exc:
                    observation = f"ошибка: {exc}"
                if view:
                    seen[view] = step
            if action in ("edit_file", "create_file", "write_file"):
                seen = {k: v for k, v in seen.items() if k[0] != path}  # the file changed: it may be read again
            idle = 0 if action in ("edit_file", "create_file", "write_file", "run", "run_code") else idle + 1
            if res.failed_runs > failures:  # Reflexion: the lessons so far, right after the new failure
                observation += "\n\nВыводы из прошлых попыток в этой задаче:\n" + "\n".join(
                    f"- {lesson}" for lesson in res.lessons[-6:])
            if idle >= 6:
                observation += (f"\n\nТы уже {idle} шагов подряд только читаешь. Хватит изучать: сделай правку или "
                                "запусти проверку, чтобы увидеть ошибку.")
            step_log.update(observation=observation[:3000], duration_s=round(time.perf_counter() - t1, 2))
            res.steps.append(step_log)
            history += [{"role": "assistant", "content": json.dumps(call, ensure_ascii=False)},
                        {"role": "user", "content": f"Результат {action}:\n{observation[:6000]}"}]
        res.status = "limit"
        res.summary = f"Достигнут лимит шагов ({max_steps})."


HISTORY_BUDGET = 90_000  # characters of the step history (~30k tokens) before old outputs are cut
KEEP_WHOLE = 6  # the last steps keep their outputs whole


def compact_history(history: list[dict]) -> list[dict]:
    """Old tool outputs cut to their first line: done rarely, since every change of the prompt's beginning
    makes the model read the whole prompt again."""
    cut = len(history) - 2 * KEEP_WHOLE
    out = []
    for i, m in enumerate(history):
        if i < cut and m["role"] == "user" and len(m["content"]) > 400:
            lines = m["content"].split("\n")
            head = "\n".join(lines[:2])[:300]
            m = {"role": "user", "content": f"{head}\n(вывод сокращён, чтобы освободить контекст; если нужен — запроси снова)"}
        out.append(m)
    return out


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
