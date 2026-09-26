"""Э15: git working copy, ACI edits, the fix loop, pipelines, apply only on confirmation, the Docker sandbox."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag_agent.code.agent import CodeAgent, parse_command
from rag_agent.code.sandbox import DockerSandbox, RunResult
from rag_agent.code.workspace import Workspace, WorkspaceError
from rag_agent.config import CodeConfig
from rag_agent.engine import Engine
from tests.conftest import FakeLLM


class HostSandbox:
    """Runs commands with the host Python inside the working copy: tests only, no isolation."""

    def __init__(self):
        self.commands: list[list[str]] = []

    def run(self, command, work, corpus=None, timeout_s=None) -> RunResult:
        self.commands.append(command)
        argv = [sys.executable, *command[1:]] if command[0] == "python" else command
        env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(argv, cwd=work, capture_output=True, text=True, encoding="utf-8", timeout=120, env=env)
        return RunResult(command=command, exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "ratio.py").write_text("def ratio(a, b):\n    return a / b\n\n\nprint(ratio(1, 0))\n",
                                               encoding="utf-8")
    (root / "logs").mkdir()
    (root / "logs" / "train.log").write_text(
        "start lr=0.1\n" + "".join(f"epoch={i} loss={1 / (i + 1):.3f} acc={0.5 + i / 20:.3f}\n" for i in range(6)),
        encoding="utf-8")
    (root / "Аккаунты").mkdir()
    (root / "Аккаунты" / "keys.txt").write_text("secret", encoding="utf-8")
    return root


def test_workspace_checkpoints_aci_and_apply(tmp_path: Path, project: Path):
    ws = Workspace.create(tmp_path / "ws", project, exclude=["Аккаунты"])
    assert "scripts/ratio.py" in ws.files() and not any("Аккаунты" in f for f in ws.files())  # exclusions hold
    assert "    2|     return a / b" in ws.view("scripts/ratio.py")
    with pytest.raises(WorkspaceError, match="отклонена"):
        ws.edit("scripts/ratio.py", "return a / b", "return a / (b")  # broken syntax is not written
    assert "return a / b" in (ws.root / "scripts" / "ratio.py").read_text(encoding="utf-8")
    with pytest.raises(WorkspaceError, match="ровно 1"):
        ws.edit("scripts/ratio.py", "a", "c")
    ws.edit("scripts/ratio.py", "return a / b", "return a / b if b else float('nan')")
    commit = ws.commit("fix")
    assert commit and "+    return a / b if b else float('nan')" in ws.diff()
    for bad in ("../outside.py", r"..\..\x.py", ".git/config"):
        with pytest.raises(WorkspaceError):
            ws.path(bad)
    assert ws.root in ws.path("/etc/passwd").parents  # an absolute path is taken inside the working copy
    target = tmp_path / "user_folder"
    target.mkdir()
    assert ws.apply_to(target) == ["scripts/ratio.py"]
    ws.rollback("initial")
    assert ws.diff() == "" and ws.changed() == []


def test_commands_are_restricted():
    assert parse_command("python scripts/x.py --fast") == ["python", "scripts/x.py", "--fast"]
    assert parse_command("pytest -q tests")[:3] == ["python", "-m", "pytest"]
    assert parse_command("notebook experiments/a.ipynb")[:2] == ["python", "-c"]
    for bad in ("rm -rf /", "curl http://x", "python -c 'import os'", "bash -c ls"):
        with pytest.raises(WorkspaceError):
            parse_command(bad)


def make_agent(settings, tmp_path, project, script, **cfg) -> CodeAgent:
    for k, v in cfg.items():
        setattr(settings.code, k, v)
    ws = Workspace.create(tmp_path / "ws", project, exclude=["Аккаунты"])
    return CodeAgent(settings, FakeLLM(settings, agent=script), HostSandbox(), ws, "Исправь падение ratio.py", "t1")


FIX = [
    {"thought": "", "action": "run", "args": {"command": "python scripts/ratio.py"}},
    {"thought": "деление на ноль", "action": "edit_file",
     "args": {"path": "scripts/ratio.py", "old": "print(ratio(1, 0))", "new": "print(ratio(1, 2))"}},
    {"thought": "", "action": "run", "args": {"command": "python scripts/ratio.py"}},
    {"thought": "", "action": "finish", "args": {"summary": "исправлен вызов, скрипт печатает 0.5"}},
]


def test_fix_loop_uses_execution_feedback(settings, tmp_path: Path, project: Path):
    agent = make_agent(settings, tmp_path, project, list(FIX))
    res = agent.run()
    assert res.status == "done" and res.runs == 2 and res.failed_runs == 1
    assert "ZeroDivisionError" in res.lessons[0] and "0.5" in res.steps[2]["observation"]
    # the lesson from the failed run is in the system prompt of the following steps (Reflexion)
    assert "Выводы из прошлых попыток" in agent.llm.calls[-1][0]["content"]
    assert "+print(ratio(1, 2))" in res.diff and res.broken_files == []
    assert len(res.checkpoints) == 2 and (project / "scripts" / "ratio.py").read_text(encoding="utf-8").endswith("(1, 0))\n")


def test_single_attempt_stops_after_the_first_failure(settings, tmp_path: Path, project: Path):
    res = make_agent(settings, tmp_path, project, list(FIX), max_iterations=1).run()  # H13 baseline
    assert res.status == "failed" and res.runs == 1 and "+print(ratio(1, 2))" not in res.diff


def test_overwrite_baseline_can_break_files(settings, tmp_path: Path, project: Path):
    script = [{"thought": "", "action": "write_file", "args": {"path": "scripts/ratio.py", "content": "def ratio(a, b:\n"}},
              {"thought": "", "action": "finish", "args": {"summary": ""}}]
    agent = make_agent(settings, tmp_path, project, script, aci=False)  # H14 baseline: no syntax check
    assert "edit_file" not in agent.tools and "write_file" in agent.tools
    assert agent.run().broken_files == ["scripts/ratio.py"]


def test_plot_logs_pipeline(settings, tmp_path: Path, project: Path):
    pytest.importorskip("pandas")  # the generated script runs with this Python (the sandbox has both)
    pytest.importorskip("matplotlib")
    ws = Workspace.create(tmp_path / "ws", project)
    llm = FakeLLM(settings, pipeline={"pipeline": "plot_logs", "target": "logs/train.log"})
    res = CodeAgent(settings, llm, HostSandbox(), ws, "Построй график по логу обучения", "t2").run()
    assert res.status == "done" and res.pipeline == "plot_logs" and "agent" not in llm.kinds
    assert "reports/train.png" in res.artifacts and "6 эпох" in res.summary


def test_engine_code_task_applies_only_on_request(settings, fake_embedder, project: Path):
    settings.corpus.include_ext = [".py", ".log"]
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings, agent=list(FIX)))
    eng.index_folder(project)
    res = eng.code_task("Исправь падение ratio.py", sandbox=HostSandbox())
    assert res.status == "done" and res.changed == [["M", "scripts/ratio.py"]]
    assert "(1, 0))" in (project / "scripts" / "ratio.py").read_text(encoding="utf-8")  # untouched before apply
    assert eng.code_result(res.task_id).diff == res.diff
    applied = eng.apply_code(res.task_id)
    assert applied.applied == ["scripts/ratio.py"]
    assert "(1, 2))" in (project / "scripts" / "ratio.py").read_text(encoding="utf-8")
    eng._thread.join(30)
    eng.close()


docker_ready = DockerSandbox(CodeConfig(image="python:3.12-slim")).available()


@pytest.mark.skipif(not docker_ready, reason="Docker or the python:3.12-slim image is not available")
def test_docker_sandbox_isolation(tmp_path: Path):
    box = DockerSandbox(CodeConfig(image="python:3.12-slim", timeout_s=60))
    (tmp_path / "probe.py").write_text(
        "import os, urllib.request\n"
        "print('uid', os.getuid())\n"
        "try:\n    urllib.request.urlopen('http://example.com', timeout=3); print('NET OPEN')\n"
        "except Exception as e:\n    print('net blocked')\n"
        "try:\n    open('/usr/x', 'w'); print('ROOT WRITABLE')\n"
        "except OSError:\n    print('root read-only')\n"
        "print('secret' in os.environ.get('RAG_TEST_SECRET', ''))\n"
        "open('out.txt', 'w').write('ok')\n", encoding="utf-8")
    os.environ["RAG_TEST_SECRET"] = "secret"  # host variables must not reach the container
    run = box.run(["python", "probe.py"], tmp_path)
    assert run.ok and run.stdout.split() == ["uid", "1000", "net", "blocked", "root", "read-only", "False"]
    assert (tmp_path / "out.txt").read_text() == "ok"
    slow = box.run(["python", "-c", "import time; time.sleep(30)"], tmp_path, timeout_s=3)
    assert slow.timed_out and not slow.ok


def test_code_eval_harness(settings, tmp_path: Path, project: Path):
    from rag_agent.evaluation.code_eval import CodeTaskSet, run_code_eval, summarize_code, validate_tasks

    ts = CodeTaskSet.model_validate({"name": "t", "corpus": str(project), "tasks": [{
        "id": "fix-ratio", "task": "Исправь падение ratio.py",
        "patch": [{"file": "scripts/ratio.py", "old": "return a / b", "new": "return a / b"}],
        "check": "import subprocess, sys\n\ndef test_runs():\n"
                 "    assert subprocess.run([sys.executable, 'scripts/ratio.py']).returncode == 0\n"}]})
    # the "original" project is itself broken, so validation flags the task: a bug must be planted by the patch
    [problem] = validate_tasks(ts, HostSandbox(), log=lambda _: None)
    assert problem.startswith("fix-ratio: check fails on the original corpus")
    results = run_code_eval(settings, ts, FakeLLM(settings, agent=list(FIX)), HostSandbox(), log=lambda _: None,
                            work_root=tmp_path)
    assert [(r.id, r.solved, r.failed_runs) for r in results] == [("fix-ratio", True, 1)]
    s = summarize_code(results, n_boot=50)
    assert s["solved"]["mean"] == 1.0 and s["broken_files_share"] == 0.0
