"""One agent in the chat: hand-off of code tasks, run_code in the research agent, the sandbox network
toggle with confirmation, capabilities in the general answer, the web suggestion, the model picker."""

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_agent.api import create_app
from rag_agent.code.agent import parse_command
from rag_agent.code.workspace import WorkspaceError
from rag_agent.engine import Engine
from rag_agent.llm import LLMError, ModelSwitch
from rag_agent.router import needs_code
from tests.conftest import FakeLLM
from tests.test_code_agent import FIX, HostSandbox


class RecordingSandbox(HostSandbox):
    def __init__(self):
        super().__init__()
        self.network: list[bool] = []

    def run(self, command, work, corpus=None, timeout_s=None, network=False):
        self.network.append(network)
        return super().run(command, work, corpus, timeout_s)


def chat_engine(settings, fake_embedder, folder: Path, sandbox=None, **llm) -> Engine:
    settings.code.chat = "auto"
    settings.corpus.include_ext = [".py", ".log"]
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings, **llm))
    eng.index_folder(folder)
    eng._sandbox = sandbox or RecordingSandbox()
    eng._sandbox_ok = (True, time.monotonic())  # Docker "running"
    return eng


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "ratio.py").write_text("def ratio(a, b):\n    return a / b\n\n\nprint(ratio(1, 0))\n",
                                               encoding="utf-8")
    (root / "logs").mkdir()
    (root / "logs" / "train.log").write_text("epoch=1 loss=0.5\nepoch=2 loss=0.25\n", encoding="utf-8")
    return root


def test_code_check_runs_only_for_action_requests(settings):
    llm = FakeLLM(settings, code=True)
    assert not needs_code("Какой learning rate в эксперименте A?", llm).checked  # no action word: no model call
    assert llm.kinds == []
    decision = needs_code("Исправь падение ratio.py", llm)
    assert decision.checked and decision.code and llm.kinds == ["route_code"]


def test_chat_hands_a_code_task_to_the_code_agent(settings, fake_embedder, project: Path):
    eng = chat_engine(settings, fake_embedder, project, agent=list(FIX), code=True)
    history = [{"role": "user", "content": "Запусти scripts/ratio.py"},
               {"role": "assistant", "content": "[Код-агент, ошибка] docker не запущен"}]
    ans = eng.ask("Исправь падение ratio.py", history=history)
    assert ans.route == "code" and ans.answerable and ans.code["status"] == "done"
    assert ans.code["changed"] == [["M", "scripts/ratio.py"]] and ans.answer == ans.code["summary"]
    assert [s.name for s in ans.trace][:3] == ["route", "route_code", "code_agent"]
    # the code agent saw the dialogue: "try again" continues the previous work
    first = next(c for c in eng.llm.calls if "Задача:" in str(c[-1].get("content")))
    assert "docker не запущен" in first[1]["content"]
    assert "(1, 0))" in (project / "scripts" / "ratio.py").read_text(encoding="utf-8")  # nothing applied
    assert eng._sandbox.network and not any(eng._sandbox.network)  # no network unless the user allows it
    eng.close()


def test_sandbox_network_needs_a_confirmation(settings, fake_embedder, project: Path):
    eng = chat_engine(settings, fake_embedder, project, agent=list(FIX), code=True)
    ans = eng.ask("Исправь падение ratio.py", sandbox_net=True)
    [pending] = ans.pending
    assert ans.route == "code" and not ans.code and pending["args"]["network"] and pending["rule"] == "S17"
    assert eng._sandbox.network == []  # nothing ran before the user's yes
    ans = eng.ask("Исправь падение ratio.py", sandbox_net=True, confirmed=[pending["key"]])
    assert ans.code["network"] and eng._sandbox.network and all(eng._sandbox.network)
    settings.code.network = "never"  # the config can forbid the toggle altogether
    eng.llm.agent = list(FIX)
    ans = eng.ask("Исправь падение ratio.py", sandbox_net=True)
    assert not ans.pending and not ans.code["network"]
    eng.close()


def test_pip_install_only_with_network():
    with pytest.raises(WorkspaceError, match="нет сети"):
        parse_command("pip install polars")
    cmd = parse_command("pip install polars", network=True)
    assert cmd[:4] == ["python", "-m", "pip", "install"] and "--target" in cmd and cmd[-1] == "polars"
    assert parse_command("python -m pip install tqdm", network=True)[-1] == "tqdm"
    with pytest.raises(WorkspaceError):
        parse_command("pip uninstall numpy", network=True)


def test_no_sandbox_answers_without_running_code(settings, fake_embedder, project: Path):
    eng = chat_engine(settings, fake_embedder, project, code=True)
    eng._sandbox_ok = (False, time.monotonic())
    ans = eng.ask("Исправь падение ratio.py")
    assert ans.route == "corpus" and not ans.code and "Docker" in ans.notice
    eng.close()


def test_research_agent_runs_code_for_a_calculation(settings, fake_embedder, project: Path):
    settings.agent.mode = "always"
    script = [{"thought": "посчитаю", "action": "run_code",
               "args": {"code": "print(sum(float(l.split('loss=')[1]) for l in open('logs/train.log')))"}},
              {"thought": "", "action": "answer", "args": {}}]
    eng = chat_engine(settings, fake_embedder, project, agent=script,
                      reply={"answerable": True, "answer": "Сумма loss — 0.75 [1].", "general": ""})
    ans = eng.ask("Какая сумма loss по логу?")
    step = next(s.detail for s in ans.trace if s.name == "agent" and s.detail.get("action") == "run_code")
    assert step["exit_code"] == 0 and "0.75" in step["observation"] and step["policy"]["decision"] == "allow"
    computed = next(s for s in ans.sources if s.file_type == "sandbox")  # the output is evidence for the answer
    assert "0.75" in computed.text and "open('logs/train.log')" in computed.text
    assert not list((settings.data_dir / "workspaces").glob("qa-*"))  # the throwaway copy is removed
    eng.close()


def test_run_code_is_off_for_the_eval_sets(settings, fake_embedder, project: Path):
    settings.agent.mode = "always"
    eng = chat_engine(settings, fake_embedder, project, code=True)
    eng.ask("Исправь падение ratio.py", code="off")
    assert "route_code" not in eng.llm.kinds
    prompt = next(c[0]["content"] for c in eng.llm.calls if "Инструменты:" in c[0]["content"])
    assert "run_code" not in prompt
    eng.close()


def test_general_answer_knows_the_capabilities(settings, fake_embedder, project: Path):
    route = {"route": "general", "standalone_question": "Какой сейчас курс евро?", "web": True}
    eng = chat_engine(settings, fake_embedder, project, route=route)
    ans = eng.ask("Какой сейчас курс евро?", web="off")
    assert ans.route == "general" and ans.suggest == ["web"]  # the UI offers to turn the internet on
    system = eng.llm.calls[-1][0]["content"]
    assert "интернет: выключен" in system and f"«{project.name}»" in system and "песочнице" in system
    eng.llm.route = {**route, "web": False, "standalone_question": "Привет"}
    assert eng.ask("Привет", web="off").suggest == []
    eng.close()


def test_model_switch_is_per_thread(settings):
    base, other = FakeLLM(settings), FakeLLM(settings)
    other.cfg = settings.llm.model_copy(update={"model": "qwen3.6:27b"})
    switch = ModelSwitch(base)
    switch._models["qwen3.6:27b"] = other
    seen = {}

    def worker():
        seen["thread"] = switch.name

    with switch.use("qwen3.6:27b"):
        assert switch.name == "ollama:qwen3.6:27b" and switch.current is other
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert seen["thread"] == base.name and switch.current is base  # other requests keep the configured model
    with switch.use(settings.llm.model):
        assert switch.current is base
    with pytest.raises(LLMError):
        switch.get("qwen; rm -rf /")


def test_api_model_and_code_fields(settings, fake_embedder, project: Path, monkeypatch):
    eng = chat_engine(settings, fake_embedder, project)
    other = FakeLLM(settings, general="Ответ другой модели.")
    other.cfg = settings.llm.model_copy(update={"model": "qwen3.6:27b"})
    eng.llm._models["qwen3.6:27b"] = other
    monkeypatch.setattr(ModelSwitch, "installed", lambda self: [{"name": "qwen3.5:9b"}, {"name": "qwen3.6:27b"}])
    with TestClient(create_app(eng)) as client:
        m = client.get("/models").json()
        assert m["default"] == settings.llm.model and [x["name"] for x in m["models"]] == ["qwen3.5:9b", "qwen3.6:27b"]
        eng.llm.route = {"route": "general", "standalone_question": "Привет"}
        other.route = eng.llm.route
        d = client.post("/dialogs", json={}).json()
        a = client.post("/ask", json={"question": "Привет", "dialog_id": d["id"], "model": "qwen3.6:27b",
                                      "sandbox_net": True}).json()
        assert a["answer"] == "Ответ другой модели." and a["model"] == "ollama:qwen3.6:27b"
        q = client.get(f"/dialogs/{d['id']}").json()["turns"][0]["payload"]
        assert q["model"] == "qwen3.6:27b" and q["sandbox_net"] is True  # a confirmation re-asks the same way
        p = client.get("/policy").json()
        assert p["sandbox"] == "no_network" and p["sandbox_network"] == "confirm" and p["code"] == "auto"
    eng.close()
