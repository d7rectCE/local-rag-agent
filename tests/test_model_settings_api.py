"""The settings panel: per-model settings through the API, the chat model's sampling on the free text only,
its context and answer length on every call."""

import json

import httpx
from fastapi.testclient import TestClient

from rag_agent.api import create_app
from rag_agent.config import LLMConfig
from rag_agent.engine import Engine
from rag_agent.llm import ModelSwitch, OllamaLLM
from tests.conftest import FakeLLM


def ollama_with_log(cfg: LLMConfig, bodies: list[dict]) -> OllamaLLM:
    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}, "done_reason": "stop"})

    llm = OllamaLLM(cfg)
    llm._http = httpx.Client(transport=httpx.MockTransport(handler))
    return llm


def test_the_ui_sampling_reaches_the_free_text_only():
    bodies: list[dict] = []
    llm = ollama_with_log(LLMConfig(model="m", options={"temperature": 0.8, "top_p": 0.9}), bodies)
    llm.chat([{"role": "user", "content": "hi"}])
    llm.chat([{"role": "user", "content": "route"}], json_schema={"type": "object"})
    llm.chat([{"role": "user", "content": "fixed"}], temperature=0.0)
    free, schema, fixed = (b["options"] for b in bodies)
    assert free["temperature"] == 0.8 and free["top_p"] == 0.9
    assert schema["temperature"] == 0.0 and "top_p" not in schema  # routing and plans stay reliable
    assert fixed["temperature"] == 0.0 and "top_p" not in fixed


def test_a_tuned_model_keeps_one_context_for_every_call():
    switch = ModelSwitch(OllamaLLM(LLMConfig(model="base", num_ctx=65536)))
    tuned = switch.tuned(None, {"temperature": 0.5, "num_ctx": 16384, "max_tokens": 2048})
    assert tuned.cfg.model == "base" and tuned.cfg.num_ctx == 16384 and tuned.cfg.max_tokens == 2048
    assert tuned.cfg.options == {"temperature": 0.5}
    assert switch.tuned(None, {"temperature": 0.5, "num_ctx": 16384, "max_tokens": 2048}) is tuned  # cached
    with switch.use(None, {"top_k": 40}):
        assert switch.current.cfg.options == {"top_k": 40}
    assert switch.current is switch.base


def test_settings_endpoints(settings, fake_embedder):
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    with TestClient(create_app(eng)) as client:
        text = client.get("/settings/text").json()
        assert text["name"] == settings.llm.model and text["values"] == {}
        assert text["recommended"]["num_ctx"] == settings.llm.num_ctx and 65536 in text["choices"]["num_ctx"]
        saved = client.put("/settings/text", json={"values": {"temperature": 0.7, "top_k": "40"}}).json()
        assert saved["values"] == {"temperature": 0.7, "top_k": 40}
        assert client.put("/settings/text", json={"values": {"steps": 3}}).status_code == 400
        assert client.delete("/settings/text").json()["values"] == {}
        assert client.get("/settings/image").status_code == 404  # no generator configured
        assert client.get("/settings/audio").status_code == 404
    eng.close()


def test_chat_models_of_models_text_are_imported_into_ollama(settings, fake_embedder, monkeypatch):
    import subprocess as sp

    from rag_agent import api as api_module

    folder = settings.models_dir / "text"
    folder.mkdir(parents=True)
    (folder / "My-Chat 8B.Q4_K_M.gguf").write_bytes(b"x")
    calls = []

    def fake_run(args, **kwargs):
        if args[:2] == ["ollama", "create"]:
            calls.append((args, open(args[-1], encoding="utf-8").read()))
            return sp.CompletedProcess(args, 0, "success", "")
        return real(args, **kwargs)

    real = sp.run
    monkeypatch.setattr(api_module.subprocess, "run", fake_run)
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    with TestClient(create_app(eng)) as client:
        [f] = client.get("/models").json()["files"]
        assert f["ollama"] == "my-chat-8b.q4_k_m" and not f["imported"]
        assert client.post("/models/import", json={"file": "../secret.gguf"}).status_code == 404  # names only
        assert client.post("/models/import", json={"file": f["name"]}).json() == {"model": "my-chat-8b.q4_k_m"}
    [(args, modelfile)] = calls
    assert args[2] == "my-chat-8b.q4_k_m" and modelfile.startswith('FROM "') and "My-Chat 8B.Q4_K_M.gguf" in modelfile
    eng.close()


def test_a_file_imported_under_another_name_is_known_by_its_size(settings, fake_embedder, monkeypatch):
    folder = settings.models_dir / "text"
    folder.mkdir(parents=True)
    (folder / "Gemma4-26B-Q4_K_M.gguf").write_bytes(b"x" * 1234)
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    monkeypatch.setattr(type(eng.llm), "installed", lambda self: [{"name": "gemma4-uncensored:latest", "size": 1234}])
    with TestClient(create_app(eng)) as client:
        [f] = client.get("/models").json()["files"]
    assert f["imported"] and f["ollama"] == "gemma4-uncensored:latest"
    eng.close()
