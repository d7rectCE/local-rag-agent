"""Web UI: the static page with its CSP, stored dialogs, /ask with a dialog, the access summary."""

from pathlib import Path

from fastapi.testclient import TestClient

from rag_agent.api import WEBUI, create_app
from rag_agent.dialogs import DialogStore, make_title
from rag_agent.engine import Engine
from tests.conftest import FakeLLM


def test_page_and_assets_are_served_with_a_strict_csp(settings, fake_embedder):
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    with TestClient(create_app(eng)) as client:
        page = client.get("/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        csp = page.headers["content-security-policy"]
        assert "script-src 'self'" in csp and "img-src 'self' data: blob:" in csp and "connect-src 'self'" in csp
        for asset in ("app.js", "app.css", "theme.js", "fonts/fonts.css", "fonts/manrope-cyrillic.woff2",
                      "vendor/marked.min.js", "vendor/purify.min.js"):
            r = client.get(f"/ui/{asset}")
            assert r.status_code == 200, asset
            assert r.headers["content-security-policy"] == csp
        assert client.get("/health").headers.get("content-security-policy") is None  # API responses untouched
    # every asset the page references exists (no external URLs: the UI works offline)
    html = (WEBUI / "index.html").read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html
    for ref in ("/ui/app.js", "/ui/app.css", "/ui/theme.js", "/ui/fonts/fonts.css"):
        assert ref in html and (WEBUI / ref.removeprefix("/ui/")).exists()
    assert "googleapis" not in (WEBUI / "app.css").read_text(encoding="utf-8")


def test_dialog_store(tmp_path: Path):
    store = DialogStore(tmp_path / "d.sqlite")
    d = store.create()
    assert d["title"] == "" and d["n_turns"] == 0
    store.add_turn(d["id"], "user", "question", "Какой learning rate был в эксперименте с аугментациями и почему такой?")
    store.add_turn(d["id"], "assistant", "answer", "0.05 [1]", payload={"answer": "0.05 [1]", "sources": []})
    got = store.get(d["id"])
    assert got["title"].startswith("Какой learning rate") and len(got["title"]) <= 60
    assert [t["kind"] for t in got["turns"]] == ["question", "answer"] and got["turns"][1]["payload"]["answer"] == "0.05 [1]"
    assert store.history(d["id"]) == [{"role": "user", "content": got["turns"][0]["content"]},
                                      {"role": "assistant", "content": "0.05 [1]"}]
    store.add_turn(d["id"], "assistant", "code", "done", payload={"task_id": "t1", "applied": []}, ref="t1")
    assert store.update_ref("t1", {"task_id": "t1", "status": "done", "summary": "Построил график",
                                   "changed": [["A", "plot.py"]], "applied": ["plot.py"]}) == 1
    assert store.get(d["id"])["turns"][-1]["payload"]["applied"] == ["plot.py"]
    # a code task is history too: "what did you do?" is answered from its summary and the user's decision
    last = store.history(d["id"])[-1]["content"]
    assert "Построил график" in last and "A plot.py" in last and "применил" in last
    # a code result inside a chat answer: apply / reject update the answer's "code" field, not the answer
    store.add_turn(d["id"], "assistant", "answer", "Готово", payload={"answer": "Готово", "code": {"task_id": "t2"}},
                   ref="t2")
    assert store.update_ref("t2", {"task_id": "t2", "rejected": True, "changed": [["M", "a.py"]]}) == 1
    payload = store.get(d["id"])["turns"][-1]["payload"]
    assert payload["answer"] == "Готово" and payload["code"]["rejected"]
    assert "отклонил" in store.history(d["id"])[-1]["content"]
    store.drop_last_answer(d["id"])
    store.drop_last_answer(d["id"])
    assert [t["kind"] for t in store.get(d["id"])["turns"]] == ["question", "answer"]
    assert store.rename(d["id"], "Новое имя")["title"] == "Новое имя"
    assert [x["id"] for x in store.list()] == [d["id"]]
    store.delete(d["id"])
    assert store.list() == []
    assert make_title("  a\n b ") == "a b"


def test_ask_in_a_dialog_keeps_the_history(settings, fake_embedder, corpus: Path):
    llm = FakeLLM(settings)
    eng = Engine(settings, embedder=fake_embedder, llm=llm)
    eng.index_folder(corpus)
    with TestClient(create_app(eng)) as client:
        d = client.post("/dialogs", json={}).json()
        assert d["corpus"] == str(corpus.resolve()) or d["corpus"].endswith(corpus.name)
        a1 = client.post("/ask", json={"question": "Какой learning rate?", "dialog_id": d["id"], "route": "corpus"}).json()
        assert a1["answer"]
        stored = client.get(f"/dialogs/{d['id']}").json()
        assert stored["title"] == "Какой learning rate?"
        assert [t["kind"] for t in stored["turns"]] == ["question", "answer"]
        assert stored["turns"][1]["payload"]["sources"] and stored["turns"][0]["payload"]["route"] == "corpus"
        # the follow-up gets the stored turns as its history (the router sees them)
        llm.calls.clear()
        client.post("/ask", json={"question": "А в другом эксперименте?", "dialog_id": d["id"]})
        seen = "\n".join(m["content"] for call in llm.calls for m in call if isinstance(m.get("content"), str))
        assert "Какой learning rate?" in seen
        assert len(client.get(f"/dialogs/{d['id']}").json()["turns"]) == 4
        # regenerating the last answer (after a confirmation) does not add the question again
        client.post("/ask", json={"question": "А в другом эксперименте?", "dialog_id": d["id"], "replace_last": True})
        turns = client.get(f"/dialogs/{d['id']}").json()["turns"]
        assert [t["kind"] for t in turns] == ["question", "answer", "question", "answer"]
        listed = client.get("/dialogs").json()
        assert listed[0]["id"] == d["id"] and listed[0]["n_turns"] == 4
        assert client.patch(f"/dialogs/{d['id']}", json={"title": "LR"}).json()["title"] == "LR"
        assert client.post("/ask", json={"question": "x", "dialog_id": "nope"}).status_code == 404
        assert client.delete(f"/dialogs/{d['id']}").status_code == 200
        assert client.get(f"/dialogs/{d['id']}").status_code == 404


def test_policy_summary(settings, fake_embedder, corpus: Path):
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    eng.index_folder(corpus)
    with TestClient(create_app(eng)) as client:
        p = client.get("/policy").json()
        assert p["corpus"] == {"name": corpus.name, "access": "read"} and p["writes"] == "confirm"
        assert p["sandbox"] in ("no_network", "unavailable") and p["web"] == settings.web.mode
        status = client.get("/status").json()
        assert "gpu" in status and status["llm"]["name"]
