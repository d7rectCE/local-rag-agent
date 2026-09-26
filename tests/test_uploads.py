"""Э13: uploads — signature-based type detection, session scope, routing (whole / index / Self-Route)."""

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rag_agent.engine import Engine
from rag_agent.retrieval import Hit
from rag_agent.schema import FileType, Node, NodeType
from rag_agent.uploads import UploadError, detect_type, exact_keys, exact_matches, fit_budget
from tests.conftest import FakeLLM

NOTE = "# Договор\n\nСрок сдачи этапа 1 — 10 апреля 2026 г.\n\nСумма договора — 150 000 рублей.\n"


def zip_bytes(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in names:
            zf.writestr(n, "<x/>")
    return buf.getvalue()


def test_type_is_detected_by_content():
    assert detect_type(b"%PDF-1.7 ...", "report.txt") == ".pdf"
    assert detect_type(zip_bytes(["[Content_Types].xml", "word/document.xml"]), "x.bin") == ".docx"
    assert detect_type(json.dumps({"cells": [], "nbformat": 4}).encode(), "nb.json") == ".ipynb"
    assert detect_type("Заметки: дедлайн 15 апреля".encode("cp1251"), "notes") == ".txt"
    assert detect_type(b"def f():\n    return 1\n", "a.py") == ".py"
    for data, name in [(zip_bytes(["a.txt"]), "a.zip"), (b"\x89PNG\r\n\x1a\n....", "a.png"),
                       (b"MZ\x90\x00" + b"\x00" * 100, "setup.exe"), (b"\x00\x01\x02\x03" * 100, "blob.txt")]:
        with pytest.raises(UploadError):
            detect_type(data, name)


def upload_step(ans) -> dict:
    return next(s.detail for s in ans.trace if s.name == "upload")


def upload_engine(settings, fake_embedder, **llm) -> Engine:
    return Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings, **llm))


def test_upload_lifecycle(settings, fake_embedder, corpus: Path):
    eng = upload_engine(settings, fake_embedder)
    eng.index_folder(corpus)
    info = eng.upload("s1", "../../evil name?.md", NOTE.encode("utf-8"))
    assert info.name == "evil name_.md" and info.file_type == ".md" and info.fits_context and info.n_fragments >= 1
    assert eng.uploads.file_path(info).parent.parent.name == "s1"  # stays inside the session folder
    assert [u.id for u in eng.uploads.list("s1")] == [info.id] and eng.uploads.list("s2") == []
    # a copy of a corpus file is recognised by its hash
    dup = eng.upload("s1", "copy.py", (corpus / "pkg" / "metrics.py").read_bytes())
    assert dup.duplicate_of == "pkg/metrics.py"
    with pytest.raises(UploadError):
        eng.upload("../s1", "x.txt", b"text")
    settings.uploads.max_mb = 0.00001
    with pytest.raises(UploadError, match="больше"):
        eng.upload("s1", "big.txt", b"x" * 100)
    # expired uploads disappear
    f = eng.uploads._dir("s1", info.id) / "info.json"
    expired = info.model_copy(update={"expires_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()})
    f.write_text(expired.model_dump_json(), encoding="utf-8")
    assert eng.uploads.cleanup() == 1 and [u.id for u in eng.uploads.list("s1")] == [dup.id]
    eng.close()


def test_small_upload_goes_whole_without_a_corpus(settings, fake_embedder):
    eng = upload_engine(settings, fake_embedder, reply={"answerable": True, "answer": "10 апреля [1].", "general": ""})
    info = eng.upload("s1", "contract.md", NOTE.encode("utf-8"))
    ans = eng.ask("Когда срок сдачи этапа 1?", uploads=[info.id], session="s1")
    assert ans.route == "upload" and upload_step(ans)["route"] == "whole"
    assert upload_step(ans)["trust"] == "untrusted" and ans.sources[0].file_path == "upload:contract.md"
    assert "недоверенный" in eng.llm.calls[-1][0]["content"]
    eng.close()


def test_large_upload_uses_the_index_then_self_route(settings, fake_embedder):
    big = "\n\n".join(f"## Запись {i}\n\nЭксперимент {i}: accuracy 0.{900 + i}, датасет digits." for i in range(60))
    eng = upload_engine(settings, fake_embedder)
    settings.uploads.context_share = 0.02  # a budget of ~1000 chars: the file does not fit
    info = eng.upload("s1", "journal.md", big.encode("utf-8"))
    assert not info.fits_context
    ans = eng.ask("Какая accuracy в записи 7?", uploads=[info.id], session="s1")
    assert upload_step(ans)["route"] == "index" and eng.llm.kinds[-1] == "answer"
    # the model says the fragments are not enough -> the whole file is read in parts
    eng.llm.reply = {"answerable": False, "answer": "Не хватает данных.", "general": ""}
    ans = eng.ask("Какая accuracy в записи 7?", uploads=[info.id], session="s1")
    assert upload_step(ans)["route"] == "index+parts" and eng.llm.kinds.count("map") > 1
    parts = next(s for s in ans.trace if s.name == "upload_parts").detail
    assert parts["parts"] == eng.llm.kinds.count("map") and parts["selected"]
    settings.uploads.route = "index"  # H12 ablation: retrieval only
    eng.llm.kinds.clear()
    eng.ask("Какая accuracy в записи 7?", uploads=[info.id], session="s1")
    assert "map" not in eng.llm.kinds
    eng.close()


def test_add_to_corpus_is_explicit_and_never_overwrites(settings, fake_embedder, corpus: Path):
    eng = upload_engine(settings, fake_embedder)
    eng.index_folder(corpus)
    info = eng.upload("s1", "contract.md", NOTE.encode("utf-8"))
    assert not (corpus / "uploads").exists()  # nothing is written to the corpus by the upload itself
    first = eng.add_upload_to_corpus("s1", info.id)
    eng._thread.join(30)
    second = eng.add_upload_to_corpus("s1", info.id)
    eng._thread.join(30)
    assert first["path"] == "uploads/contract.md" and second["path"] == "uploads/contract (1).md"
    assert (corpus / "uploads" / "contract.md").read_text(encoding="utf-8") == NOTE
    eng.close()


def test_exact_keys_and_matches():
    assert exact_keys("Что произошло в лаборатории 14 марта 2025 года?") == ["2025-03-14", "14.03.2025"]
    assert exact_keys("Какие параметры были у запуска run-431 на cifar10-subset-v2?") == ["run-431", "cifar10-subset-v2"]
    assert exact_keys("Что было 2 июня?") == ["-06-02"]
    assert exact_keys("Сколько запусков в 2025 году?") == []  # a bare number is not a key
    rows = [f"2025-03-{d:02d} | run-{400 + d} | ResNet-18 | val accuracy 0.9{d:02d}" for d in range(1, 21)]
    nodes = [Node(id=f"f#L{k}", file_path="f.txt", file_type=FileType.TXT, node_type=NodeType.TABLE, text="\n".join(rows[k:k + 4]))
             for k in range(0, 20, 4)]
    assert [n.id for n in exact_matches(nodes, exact_keys("Что было у run-414?"), 3)] == ["f#L12"]
    assert [n.id for n in exact_matches(nodes, exact_keys("Что было 14 марта 2025?"), 3)] == ["f#L12"]
    assert exact_matches(nodes, exact_keys("Какой ResNet-18 лучший?"), 3) == []  # in every fragment: no signal
    hits = [Hit(node=n, score=1.0, rank=k) for k, n in enumerate(nodes, start=1)]
    size = len(nodes[0].text)
    assert len(fit_budget(hits, 2 * size + 1)) == 2 and len(fit_budget(hits, 1)) == 1  # whole fragments only

