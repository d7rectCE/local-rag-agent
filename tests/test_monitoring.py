"""Э9: canary questions for monitoring are built from the catalog, without a model."""

from pathlib import Path

from rag_agent.engine import Engine
from rag_agent.monitoring.canaries import build_canaries
from tests.conftest import FakeLLM


def test_canaries_from_the_catalog(settings, fake_embedder, corpus: Path):
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    eng.index_folder(corpus)
    catalog = eng.index.catalog
    canaries = build_canaries(catalog, min_per_segment=1)
    by_id = {c.id: c for c in canaries}
    f1 = by_id["def:pkg/metrics.py:compute_f1"]
    assert f1.segment == "py:def:pkg" and f1.question == "Где определена функция compute_f1?"
    nodes = list(catalog.get_nodes([r["id"] for r in catalog.query("SELECT id FROM nodes WHERE embed = 1")]).values())
    answering = [n for n in nodes if f1.answered_by(n)]
    assert answering and all(n.file_path == "pkg/metrics.py" for n in answering)
    assert "def compute_f1" in "".join(n.text for n in answering)
    assert by_id["def:pkg/metrics.py:Trainer"].question == "Где определён класс Trainer?"
    assert len({c.id for c in canaries}) == len(canaries)  # ids are unique
    assert build_canaries(catalog, min_per_segment=10**6) == []  # thin segments are dropped
    assert build_canaries(catalog, min_per_segment=1) == canaries  # deterministic
    eng.close()
