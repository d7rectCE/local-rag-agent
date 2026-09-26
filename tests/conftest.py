"""Shared fixtures: synthetic corpus, fake embedder and fake LLM (no models, no network)."""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import nbformat
import numpy as np
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

from rag_agent.config import Settings
from rag_agent.index.embedder import Encoded
from rag_agent.llm import BaseLLM, LLMResponse

PY_SOURCE = '''"""Metrics helpers."""
import numpy as np


def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


class Trainer:
    def fit(self, X, y):
        return self
'''


def make_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "language": "python", "display_name": "Python 3"}
    nb.cells = [
        new_markdown_cell("# Experiment A\nGradient boosting on breast cancer."),
        new_code_cell("lr = 0.05\nprint('roc_auc', 0.9925)", execution_count=2,
                      outputs=[new_output("stream", name="stdout", text="roc_auc 0.9925\n")]),
        new_code_cell("df.head()", execution_count=1,
                      outputs=[new_output("execute_result", data={"text/plain": "   a  b\n0  1  2", "image/png": "iVBORw0KGgo="},
                                          execution_count=1)]),
        new_code_cell("1 / 0", execution_count=3,
                      outputs=[new_output("error", ename="ZeroDivisionError", evalue="division by zero", traceback=[])]),
        new_markdown_cell(""),
    ]
    return nb


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "metrics.py").write_text(PY_SOURCE, encoding="utf-8")
    (root / "notebooks").mkdir()
    nbformat.write(make_notebook(), str(root / "notebooks" / "exp_a.ipynb"))
    for excluded in ("__pycache__", ".git", "Аккаунты"):
        (root / excluded).mkdir()
        (root / excluded / "secret.py").write_text("PASSWORD = 'x'\n", encoding="utf-8")
    (root / "notes.txt").write_text("not indexed", encoding="utf-8")
    return root


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.storage.data_dir = str(tmp_path / "data")
    s.corpus.exclude = ["__pycache__", "Аккаунты"]
    s.corpus.include_ext = [".py", ".ipynb"]  # document formats are tested in test_documents.py
    s.chunking.lines_per_chunk = 6
    s.chunking.overlap_lines = 2
    # no real models in tests: the reranker is injected explicitly where needed
    s.retrieval.rerank = False
    s.retrieval.symbols = False
    s.reasoning.mode = "off"  # reasoning is tested explicitly in test_reasoning.py
    s.catalog.extract = False  # the catalog of experiments is tested in test_catalog.py
    s.agent.mode = "off"  # the agent and the CRAG check are tested in test_agent.py
    s.agent.crag = False
    return s


class FakeEmbedder:
    """Hashing bag-of-words: deterministic, lexical, good enough to test plumbing."""

    name = "fake-hash-embedder"
    device = "cpu"
    dim = 128
    has_sparse = True

    def encode(self, texts: list[str], batch_size: int | None = None) -> Encoded:
        dense = np.zeros((len(texts), self.dim), dtype=np.float32)
        sparse = []
        for i, text in enumerate(texts):
            vec: dict[int, float] = {}
            for tok in re.findall(r"\w+", text.lower()):
                h = zlib.crc32(tok.encode("utf-8"))
                dense[i, h % self.dim] += 1.0
                vec[h % 100_000] = 1.0
            norm = np.linalg.norm(dense[i])
            if norm:
                dense[i] /= norm
            sparse.append(vec)
        return Encoded(dense=dense, sparse=sparse)


EXTRACTION = {"experiments": [{
    "title": "Experiment A", "task": "binary classification", "dataset": "Breast Cancer", "model": "HistGradientBoosting",
    "cell": 1,
    "metrics": [
        {"name": "ROC-AUC", "value": 0.9925, "split": "validation", "variant": "", "cell": 2},
        {"name": "accuracy", "value": 0.97, "split": "val", "variant": "", "cell": 2},  # not printed anywhere
    ],
    "hyperparameters": [
        {"name": "lr", "value": "0.05", "variant": "", "cell": 3},  # wrong cell: found in cell 2
        {"name": "max_depth", "value": "7", "variant": "", "cell": 2},  # not in the code
    ],
}]}


class FakeLLM(BaseLLM):
    """Answers by request kind: router (schema with "route"), grounded answer
    (schema with "answerable"), judge, catalog extraction, SQL, or a free-text general answer."""

    def __init__(self, settings: Settings, reply: dict | str | None = None, route: dict | str | None = None,
                 general: str = "Общий ответ.", extraction: dict | None = None, sql: list[dict] | None = None,
                 agent: list[dict] | None = None):
        super().__init__(settings.llm)
        self.reply = reply if reply is not None else {"answerable": True, "answer": "Learning rate был 0.05 [1].", "general": ""}
        self.route = route  # None -> corpus with the question unchanged
        self.general = general
        self.extraction = extraction if extraction is not None else EXTRACTION
        self.sql = list(sql or [])  # replies of successive SQL calls; the last one repeats
        self.agent = list(agent or [])  # scripted agent steps; after the script: answer
        self.calls: list[list[dict]] = []
        self.kinds: list[str] = []
        self.budgets: list[int] = []  # reasoning budgets of chat_reasoning calls

    def _chat(self, messages, json_schema, tools, temperature, max_tokens, think) -> LLMResponse:
        self.calls.append(messages)
        props = (json_schema or {}).get("properties", {})
        if "route" in props:
            self.kinds.append("route")
            question = messages[-1]["content"].split("Последний вопрос: ", 1)[-1]
            reply = self.route if self.route is not None else {"route": "corpus", "standalone_question": question}
        elif "answerable" in props:
            self.kinds.append("answer")
            reply = self.reply
        elif "experiments" in props:
            self.kinds.append("extract")
            reply = self.extraction
        elif "sql" in props:
            self.kinds.append("sql")
            reply = (self.sql.pop(0) if len(self.sql) > 1 else self.sql[0]) if self.sql else {
                "sql": "SELECT path, cell, name, value FROM metrics", "reason": ""}
        elif "action" in props:
            self.kinds.append("agent")
            reply = self.agent.pop(0) if self.agent else {"thought": "достаточно", "action": "answer", "args": {}}
        elif set(props) == {"query"}:
            self.kinds.append("rewrite")
            reply = {"query": "learning rate lr"}
        elif "correctness" in props:
            self.kinds.append("judge")
            reply = {"correctness": "correct", "faithful": True, "supported_citations": [1], "relevant": True,
                     "rationale": "ok"}
        else:
            self.kinds.append("general")
            reply = self.general
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return LLMResponse(content=content, usage={"prompt_tokens": 10, "completion_tokens": 5})

    def chat_reasoning(self, messages, *, budget_tokens, json_schema=None, purpose="chat") -> LLMResponse:
        resp = self._chat(messages, json_schema, None, None, None, True)
        self.kinds[-1] += "+reasoning"
        self.budgets.append(budget_tokens)
        resp.thinking = "draft " * min(budget_tokens, 3)
        resp.usage["thinking_tokens"] = min(budget_tokens, 3)
        resp.usage["completion_tokens"] += resp.usage["thinking_tokens"]  # as in Ollama: eval_count includes thinking
        resp.thinking_truncated = budget_tokens < 3
        return resp

    def is_available(self) -> bool:
        return True


class FakeReranker:
    """Relevance by word overlap with the query: 0.9 with a shared word of 2+ letters, else 0.02."""

    def __init__(self, relevant: bool | None = None):
        self.relevant = relevant  # force every score high (True) or low (False)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        words = {w for w in re.findall(r"\w+", query.lower()) if len(w) >= 2}
        out = []
        for p in passages:
            hit = bool(words & set(re.findall(r"\w+", p.lower()))) if self.relevant is None else self.relevant
            out.append(0.9 if hit else 0.02)
        return np.asarray(out, dtype=np.float32)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
