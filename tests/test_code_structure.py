"""Э3: static analysis, AST chunking, notebook graph, exact-name retrieval, reranking."""

from pathlib import Path

import nbformat
import numpy as np
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from rag_agent.config import ChunkingConfig
from rag_agent.engine import Engine
from rag_agent.ingest import iter_corpus, parse_file
from rag_agent.ingest.code_analysis import analyze, strip_magics
from rag_agent.retrieval import rrf
from rag_agent.schema import EdgeType, NodeType
from tests.conftest import FakeLLM

MODULE = '''"""Module doc."""
import numpy as np

LIMIT = 3


def small_a(x):
    return x + 1


def small_b(x):
    return x * 2


# comment that belongs to the decorated function
@staticmethod
def decorated(y):
    return y


class Big:
    """A class that does not fit into one chunk."""

    scale = 2

    def first(self, values):
        total = 0
        for v in values:
            total += v * self.scale + 1000000
        return total

    def second(self, values):
        return [small_a(v) for v in values if v > 1000000 and v < 2000000 and v != 1500000]

    def third(self):
        return self.first([1, 2, 3]) + self.second([4, 5, 6]) + 1234567890 + 9876543210
'''


def parse(tmp_path: Path, name: str, content: str, **cfg):
    (tmp_path / name).write_text(content, encoding="utf-8")
    cf = next(f for f in iter_corpus(tmp_path, [".py", ".ipynb"]) if f.rel_path == name)
    return parse_file(cf, ChunkingConfig(**cfg))


def test_analyze_symbols_calls_and_def_use():
    a = analyze("import pandas as pd\ndf = df.dropna()\nrows = [x for x in df]\n\ndef f(a):\n    return helper(a)\n\nclass K:\n    def m(self):\n        return f(1)\n")
    assert {"pd", "df", "rows", "f", "K"} <= a.defines
    assert a.uses == {"df"}  # read before being rebound; x is comprehension-local, helper is inside a body
    kinds = {s.qualname: s.kind for s in a.symbols}
    assert kinds == {"f": "function", "K": "class", "K.m": "method"}
    assert {(c.name, c.caller) for c in a.calls} >= {("helper", "f"), ("f", "K.m"), ("dropna", None)}
    assert next(s for s in a.symbols if s.name == "f").signature == "def f(a)"


def test_strip_magics_keeps_line_numbers():
    src = "%matplotlib inline\n!pip install x\nvalue = 1"
    assert strip_magics(src).split("\n") == ["", "", "value = 1"]
    assert analyze(strip_magics(src)).defines == {"value"}


def test_ast_chunks_keep_definitions_whole(tmp_path):
    parsed = parse(tmp_path, "m.py", MODULE, python="ast", ast_max_chars=260)
    chunks = [n for n in parsed.nodes if n.node_type != NodeType.FILE]
    lines = MODULE.split("\n")
    # every function and method lies entirely inside one chunk (a class too big for one chunk is split by design)
    for sym in parsed.symbols:
        if sym.kind != "class":
            assert any(c.location.line_start <= sym.line_start and sym.line_end <= c.location.line_end for c in chunks), sym
    # the comment and the decorator travel with the function
    decorated = next(c for c in chunks if "decorated" in c.title)
    assert "# comment that belongs" in decorated.text and "@staticmethod" in decorated.text
    # the class is split into a header and methods that carry the class as context
    header = next(c for c in chunks if c.title == "class Big")
    assert header.node_type == NodeType.CLASS and "scale = 2" in header.text and "def first" not in header.text
    methods = [c for c in chunks if c.title.startswith("Big.")]
    assert methods and all("class Big" in c.context for c in methods)
    assert all(c.id == f"m.py#L{c.location.line_start}-{c.location.line_end}" for c in chunks)
    assert chunks[-1].location.line_end == len(lines) - 1  # the trailing newline adds an empty last line


def test_ast_merges_small_neighbours(tmp_path):
    parsed = parse(tmp_path, "m.py", MODULE, python="ast", ast_max_chars=5000)
    chunks = [n for n in parsed.nodes if n.node_type != NodeType.FILE]
    assert len(chunks) == 1 and "small_a" in chunks[0].title and "class Big" in chunks[0].title


def test_ast_falls_back_on_syntax_error(tmp_path):
    parsed = parse(tmp_path, "bad.py", "def broken(:\n    pass\n", python="ast")
    assert any("fell back" in w for w in parsed.warnings)
    assert any(n.node_type == NodeType.CODE_CHUNK for n in parsed.nodes)


def test_oversized_function_is_windowed_with_signature_context(tmp_path):
    body = "\n".join(f"    x{i} = {i} * 1234567" for i in range(40))
    parsed = parse(tmp_path, "long.py", f"def long_one(a, b=2):\n{body}\n    return a\n", python="ast",
                   ast_max_chars=200, lines_per_chunk=15, overlap_lines=3)
    parts = [n for n in parsed.nodes if "part" in n.title]
    assert len(parts) >= 3 and all(p.title.startswith("long_one (part") for p in parts)
    assert "def long_one(a, b=2)" in parts[1].context


def test_embedding_text_without_header_is_bare_text(tmp_path):
    parsed = parse(tmp_path, "m.py", MODULE, python="ast", ast_max_chars=260)
    method = next(n for n in parsed.nodes if n.title.startswith("Big."))
    assert method.embedding_text(False) == method.text
    assert method.embedding_text(True).startswith("File: m.py | Big.") and "class Big" in method.embedding_text(True)


def notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell("# Title\nintro"),
        new_code_cell("import pandas as pd\nfrom lib import score"),
        new_markdown_cell("## Data"),
        new_code_cell("%matplotlib inline\ndf = pd.read_csv('x.csv')"),
        new_markdown_cell("### Cleaning"),
        new_code_cell("df = df.dropna()\nresult = score(df)"),
        new_code_cell("def local(z):\n    return z\nprint(result"),  # syntax error: no edges, a warning
    ]
    return nb


def test_notebook_graph_and_context(tmp_path):
    nbformat.write(notebook(), str(tmp_path / "n.ipynb"))
    cf = next(iter_corpus(tmp_path, [".ipynb"]))
    parsed = parse_file(cf, ChunkingConfig(notebook_context=True))
    by_id = {n.id: n for n in parsed.nodes}
    assert by_id["n.ipynb#cell6"].metadata["section"] == "Title > Data > Cleaning"
    assert by_id["n.ipynb#cell2"].metadata["section"] == "Title"
    flows = {(e.src.split("#")[1], e.dst.split("#")[1]): e.label for e in parsed.edges if e.type == EdgeType.USES_VAR}
    assert flows == {("cell2", "cell4"): "pd", ("cell4", "cell6"): "df", ("cell2", "cell6"): "score"}
    assert {(c.name, c.cell) for c in parsed.calls} >= {("read_csv", 4), ("score", 6)}
    assert any("cell 7: SyntaxError" in w for w in parsed.warnings)

    legacy = parse_file(cf, ChunkingConfig(notebook_context=False))
    assert {n.id: n for n in legacy.nodes}["n.ipynb#cell6"].metadata["section"] == "Cleaning"


def test_rrf_prefers_items_in_both_lists():
    fused = [k for k, _ in rrf([["a", "b", "c"], ["c", "d"]])]
    assert fused[0] == "c" and set(fused) == {"a", "b", "c", "d"}


class FakeReranker:
    name = "fake-reranker"

    def score(self, query, passages):
        return np.array([1.0 if "compute_f1" in p else 0.0 for p in passages], dtype=np.float32)


def test_symbols_channel_and_reranker(settings, fake_embedder, corpus: Path):
    (corpus / "pkg" / "use.py").write_text("from pkg.metrics import compute_f1\n\nprint(compute_f1(1, 2, 3))\n", encoding="utf-8")
    settings.chunking.python = "ast"
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings), reranker=FakeReranker())
    eng.index_folder(corpus)
    defs = eng.index.catalog.find_symbols("COMPUTE_F1")  # case-insensitive
    assert defs and defs[0]["file_path"] == "pkg/metrics.py"
    assert {c["file_path"] for c in eng.index.catalog.find_calls("compute_f1")} == {"pkg/use.py"}

    hits = eng.search("where is compute_f1 called", top_k=5, symbols=True)
    files = [h.node.file_path for h in hits]
    assert "pkg/metrics.py" in files and "pkg/use.py" in files

    found = eng.lookup_symbol("compute_f1")
    assert found["definitions"][0]["signature"] == "def compute_f1(tp, fp, fn)"
    assert [(c["file_path"], c["line"]) for c in found["calls"]] == [("pkg/use.py", 3)]

    reranked = eng.search("anything", top_k=3, rerank=True)
    assert "compute_f1" in reranked[0].node.text and reranked[0].score == 1.0
    eng.close()
