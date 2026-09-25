from pathlib import Path

from rag_agent.config import ChunkingConfig
from rag_agent.ingest import iter_corpus, parse_file
from rag_agent.ingest.python_parser import line_windows
from rag_agent.ingest.text_utils import truncate_keep_tail
from rag_agent.ingest.walker import ExcludeRules
from rag_agent.schema import EdgeType, NodeType


def rel_paths(root: Path, **kwargs) -> list[str]:
    return [f.rel_path for f in iter_corpus(root, include_ext=[".py", ".ipynb"], **kwargs)]


def test_walker_applies_excludes_and_hidden(corpus: Path):
    found = rel_paths(corpus, exclude=["__pycache__", "Аккаунты"])
    assert found == ["notebooks/exp_a.ipynb", "pkg/metrics.py"]


def test_walker_path_pattern_and_protected_dirs(corpus: Path):
    assert rel_paths(corpus, exclude=["notebooks/**", "__pycache__", "Аккаунты"]) == ["pkg/metrics.py"]
    assert rel_paths(corpus, exclude=["__pycache__", "Аккаунты"], protected_dirs=[corpus / "pkg"]) == [
        "notebooks/exp_a.ipynb"
    ]
    # a protected dir that contains the root must not hide the whole corpus
    assert len(rel_paths(corpus, exclude=["__pycache__", "Аккаунты"], protected_dirs=[corpus.parent])) == 2


def test_walker_size_limit(corpus: Path):
    skipped = []
    found = list(iter_corpus(corpus, [".py"], exclude=["__pycache__", "Аккаунты"], max_file_mb=1e-5, skipped=skipped))
    assert found == []
    assert skipped and "too large" in skipped[0].reason


def test_exclude_rules_name_vs_path():
    rules = ExcludeRules(["*.egg-info", "data/raw/**"], skip_hidden=True)
    assert rules.excludes("pkg.egg-info")
    assert rules.excludes("data/raw/x.py")
    assert not rules.excludes("data/clean/x.py")
    assert rules.excludes(".hidden/x.py")


def test_line_windows_cover_all_lines():
    assert line_windows(0, 5, 1) == []
    assert line_windows(3, 5, 1) == [(0, 3)]
    assert line_windows(10, 4, 1) == [(0, 4), (3, 7), (6, 10)]


def test_python_parser_chunks_and_outline(corpus: Path):
    cf = next(f for f in iter_corpus(corpus, [".py"], exclude=["__pycache__", "Аккаунты"]))
    parsed = parse_file(cf, ChunkingConfig(python="lines", lines_per_chunk=6, overlap_lines=2))
    file_node = parsed.nodes[0]
    assert file_node.node_type == NodeType.FILE and not file_node.embed
    assert file_node.metadata["functions"] == ["compute_f1"]
    assert file_node.metadata["classes"] == ["Trainer"]
    assert "numpy" in file_node.metadata["imports"]
    chunks = [n for n in parsed.nodes if n.node_type == NodeType.CODE_CHUNK]
    assert chunks[0].location.line_start == 1 and chunks[0].location.line_end == 6
    assert all(e.type == EdgeType.CONTAINS for e in parsed.edges)
    assert parsed.warnings == []


def test_python_parser_reports_syntax_error(tmp_path: Path):
    (tmp_path / "bad.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    cf = next(iter_corpus(tmp_path, [".py"]))
    parsed = parse_file(cf, ChunkingConfig())
    assert any("SyntaxError" in w for w in parsed.warnings)
    assert any(n.node_type == NodeType.CODE_CHUNK for n in parsed.nodes)


def test_python_parser_reads_cp1251(tmp_path: Path):
    (tmp_path / "old.py").write_bytes("# старый комментарий\nx = 1\n".encode("cp1251"))
    parsed = parse_file(next(iter_corpus(tmp_path, [".py"])), ChunkingConfig())
    assert "старый комментарий" in parsed.nodes[1].text
    assert parsed.nodes[0].metadata["encoding"] == "cp1251"


def test_notebook_parser_cells_outputs_edges(corpus: Path):
    cf = next(f for f in iter_corpus(corpus, [".ipynb"]))
    parsed = parse_file(cf, ChunkingConfig())
    by_id = {n.id: n for n in parsed.nodes}
    rel = "notebooks/exp_a.ipynb"

    assert by_id[rel].metadata["out_of_order"] is True
    assert by_id[rel].text == "Experiment A"
    md = by_id[f"{rel}#cell1"]
    assert md.node_type == NodeType.MARKDOWN_CELL and md.location.cell == 1
    code = by_id[f"{rel}#cell2"]
    assert code.metadata["section"] == "Experiment A"
    out = by_id[f"{rel}#cell2/out"]
    assert out.node_type == NodeType.CELL_OUTPUT and "0.9925" in out.text
    assert "lr = 0.05" in out.context  # producing code is attached for retrieval
    table = by_id[f"{rel}#cell3/out"]
    assert table.metadata["n_images"] == 1 and "a  b" in table.text
    err = by_id[f"{rel}#cell4/out"]
    assert err.text == "ZeroDivisionError: division by zero"
    assert f"{rel}#cell5" not in by_id  # empty markdown cell skipped
    produces = {(e.src, e.dst) for e in parsed.edges if e.type == EdgeType.PRODUCES}
    assert (f"{rel}#cell2", f"{rel}#cell2/out") in produces


def test_truncate_keeps_tail():
    text = "header\n" + "x" * 1000 + "\nFINAL roc_auc=0.99"
    out = truncate_keep_tail(text, 200)
    assert len(out) <= 200
    assert out.endswith("FINAL roc_auc=0.99")
    assert "обрезано" in out
