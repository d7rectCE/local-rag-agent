from rag_agent.config import load_settings
from rag_agent.generation import extract_citations


def test_citations_normalised_and_filtered():
    text, cited = extract_citations("LR = 0.05 [2, 1]; see also [7] and [3].", valid={1, 2, 3})
    assert text == "LR = 0.05 [2][1]; see also  and [3]."
    assert cited == [2, 1, 3]


def test_citations_ignore_code():
    text, cited = extract_citations("Use `x[1]` here [1].\n```python\ny = arr[2]\n```", valid={1, 2})
    assert "`x[1]`" in text and "arr[2]" in text
    assert cited == [1]


def test_env_overrides_nested_values(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("llm:\n  model: a\nretrieval:\n  top_k: 3\n", encoding="utf-8")
    s = load_settings(cfg, environ={"RAG__LLM__MODEL": "qwen3.6:27b", "RAG__RETRIEVAL__TOP_K": "9", "OTHER": "1"},
                      local_path=None)
    assert s.llm.model == "qwen3.6:27b"
    assert s.retrieval.top_k == 9


def test_missing_model_error_mentions_download(monkeypatch):
    import pytest

    from rag_agent.index.embedder import load_pretrained

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    class Loader:
        @staticmethod
        def from_pretrained(name, local_files_only, **kwargs):
            assert local_files_only is True
            raise OSError("not in cache")

    with pytest.raises(RuntimeError, match="hf download org/missing"):
        load_pretrained(Loader, "org/missing", True)
