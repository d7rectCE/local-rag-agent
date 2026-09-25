"""Configuration: YAML file + environment overrides (NFR6).

Resolution order: defaults in the models below < YAML file (``RAG_CONFIG`` or
``configs/default.yaml``) < machine-specific ``configs/local.yaml`` (not
committed) < environment variables ``RAG__SECTION__KEY``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
LOCAL_CONFIG_PATH = REPO_ROOT / "configs" / "local.yaml"
ENV_PREFIX = "RAG__"


class CorpusConfig(BaseModel):
    include_ext: list[str] = [".py", ".ipynb"]
    exclude: list[str] = []
    skip_hidden: bool = True
    max_file_mb: float = 20.0


class StorageConfig(BaseModel):
    data_dir: str = "~/.local-rag-agent"
    qdrant_url: str | None = None


class ChunkingConfig(BaseModel):
    python: Literal["lines", "ast"] = "lines"
    lines_per_chunk: int = 60
    overlap_lines: int = 10
    max_chunk_chars: int = 6000
    output_max_chars: int = 2000
    context_header: bool = True


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "auto"
    fp16: bool = True
    batch_size: int = 16
    max_length: int = 1024
    pooling: Literal["cls", "mean"] = "cls"


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "sparse", "hybrid"] = "dense"
    top_k: int = 6


class LLMConfig(BaseModel):
    provider: Literal["ollama", "openai"] = "ollama"
    model: str = "qwen3.5:9b"
    base_url: str = "http://localhost:11434"
    api_key_env: str | None = None
    temperature: float = 0.0
    think: bool = False
    num_ctx: int = 16384
    max_tokens: int = 1024
    timeout_s: float = 300.0


class GenerationConfig(BaseModel):
    max_source_chars: int = 2500


class TracingConfig(BaseModel):
    enabled: bool = True
    log_prompts: bool = True


class EvaluationConfig(BaseModel):
    judge_model: str = "qwen3.6:27b"
    judge_think: bool = False
    retrieval_k: int = 10
    bootstrap: int = 1000
    output_dir: str = "runs/eval"


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class Settings(BaseModel):
    corpus: CorpusConfig = CorpusConfig()
    storage: StorageConfig = StorageConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    generation: GenerationConfig = GenerationConfig()
    tracing: TracingConfig = TracingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    api: ApiConfig = ApiConfig()

    @property
    def data_dir(self) -> Path:
        return Path(os.path.expandvars(self.storage.data_dir)).expanduser().resolve()


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _apply_env_overrides(data: dict[str, Any], environ: dict[str, str]) -> None:
    for key, raw in environ.items():
        if not key.upper().startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in key[len(ENV_PREFIX):].split("__") if p]
        if not path:
            continue
        node = data
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = yaml.safe_load(raw) if raw != "" else None


def _read_yaml(path: Path) -> dict[str, Any]:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}


def load_settings(
    path: str | Path | None = None,
    environ: dict[str, str] | None = None,
    local_path: Path | None = LOCAL_CONFIG_PATH,
) -> Settings:
    environ = dict(os.environ) if environ is None else environ
    data = _read_yaml(Path(path or environ.get("RAG_CONFIG") or DEFAULT_CONFIG_PATH))
    if local_path is not None:
        _deep_merge(data, _read_yaml(local_path))
    _apply_env_overrides(data, environ)
    return Settings.model_validate(data)
