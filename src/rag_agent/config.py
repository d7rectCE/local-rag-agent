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
    include_ext: list[str] = [".py", ".ipynb", ".pdf", ".docx", ".txt", ".md", ".log"]
    exclude: list[str] = []
    skip_hidden: bool = True
    max_file_mb: float = 20.0


class StorageConfig(BaseModel):
    data_dir: str = "~/.local-rag-agent"
    qdrant_url: str | None = None


class ChunkingConfig(BaseModel):
    python: Literal["lines", "ast"] = "ast"
    lines_per_chunk: int = 60
    overlap_lines: int = 10
    ast_max_chars: int = 1800  # AST chunk budget in non-whitespace characters
    max_chunk_chars: int = 6000
    output_max_chars: int = 2000
    context_header: bool = True
    notebook_context: bool = True  # notebook title and heading path in every cell's header
    text_max_chars: int = 1500  # prose chunks of PDF / DOCX / TXT, cut at paragraph boundaries
    docx: Literal["structured", "plain"] = "structured"  # plain = flat text in fixed windows (H10 baseline)


DEFAULT_LAYOUT_MODEL = "docling-project/docling-layout-heron"


class DocumentsConfig(BaseModel):
    """PDF conversion (Docling) and OCR (ТЗ S1)."""

    models_dir: str = "~/.cache/docling/models"  # filled by `docling-tools models download`
    device: Literal["auto", "cpu", "cuda"] = "auto"
    ocr: Literal["auto", "always", "never"] = "auto"  # auto: only pages without a text layer
    ocr_langs: list[str] = ["ru", "en"]
    min_text_chars: int = 30  # a page with fewer extractable characters counts as scanned
    table_mode: Literal["accurate", "fast"] = "accurate"
    # PDF text backend: docling-parse (Docling's default) cannot open its resources from a
    # non-ASCII install path on Windows; auto falls back to pdfium there
    backend: Literal["auto", "docling_parse", "pypdfium"] = "auto"
    # layout detector (H8): Hugging Face repo id, looked up as <models_dir>/<org>--<name>; the default is
    # Docling's Heron, `scripts/train_layout_detector.py export` installs our DocLayNet model next to it
    layout_model: str = DEFAULT_LAYOUT_MODEL


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "auto"
    fp16: bool = True
    batch_size: int = 16
    max_length: int = 1024
    pooling: Literal["cls", "mean"] = "cls"
    # load models only from the local Hugging Face cache: no network calls at runtime (NFR1);
    # a missing model raises an error with the download command
    local_files_only: bool = True


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "sparse", "hybrid"] = "dense"
    top_k: int = 6
    symbols: bool = True  # exact-name channel: definitions and call sites of identifiers in the query
    rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_pool: int = 40  # first-stage candidates passed to the cross-encoder
    rerank_max_length: int = 512
    rerank_batch_size: int = 16


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


class ReasoningConfig(BaseModel):
    """Reasoning mode (ТЗ ч.2, S15): off, on, or auto (the router decides per question)."""

    mode: Literal["off", "on", "auto"] = "auto"
    # hard caps on reasoning tokens by the router's difficulty estimate; past the cap the answer
    # is forced (budget forcing). "on" without an estimate uses the deep budget.
    budget_tokens: dict[str, int] = {"light": 512, "deep": 1536}
    # auto mode: a fast answer that is not grounded in the sources is retried with deep reasoning
    escalate: bool = True

    def budget(self, level: str) -> int:
        return self.budget_tokens.get(level) or max(self.budget_tokens.values())


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


class CatalogConfig(BaseModel):
    """Relational catalog of experiments and the SQL tool (ТЗ S6)."""

    # LLM extraction of experiments, metrics and hyperparameters from notebooks after indexing;
    # cached per file content and model, every value is checked against its source cell
    extract: bool = True
    notebook_chars: int = 14000  # notebook text shown to the extractor
    sql: bool = True  # aggregate questions also query the catalog
    max_rows: int = 50  # forced LIMIT
    timeout_s: float = 5.0
    attempts: int = 3  # generation + self-correction on an error or an empty result
    examples: str | None = None  # extra "question -> SQL" examples (YAML list), added to the built-in ones


class AgentConfig(BaseModel):
    """ReAct agent with tools and the CRAG relevance check (ТЗ S7, Э7)."""

    # auto: aggregate and multi-step questions (router: aggregate or reasoning light/deep) go through
    # the agent, simple ones are answered by direct retrieval; always / off force one path
    mode: Literal["off", "auto", "always"] = "auto"
    max_steps: int = 8
    max_generated_tokens: int = 6000  # tool-call tokens over the whole trajectory
    timeout_s: float = 60.0  # the loop stops and answers from the evidence gathered so far
    observation_chars: int = 500  # per fragment shown to the agent
    evidence: int = 8  # fragments passed to the final answer
    # CRAG: relevance of the retrieved fragments by the cross-encoder (reranker) score in [0, 1]
    crag: bool = True
    crag_upper: float = 0.5  # >= upper: relevant
    crag_lower: float = 0.1  # < lower: irrelevant -> rewrite the query; after the retries -> refuse
    crag_retries: int = 1


class UploadsConfig(BaseModel):
    """Files uploaded into a conversation (ТЗ ч.2 S16, Э13)."""

    max_mb: float = 50.0
    ttl_hours: float = 24.0  # session scope: uploads and their index are deleted after this
    parse_timeout_s: float = 600.0  # parsing runs in a separate process
    # routing: whole (always the whole file, in parts when it does not fit), index (retrieval only),
    # self_route (whole when it fits, otherwise retrieval, and the whole file in parts when the model
    # says the retrieved fragments are not enough)
    route: Literal["self_route", "whole", "index"] = "self_route"
    context_share: float = 0.33  # effective context budget: a third of llm.num_ctx (RULER)
    chars_per_token: float = 3.0  # rough size estimate before tokenization
    top_k: int = 8


class Settings(BaseModel):
    corpus: CorpusConfig = CorpusConfig()
    storage: StorageConfig = StorageConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    documents: DocumentsConfig = DocumentsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    generation: GenerationConfig = GenerationConfig()
    reasoning: ReasoningConfig = ReasoningConfig()
    catalog: CatalogConfig = CatalogConfig()
    agent: AgentConfig = AgentConfig()
    uploads: UploadsConfig = UploadsConfig()
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
