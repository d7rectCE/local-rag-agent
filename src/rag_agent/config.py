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
    include_ext: list[str] = [".py", ".ipynb", ".pdf", ".docx", ".txt", ".md", ".log", ".png", ".jpg", ".jpeg"]
    exclude: list[str] = []
    skip_hidden: bool = True
    max_file_mb: float = 20.0


class StorageConfig(BaseModel):
    data_dir: str = "~/.local-rag-agent"
    qdrant_url: str | None = None
    # weights the user drops in: models/text (GGUF for Ollama), models/image and models/video (a folder per
    # model with its VAE and text encoder); relative to the project folder, ignored by git
    models_dir: str = "models"


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
    num_ctx: int = 65536  # prompt + answer (Ollama's window); qwen35 models keep a small KV cache
    # generated tokens per call when the caller sets no limit: answers (1024 cut long answers mid-sentence)
    max_tokens: int = 16384
    timeout_s: float = 1500.0
    # sampling of the free text of answers set in the UI (temperature, top_p, top_k, repeat_penalty): calls with
    # a JSON schema (routing, plans, extraction) keep temperature and the defaults, so they stay reliable
    options: dict[str, float] = {}


class GenerationConfig(BaseModel):
    max_source_chars: int = 2500
    # detailed: explained answers with structure and tables (the chat); concise: the short answers of the
    # evaluation runs up to Э17 (set it in an ablation spec to reproduce them)
    style: Literal["detailed", "concise"] = "detailed"


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
    # < lower: irrelevant -> rewrite the query; after the retries -> refuse. 0.02 from the calibration on
    # demo_v4 (reports/e7/crag_calibration.md): half of Q6 caught, no false refusals
    crag_lower: float = 0.02
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


class CodeConfig(BaseModel):
    """Code agent and its sandbox (ТЗ ч.2 S17, Э15)."""

    image: str = "rag-sandbox:py312"  # docker/sandbox.Dockerfile
    timeout_s: float = 120.0  # per run in the sandbox
    memory: str = "2g"
    cpus: float = 2.0
    pids: int = 256
    max_iterations: int = 5  # failed runs before the agent stops (H13: 1 = no fixing from execution feedback)
    max_steps: int = 20
    # a new multi-module program (the "project" pipeline): the fix loop after the planned files gets more room
    project_max_steps: int = 40
    project_max_iterations: int = 8
    aci: bool = True  # view / search / edit with a syntax check; false: whole-file overwrite (H14 baseline)
    workspace_max_mb: float = 50.0  # text files copied from the corpus into the working copy
    file_max_mb: float = 2.0
    # databases and tables (SQLite, Parquet, Excel) are copied too, to be read by the code; they are ignored by
    # git, so they never show in the diff and are never copied back into the user's folder
    data_max_mb: float = 200.0
    # the whole corpus read-only at /corpus: off by default — it would expose folders excluded from indexing
    # (e.g. "Аккаунты"); the working copy already holds the filtered text files
    mount_corpus: bool = False
    # the chat hands tasks that need running code to the code agent and gives the research agent run_code
    # (one agent for the user, no separate code mode); off: code runs only through /code and `rag code`
    chat: Literal["off", "auto"] = "auto"
    # network in the sandbox: never (NFR8 as written) or confirm — the user turns it on per request and
    # confirms every task that runs with it (downloads, pip install into the working copy)
    network: Literal["never", "confirm"] = "confirm"


class WebConfig(BaseModel):
    """Web gateway (ТЗ ч.2 S19, Э16): SearXNG on this machine, page reading, citations with dates."""

    # off: never; auto: temporal markers / external facts (router) or irrelevant corpus results;
    # always: every question also searches the web. Off by default: the web is an explicit toggle (S21)
    mode: Literal["off", "auto", "always"] = "off"
    searxng_url: str = "http://127.0.0.1:8888"
    searxng_image: str = "searxng/searxng"
    results: int = 8
    pages: int = 4  # pages read per question
    page_max_mb: float = 3.0
    timeout_s: float = 15.0
    cache_days: float = 7.0
    search_cache_hours: float = 24.0  # the same query within a day is answered from the disk cache
    min_interval_s: float = 1.5  # between queries to SearXNG: its engines suspend clients that send bursts
    passages: int = 6  # pages (their compressed fragments) given to the answer
    # primary sources first, aggregators and SEO sites last (S19)
    prefer: list[str] = ["arxiv.org", "aclanthology.org", "openreview.net", "github.com", "pypi.org", "docs.python.org",
                         "readthedocs.io", "pytorch.org", "scikit-learn.org", "numpy.org", "pandas.pydata.org",
                         "huggingface.co", "python.org", "wikipedia.org", "ollama.com", "docs."]
    demote: list[str] = ["medium.com", "towardsdatascience.com", "geeksforgeeks.org", "w3schools.com", "tutorialspoint.com",
                         "javatpoint.com", "quora.com", "pinterest.", "dev.to", "habr.com/ru/companies", "analyticsvidhya.com"]


class ImagesConfig(BaseModel):
    """Image pipeline (ТЗ S4, Э5): type classifier, VLM descriptions, OCR, optional visual index (H4)."""

    enabled: bool = True  # describe the images of the corpus after indexing
    vlm_model: str | None = None  # a vision model in Ollama; None: llm.model
    classify: bool = True  # the fine-tuned CNN of scripts/train_image_classifier.py (if exported)
    ocr: bool = True  # EasyOCR for screenshots and scans, with the models Docling uses
    max_images: int = 500  # described per indexing run (the rest on the next run)
    # ColQwen2 multi-vector index of images and PDF pages; ~4.5 GB of VRAM, so off next to a 27B LLM
    visual_index: bool = False
    visual_model: str = "vidore/colqwen2-v1.0-hf"
    visual_pages: bool = True
    # how image evidence is retrieved (H4): text — the descriptions in the text index; visual — the
    # visual index instead of the descriptions; fusion — both, by RRF
    channel: Literal["text", "visual", "fusion"] = "text"


class ImageModelProfile(BaseModel):
    """A diffusion model with its own companions and sampling: another family (its own VAE and text encoder)
    or a distilled "turbo" build (few steps, CFG 1). Empty fields fall back to the imagegen section."""

    path: str
    vae: str = ""
    llm: str = ""
    steps: int | None = None
    cfg_scale: float | None = None
    sampler: str = ""
    scheduler: str = ""  # sd-cli --scheduler (simple, discrete, karras, ...); empty: the model's default
    # None: as the imagegen section. A small model fits the GPU whole; an fp8 encoder must not run on the CPU
    # (the CPU backend of sd.cpp crashed on Qwen3-VL-4B fp8 weights, the Vulkan one runs them)
    text_encoder_on_cpu: bool | None = None


class ImageGenConfig(BaseModel):
    """Image generation and editing in the chat (Qwen-Image-2.1 through stable-diffusion.cpp's sd-cli).
    Off by default: the engine and the weights are the user's local choice; the paths belong in
    configs/local.yaml."""

    enabled: bool = False
    sd_cli: str = ""  # stable-diffusion.cpp's sd-cli executable
    diffusion_model: str = ""  # the DiT (GGUF or safetensors): the default one
    models_dir: str = ""  # more DiT files to choose from in the UI (besides the folder of diffusion_model)
    vae: str = ""
    llm: str = ""  # the text encoder (Qwen3-VL-8B for Qwen-Image-2.1)
    width: int = 1024
    height: int = 1024
    steps: int = 20
    # sd.cpp's docs give 6.0; with this model it burns the picture (dark, crushed shadows) and draws vertical 8 px
    # stripes: their strength (FFT at 8 px) fell 117 -> 44 -> 23 at CFG 6 -> 4 -> 3, the same on Vulkan and ROCm
    cfg_scale: float = 3.0
    sampler: str = "euler"
    scheduler: str = ""  # sd-cli --scheduler; empty: the model's default
    # more models for the UI picker, each with its own VAE, text encoder and sampling (see ImageModelProfile)
    models: list[ImageModelProfile] = []
    strength: float = 0.75  # redraw: how far the result may move from the source
    negative_prompt: str = ""
    # the text encoder (8B) runs on the CPU (~9 s per prompt) and the DiT + VAE stay on the GPU: offloading all
    # weights to RAM made the Vulkan build stage the quantized DiT for ~230 s per picture
    text_encoder_on_cpu: bool = True
    offload_to_cpu: bool = False
    flash_attention: bool = True
    vae_tiling: bool = True  # the Vulkan build cannot allocate the whole 1024x1024 decode and falls back to the CPU
    unload_llm: bool = True  # the chat model leaves the GPU before an image is made
    work_dir: str = ""  # sd-cli's inputs and output; must have an ASCII path (default: the system temp folder)
    timeout_s: float = 900.0


class SecurityConfig(BaseModel):
    """Policy layer switch (ТЗ ч.2 S20). Turning it off is only for the H16 comparison in red-team runs."""

    policies: bool = True


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
    code: CodeConfig = CodeConfig()
    web: WebConfig = WebConfig()
    images: ImagesConfig = ImagesConfig()
    imagegen: ImageGenConfig = ImageGenConfig()
    security: SecurityConfig = SecurityConfig()
    tracing: TracingConfig = TracingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    api: ApiConfig = ApiConfig()

    @property
    def data_dir(self) -> Path:
        return Path(os.path.expandvars(self.storage.data_dir)).expanduser().resolve()

    @property
    def models_dir(self) -> Path:
        p = Path(os.path.expandvars(self.storage.models_dir)).expanduser()
        return (p if p.is_absolute() else REPO_ROOT / p).resolve()


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
        # YAML 1.1 reads on/off/yes/no as booleans; here they are mode names (reasoning, agent, web)
        node[path[-1]] = raw if raw.lower() in {"on", "off", "yes", "no"} else (yaml.safe_load(raw) if raw != "" else None)


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
