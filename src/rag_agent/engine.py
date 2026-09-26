"""Engine: holds the models and the active corpus index; shared by API and CLI."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_agent.config import REPO_ROOT, Settings, load_settings
from rag_agent.generation import Answer, TraceStep, generate_answer, generate_general
from rag_agent.index.embedder import Embedder
from rag_agent.index.indexer import CorpusIndex, IndexProgress, corpus_key, run_indexing
from rag_agent.llm import BaseLLM, make_llm
from rag_agent.index.reranker import Reranker
from rag_agent.retrieval import Hit, Mode, corpus_mentions, search
from rag_agent.router import RouteChoice, route_question, trim_history
from rag_agent.tracing import NULL_TRACER, Tracer

log = logging.getLogger(__name__)


class NoCorpusError(RuntimeError):
    pass


class IndexingBusyError(RuntimeError):
    pass


class CorpusRegistry:
    """Known corpora and their per-folder indexing preferences (corpora.json)."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write(self, data: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def all(self) -> list[dict]:
        with self._lock:
            items = list(self._read().values())
        return sorted(items, key=lambda d: d.get("last_used", ""), reverse=True)

    def get(self, root: Path) -> dict | None:
        with self._lock:
            return self._read().get(corpus_key(root))

    def update(self, root: Path, **fields) -> dict:
        with self._lock:
            data = self._read()
            entry = data.setdefault(corpus_key(root), {"root": str(root)})
            entry.update(fields)
            entry["last_used"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._write(data)
            return entry


class Engine:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: BaseLLM | None = None,
        reranker: Reranker | None = None,
    ):
        self.settings = settings or load_settings()
        data_dir = self.settings.data_dir
        tr = self.settings.tracing
        self.tracer = Tracer(data_dir / "traces", tr.log_prompts) if tr.enabled else NULL_TRACER
        self.embedder = embedder or Embedder(self.settings.embedding)
        self.llm = llm or make_llm(self.settings.llm, self.tracer)
        self._reranker = reranker
        self.registry = CorpusRegistry(data_dir / "corpora.json")
        self.progress = IndexProgress()
        self._index: CorpusIndex | None = None
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    # --- corpus management ------------------------------------------------
    @property
    def protected_dirs(self) -> list[Path]:
        """Never index the tool itself or its own data, even if they lie inside the corpus."""
        return [REPO_ROOT, self.settings.data_dir]

    def open_corpus(self, root: str | Path) -> CorpusIndex:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"folder not found: {root}")
        with self._lock:
            if self._index is not None and self._index.root == root:
                return self._index
            if self.progress.running:
                raise IndexingBusyError("indexing is in progress")
            if self._index is not None:
                self._index.close()
            self._index = CorpusIndex(root, self.settings, self.embedder)
            self.registry.update(root)
            return self._index

    @property
    def index(self) -> CorpusIndex:
        if self._index is None:
            known = self.registry.all()
            if not known:
                raise NoCorpusError("no corpus indexed yet")
            self.open_corpus(known[0]["root"])
        return self._index

    def corpus_prefs(self, root: str | Path) -> dict:
        """Per-folder choices of the user; defaults from the config otherwise. Only
        explicit choices are remembered, so new default file types reach old folders."""
        entry = self.registry.get(Path(root).expanduser().resolve()) or {}
        explicit = entry.get("explicit", {})
        return {
            "include_ext": entry["include_ext"] if explicit.get("include_ext") else self.settings.corpus.include_ext,
            "exclude": entry["exclude"] if explicit.get("exclude") else self.settings.corpus.exclude,
        }

    # --- indexing ---------------------------------------------------------
    def index_folder(
        self,
        root: str | Path,
        *,
        include_ext: list[str] | None = None,
        exclude: list[str] | None = None,
        force: bool = False,
        background: bool = False,
    ) -> IndexProgress:
        with self._lock:
            if self.progress.running:
                raise IndexingBusyError("indexing is already running")
            index = self.open_corpus(root)
            prefs = self.corpus_prefs(index.root)
            explicit = dict((self.registry.get(index.root) or {}).get("explicit", {}))
            if include_ext:
                explicit["include_ext"] = True
            if exclude is not None:
                explicit["exclude"] = True
            include_ext = include_ext or prefs["include_ext"]
            exclude = prefs["exclude"] if exclude is None else exclude
            self.registry.update(index.root, include_ext=include_ext, exclude=exclude, explicit=explicit)
            progress = IndexProgress(state="scanning", root=str(index.root))
            self.progress = progress
            self._cancel.clear()

        def job() -> None:
            try:
                run_indexing(
                    index,
                    self.settings,
                    self.embedder,
                    progress,
                    include_ext=include_ext,
                    exclude=exclude,
                    protected_dirs=self.protected_dirs,
                    force=force,
                    cancel=self._cancel,
                )
            except Exception:
                log.exception("indexing failed")
            finally:
                summary = progress.model_dump(exclude={"skipped"})
                self.registry.update(index.root, last_indexed=progress.finished_at, last_state=progress.state)
                self.tracer.log("index_run", **summary)

        if background:
            self._thread = threading.Thread(target=job, name="indexer", daemon=True)
            self._thread.start()
        else:
            job()
        return progress

    def cancel_indexing(self) -> None:
        self._cancel.set()

    @property
    def indexing_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- question answering -------------------------------------------------
    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            emb = self.settings.embedding
            self._reranker = Reranker(self.settings.retrieval, device=emb.device, fp16=emb.fp16,
                                      local_files_only=emb.local_files_only)
        return self._reranker

    def search(
        self,
        question: str,
        top_k: int | None = None,
        mode: Mode | None = None,
        rerank: bool | None = None,
        symbols: bool | None = None,
    ) -> list[Hit]:
        cfg = self.settings.retrieval
        use_rerank = cfg.rerank if rerank is None else rerank
        return search(
            self.index,
            self.embedder,
            question,
            top_k or cfg.top_k,
            mode or cfg.mode,
            reranker=self.reranker if use_rerank else None,
            rerank_pool=cfg.rerank_pool,
            symbols=cfg.symbols if symbols is None else symbols,
            with_header=self.settings.chunking.context_header,
        )

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
        mode: Mode | None = None,
        route: RouteChoice = "auto",
        rerank: bool | None = None,
        symbols: bool | None = None,
        reasoning: str | None = None,
    ) -> Answer:
        """Route the question, then answer it from the user's files (with citations)
        or from general knowledge. ``history`` is the previous chat turns
        (``{"role": "user"|"assistant", "content": ...}``) for follow-up questions.
        ``reasoning`` is off / on / auto (the router decides); defaults to the config."""
        question = question.strip()
        if not question:
            raise ValueError("empty question")
        t0 = time.perf_counter()
        turns = trim_history(history)
        steps: list[TraceStep] = []

        index = None
        try:
            index = self.index
        except NoCorpusError:
            if route == "corpus":
                raise

        standalone = question
        chosen = route
        reasoning_mode = reasoning or self.settings.reasoning.mode
        think = reasoning_mode == "on"
        # the router also rewrites follow-ups into standalone questions and decides on reasoning in auto mode
        if route == "auto" or turns or reasoning_mode == "auto":
            decision = route_question(question, turns, self.llm)
            standalone = decision.standalone_question
            if reasoning_mode == "auto":
                think = decision.needs_reasoning
            detail = {"route": decision.route, "standalone_question": standalone, "fallback": decision.fallback,
                      "needs_reasoning": decision.needs_reasoning}
            if route == "auto":
                chosen = decision.route
                # corpus signal: a question that names a function, class or file of the corpus is about the files
                mentions = corpus_mentions(index, standalone) if index is not None and chosen == "general" else []
                if mentions:
                    chosen = "corpus"
                    detail.update(override="corpus", corpus_mentions=mentions)
            steps.append(TraceStep(name="route", duration_s=round(decision.latency_s, 3), detail=detail))

        notice = None
        if chosen == "corpus" and index is None:
            chosen, notice = "general", "Папка ещё не проиндексирована, поэтому ответ дан из общих знаний модели."

        budget = self.settings.reasoning.budget_tokens if think else None
        if chosen == "general":
            answer = generate_general(question, turns, self.llm, reasoning_budget=budget)
        else:
            top_k = top_k or self.settings.retrieval.top_k
            mode = mode or self.settings.retrieval.mode
            t1 = time.perf_counter()
            hits = self.search(standalone, top_k, mode, rerank=rerank, symbols=symbols)
            cfg = self.settings.retrieval
            steps.append(
                TraceStep(
                    name="retrieve",
                    duration_s=round(time.perf_counter() - t1, 3),
                    detail={
                        "mode": mode,
                        "top_k": top_k,
                        "rerank": cfg.rerank if rerank is None else rerank,
                        "symbols": cfg.symbols if symbols is None else symbols,
                        "hits": [[h.node.id, round(h.score, 4)] for h in hits],
                    },
                )
            )
            answer = generate_answer(standalone, hits, self.llm, self.settings.generation.max_source_chars,
                                     reasoning_budget=budget)
            answer.question = question

        answer.standalone_question = standalone if standalone != question else None
        answer.notice = notice
        answer.trace[:0] = steps
        answer.latency_s = round(time.perf_counter() - t0, 3)
        self.tracer.log(
            "ask",
            corpus=str(index.root) if index is not None else None,
            question=question if self.tracer.log_prompts else None,
            route=answer.route,
            answerable=answer.answerable,
            grounded=answer.grounded,
            citations=[c.node_id for c in answer.citations],
            trace=[s.model_dump() for s in answer.trace],
            latency_s=answer.latency_s,
        )
        return answer

    def lookup_symbol(self, name: str) -> dict:
        """Definitions and call sites of a name from the static index (no LLM, ТЗ S2)."""
        catalog = self.index.catalog
        return {"name": name, "definitions": catalog.find_symbols(name), "calls": catalog.find_calls(name)}

    # --- introspection ------------------------------------------------------
    def file_view(self, rel_path: str) -> list[dict]:
        """Parsed content of one indexed file (served from the catalog, never from disk)."""
        return [n.model_dump(mode="json") for n in self.index.catalog.file_nodes(rel_path)]

    def status(self) -> dict:
        idx = self._index
        if idx is None:
            try:
                idx = self.index
            except NoCorpusError:
                idx = None
        return {
            "corpus": None
            if idx is None
            else {
                "root": str(idx.root),
                "signature": idx.signature,
                "index_dir": str(idx.dir),
                "stats": idx.catalog.stats(),
                "vectors": idx.store.count(),
                "prefs": self.corpus_prefs(idx.root),
            },
            "progress": self.progress.model_dump(),
            "embedder": {"model": self.embedder.name, "device": self.embedder.device},
            "llm": {"name": self.llm.name, "available": getattr(self.llm, "is_available", lambda: True)()},
            "retrieval": self.settings.retrieval.model_dump(),
            "reasoning": self.settings.reasoning.model_dump(),
            "defaults": {"include_ext": self.settings.corpus.include_ext, "exclude": self.settings.corpus.exclude},
        }

    def close(self) -> None:
        with self._lock:
            if self._index is not None:
                self._index.close()
                self._index = None
        self.llm.close()
