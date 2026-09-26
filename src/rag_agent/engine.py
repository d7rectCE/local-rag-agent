"""Engine: holds the models and the active corpus index; shared by API and CLI."""

from __future__ import annotations

import json
import re
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_agent.agent import Agent, RelevanceEvaluator, rewrite_query
from rag_agent.code.agent import CodeAgent, CodeResult, catalog_metric_rows, new_task_id
from rag_agent.code.sandbox import DockerSandbox
from rag_agent.code.workspace import Workspace, WorkspaceError
from rag_agent.config import REPO_ROOT, Settings, load_settings
from rag_agent.policy import Policy, Provenance, defang_markdown
from rag_agent.generation import Answer, TraceStep, generate_answer, generate_general
from rag_agent.index.embedder import Embedder
from rag_agent.index.indexer import CorpusIndex, IndexProgress, corpus_key, run_indexing
from rag_agent.llm import BaseLLM, make_llm
from rag_agent.index.reranker import Reranker
from rag_agent.retrieval import Hit, Mode, corpus_mentions, search
from rag_agent.router import RouteChoice, route_question, trim_history
from rag_agent.schema import FileType, Location, Node, NodeType
from rag_agent.structured.analytics import ANALYTICS_FILE, build_analytics
from rag_agent.structured.extract import update_catalog
from rag_agent.structured.sql import SQLResult, SQLTool
from rag_agent.tracing import NULL_TRACER, Tracer
from rag_agent.uploads import UploadError, UploadInfo, UploadStore, answer_from_uploads
from rag_agent.web_qa import answer_from_web

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
        self.uploads = UploadStore(self.settings, self.embedder)
        try:
            self.uploads.cleanup()  # expired session uploads
        except OSError:
            log.warning("could not clean up expired uploads", exc_info=True)

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
                if progress.state == "done":
                    self._update_catalog(index, progress)
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

    def _update_catalog(self, index: CorpusIndex, progress: IndexProgress) -> None:
        """Э6: extract experiments from new or changed notebooks, then rebuild the analytics database."""
        t0 = time.perf_counter()
        if self.settings.catalog.extract:
            progress.state = "extracting"
            try:
                progress.catalog = update_catalog(index, self.llm, self.settings.catalog, progress, self._cancel)
            except Exception as exc:  # the index itself is fine: answer without the catalog
                log.exception("catalog extraction failed")
                progress.catalog = {"error": str(exc)}
        try:
            progress.catalog["tables"] = build_analytics(index.catalog, index.dir / ANALYTICS_FILE)
        except Exception as exc:
            log.exception("analytics database build failed")
            progress.catalog["error"] = str(exc)
        progress.current = None
        progress.elapsed_s = round(progress.elapsed_s + time.perf_counter() - t0, 2)
        progress.state = "cancelled" if self._cancel.is_set() else "done"

    def sql_tool(self) -> SQLTool:
        return SQLTool(self.index.dir / ANALYTICS_FILE, self.llm, self.settings.catalog, self.embedder)

    def query_catalog(self, question: str) -> SQLResult:
        """Answer an aggregate question with SQL over the catalog (no answer text)."""
        res = self.sql_tool().run(question)
        self.tracer.log("sql", question=question if self.tracer.log_prompts else None, **res.model_dump(exclude={"question", "rows"}),
                        n_rows=len(res.rows))
        return res

    def _sql_hits(self, res: SQLResult, index: CorpusIndex) -> list[Hit]:
        """The SQL result as a source for the answer, followed by the notebook cells its rows come from."""
        node = Node(id="catalog:sql", file_path="каталог экспериментов (SQL)", file_type=FileType.CATALOG,
                    node_type=NodeType.TABLE, title="Результат запроса к каталогу",
                    text=f"SQL: {res.sql}\n\n{res.markdown()}", location=Location())
        hits = [Hit(node=node, score=1.0, rank=1)]
        for path, cell in res.provenance()[:5]:
            nodes = index.catalog.file_nodes(path) if cell is not None else []
            out = next((n for n in nodes if n.location.cell == cell and n.node_type == NodeType.CELL_OUTPUT), None)
            src = out or next((n for n in nodes if n.location.cell == cell), None)
            if src is not None:
                hits.append(Hit(node=src, score=0.5, rank=len(hits) + 1))
        return hits

    # --- uploads (ТЗ ч.2 S16) ------------------------------------------------------
    def upload(self, session: str, name: str, data: bytes) -> UploadInfo:
        known = {}
        if self._index is not None:
            known = {r["content_hash"]: r["path"] for r in self._index.catalog.query("SELECT path, content_hash FROM files")}
        info = self.uploads.add(session, name, data, known)
        self.tracer.log("upload", session=session, name=name if self.tracer.log_prompts else None,
                        file_type=info.file_type, size=info.size, fragments=info.n_fragments,
                        fits_context=info.fits_context, duplicate=bool(info.duplicate_of), parse_s=info.parse_s)
        return info

    def add_upload_to_corpus(self, session: str, upload_id: str, subdir: str = "uploads") -> dict:
        """The explicit "add to corpus" action: the file is copied into the corpus folder (never
        overwriting a file there) and goes through the usual incremental indexing."""
        info = self.uploads.get(session, upload_id)
        index = self.index
        target_dir = (index.root / subdir).resolve()
        if index.root not in target_dir.parents and target_dir != index.root:
            raise UploadError("папка назначения вне корпуса")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / info.name
        k = 1
        while target.exists():
            target = target_dir / f"{Path(info.name).stem} ({k}){Path(info.name).suffix}"
            k += 1
        target.write_bytes(self.uploads.file_path(info).read_bytes())
        progress = self.index_folder(index.root, background=True)
        return {"path": target.relative_to(index.root).as_posix(), "progress": progress.model_dump()}

    # --- code agent (ТЗ ч.2 S17) ------------------------------------------------------
    def _workspace(self, task_id: str) -> Workspace:
        if not re.fullmatch(r"[0-9A-Za-z-]{6,64}", task_id or ""):
            raise WorkspaceError("неверный идентификатор задачи")
        root = self.settings.data_dir / "workspaces" / task_id
        if not (root / ".git").exists():
            raise WorkspaceError(f"задача {task_id} не найдена")
        return Workspace(root)

    def code_task(self, task: str, sandbox: DockerSandbox | None = None) -> CodeResult:
        """Solve a code task in a fresh git working copy of the corpus; the user's folder is untouched."""
        index = self.index
        cfg = self.settings.code
        task_id = new_task_id()
        ws = Workspace.create(self.settings.data_dir / "workspaces" / task_id, index.root,
                              exclude=self.corpus_prefs(index.root)["exclude"], max_mb=cfg.workspace_max_mb,
                              file_max_mb=cfg.file_max_mb)
        agent = CodeAgent(self.settings, self.llm, sandbox or DockerSandbox(cfg), ws, task, task_id,
                          corpus=index.root, catalog_rows=catalog_metric_rows(self))
        res = agent.run()
        self._save_code(ws, res)
        self.tracer.log("code_task", task=task if self.tracer.log_prompts else None, task_id=task_id, status=res.status,
                        pipeline=res.pipeline, steps=len(res.steps), runs=res.runs, failed_runs=res.failed_runs,
                        changed=len(res.changed), broken=len(res.broken_files), latency_s=res.latency_s)
        return res

    @staticmethod
    def _save_code(ws: Workspace, res: CodeResult) -> None:
        (ws.root / ".agent").mkdir(exist_ok=True)
        (ws.root / ".agent" / "result.json").write_text(res.model_dump_json(indent=2), encoding="utf-8")

    def code_result(self, task_id: str) -> CodeResult:
        ws = self._workspace(task_id)
        return CodeResult.model_validate_json((ws.root / ".agent" / "result.json").read_text(encoding="utf-8"))

    def apply_code(self, task_id: str) -> CodeResult:
        """apply_changes (FR14): called only by the user's explicit confirmation in the UI or API."""
        ws, res = self._workspace(task_id), self.code_result(task_id)
        res.applied = ws.apply_to(self.index.root)
        self._save_code(ws, res)
        self.tracer.log("apply_changes", task_id=task_id, files=res.applied)
        if res.applied and not self.progress.running:
            self.index_folder(self.index.root, background=True)
        return res

    def rollback_code(self, task_id: str, commit: str) -> CodeResult:
        ws, res = self._workspace(task_id), self.code_result(task_id)
        ws.rollback(commit)
        res.diff, res.changed, res.checkpoints = ws.diff(), [list(c) for c in ws.changed()], ws.checkpoints()
        self._save_code(ws, res)
        return res

    def code_file(self, task_id: str, rel: str) -> Path:
        p = self._workspace(task_id).path(rel)
        if not p.is_file():
            raise WorkspaceError(f"нет файла {rel}")
        return p

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
        file_types: list[str] | None = None,
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
            file_types=file_types,
        )

    def _answer_web(self, question: str, index: CorpusIndex | None, budget: int | None,
                    confirmed: list[str] | None) -> Answer:
        """Э16: the web, next to relevant fragments of the user's files (for explicit conflicts)."""
        archive = []
        if index is not None:
            hits = self.search(question, 4)
            acfg = self.settings.agent
            if hits and acfg.crag:
                scores = RelevanceEvaluator(self, acfg).scores(question, [h.node for h in hits])
                hits = [h for h, s in zip(hits, scores) if s >= acfg.crag_upper][:3]
            archive = hits[:3]
        policy = Policy.for_catalog(index.catalog if index is not None else None, web=True,
                                    enabled=self.settings.security.policies)
        return answer_from_web(question, self.llm, self.settings, policy=policy, prov=Provenance(), archive=archive,
                               client=self.web_client, fetcher=self.page_fetcher, confirmed=set(confirmed or []),
                               reasoning_budget=budget)

    @property
    def web_client(self):
        """SearXNG client (replaceable: the red-team bench swaps it for a recording fake)."""
        if getattr(self, "_web_client", None) is None:
            from rag_agent.web import SearxClient

            self._web_client = SearxClient(self.settings.web)
        return self._web_client

    @property
    def page_fetcher(self):
        if getattr(self, "_page_fetcher", None) is None:
            from rag_agent.web import PageFetcher

            self._page_fetcher = PageFetcher(self.settings)
        return self._page_fetcher

    def _answer_direct(self, standalone: str, index: CorpusIndex, steps: list[TraceStep], *, top_k: int, mode: str,
                       rerank: bool | None, symbols: bool | None, aggregate: bool, budget: int | None,
                       escalate: bool) -> tuple[Answer, bool]:
        """Direct retrieval (no agent loop): search, the CRAG check, SQL for an aggregate question, the answer."""
        t1 = time.perf_counter()
        hits = self.search(standalone, top_k, mode, rerank=rerank, symbols=symbols)
        cfg = self.settings.retrieval
        steps.append(TraceStep(name="retrieve", duration_s=round(time.perf_counter() - t1, 3), detail={
            "mode": mode, "top_k": top_k, "rerank": cfg.rerank if rerank is None else rerank,
            "symbols": cfg.symbols if symbols is None else symbols,
            "hits": [[h.node.id, round(h.score, 4)] for h in hits]}))
        # Э6: an aggregate question also queries the catalog; the result leads the sources
        sql_rows = False
        if aggregate:
            res = self.query_catalog(standalone)
            steps.append(TraceStep(name="sql", duration_s=res.latency_s, detail={
                "sql": res.sql, "rows": len(res.rows), "attempts": res.attempts, "error": res.error,
                "tables": res.tables, "hints": res.hints}))
            if res.ok and res.rows:
                sql_rows = True
                extra = self._sql_hits(res, index)
                seen = {h.node.id for h in extra}
                merged = extra + [h for h in hits if h.node.id not in seen]
                hits = [Hit(node=h.node, score=h.score, rank=i) for i, h in enumerate(merged, start=1)]
        # CRAG (ТЗ S7): irrelevant fragments -> one rewritten query -> still irrelevant -> honest refusal
        acfg = self.settings.agent
        if acfg.crag and hits and not sql_rows:
            evaluator = RelevanceEvaluator(self, acfg)
            t2 = time.perf_counter()
            best = max(evaluator.scores(standalone, [h.node for h in hits]))
            detail = {"best": round(best, 4), "verdict": evaluator.verdict(best)}
            for _ in range(acfg.crag_retries):
                if detail["verdict"] != "incorrect":
                    break
                query = rewrite_query(self.llm, standalone)
                retry = self.search(query, top_k, mode, rerank=rerank, symbols=symbols)
                retry_best = max(evaluator.scores(standalone, [h.node for h in retry]), default=0.0)
                detail.setdefault("rewrites", []).append({"query": query, "best": round(retry_best, 4)})
                if retry_best > best:
                    hits, best = retry, retry_best
                    detail.update(best=round(best, 4), verdict=evaluator.verdict(best))
            steps.append(TraceStep(name="crag", duration_s=round(time.perf_counter() - t2, 3), detail=detail))
            if detail["verdict"] == "incorrect":
                return Answer(question=standalone, answer="В архиве не нашлось фрагментов, относящихся к вопросу, "
                              "поэтому ответа по файлам нет.", answerable=False, grounded=True,
                              model=self.llm.name), False
        answer = generate_answer(standalone, hits, self.llm, self.settings.generation.max_source_chars,
                                 reasoning_budget=budget)
        # escalation (ТЗ ч.2 S15): the fast answer claims an answer but cites nothing valid
        if escalate and budget is None and not answer.grounded:
            fast_trace = answer.trace
            answer = generate_answer(standalone, hits, self.llm, self.settings.generation.max_source_chars,
                                     reasoning_budget=self.settings.reasoning.budget("deep"))
            answer.trace[:0] = [*fast_trace, TraceStep(name="escalate", duration_s=0.0, detail={"reason": "ungrounded"})]
            return answer, True
        return answer, False

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
        agent: str | None = None,
        uploads: list[str] | None = None,
        session: str | None = None,
        confirmed: list[str] | None = None,
        web: str | None = None,
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
            if route == "corpus" and not uploads:
                raise

        standalone = question
        chosen = route
        rcfg = self.settings.reasoning
        reasoning_mode = reasoning or rcfg.mode
        # reasoning level: "on" reasons deep unless the router estimated a lighter level
        level = "deep" if reasoning_mode == "on" else "none"
        aggregate = False
        complexity = "none"  # the router's difficulty estimate: multi-step questions go to the agent (Э7)
        wants_web = False  # the router: fresh or external information is needed (Э16)
        web_mode = web or self.settings.web.mode
        # the router also rewrites follow-ups into standalone questions and estimates the reasoning level
        if route == "auto" or turns or reasoning_mode == "auto":
            decision = route_question(question, turns, self.llm)
            standalone = decision.standalone_question
            if reasoning_mode == "auto" or (reasoning_mode == "on" and decision.needs_reasoning):
                level = decision.reasoning
            aggregate, complexity, wants_web = decision.aggregate, decision.reasoning, decision.web
            detail = {"route": decision.route, "standalone_question": standalone, "fallback": decision.fallback,
                      "reasoning": decision.reasoning, "aggregate": decision.aggregate, "web": decision.web}
            if route == "auto":
                chosen = decision.route
                # corpus signal: a question that names a function, class or file of the corpus is about the files
                mentions = corpus_mentions(index, standalone) if index is not None and chosen == "general" else []
                if mentions:
                    chosen = "corpus"
                    detail.update(override="corpus", corpus_mentions=mentions)
            steps.append(TraceStep(name="route", duration_s=round(decision.latency_s, 3), detail=detail))

        notice = None
        if chosen == "corpus" and index is None and not uploads:
            chosen, notice = "general", "Папка ещё не проиндексирована, поэтому ответ дан из общих знаний модели."

        budget = rcfg.budget(level) if level != "none" else None
        if uploads:  # Э13: the question is about files uploaded into this conversation
            infos = [self.uploads.get(session or "", u) for u in uploads]
            use_rerank = self.settings.retrieval.rerank if rerank is None else rerank
            answer = answer_from_uploads(self.uploads, infos, standalone, self.llm,
                                         reranker=self.reranker if use_rerank else None,
                                         max_source_chars=self.settings.generation.max_source_chars,
                                         reasoning_budget=budget)
            answer.question = question
            dups = [f"{i.name} = {i.duplicate_of}" for i in infos if i.duplicate_of]
            if dups:
                notice = "Загруженный файл уже есть в корпусе: " + "; ".join(dups)
        elif web_mode == "always" or (web_mode == "auto" and wants_web):
            answer = self._answer_web(standalone, index, budget, confirmed)
            answer.question = question
        elif chosen == "general":
            answer = generate_general(question, turns, self.llm, reasoning_budget=budget)
        else:
            top_k = top_k or self.settings.retrieval.top_k
            mode = mode or self.settings.retrieval.mode
            agent_mode = agent or self.settings.agent.mode
            sql_ok = self.settings.catalog.sql and (index.dir / ANALYTICS_FILE).exists()
            # ТЗ S7: only aggregate and multi-step questions go through the agent loop
            if agent_mode == "always" or (agent_mode == "auto" and (aggregate or complexity != "none")):
                agent = Agent(self, index, top_k=top_k, mode=mode, sql=sql_ok, reasoning_budget=budget,
                              confirmed=set(confirmed or []), web=web_mode != "off")
                answer = agent.run(standalone, aggregate=aggregate)
            else:
                answer, escalated = self._answer_direct(standalone, index, steps, top_k=top_k, mode=mode, rerank=rerank,
                                                        symbols=symbols, aggregate=aggregate and sql_ok, budget=budget,
                                                        escalate=reasoning_mode == "auto" and rcfg.escalate)
                if escalated:
                    level = "deep"
            # S19: irrelevant corpus results send the question to the web in auto mode
            if web_mode == "auto" and not answer.answerable and not answer.pending and any(
                    s.name == "crag" and s.detail.get("verdict") == "incorrect" for s in answer.trace):
                first = answer.trace
                answer = self._answer_web(standalone, index, budget, confirmed)
                answer.trace[:0] = [*first, TraceStep(name="web_fallback", duration_s=0.0,
                                                      detail={"reason": "irrelevant corpus results"})]
            answer.question = question

        answer.standalone_question = standalone if standalone != question else None
        # rule 4 (ТЗ ч.2 S20): rendering the answer must not load images or follow links by itself
        if self.settings.security.policies:
            answer.answer, answer.general = defang_markdown(answer.answer), defang_markdown(answer.general)
        answer.reasoning_level = level
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

    def catalog_summary(self, idx: CorpusIndex | None = None) -> dict:
        """Experiments extracted into the catalog: counts and per-notebook extraction state."""
        idx = idx or self.index
        states = idx.catalog.query("SELECT file_path, model, n_experiments, n_values, n_dropped, error FROM x_state "
                                   "ORDER BY file_path")
        return {
            "analytics": (idx.dir / ANALYTICS_FILE).exists(),
            "notebooks": len(states),
            "experiments": sum(r["n_experiments"] or 0 for r in states),
            "values": sum(r["n_values"] or 0 for r in states),
            "dropped": sum(r["n_dropped"] or 0 for r in states),
            "files": [dict(r) for r in states],
        }

    def experiments(self) -> list[dict]:
        """The extracted experiments with their metric values (for the UI and ``rag experiments``)."""
        cat = self.index.catalog
        out = []
        for e in cat.query("SELECT * FROM x_experiments ORDER BY file_path, idx"):
            metrics = cat.query("SELECT name, value, split, variant, cell FROM x_metrics WHERE file_path=? AND exp_idx=?",
                                (e["file_path"], e["idx"]))
            hparams = cat.query("SELECT name, value, variant, cell FROM x_hparams WHERE file_path=? AND exp_idx=?",
                                (e["file_path"], e["idx"]))
            out.append({**dict(e), "metrics": [dict(m) for m in metrics], "hyperparameters": [dict(h) for h in hparams]})
        return out

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
                "catalog": self.catalog_summary(idx),
            },
            "progress": self.progress.model_dump(),
            "embedder": {"model": self.embedder.name, "device": self.embedder.device},
            "llm": {"name": self.llm.name, "available": getattr(self.llm, "is_available", lambda: True)()},
            "retrieval": self.settings.retrieval.model_dump(),
            "reasoning": self.settings.reasoning.model_dump(),
            "agent": self.settings.agent.model_dump(),
            "web": {"mode": self.settings.web.mode, "searxng_url": self.settings.web.searxng_url},
            "defaults": {"include_ext": self.settings.corpus.include_ext, "exclude": self.settings.corpus.exclude},
        }

    def close(self) -> None:
        with self._lock:
            if self._index is not None:
                self._index.close()
                self._index = None
        self.llm.close()
