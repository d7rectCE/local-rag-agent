"""HTTP API (FastAPI). Run with ``rag serve``; binds to localhost by default."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from rag_agent.engine import Engine, IndexingBusyError, NoCorpusError
from rag_agent.generation import Answer
from rag_agent.index.indexer import IndexProgress
from rag_agent.llm import LLMError
from rag_agent.structured.sql import SQLResult
from rag_agent.uploads import UploadError, UploadInfo


class IndexRequest(BaseModel):
    root: str
    include_ext: list[str] | None = None
    exclude: list[str] | None = None
    force: bool = False


class OpenRequest(BaseModel):
    root: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SQLRequest(BaseModel):
    question: str


class AskRequest(BaseModel):
    question: str
    history: list[ChatTurn] = []
    top_k: int | None = None
    mode: Literal["dense", "sparse", "hybrid"] | None = None
    route: Literal["auto", "corpus", "general"] = "auto"
    rerank: bool | None = None
    symbols: bool | None = None
    reasoning: Literal["off", "on", "auto"] | None = None
    agent: Literal["off", "auto", "always"] | None = None
    uploads: list[str] = []  # ids of files uploaded into this session: the question is about them
    confirmed: list[str] = []  # keys of agent actions the user approved (FR17)
    session: str | None = None


def create_app(engine: Engine | None = None) -> FastAPI:
    state: dict[str, Engine] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["engine"] = engine or Engine()
        yield
        state["engine"].close()

    app = FastAPI(title="local-rag-agent", version="0.1.0", lifespan=lifespan)

    def eng() -> Engine:
        return state["engine"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict:
        return eng().status()

    @app.get("/corpora")
    def corpora() -> list[dict]:
        return eng().registry.all()

    @app.get("/corpus/prefs")
    def prefs(root: str) -> dict:
        return eng().corpus_prefs(root)

    @app.post("/corpus/open")
    def open_corpus(req: OpenRequest) -> dict:
        try:
            idx = eng().open_corpus(req.root)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except IndexingBusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"root": str(idx.root), "stats": idx.catalog.stats()}

    @app.post("/index", status_code=202)
    def start_index(req: IndexRequest) -> IndexProgress:
        try:
            return eng().index_folder(
                req.root, include_ext=req.include_ext, exclude=req.exclude, force=req.force, background=True
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except IndexingBusyError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/index/progress")
    def progress() -> IndexProgress:
        return eng().progress

    @app.post("/index/cancel")
    def cancel() -> dict:
        eng().cancel_indexing()
        return {"cancelled": True}

    @app.get("/index/problems")
    def problems() -> list[dict]:
        try:
            return eng().index.catalog.problems()
        except NoCorpusError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/file")
    def file_view(path: str) -> list[dict]:
        try:
            return eng().file_view(path)
        except NoCorpusError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/symbols")
    def symbols(name: str) -> dict:
        try:
            return eng().lookup_symbol(name)
        except NoCorpusError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/catalog")
    def catalog() -> dict:
        try:
            return {**eng().catalog_summary(), "experiments": eng().experiments()}
        except NoCorpusError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/sql")
    def sql(req: SQLRequest) -> SQLResult:
        try:
            return eng().query_catalog(req.question)
        except NoCorpusError as exc:
            raise HTTPException(404, "Сначала проиндексируйте папку") from exc

    @app.post("/uploads")
    def upload(session: str, file: UploadFile) -> UploadInfo:
        try:
            return eng().upload(session, file.filename or "file", file.file.read())
        except UploadError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/uploads")
    def uploads(session: str) -> list[UploadInfo]:
        try:
            return eng().uploads.list(session)
        except UploadError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/uploads/{upload_id}")
    def delete_upload(upload_id: str, session: str) -> dict:
        try:
            eng().uploads.delete(session, upload_id)
        except UploadError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"deleted": upload_id}

    @app.post("/uploads/{upload_id}/corpus")
    def upload_to_corpus(upload_id: str, session: str) -> dict:
        """Explicit user action: copy the upload into the corpus folder and index it."""
        try:
            return eng().add_upload_to_corpus(session, upload_id)
        except UploadError as exc:
            raise HTTPException(400, str(exc)) from exc
        except NoCorpusError as exc:
            raise HTTPException(404, "Сначала выберите и проиндексируйте папку") from exc
        except IndexingBusyError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/ask")
    def ask(req: AskRequest) -> Answer:
        try:
            return eng().ask(
                req.question,
                history=[t.model_dump() for t in req.history],
                top_k=req.top_k,
                mode=req.mode,
                route=req.route,
                rerank=req.rerank,
                symbols=req.symbols,
                reasoning=req.reasoning,
                agent=req.agent,
                uploads=req.uploads,
                session=req.session,
                confirmed=req.confirmed,
            )
        except NoCorpusError as exc:
            raise HTTPException(404, "Сначала проиндексируйте папку") from exc
        except (ValueError, UploadError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except LLMError as exc:
            logging.getLogger(__name__).exception("LLM failure")
            raise HTTPException(502, f"LLM недоступна: {exc}") from exc

    return app
