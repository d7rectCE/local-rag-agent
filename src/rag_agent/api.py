"""HTTP API (FastAPI). Run with ``rag serve``; binds to localhost by default."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_agent.engine import Engine, IndexingBusyError, NoCorpusError
from rag_agent.generation import Answer
from rag_agent.index.indexer import IndexProgress
from rag_agent.llm import LLMError


class IndexRequest(BaseModel):
    root: str
    include_ext: list[str] | None = None
    exclude: list[str] | None = None
    force: bool = False


class OpenRequest(BaseModel):
    root: str


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    mode: Literal["dense", "sparse", "hybrid"] | None = None


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

    @app.post("/ask")
    def ask(req: AskRequest) -> Answer:
        try:
            return eng().ask(req.question, top_k=req.top_k, mode=req.mode)
        except NoCorpusError as exc:
            raise HTTPException(404, "Сначала проиндексируйте папку") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except LLMError as exc:
            logging.getLogger(__name__).exception("LLM failure")
            raise HTTPException(502, f"LLM недоступна: {exc}") from exc

    return app
