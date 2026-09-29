"""HTTP API (FastAPI) and the web UI. Run with ``rag serve`` (``rag ui`` also opens the browser);
binds to localhost by default. The UI is a static page at ``/`` (``src/rag_agent/webui``) that talks
to this API; its dialogs are stored server-side (``/dialogs``)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_agent.code.agent import CodeResult
from rag_agent.code.sandbox import SandboxError
from rag_agent.code.workspace import WorkspaceError
from rag_agent.dialogs import DialogNotFound, DialogStore
from rag_agent.engine import Engine, IndexingBusyError, NoCorpusError
from rag_agent.generation import Answer
from rag_agent.hardware import gpu_name
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


class CodeRequest(BaseModel):
    task: str
    dialog_id: str | None = None
    model: str | None = None
    network: bool = False  # an explicit request for a run with network (the chat asks for confirmation first)


class DialogRequest(BaseModel):
    title: str = ""


class ModelSettingsRequest(BaseModel):
    values: dict = {}


class PickFolderRequest(BaseModel):
    initial: str = ""


class RollbackRequest(BaseModel):
    commit: str


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
    web: Literal["off", "auto", "always"] | None = None
    session: str | None = None
    # web UI: the history comes from the stored dialog and the turns are saved there; replace_last
    # regenerates the last answer (after a confirmation) instead of adding the question again
    dialog_id: str | None = None
    replace_last: bool = False
    code: Literal["off", "auto"] | None = None  # the chat may run code (code agent, run_code); config default
    sandbox_net: bool = False  # the UI toggle: code tasks may run with network, each after a confirmation
    model: str | None = None  # another installed chat model for this request
    upload_fallback: bool = False  # files stay attached in the chat: when they do not answer, route as usual
    # a picture action from the UI: {mode: generate|edit|redraw|inpaint|outpaint, source: "generated:<id>" |
    # "upload:<id>", mask: PNG as base64 (white = redraw), sides: [left, right, top, bottom], strength, seed}
    image: dict | None = None
    video: dict | None = None  # {"mode": "animate", "source": ..., "model": ...}


WEBUI = Path(__file__).with_name("webui")
# rule 4 (ТЗ ч.2 S20) for the page itself: nothing is loaded from outside the machine, whatever an
# answer contains; inline styles are allowed for the few computed widths (trace bars)
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
       "media-src 'self'; "
       "font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
       "form-action 'self'")
_PICKER = """
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
path = filedialog.askdirectory(initialdir=sys.argv[1] or None, title="Выберите рабочую папку")
sys.stdout.write(path or "")
"""


def create_app(engine: Engine | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["engine"] = engine or Engine()
        state["dialogs"] = DialogStore(state["engine"].settings.data_dir / "dialogs.sqlite")
        yield
        state["dialogs"].close()
        state["engine"].close()

    app = FastAPI(title="local-rag-agent", version="0.1.0", lifespan=lifespan)

    def eng() -> Engine:
        return state["engine"]

    def dialogs() -> DialogStore:
        return state["dialogs"]

    @app.middleware("http")
    async def csp(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/ui/"):
            # revalidate every time (ETag): after an update the browser must not keep an old app.js
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Content-Security-Policy"] = CSP
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
        return response

    if WEBUI.is_dir():
        app.mount("/ui", StaticFiles(directory=WEBUI), name="ui")

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        # the page's own script and styles carry their modification time: an update is a new URL, so no
        # browser keeps an old app.js (vendored libraries change only with their version)
        html = (WEBUI / "index.html").read_text(encoding="utf-8")
        for name in ("app.js", "app.css", "theme.js"):
            stamp = int((WEBUI / name).stat().st_mtime)
            html = html.replace(f'"/ui/{name}"', f'"/ui/{name}?v={stamp}"')
        return HTMLResponse(html)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict:
        return {**eng().status(), "gpu": gpu_name()}  # this machine's card, detected (not configured)

    @app.get("/policy")
    def policy() -> dict:
        """What the agent may do, for the "access" panel of the UI (ТЗ ч.2 S20, FR17)."""
        e = eng()
        corpus = e.status()["corpus"]
        return {
            "corpus": {"name": Path(corpus["root"]).name if corpus else None, "access": "read"},
            "writes": "confirm",  # apply_changes only after the user's approval
            "sandbox": "no_network" if e.sandbox_available() else "unavailable",
            "sandbox_network": e.settings.code.network,  # never | confirm (the toggle may be offered)
            "imagegen": e.imagegen.available,  # pictures can be drawn and edited
            "code": e.settings.code.chat,
            "web": e.settings.web.mode,
            "policies": e.settings.security.policies,
        }

    @app.get("/videos/{dialog_id}/{name}")
    def generated_video(dialog_id: str, name: str) -> FileResponse:
        """A generated video of a dialog (``<id>.webm``)."""
        from rag_agent.videogen import VideoGenError

        try:
            p = eng().videogen.path(dialog_id, name.removesuffix(".webm"))
        except VideoGenError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not p.exists():
            raise HTTPException(404, "видео не найдено")
        return FileResponse(p, media_type="video/webm")

    @app.get("/videogen")
    def videogen_info() -> dict:
        """The video generator for the UI: its models and the device it computes on."""
        v = eng().videogen
        if not v.available:
            return {"available": False, "missing": v.missing(), "models": v.models()}
        return {"available": True, "models": v.models(), "device": eng().imagegen.device()}

    @app.get("/imagegen")
    def imagegen_info() -> dict:
        """The picture generator for the UI: its models, the device it computes on, the sizes of the model
        card with the expected time of each on this machine (learned from the pictures made here)."""
        from rag_agent.imagegen import MAX_AREA, MAX_SIDE, OFFICIAL_SIZES

        g = eng().imagegen
        if not g.available:
            return {"available": False, "missing": g.missing()}
        return {
            "available": True,
            "models": g.models(),
            "device": g.device(),
            "sizes": [{"id": ratio, "width": w, "height": h} for ratio, (w, h) in OFFICIAL_SIZES.items()],
            "max_side": MAX_SIDE,
            "max_area": MAX_AREA,
        }

    def model_settings(kind: str, name: str) -> dict:
        """What the settings panel shows for a model: the recommended values, the ones set in the UI, the
        choices of the fields and what the model is."""
        from rag_agent.imagegen import SAMPLERS, SCHEDULERS

        e = eng()
        if kind == "text":
            name = name or e.settings.llm.model
            c = e.settings.llm
            return {"kind": kind, "name": name, "values": e.model_settings.get("text", name),
                    "recommended": {"temperature": c.temperature, "top_p": None, "top_k": None, "repeat_penalty": None,
                                    "num_ctx": c.num_ctx, "max_tokens": c.max_tokens},
                    "choices": {"num_ctx": [8192, 16384, 32768, 65536, 131072]}, "info": {"label": name}}
        if kind == "image":
            g = e.imagegen
            name = name or g.default_name()
            entry = g.catalog().get(name)
            if entry is None:
                raise HTTPException(404, f"модель картинок «{name}» не найдена")
            from rag_agent import modelfiles

            return {"kind": kind, "name": name, "values": g.store.get("image", name),
                    "recommended": g.recommended(name), "choices": {"sampler": SAMPLERS, "scheduler": SCHEDULERS},
                    "info": {"label": modelfiles.label(entry["path"]), "vae": Path(entry["vae"]).name if entry["vae"] else "",
                             "llm": Path(entry["llm"]).name if entry["llm"] else "", "source": entry["source"]}}
        if kind == "video":
            v = e.videogen
            name = name or v.default_name()
            entry = v.catalog().get(name)
            if entry is None:
                raise HTTPException(404, f"видеомодель «{name}» не найдена")
            from rag_agent import modelfiles

            files = {k: Path(entry[k]).name for k in ("vae", "llm", "audio_vae", "connectors") if entry.get(k)}
            return {"kind": kind, "name": name, "values": v.store.get("video", name),
                    "recommended": v.recommended(name), "choices": {"sampler": SAMPLERS, "scheduler": SCHEDULERS},
                    "info": {"label": modelfiles.label(entry["path"]), **files, "source": entry["source"]}}
        raise HTTPException(404, f"настройки для «{kind}» пока не поддерживаются")

    @app.get("/settings/{kind}")
    def get_model_settings(kind: str, name: str = "") -> dict:
        return model_settings(kind, name)

    @app.put("/settings/{kind}")
    def put_model_settings(kind: str, req: ModelSettingsRequest, name: str = "") -> dict:
        from rag_agent.model_settings import ModelSettingsError

        info = model_settings(kind, name)
        try:
            eng().model_settings.set(kind, info["name"], req.values)
        except ModelSettingsError as exc:
            raise HTTPException(400, str(exc)) from exc
        return model_settings(kind, info["name"])

    @app.delete("/settings/{kind}")
    def reset_model_settings(kind: str, name: str = "") -> dict:
        info = model_settings(kind, name)
        eng().model_settings.reset(kind, info["name"])
        return model_settings(kind, info["name"])

    @app.get("/models")
    def models() -> dict:
        """Installed chat models for the UI picker; ``default`` is the configured one (the eval runs use it)."""
        e = eng()
        return {"default": e.settings.llm.model, "models": e.llm.installed()}

    @app.post("/pick-folder")
    def pick_folder(req: PickFolderRequest) -> dict:
        """The native folder dialog of this machine (the UI is served to localhost only)."""
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            res = subprocess.run([sys.executable, "-c", _PICKER, req.initial], capture_output=True, text=True,
                                 encoding="utf-8", env=env, timeout=900)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(500, f"диалог выбора папки недоступен: {exc}") from exc
        path = res.stdout.strip()
        return {"path": os.path.normpath(path) if path else None}

    # --- dialogs of the web UI ---------------------------------------------------------
    @app.get("/dialogs")
    def list_dialogs() -> list[dict]:
        return dialogs().list()

    @app.post("/dialogs")
    def create_dialog(req: DialogRequest) -> dict:
        try:
            root = str(eng().index.root)  # the index only: status() would also load the embedder
        except NoCorpusError:
            root = None
        return dialogs().create(req.title, root)

    @app.get("/dialogs/{dialog_id}")
    def get_dialog(dialog_id: str) -> dict:
        try:
            return dialogs().get(dialog_id)
        except DialogNotFound as exc:
            raise HTTPException(404, "диалог не найден") from exc

    @app.patch("/dialogs/{dialog_id}")
    def rename_dialog(dialog_id: str, req: DialogRequest) -> dict:
        try:
            return dialogs().rename(dialog_id, req.title)
        except DialogNotFound as exc:
            raise HTTPException(404, "диалог не найден") from exc

    @app.delete("/dialogs/{dialog_id}")
    def delete_dialog(dialog_id: str) -> dict:
        try:
            dialogs().delete(dialog_id)
        except DialogNotFound as exc:
            raise HTTPException(404, "диалог не найден") from exc
        for info in eng().uploads.list(dialog_id):  # the dialog is the upload session
            eng().uploads.delete(dialog_id, info.id)
        eng().imagegen.delete_dialog(dialog_id)
        eng().videogen.delete_dialog(dialog_id)
        return {"deleted": dialog_id}

    @app.get("/images/{dialog_id}/{name}")
    def generated_image(dialog_id: str, name: str) -> FileResponse:
        """A generated picture of a dialog (``<id>.png``)."""
        from rag_agent.imagegen import ImageGenError

        try:
            path = eng().imagegen.path(dialog_id, name.removesuffix(".png"))
        except ImageGenError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not path.exists():
            raise HTTPException(404, "картинка не найдена")
        return FileResponse(path, media_type="image/png")

    @app.get("/uploads/{upload_id}/file")
    def upload_file(upload_id: str, session: str) -> FileResponse:
        """The uploaded file itself (the UI shows uploaded pictures)."""
        try:
            info = eng().uploads.get(session, upload_id)
        except UploadError as exc:
            raise HTTPException(404, str(exc)) from exc
        types = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}
        return FileResponse(eng().uploads.file_path(info), media_type=types.get(info.file_type, "application/octet-stream"))

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

    @app.post("/code")
    def code(req: CodeRequest) -> CodeResult:
        try:
            context = ""
            if req.dialog_id:
                dialogs().get(req.dialog_id)  # 404 before the long run
                context = "\n".join(f"{'Пользователь' if t['role'] == 'user' else 'Ассистент'}: {t['content']}"
                                     for t in dialogs().history(req.dialog_id, 6))[-4000:]
            res = eng().code_task(req.task, context=context, network=req.network, model=req.model)
            if req.dialog_id:
                dialogs().add_turn(req.dialog_id, "user", "code_task", req.task)
                dialogs().add_turn(req.dialog_id, "assistant", "code", res.summary or res.status,
                                   payload=res.model_dump(), ref=res.task_id)
            return res
        except DialogNotFound as exc:
            raise HTTPException(404, "диалог не найден") from exc
        except NoCorpusError as exc:
            raise HTTPException(404, "Сначала выберите и проиндексируйте папку") from exc
        except SandboxError as exc:
            raise HTTPException(503, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(502, f"LLM недоступна: {exc}") from exc

    @app.get("/code/{task_id}")
    def code_result(task_id: str) -> CodeResult:
        try:
            return eng().code_result(task_id)
        except WorkspaceError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/code/{task_id}/apply")
    def code_apply(task_id: str) -> CodeResult:
        """apply_changes: the explicit confirmation of the user (FR14)."""
        try:
            res = eng().apply_code(task_id)
        except WorkspaceError as exc:
            raise HTTPException(400, str(exc)) from exc
        dialogs().update_ref(task_id, res.model_dump())
        return res

    @app.post("/code/{task_id}/reject")
    def code_reject(task_id: str) -> dict:
        """The user declined the changes: nothing reaches the folder; the dialog remembers the decision."""
        try:
            res = eng().code_result(task_id)
        except WorkspaceError as exc:
            raise HTTPException(404, str(exc)) from exc
        payload = {**res.model_dump(), "rejected": True}
        dialogs().update_ref(task_id, payload)
        return payload

    @app.post("/code/{task_id}/rollback")
    def code_rollback(task_id: str, req: RollbackRequest) -> CodeResult:
        try:
            res = eng().rollback_code(task_id, req.commit)
        except WorkspaceError as exc:
            raise HTTPException(400, str(exc)) from exc
        dialogs().update_ref(task_id, res.model_dump())
        return res

    @app.get("/code/{task_id}/file")
    def code_file(task_id: str, path: str) -> FileResponse:
        try:
            return FileResponse(eng().code_file(task_id, path))
        except WorkspaceError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/ask")
    def ask(req: AskRequest) -> Answer:
        history = [t.model_dump() for t in req.history]
        try:
            if req.dialog_id:
                if req.replace_last:
                    dialogs().drop_last_answer(req.dialog_id)
                stored = dialogs().history(req.dialog_id)
                if req.replace_last and stored and stored[-1]["role"] == "user":
                    stored = stored[:-1]  # the question being answered again is not its own history
                history = history or stored
            answer = eng().ask(
                req.question,
                history=history,
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
                web=req.web,
                code=req.code,
                sandbox_net=req.sandbox_net,
                model=req.model,
                upload_fallback=req.upload_fallback,
                image=req.image,
                video=req.video,
            )
            if req.dialog_id:
                if not req.replace_last:
                    dialogs().add_turn(req.dialog_id, "user", "question", req.question,
                                       payload={"uploads": req.uploads, "route": req.route, "web": req.web,
                                                "reasoning": req.reasoning, "code": req.code,
                                                "sandbox_net": req.sandbox_net, "model": req.model,
                                                "image": {k: v for k, v in (req.image or {}).items() if k != "mask"}
                                                or None, "video": req.video})
                dialogs().add_turn(req.dialog_id, "assistant", "answer", answer.answer, payload=answer.model_dump(),
                                   ref=answer.code["task_id"] if answer.code else None)
            return answer
        except DialogNotFound as exc:
            raise HTTPException(404, "диалог не найден") from exc
        except NoCorpusError as exc:
            raise HTTPException(404, "Сначала проиндексируйте папку") from exc
        except (ValueError, UploadError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except LLMError as exc:
            logging.getLogger(__name__).exception("LLM failure")
            raise HTTPException(502, f"LLM недоступна: {exc}") from exc

    return app

