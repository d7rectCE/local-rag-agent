"""Streamlit UI: choose a working folder, index it, ask questions (thin client of the API).

Run with ``rag ui`` (the API must be running: ``rag serve``).
"""

from __future__ import annotations

import os
import subprocess
import sys

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8765")
TYPE_LABELS = {
    ".py": "Python (.py)",
    ".ipynb": "Ноутбуки (.ipynb)",
    ".pdf": "PDF (этап 4)",
    ".png": "Изображения (этап 5)",
}
SUPPORTED_TYPES = [".py", ".ipynb"]
ROUTE_LABELS = {"auto": "Авто", "corpus": "Мои файлы", "general": "Общие знания"}
STATE_LABELS = {
    "idle": "ожидание",
    "scanning": "сканирование папки",
    "indexing": "индексация",
    "done": "готово",
    "error": "ошибка",
    "cancelled": "остановлено",
}
_PICKER = """
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
path = filedialog.askdirectory(initialdir=sys.argv[1] or None, title="Выберите рабочую папку")
sys.stdout.write(path or "")
"""


@st.cache_resource
def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=900)


def api(method: str, path: str, **kwargs):
    """Returns (data, error_message)."""
    try:
        r = _client().request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        return None, f"API недоступен ({API_URL}): {exc}"
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"{r.status_code}: {detail}"
    return r.json(), None


def pick_folder() -> None:
    """Native folder dialog in a subprocess (tkinter must not run in Streamlit's thread)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        res = subprocess.run(
            [sys.executable, "-c", _PICKER, st.session_state.get("root", "")],
            capture_output=True, text=True, encoding="utf-8", env=env, timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if res.stdout.strip():
        st.session_state.root = os.path.normpath(res.stdout.strip())


def open_corpus(root: str) -> None:
    _, err = api("POST", "/corpus/open", json={"root": root})
    if err:
        st.session_state.open_error = err
    else:
        st.session_state.root = root
        st.session_state.messages = []


def fmt_seconds(s: float) -> str:
    return f"{s:.0f} с" if s < 120 else f"{s / 60:.1f} мин"


def progress_panel() -> None:
    prog, err = api("GET", "/index/progress")
    if err or prog is None:
        return
    state = prog["state"]
    if state == "idle":
        return
    running = state in ("scanning", "indexing")
    total, done = prog["files_total"], prog["files_done"]
    label = STATE_LABELS.get(state, state)
    if running:
        st.progress(done / total if total else 0.0, text=f"{label}: {done}/{total}")
        if prog.get("current"):
            st.caption(f"`{prog['current']}`")
        if st.button("Остановить", icon=":material/stop:", use_container_width=True):
            api("POST", "/index/cancel")
    elif state == "done":
        st.success(f"Готово за {fmt_seconds(prog['elapsed_s'])}: {total} файлов, {prog['nodes_embedded']} новых фрагментов")
    elif state == "error":
        st.error(f"Ошибка индексации: {prog.get('message')}")
    else:
        st.warning(f"Индексация {label}")
    st.caption(
        f"новых {prog['new']} · изменённых {prog['changed']} · без изменений {prog['unchanged']}"
        f" · удалённых {prog['deleted']} · ошибок {prog['errors']}"
    )
    if not running and st.session_state.get("was_running"):
        st.session_state.was_running = False
        st.rerun()  # refresh stats in the rest of the page
    st.session_state.was_running = running


def sidebar(status: dict) -> None:
    corpus = status.get("corpus")
    running = status["progress"]["state"] in ("scanning", "indexing")
    if "root" not in st.session_state:
        st.session_state.root = corpus["root"] if corpus else ""

    st.header("Рабочая папка")
    st.text_input("Путь", key="root", placeholder=r"C:\Users\...\WorkSpace")
    st.button("Выбрать папку…", icon=":material/folder_open:", on_click=pick_folder, use_container_width=True)
    root = st.session_state.root.strip()

    prefs = status["defaults"]
    if root:
        data, _ = api("GET", "/corpus/prefs", params={"root": root})
        prefs = data or prefs
    with st.expander("Что индексировать"):
        types = st.multiselect(
            "Типы файлов",
            options=SUPPORTED_TYPES,
            default=[t for t in prefs["include_ext"] if t in SUPPORTED_TYPES],
            format_func=TYPE_LABELS.get,
        )
        st.caption("PDF и изображения подключатся на этапах 4–5.")
        exclude = st.text_area(
            "Исключения",
            value="\n".join(prefs["exclude"]),
            height=200,
            help="По одному шаблону в строке. Шаблон без «/» исключает любую папку или файл с таким именем "
            "(например, «Аккаунты», «*.egg-info»); шаблон с «/» сопоставляется с путём (например, «data/raw/**»). "
            "Скрытые папки (начинающиеся с точки) пропускаются всегда.",
        )
        force = st.checkbox("Переиндексировать всё с нуля")
    if st.button("Индексировать", type="primary", icon=":material/play_arrow:", disabled=running or not root or not types,
                 use_container_width=True):
        payload = {
            "root": root,
            "include_ext": types,
            "exclude": [line.strip() for line in exclude.splitlines() if line.strip()],
            "force": force,
        }
        _, err = api("POST", "/index", json=payload)
        if err:
            st.error(err)
        else:
            st.session_state.was_running = True
            st.rerun()

    st.fragment(progress_panel, run_every=1.0 if running else None)()

    known = [c["root"] for c in status.get("corpora", [])]
    if len(known) > 1:
        st.divider()
        st.subheader("Ранее проиндексированные")
        choice = st.selectbox("Папка", known, index=known.index(corpus["root"]) if corpus and corpus["root"] in known else 0,
                              label_visibility="collapsed")
        st.button("Открыть", disabled=running or not choice, on_click=open_corpus, args=(choice,), use_container_width=True)
        if st.session_state.get("open_error"):
            st.error(st.session_state.pop("open_error"))

    st.divider()
    st.subheader("Ответы")
    st.segmented_control(
        "Источник ответа", list(ROUTE_LABELS), default="auto", format_func=ROUTE_LABELS.get, key="route",
        help="«Авто» сам решает, нужен ли поиск по файлам. «Мои файлы» всегда отвечает по файлам со ссылками. "
        "«Общие знания» отвечает без поиска, как обычный ассистент.",
    )
    st.slider("Фрагментов в контексте", 2, 12, status["retrieval"]["top_k"], key="top_k")
    st.selectbox("Режим", ["dense", "sparse", "hybrid"], index=["dense", "sparse", "hybrid"].index(status["retrieval"]["mode"]),
                 key="mode", help="sparse и hybrid работают только с BGE-M3")


def answer_text(ans: dict) -> str:
    """What the assistant said, as plain text for the dialogue history."""
    return ans["answer"] + (f"\n\n{ans['general']}" if ans.get("general") else "")


def render_answer(ans: dict) -> None:
    if ans.get("notice"):
        st.caption(f":material/info: {ans['notice']}")
    if ans.get("standalone_question"):
        st.caption(f":material/edit_note: Понял вопрос как: «{ans['standalone_question']}»")
    st.markdown(ans["answer"])
    if ans.get("route") == "general":
        st.caption(":material/school: Ответ из общих знаний модели, файлы не использовались.")
        return
    if not ans["answerable"]:
        st.caption(":material/help: В найденных фрагментах ответа нет.")
    elif not ans["grounded"]:
        st.caption(":material/warning: Ответ без ссылок на источники, проверьте его вручную.")
    if ans.get("general"):
        with st.container(border=True):
            st.caption(":material/school: Из общих знаний модели (не из ваших файлов)")
            st.markdown(ans["general"])
    if ans["citations"]:
        st.markdown("  \n".join(f"**[{c['n']}]** `{c['file_path']}` — {c['location']}" for c in ans["citations"]))
    with st.expander(f"Найденные фрагменты ({len(ans['sources'])})"):
        for s in ans["sources"]:
            mark = " · процитирован" if s["cited"] else ""
            st.markdown(f"**[{s['n']}]** `{s['file_path']}` — {s['location'] or s['title']} · score {s['score']:.3f}{mark}")
            lang = "python" if s["node_type"] in ("code_chunk", "code_cell", "function", "class") else "text"
            st.code(s["text"], language=lang)
    with st.expander("Трасса"):
        for step in ans["trace"]:
            st.markdown(f"**{step['name']}** — {step['duration_s']:.2f} с")
            st.json(step["detail"], expanded=False)
        st.caption(f"Всего {ans['latency_s']:.2f} с · {ans.get('model') or ''}")


def main() -> None:
    st.set_page_config(page_title="Local RAG Agent", page_icon=":material/manage_search:", layout="wide")
    status, err = api("GET", "/status")
    if err:
        st.error(err)
        st.markdown("Запустите API в отдельном терминале:")
        st.code("rag serve", language="bash")
        st.stop()
    corpora, _ = api("GET", "/corpora")
    status["corpora"] = corpora or []

    with st.sidebar:
        sidebar(status)

    corpus = status.get("corpus")
    title_col, clear_col = st.columns([5, 1], vertical_alignment="bottom")
    title_col.title("Local RAG Agent")
    if clear_col.button("Новый диалог", icon=":material/add_comment:", use_container_width=True):
        st.session_state.messages = []
    if corpus:
        stats = corpus["stats"]
        st.caption(
            f"`{corpus['root']}` · файлов {stats['n_files']} · фрагментов {corpus['vectors']}"
            f" · эмбеддер {status['embedder']['model']} · LLM {status['llm']['name']}"
        )
    if not status["llm"]["available"]:
        st.warning(f"LLM {status['llm']['name']} недоступна. Запущена ли Ollama и скачана ли модель?")

    if st.session_state.get("corpus_root") != (corpus or {}).get("root"):
        st.session_state.corpus_root = (corpus or {}).get("root")
        st.session_state.messages = []
    messages = st.session_state.setdefault("messages", [])

    if not corpus or not corpus["stats"]["n_files"]:
        st.info("Выберите папку слева и нажмите «Индексировать», чтобы задавать вопросы по своим файлам. "
                "Общие вопросы можно задавать и без индексации.")
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer(msg["answer"])
            else:
                st.markdown(msg["content"])

    question = st.chat_input("Спросите про свои файлы или задайте общий вопрос…")
    if question:
        history = [
            {"role": m["role"], "content": m["content"] if m["role"] == "user" else answer_text(m["answer"])}
            for m in messages
        ]
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Думаю…"):
                ans, err = api("POST", "/ask", json={
                    "question": question,
                    "history": history,
                    "top_k": st.session_state.top_k,
                    "mode": st.session_state.mode,
                    "route": st.session_state.get("route") or "auto",
                })
            if err:
                st.error(err)
                messages.pop()
            else:
                render_answer(ans)
                messages.append({"role": "assistant", "answer": ans})


main()
