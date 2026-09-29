"""Dialogs of the web UI: the list in the sidebar and the turns of each conversation.

Stored in ``data_dir/dialogs.sqlite``. A turn keeps its text (the history for follow-up
questions is the plain text of the turns; a code task is summarised with its status, files
and whether the user applied it, so "what did you do?" has an answer) and a JSON payload with
everything the UI shows: the whole answer with sources and trace, or the result of a code
task. A reopened dialog is rendered exactly as it was answered; nothing is recomputed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

TITLE_CHARS = 60
KINDS = ("question", "answer", "code_task", "code")

SCHEMA = """
CREATE TABLE IF NOT EXISTS dialogs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    corpus TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dialog_id TEXT NOT NULL REFERENCES dialogs(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    ref TEXT,
    payload TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_dialog ON turns(dialog_id, id);
CREATE INDEX IF NOT EXISTS turns_ref ON turns(ref);
"""


class DialogNotFound(KeyError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_title(text: str) -> str:
    one_line = " ".join((text or "").split())
    return one_line if len(one_line) <= TITLE_CHARS else one_line[: TITLE_CHARS - 1].rstrip() + "…"


class DialogStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- dialogs ---------------------------------------------------------------------
    def create(self, title: str = "", corpus: str | None = None) -> dict:
        did, now = uuid.uuid4().hex[:16], _now()
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO dialogs VALUES (?, ?, ?, ?, ?)", (did, make_title(title), corpus, now, now))
        return self._dialog(did)

    def list(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT d.*, (SELECT COUNT(*) FROM turns t WHERE t.dialog_id = d.id) AS n_turns FROM dialogs d "
                "ORDER BY d.updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def _dialog(self, did: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT d.*, (SELECT COUNT(*) FROM turns t WHERE t.dialog_id = d.id) AS n_turns FROM dialogs d "
                "WHERE d.id = ?", (did,)).fetchone()
        if row is None:
            raise DialogNotFound(did)
        return dict(row)

    def get(self, did: str) -> dict:
        dialog = self._dialog(did)
        with self._lock:
            rows = self._conn.execute("SELECT * FROM turns WHERE dialog_id = ? ORDER BY id", (did,)).fetchall()
        dialog["turns"] = [self._turn(r) for r in rows]
        return dialog

    def rename(self, did: str, title: str) -> dict:
        with self._lock, self._conn:
            cur = self._conn.execute("UPDATE dialogs SET title = ? WHERE id = ?", (make_title(title), did))
        if cur.rowcount == 0:
            raise DialogNotFound(did)
        return self._dialog(did)

    def delete(self, did: str) -> None:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM dialogs WHERE id = ?", (did,))
        if cur.rowcount == 0:
            raise DialogNotFound(did)

    # --- turns -------------------------------------------------------------------------
    @staticmethod
    def _turn(row: sqlite3.Row) -> dict:
        turn = dict(row)
        turn["payload"] = json.loads(turn["payload"]) if turn["payload"] else None
        return turn

    def add_turn(self, did: str, role: str, kind: str, content: str, payload: dict | None = None,
                 ref: str | None = None) -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown turn kind {kind}")
        dialog = self._dialog(did)
        now = _now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO turns (dialog_id, role, kind, content, ref, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (did, role, kind, content, ref, json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                 now))
            if role == "user" and not dialog["title"]:  # the first question names the dialog
                self._conn.execute("UPDATE dialogs SET title = ?, updated_at = ? WHERE id = ?",
                                   (make_title(content), now, did))
            else:
                self._conn.execute("UPDATE dialogs SET updated_at = ? WHERE id = ?", (now, did))
            row = self._conn.execute("SELECT * FROM turns WHERE id = ?", (cur.lastrowid,)).fetchone()
        return self._turn(row)

    def drop_last_answer(self, did: str) -> None:
        """Before an answer is regenerated for the same question (e.g. after a confirmation)."""
        with self._lock, self._conn:
            row = self._conn.execute("SELECT id, role FROM turns WHERE dialog_id = ? ORDER BY id DESC LIMIT 1",
                                     (did,)).fetchone()
            if row is not None and row["role"] == "assistant":
                self._conn.execute("DELETE FROM turns WHERE id = ?", (row["id"],))

    def update_ref(self, ref: str, payload: dict) -> int:
        """Replace the code result of the turns that refer to ``ref`` (a code task after apply / reject /
        rollback): the whole payload of a code turn, the ``code`` field of an answer from the chat."""
        with self._lock, self._conn:
            rows = self._conn.execute("SELECT id, kind, payload FROM turns WHERE ref = ?", (ref,)).fetchall()
            for r in rows:
                new = payload
                if r["kind"] == "answer":
                    new = {**(json.loads(r["payload"]) if r["payload"] else {}), "code": payload}
                self._conn.execute("UPDATE turns SET payload = ? WHERE id = ?",
                                   (json.dumps(new, ensure_ascii=False), r["id"]))
        return len(rows)

    def history(self, did: str, max_turns: int = 12) -> list[dict]:
        """Turns as plain text for follow-up questions, oldest first; code tasks with their outcome."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, kind, content, payload FROM turns WHERE dialog_id = ? ORDER BY id DESC LIMIT ?",
                (did, max_turns)).fetchall()
        out = []
        for r in reversed(rows):
            payload = json.loads(r["payload"]) if r["payload"] else {}
            code = payload if r["kind"] == "code" else (payload or {}).get("code") if r["kind"] == "answer" else None
            image = (payload or {}).get("image") if r["kind"] == "answer" else None
            video = (payload or {}).get("video") if r["kind"] == "answer" else None
            content = (code_summary(code) if code else image_summary(image) if image else
                       video_summary(video) if video else r["content"])
            out.append({"role": r["role"], "content": content})
        return out


IMAGE_DONE = {"generate": "нарисовал новую картинку", "edit": "изменил картинку по инструкции",
              "redraw": "перерисовал картинку", "inpaint": "дорисовал выделенную область",
              "outpaint": "расширил картинку за края"}


def image_summary(img: dict) -> str:
    """A picture as one assistant turn: a note of what the generator did rather than the chat's reply, which a model
    would copy ("Перерисовал картинку: …") when asked for another change without drawing anything."""
    return (f"[Генератор картинок {IMAGE_DONE.get(img.get('mode'), 'сделал картинку')} "
            f"{img.get('width')}×{img.get('height')}, она показана пользователю. Промпт: {img.get('prompt') or ''}]")


def video_summary(v: dict) -> str:
    """A video as one assistant turn: what the generator made (see image_summary)."""
    what = "оживил картинку в ролик" if v.get("mode") == "image" else "снял ролик"
    return (f"[Генератор видео {what} {v.get('width')}×{v.get('height')}, {v.get('frames')} кадров при "
            f"{v.get('fps')} к/с, он показан пользователю. Промпт: {v.get('prompt') or ''}]")


CODE_STATUS = {"done": "готово", "failed": "не удалось", "limit": "лимит шагов", "error": "ошибка", "running": "идёт"}


def code_summary(res: dict) -> str:
    """A code task as one assistant turn: what was done, which files, and what the user decided."""
    parts = [f"[Код-агент, {CODE_STATUS.get(res.get('status'), res.get('status'))}] {res.get('summary') or ''}".strip()]
    changed = [f"{st} {path}" for st, path in res.get("changed") or []]
    if changed:
        parts.append("Изменения в рабочей копии: " + ", ".join(changed[:12]))
    if res.get("artifacts"):
        parts.append("Созданы: " + ", ".join(res["artifacts"][:12]))
    if res.get("applied"):
        parts.append("Пользователь применил изменения к папке.")
    elif res.get("rejected"):
        parts.append("Пользователь отклонил изменения, в папку ничего не записано.")
    elif changed:
        parts.append("Изменения ждут подтверждения пользователя, в папку пока ничего не записано.")
    if res.get("runs"):
        parts.append(f"Запусков: {res['runs']}, неудачных: {res.get('failed_runs', 0)}.")
    return "\n".join(parts)
