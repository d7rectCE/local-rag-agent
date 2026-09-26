"""Working copy of a code task (ТЗ ч.2 S17, FR13, FR14).

The agent never writes into the user's folder: text files of the corpus are copied
into ``data_dir/workspaces/<task>``, which is a git repository — every step is a
commit, any checkpoint can be restored, and the result is shown as a diff. Only
``apply_to`` (after the user's confirmation) copies changed files back.

Editing follows the agent-computer interface of SWE-agent: files are viewed in
numbered windows, searched, and edited by replacing a unique fragment; an edit
that breaks the syntax of a .py / .ipynb / .json file is rejected before writing.
The H14 baseline (``overwrite``) writes whole files without any check.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import re
import shutil
import subprocess
from pathlib import Path

TEXT_EXTS = {".py", ".ipynb", ".txt", ".md", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
INITIAL = "initial"  # tag of the starting state


class WorkspaceError(ValueError):
    pass


def check_syntax(path: str, content: str) -> str | None:
    """None if the content is valid for its type, otherwise a short error for the agent."""
    suffix = Path(path).suffix.lower()
    try:
        if suffix == ".py":
            ast.parse(content, filename=path)
        elif suffix == ".json":
            json.loads(content)
        elif suffix == ".ipynb":
            import nbformat

            nbformat.validate(nbformat.reads(content, as_version=4))
            for i, cell in enumerate(json.loads(content).get("cells", []), start=1):
                if cell.get("cell_type") == "code":
                    src = "".join(cell.get("source", []))
                    src = "\n".join("pass" if ln.lstrip().startswith(("%", "!")) else ln for ln in src.split("\n"))
                    try:
                        ast.parse(src)
                    except SyntaxError as exc:
                        return f"ячейка {i}: SyntaxError: {exc.msg} (строка {exc.lineno})"
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (строка {exc.lineno})"
    except Exception as exc:  # json / nbformat validation
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
    return None


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()

    # --- creation and git --------------------------------------------------------------
    @classmethod
    def create(cls, root: Path, source: Path | None, exclude: list[str] = (), max_mb: float = 50.0,
               file_max_mb: float = 2.0) -> Workspace:
        root.mkdir(parents=True, exist_ok=False)
        ws = cls(root)
        total = 0
        if source is not None:
            for path in sorted(source.rglob("*")):
                rel = path.relative_to(source)
                if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]) or not path.is_file():
                    continue
                if path.suffix.lower() not in TEXT_EXTS or any(fnmatch.fnmatch(p, pat) for p in rel.parts for pat in exclude):
                    continue
                size = path.stat().st_size
                if size > file_max_mb * 2**20 or total + size > max_mb * 2**20:
                    continue
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                total += size
        ws.git("init", "-q")
        ws.git("config", "user.name", "rag-agent")
        ws.git("config", "user.email", "agent@localhost")
        ws.git("config", "core.autocrlf", "false")
        (root / ".gitignore").write_text(".agent/\n__pycache__/\n.ipynb_checkpoints/\n", encoding="utf-8")
        ws.git("add", "-A")
        ws.git("commit", "-q", "--allow-empty", "-m", "начальное состояние")
        ws.git("tag", INITIAL)
        return ws

    def git(self, *args: str) -> str:
        proc = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, encoding="utf-8",
                              errors="replace", timeout=60)
        if proc.returncode != 0:
            raise WorkspaceError(f"git {' '.join(args[:2])}: {proc.stderr.strip()[:300]}")
        return proc.stdout

    def commit(self, message: str) -> str | None:
        """A checkpoint; None when nothing changed."""
        self.git("add", "-A")
        if not self.git("status", "--porcelain").strip():
            return None
        self.git("commit", "-q", "-m", message[:200])
        return self.git("rev-parse", "--short", "HEAD").strip()

    def checkpoints(self) -> list[dict]:
        out = self.git("log", "--format=%h%x09%s", "HEAD")
        return [{"commit": h, "message": m} for h, m in (ln.split("\t", 1) for ln in out.splitlines() if "\t" in ln)]

    def rollback(self, commit: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{4,40}|" + INITIAL, commit):
            raise WorkspaceError("неверная контрольная точка")
        self.git("reset", "-q", "--hard", commit)

    def diff(self, base: str = INITIAL) -> str:
        return self.git("diff", "--no-color", base, "--", ".", ":(exclude).gitignore")

    def changed(self, base: str = INITIAL) -> list[tuple[str, str]]:
        """(status A/M/D, path) against the base, including untracked files."""
        self.git("add", "-A")
        out = self.git("diff", "--cached", "--name-status", base)
        return [(ln[0], ln.split("\t", 1)[1]) for ln in out.splitlines() if "\t" in ln and not ln.endswith(".gitignore")]

    # --- files ---------------------------------------------------------------------------
    def path(self, rel: str) -> Path:
        rel = rel.replace("\\", "/").lstrip("/")
        for prefix in ("work/", "./"):
            rel = rel.removeprefix(prefix)
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise WorkspaceError(f"путь вне рабочей копии: {rel}")
        if ".git" in p.relative_to(self.root).parts:
            raise WorkspaceError("папка .git недоступна")
        return p

    def view(self, rel: str, start: int = 1, window: int = 80) -> str:
        p = self.path(rel)
        if not p.is_file():
            raise WorkspaceError(f"нет файла {rel}")
        lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
        start = max(1, min(start, len(lines)))
        end = min(len(lines), start + window - 1)
        body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(start, end + 1))
        more = f"\n(строки {start}–{end} из {len(lines)}; дальше — view_file с start={end + 1})" if end < len(lines) else ""
        return f"{rel} (строки {start}–{end} из {len(lines)}):\n{body}{more}"

    def search(self, pattern: str, glob: str = "*", limit: int = 50) -> str:
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))
        hits = []
        for p in sorted(self.root.rglob(glob)):
            if not p.is_file() or ".git" in p.relative_to(self.root).parts or p.suffix.lower() not in TEXT_EXTS:
                continue
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").split("\n"), start=1):
                if rx.search(line):
                    hits.append(f"{p.relative_to(self.root).as_posix()}:{n}: {line.strip()[:160]}")
                    if len(hits) >= limit:
                        return "\n".join(hits) + f"\n(показаны первые {limit})"
        return "\n".join(hits) or "совпадений нет"

    def files(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*")
                      if p.is_file() and ".git" not in p.relative_to(self.root).parts)

    def edit(self, rel: str, old: str, new: str) -> str:
        """ACI edit: replace a fragment that occurs exactly once; rejected if the result does not parse."""
        p = self.path(rel)
        if not p.is_file():
            raise WorkspaceError(f"нет файла {rel}; для нового файла — create_file")
        text = p.read_text(encoding="utf-8")
        count = text.count(old) if old else 0
        if count != 1:
            raise WorkspaceError(f"фрагмент для замены найден {count} раз(а), нужен ровно 1 — уточни old (скопируй "
                                 "строки из view_file без номеров)")
        updated = text.replace(old, new, 1)
        error = check_syntax(rel, updated)
        if error:
            raise WorkspaceError(f"правка отклонена, файл не записан: {error}")
        p.write_text(updated, encoding="utf-8")
        return f"{rel}: заменено {len(old.splitlines()) or 1} стр. на {len(new.splitlines()) or 1} стр., синтаксис в порядке"

    def create_file(self, rel: str, content: str, check: bool = True) -> str:
        p = self.path(rel)
        error = check_syntax(rel, content) if check else None
        if error:
            raise WorkspaceError(f"файл не создан: {error}")
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"{rel}: {'перезаписан' if existed else 'создан'} ({len(content.splitlines())} стр.)"

    def broken_files(self) -> list[str]:
        """Changed files that no longer parse (H14: share of broken files)."""
        out = []
        for status, rel in self.changed():
            if status == "D":
                continue
            p = self.root / rel
            if p.suffix.lower() in (".py", ".ipynb", ".json") and check_syntax(rel, p.read_text(encoding="utf-8", errors="replace")):
                out.append(rel)
        return out

    # --- applying to the user's folder (after confirmation) --------------------------------
    def apply_to(self, target: Path) -> list[str]:
        """Copy added and modified files into the user's folder; deletions are only reported."""
        target = target.resolve()
        applied = []
        for status, rel in self.changed():
            if status == "D":
                applied.append(f"{rel} (удалён в рабочей копии; в вашей папке не удаляется)")
                continue
            dest = (target / rel).resolve()
            if target not in dest.parents:
                raise WorkspaceError(f"путь вне папки пользователя: {rel}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / rel, dest)
            applied.append(rel)
        return applied
