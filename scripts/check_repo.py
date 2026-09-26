"""Secrets and personal data in the files of the repository (ТЗ Э10; privacy of the published repo).

Scans the files tracked by git (or the given paths) for:

- keys and tokens: private key blocks, GitHub, Hugging Face, OpenAI / Anthropic, AWS, Google,
  Slack tokens, and string assignments to password / secret / token / api_key;
- personal paths: a user profile path (C:\\Users\\<name>, /c/Users/<name>, /Users/<name>,
  /home/<name>) — reports, notebooks and logs of this project must not name the machine's user.

A finding prints the file, the line, the kind and a masked value, never the value itself.
A line that must stay (a test fixture) is allowed with the marker ``noqa: secret``.

    python scripts/check_repo.py            # tracked files; exit code 1 on findings
    python scripts/check_repo.py FILE...
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 2**20
SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".docx", ".xlsx", ".zip", ".gz", ".whl", ".pt", ".onnx",
            ".sqlite", ".ico", ".woff", ".woff2", ".ttf"}
# generic account names of CI runners, containers and docs, not a person
GENERIC_USERS = {"public", "default", "all users", "runner", "runneradmin", "user", "username", "<user>", "you",
                 "your_name", "sandbox", "root", "admin", "name", "me", "ubuntu", "vscode", "jovyan", "app", "work",
                 "$user", "%username%", "<name>", "...", "x", "someone"}

SECRETS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA |PGP |ENCRYPTED )?PRIVATE KEY( BLOCK)?-----")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("OpenAI / Anthropic key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{24,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("assigned secret", re.compile(r"(?i)\b(?:password|passwd|secret|token|api[_-]?key)\b\s*[:=]\s*[\"']"
                                   r"(?!<|\$\{|changeme|example|dummy|test|xxx)([^\"'\s]{12,})[\"']")),
]
PATHS = [
    re.compile(r"(?i)\b[a-z]:(?:\\\\|\\|/)+users(?:\\\\|\\|/)+([^\\/\s\"'`<>|:*?]+)"),
    re.compile(r"(?i)(?<![\w.])/(?:[a-z]/)?users/([^/\s\"'`<>|:*?]+)"),
    re.compile(r"(?<![\w.])/home/([^/\s\"'`<>|:*?]+)"),
]


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [ROOT / p for p in out.decode("utf-8").split("\0") if p]


def mask(value: str) -> str:
    return value[:4] + "…" + f"({len(value)} симв.)" if len(value) > 6 else "…"


def scan_text(text: str) -> list[tuple[int, str, str]]:
    found = []
    for no, line in enumerate(text.splitlines(), start=1):
        if "noqa: secret" in line:
            continue
        for kind, pat in SECRETS:
            for m in pat.finditer(line):
                found.append((no, kind, mask(m.group(m.lastindex or 0))))
        for pat in PATHS:
            for m in pat.finditer(line):
                user = m.group(1).strip().lower()
                if user not in GENERIC_USERS and re.fullmatch(r"[\w.-]{2,}", user):  # a name, not a pattern
                    found.append((no, "personal path", mask(m.group(1))))
    return found


def main(argv: list[str]) -> int:
    files = [Path(a).resolve() for a in argv] if argv else tracked_files()
    total = 0
    for f in files:
        if f.suffix.lower() in SKIP_EXT or not f.is_file() or f.stat().st_size > MAX_BYTES:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for no, kind, shown in scan_text(text):
            total += 1
            rel = f.relative_to(ROOT) if f.is_relative_to(ROOT) else f
            print(f"{rel}:{no}: {kind}: {shown}")
    print(f"{total} находок в {len(files)} файлах" if total else f"чисто: {len(files)} файлов")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
