"""Docker sandbox for code the agent writes (ТЗ ч.2 S17, NFR8).

Every run is a fresh container: no network, uid 1000, read-only root file system,
no capabilities, no privilege escalation, limits on memory, CPU, processes and
time. Only the task's working copy is mounted writable; the corpus can be mounted
read-only for scripts that read the user's data. Host environment variables (and so
secrets) are never passed in.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

from pydantic import BaseModel

from rag_agent.config import CodeConfig


class RunResult(BaseModel):
    command: list[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def report(self, limit: int = 4000) -> str:
        """What the agent sees: exit code, stdout and stderr (the tail, where tracebacks end)."""
        def tail(text: str) -> str:
            return text if len(text) <= limit else "…" + text[-limit:]

        head = "таймаут" if self.timed_out else f"код выхода {self.exit_code}"
        return f"[{head}, {self.duration_s:.1f} с]\nstdout:\n{tail(self.stdout) or '—'}\nstderr:\n{tail(self.stderr) or '—'}"


class SandboxError(RuntimeError):
    pass


def docker_path(path: Path) -> str:
    return str(path.resolve())


class DockerSandbox:
    def __init__(self, cfg: CodeConfig):
        self.cfg = cfg

    def available(self) -> bool:
        try:
            r = subprocess.run(["docker", "image", "inspect", self.cfg.image], capture_output=True, timeout=20)
            return r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def run(self, command: list[str], work: Path, corpus: Path | None = None, timeout_s: float | None = None) -> RunResult:
        timeout = timeout_s or self.cfg.timeout_s
        name = f"rag-sbx-{uuid.uuid4().hex[:12]}"
        args = [
            "docker", "run", "--rm", "--name", name,
            "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m",
            "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", self.cfg.memory, "--cpus", str(self.cfg.cpus), "--pids-limit", str(self.cfg.pids),
            "-v", f"{docker_path(work)}:/work:rw", "-w", "/work",
        ]
        if corpus is not None:
            args += ["-v", f"{docker_path(corpus)}:/corpus:ro"]
        args += [self.cfg.image, *command]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(args, capture_output=True, timeout=timeout, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
            return RunResult(command=command, exit_code=-1, stdout=_text(exc.stdout), stderr=_text(exc.stderr),
                             timed_out=True, duration_s=round(time.perf_counter() - t0, 2))
        except OSError as exc:
            raise SandboxError(f"docker недоступен: {exc}") from exc
        if proc.returncode == 125 or "Cannot connect to the Docker daemon" in proc.stderr \
                or "failed to connect to the docker API" in proc.stderr:
            raise SandboxError("docker не запущен или образ песочницы не собран: " + proc.stderr.strip()[:300])
        return RunResult(command=command, exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
                         duration_s=round(time.perf_counter() - t0, 2))


def _text(data) -> str:
    if data is None:
        return ""
    return data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)
