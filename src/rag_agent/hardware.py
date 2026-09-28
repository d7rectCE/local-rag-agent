"""The GPU of this machine, for the model badges of the UI: detected, never configured.

The name comes from PyTorch when it is already loaded (the device the embedder and the reranker use),
otherwise from the operating system: WMI on Windows, nvidia-smi or lspci on Linux, system_profiler on
macOS. A discrete card wins over an integrated one and over virtual adapters (remote desktop, basic
display drivers). The answer is cached: the hardware does not change while the server runs."""

from __future__ import annotations

import platform
import re
import subprocess
import sys
import threading

_lock = threading.Lock()
_cached: str | None = None

# adapters that do not run models: virtual displays and the integrated graphics of a CPU
_NOT_COMPUTE = re.compile(r"basic (display|render)|remote|virtual|parsec|mirage|dummy|hyper-v|"
                          r"radeon\(tm\) graphics|radeon graphics|(uhd|hd|iris\w*) graphics|vega \d+ graphics",
                          re.IGNORECASE)
_PREFIXES = re.compile(r"^(amd radeon |nvidia geforce |nvidia |intel\(r\) |intel )", re.IGNORECASE)


def short_name(name: str) -> str:
    """'AMD Radeon RX 7900 XTX' -> 'RX 7900 XTX', 'NVIDIA GeForce RTX 4090' -> 'RTX 4090'."""
    name = re.sub(r"\((tm|r)\)", "", name.strip(), flags=re.IGNORECASE).strip()
    return _PREFIXES.sub("", name).strip() or name


def pick(names: list[str]) -> str | None:
    """The card that computes: the first one that is neither virtual nor integrated, else the first one."""
    names = [n.strip() for n in names if n and n.strip()]
    for n in names:
        if not _NOT_COMPUTE.search(n):
            return n
    return names[0] if names else None


def _run(args: list[str]) -> str:
    try:
        res = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return res.stdout if res.returncode == 0 else ""


def _from_torch() -> str | None:
    if "torch" not in sys.modules:  # importing torch only for a badge would cost seconds of start-up
        return None
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return platform.processor() or "Apple GPU"
    except Exception:  # noqa: BLE001 - a badge must never break /status
        return None
    return None


def _from_os() -> str | None:
    system = platform.system()
    if system == "Windows":
        out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "(Get-CimInstance Win32_VideoController).Name"])
        return pick(out.splitlines())
    if system == "Linux":
        out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if out.strip():
            return pick(out.splitlines())
        out = _run(["lspci"])
        return pick([re.sub(r"^.*?(VGA compatible controller|3D controller|Display controller):\s*", "", ln)
                     for ln in out.splitlines() if re.search(r"VGA|3D controller|Display controller", ln)])
    if system == "Darwin":
        out = _run(["system_profiler", "SPDisplaysDataType"])
        return pick(re.findall(r"Chipset Model:\s*(.+)", out))
    return None


def gpu_name() -> str | None:
    """The short name of this machine's GPU ('RX 7900 XTX'), or None when there is none."""
    global _cached
    with _lock:
        if _cached is not None:
            return _cached
    name = _from_torch() or _from_os()
    if name:
        with _lock:
            _cached = short_name(name)
        return _cached
    return None
