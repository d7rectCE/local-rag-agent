"""Model files on this machine: what a weights file is (its family, from the GGUF metadata or the name) and
the settings its family is known to work with.

The UI names the models by what they are ("Krea 2", "Qwen-Image-2.1"), and the chat model is told which one
is selected, so it does not describe a generator the user has switched away from."""

from __future__ import annotations

import re
import struct
from functools import lru_cache
from pathlib import Path

# the families sd.cpp runs, by general.architecture of the GGUF; sampling as each family works best here
FAMILIES: dict[str, dict] = {
    "qwen_image21": {"label": "Qwen-Image-2.1", "kind": "image"},
    "qwen_image": {"label": "Qwen-Image", "kind": "image"},
    "krea2": {"label": "Krea 2", "kind": "image"},
    "flux": {"label": "FLUX", "kind": "image"},
    "z_image": {"label": "Z-Image", "kind": "image"},
    "ltx2": {"label": "LTX-2", "kind": "video"},
    "ltxv": {"label": "LTX-Video", "kind": "video"},
    "wan": {"label": "Wan", "kind": "video"},
    "hunyuan_video": {"label": "HunyuanVideo", "kind": "video"},
}
# when a file carries no architecture (safetensors), its name tells
_NAME_FAMILY = [(re.compile(p, re.IGNORECASE), f) for p, f in (
    (r"qwen[-_]?image[-_]?2\.?1", "qwen_image21"), (r"qwen[-_]?image", "qwen_image"), (r"krea[-_]?2", "krea2"),
    (r"flux", "flux"), (r"z[-_]?image", "z_image"), (r"ltxv?[-_]?2", "ltx2"), (r"ltx", "ltxv"), (r"\bwan", "wan"),
    (r"hunyuan[-_]?video", "hunyuan_video"))]
# a distilled build: few steps, no classifier-free guidance
# dmd: distribution matching distillation (the LTX-2.3 build "10Eros_v1.4_dmd-r256")
TURBO = re.compile(r"turbo|lightning|distill|schnell|hyper|lcm|\bfast\b|dmd", re.IGNORECASE)

_TYPES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


@lru_cache(maxsize=64)
def _gguf_meta(path: str, mtime: float) -> dict:
    out: dict = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return out
        f.read(4)  # version
        _, n_kv = struct.unpack("<QQ", f.read(16))

        def string() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def value(t: int):
            if t == 8:
                return string()
            if t == 9:
                item, n = struct.unpack("<IQ", f.read(12))
                vals = [value(item) for _ in range(n)]
                return vals[:8]
            fmt = _TYPES[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        for _ in range(min(n_kv, 256)):
            key = string()
            (t,) = struct.unpack("<I", f.read(4))
            val = value(t)
            if key.startswith("general."):
                out[key] = val
    return out


def gguf_meta(path: str | Path) -> dict:
    """The general.* metadata of a GGUF file ({} for other files or a broken header)."""
    p = Path(path)
    if p.suffix.lower() != ".gguf" or not p.is_file():
        return {}
    try:
        return _gguf_meta(str(p), p.stat().st_mtime)
    except (OSError, struct.error, KeyError, ValueError):
        return {}


def family(path: str | Path) -> str:
    arch = str(gguf_meta(path).get("general.architecture") or "")
    name = Path(path).name
    if arch.startswith("ltx") and re.search(r"ltxv?[-_]?2", name, re.IGNORECASE):
        return "ltx2"  # LTX-2 builds may carry a generic "ltxv" architecture
    if arch:
        return arch
    return next((f for rx, f in _NAME_FAMILY if rx.search(name)), "")


def label(path: str | Path) -> str:
    """A human name: the family and the build ("Krea 2 (Krea2_turbo_uncensored_edit_v1.1-Q4_K_M)")."""
    fam = FAMILIES.get(family(path), {}).get("label")
    stem = Path(path).stem
    return f"{fam} ({stem})" if fam else stem


def is_turbo(path: str | Path) -> bool:
    meta = gguf_meta(path)
    return bool(TURBO.search(Path(path).name) or TURBO.search(str(meta.get("general.name") or "")))


# --- the models/ folder ----------------------------------------------------------------------------------------
MODEL_EXTS = {".gguf", ".safetensors"}
_AUDIO_VAE = re.compile(r"audio[-_ ]?vae|vocoder", re.IGNORECASE)
_VAE = re.compile(r"vae|autoencoder", re.IGNORECASE)
_CONNECTORS = re.compile(r"connector|projections", re.IGNORECASE)  # LTX-2.3: "embeddings connectors" / "projections"
_ENCODER = re.compile(r"text[-_]?enc|qwen[\d._-]*vl|gemma|umt5|t5xxl|clip[-_]?(l|g|vision)?\b|llava|mistral|llama|"
                      r"qwen\d[\d._-]*[-_]\d+b", re.IGNORECASE)
_TEXT_ARCH = re.compile(r"^(llama|qwen\d*\w*|gemma\d*|mistral\w*|phi\d*|t5\w*|clip\w*|bert|glm\w*|deepseek\w*)$")


def role(path: Path) -> str:
    """What a weights file is next to a diffusion model: dit, vae, audio_vae, llm (the text encoder),
    connectors (LTX-2.3 embeddings connectors) or vision (a projector)."""
    arch = str(gguf_meta(path).get("general.architecture") or "")
    if arch in FAMILIES:
        return "dit"
    name = path.name
    if _AUDIO_VAE.search(name):
        return "audio_vae"
    if _CONNECTORS.search(name):
        return "connectors"
    if _VAE.search(name):
        return "vae"
    if "mmproj" in name.lower():
        return "vision"
    if (arch and _TEXT_ARCH.match(arch)) or _ENCODER.search(name):
        return "llm"
    return "dit"


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[-_. ]+", Path(name).stem.lower()) if t}


def _closest(candidates: list[Path], dit: Path) -> Path | None:
    """The companion made for this model: the most name tokens shared ("qwen_image_2.1_vae" for 2.1)."""
    return max(candidates, key=lambda c: len(_tokens(c.name) & _tokens(dit.name)), default=None)


def scan(root: Path, kind: str) -> list[dict]:
    """The models of models/<kind>: every diffusion model with the VAE, text encoder (and for video the audio
    VAE and connectors) of its folder. For "text": the GGUF files (the chat models to import into Ollama)."""
    base = root / kind
    if not base.is_dir():
        return []
    if kind == "text":
        return [{"name": f.name, "path": str(f), "size": f.stat().st_size, "kind": "text"}
                for f in sorted(base.rglob("*.gguf")) if "mmproj" not in f.name.lower()]
    out = []
    for folder in [base, *sorted(d for d in base.iterdir() if d.is_dir())]:
        found = folder.glob("*") if folder == base else folder.rglob("*")
        files = sorted(f for f in found if f.is_file() and f.suffix.lower() in MODEL_EXTS)
        roles = {f: role(f) for f in files}
        for dit in (f for f, r in roles.items() if r == "dit"):
            comp = {r: _closest([f for f, rr in roles.items() if rr == r], dit)
                    for r in ("vae", "llm", "audio_vae", "connectors")}
            out.append({"name": dit.name, "path": str(dit), "size": dit.stat().st_size, "kind": kind,
                        "family": family(dit), "label": label(dit),
                        **{k: str(v) if v else "" for k, v in comp.items()}})
    return out


def ollama_name(file_name: str) -> str:
    """The Ollama name of a chat model from models/text: its file name, lower case, without the extension."""
    stem = Path(file_name).stem.lower()
    return re.sub(r"[^a-z0-9._-]+", "-", stem).strip("-.")[:80] or "model"


# sampling each family is known to work with (the UI's settings go over it)
PRESETS: dict[str, dict] = {
    "qwen_image21": {"steps": 20, "cfg_scale": 3.0, "sampler": "euler"},
    "qwen_image": {"steps": 20, "cfg_scale": 2.5, "sampler": "euler"},
    "krea2": {"steps": 28, "cfg_scale": 3.5, "sampler": "euler"},
    "ltx2": {"steps": 20, "cfg_scale": 6.0, "sampler": "euler", "frames": 33, "fps": 24,
             "negative_prompt": "worst quality, low quality, blurry, distorted, artifacts"},
}
TURBO_PRESET = {"steps": 8, "cfg_scale": 1.0, "sampler": "euler", "scheduler": "simple"}


def preset(path: str | Path, llm: str = "") -> dict:
    """Recommended settings of a model: its family's, a distilled build's few steps without guidance, the
    text encoder on the GPU when it is in fp8 (the CPU backend of sd.cpp crashes on fp8 weights)."""
    out = dict(PRESETS.get(family(path), {}))
    name = Path(path).name.lower()
    if family(path) == "ltx2" and re.search(r"2[._]5", name):
        out.update(cfg_scale=3.0, frames=121)  # LTX-2.5, as in sd.cpp's docs
    if is_turbo(path):
        # a video model keeps its own noise schedule; the guidance pass is what doubles every step
        out.update({k: v for k, v in TURBO_PRESET.items() if not (family(path) == "ltx2" and k == "scheduler")})
    if "fp8" in Path(llm).name.lower():
        out["text_encoder_on_cpu"] = False
    return out
