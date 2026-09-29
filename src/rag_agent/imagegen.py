"""Image generation and editing in the chat: Qwen-Image-2.1 through stable-diffusion.cpp.

Modes (the model is one for generation and editing):

    generate — text -> image;
    edit     — an image and an instruction -> the edited image (the image is a reference, ``-r``);
    redraw   — img2img: the image is the starting point, ``strength`` sets how much changes;
    inpaint  — only the area of a mask is redrawn (white = redraw);
    outpaint — the canvas is extended on the chosen sides and the new area is drawn (inpaint of the margin).

The user's request is rewritten by the chat LLM into a prompt for the generator (English is what the
model follows best; text to be drawn stays as written) and checked: sexual content involving minors is
refused before anything runs. The engine is a subprocess of ``sd-cli`` per image: the text encoder runs
on the CPU, the DiT and the VAE on the GPU, and the chat LLM is unloaded from the GPU first (they do not
fit together in 24 GB). Results are stored per dialog under ``data_dir/generated/<dialog>/``.

The feature is off by default (``imagegen.enabled``): the weights are the user's local choice and are not
part of the project (the Qwen-Image-2.1 builds come under the Qwen Research License).
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import BaseModel

from rag_agent.hardware import short_name
from rag_agent.llm import BaseLLM, LLMError

log = logging.getLogger(__name__)

MODES = ("generate", "edit", "redraw", "inpaint", "outpaint")
ASPECTS = {"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832), "wide": (1344, 768),
           "tall": (768, 1344)}
# the sizes of the model card (Qwen/Qwen-Image-2.1): what the model is trained for; ASPECTS above are the fast
# default (about a megapixel), chosen by the chat model when the user has not picked a size
OFFICIAL_SIZES = {"1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400), "3:2": (2528, 1696),
                  "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752)}
MAX_SIDE, MAX_AREA = 3072, 2752 * 1536  # larger sizes are scaled down, keeping the proportions
_SIZE = re.compile(r"^\s*(\d{2,5})\s*[x×хX*]\s*(\d{2,5})\s*$")

# the time of a picture on this machine: overhead + steps * k * cost, where a step grows faster than the pixel
# count (attention over all the patches); k is learned from the pictures already made here
OVERHEAD_S = 16.0  # the prompt encoded on the CPU, the weights read, the VAE decode
DEFAULT_K = 3.9  # a first guess until this machine has made a picture (measured on an RX 7900 XTX, Vulkan)

# weights in the model folder that are not diffusion models: text encoders, VAEs, vision projectors
_NOT_DIT = re.compile(r"vae|text[_-]?enc|qwen[\d._-]*vl|mmproj|clip|llava|umt5|(^|[^a-z0-9])t5", re.IGNORECASE)
_QUANT = re.compile(r"(?<![A-Za-z0-9])(IQ\d\w*|Q\d(?:_K)?(?:_[SML0-9])?|BF16|FP16|F16|FP8\w*|F32|INT8)(?![A-Za-z0-9])",
                    re.IGNORECASE)
_ID = re.compile(r"^[0-9a-f]{12}$")
_SESSION = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# a cheap pre-filter: only such requests reach the model check
IMAGE_HINT = re.compile(
    r"нарису|сгенерир\w*\s+(картин|изображ|фото|рисун|арт|логотип|иконк|обо)|изобрази|перерису|дорису|"
    r"отредактир\w*\s+(картин|фото|изображ)|измени\w*\s+(на\s+)?(картин|фото|изображ)|(сделай|создай)\s+(картин|изображ|фото|рисун|арт|логотип|иконк|обо)|"
    r"картинк|рисун|иллюстрац|\b(draw|paint|illustrate|render|sketch|outpaint|inpaint)\b|"
    r"generate\s+(an?\s+)?(image|picture|photo|illustration|logo)|расшир\w*\s+(картин|фото|изображ|холст)",
    re.IGNORECASE)

PLAN_PROMPT = """Ты готовишь запрос к генератору изображений Qwen-Image-2.1 (он рисует по тексту и редактирует картинки).
По просьбе пользователя (и истории диалога) реши:
- action: "none" — картинку рисовать не нужно (вопрос, объяснение, просьба о коде); "generate" — нарисовать новую; "edit" — изменить имеющуюся картинку по инструкции (заменить фон, добавить или убрать объект, сменить стиль: «в стиле акварели», «как аниме», «маслом» — это edit); "redraw" — перерисовать ту же картинку свободнее, сохранив композицию и цвета («перерисуй получше», «сделай детальнее», «вариация»); "outpaint" — дорисовать картинку за её краями (расширить);
- prompt: подробный промпт на английском — объект, окружение, стиль, свет, ракурс, композиция; для edit — чёткая инструкция, что изменить и что сохранить. Текст, который должен быть написан на картинке, оставь как есть в кавычках;
- aspect: "square", "portrait", "landscape", "wide" или "tall";
- sides: для outpaint — какие стороны расширить (left, right, top, bottom), иначе пустой список;
- minor_sexual: true, если просьба предполагает сексуальный или эротический контент с участием несовершеннолетних или людей, выглядящих несовершеннолетними; иначе false.
Если есть картинка-источник (прикреплённая или последняя сгенерированная), «перерисуй», «измени», «сделай фон…», «добавь…» относятся к ней.
Верни JSON: {"action": "...", "prompt": "...", "aspect": "...", "sides": [], "minor_sexual": false}"""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["none", "generate", "edit", "redraw", "outpaint"]},
        "prompt": {"type": "string"},
        "aspect": {"type": "string", "enum": list(ASPECTS)},
        "sides": {"type": "array", "items": {"type": "string", "enum": ["left", "right", "top", "bottom"]}},
        "minor_sexual": {"type": "boolean"},
    },
    "required": ["action", "prompt", "aspect", "sides", "minor_sexual"],
}

MINOR_TERMS = re.compile(r"\b(child|children|kid|kids|minor|underage|teen|preteen|loli|shota|schoolgirl|schoolboy)\b|"
                         r"ребён|ребен|дет[иья]|несовершеннолет|школьни|малолет|подрост|девочк|мальчик", re.IGNORECASE)
SEXUAL_TERMS = re.compile(r"\b(nude|naked|sex|sexual|erotic|nsfw|porn|lingerie|topless)\b|"
                          r"голы|обнаж|секс|эрот|порн|нижн\w+ бель", re.IGNORECASE)


class ImagePlan(BaseModel):
    action: str = "none"
    prompt: str = ""
    aspect: str = "square"
    sides: list[str] = []
    refused: str | None = None


class GeneratedImage(BaseModel):
    id: str
    dialog: str
    mode: str
    prompt: str
    request: str = ""  # the user's words
    width: int
    height: int
    seed: int
    steps: int
    source: str | None = None  # the image it was made from: "upload:<id>" or "generated:<id>"
    model: str = ""  # the diffusion model's file name
    strength: float | None = None  # redraw / inpaint: sd.cpp runs steps x strength of them
    elapsed_s: float = 0.0
    created_at: str = ""

    @property
    def url(self) -> str:
        return f"/images/{self.dialog}/{self.id}.png"


class ImageGenError(RuntimeError):
    pass


# a change to a picture already in the dialog ("сделай её в стиле акварели", "убери фон"): these words are about a
# picture only when there is one, so they count only then, and the plan of the chat LLM still may say "none"
EDIT_HINT = re.compile(
    r"в\s+стил|стилиз|\bфон(а|у|ом|е|ы|ов)?\b|измени|поменя|замени|убери|добавь|перекрас|раскрас|акварел|аниме|"
    r"маслом|карандаш|ярче|темнее|светлее|ч[её]рно-бел|\b(make it|in the style|background|recolou?r)\b",
    re.IGNORECASE)


def wants_image(question: str) -> bool:
    return bool(IMAGE_HINT.search(question or ""))


def wants_edit(question: str) -> bool:
    return bool(EDIT_HINT.search(question or ""))


def plan_image(question: str, history: list[dict], llm: BaseLLM, has_source: bool) -> ImagePlan:
    """The chat LLM decides whether to draw and writes the generator's prompt."""
    dialogue = "\n".join(f"{'Пользователь' if t['role'] == 'user' else 'Ассистент'}: {t['content'][:600]}"
                         for t in history[-6:])
    user = (f"История:\n{dialogue or 'пусто'}\n\nКартинка-источник: {'есть' if has_source else 'нет'}\n\n"
            f"Просьба: {question}")
    try:
        data = llm.chat([{"role": "system", "content": PLAN_PROMPT}, {"role": "user", "content": user}],
                        json_schema=PLAN_SCHEMA, max_tokens=600, purpose="image_plan").json()
    except LLMError:
        return ImagePlan()
    plan = ImagePlan(action=str(data.get("action") or "none"), prompt=str(data.get("prompt") or "").strip(),
                     aspect=str(data.get("aspect") or "square"),
                     sides=[s for s in data.get("sides") or [] if s in ("left", "right", "top", "bottom")])
    if plan.action in ("edit", "redraw", "outpaint") and not has_source:
        plan.action = "generate"
    text = f"{question}\n{plan.prompt}"
    if data.get("minor_sexual") or (MINOR_TERMS.search(text) and SEXUAL_TERMS.search(text)):
        plan.refused = "Не могу создать такое изображение: сексуальный контент с участием несовершеннолетних запрещён."
    return plan


def unload_ollama(base_url: str) -> None:
    """The generator needs the GPU: the chat model leaves it (Ollama loads it back on the next question)."""
    try:
        for m in httpx.get(f"{base_url}/api/ps", timeout=5).json().get("models", []):
            httpx.post(f"{base_url}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
    except httpx.HTTPError:
        pass


def outpaint_canvas(image: bytes, sides: list[str], share: float = 0.25,
                    area: int | None = None) -> tuple[bytes, bytes, bytes, int, int]:
    """The image on a larger canvas, the mask of what the model draws (white) and the alpha of what stays from
    the old image when the result is assembled. Sizes are multiples of 32; ``area`` (pixels) scales the canvas
    to a chosen size, otherwise it keeps the picture's scale unless it exceeds the model's range.

    The mask reaches a band into the old image: the model redraws that band in the light of both sides, and
    the old image fades into the new one across it, so there is no hard seam where the canvas was cut."""
    import numpy as np
    from PIL import Image, ImageFilter

    with Image.open(io.BytesIO(image)) as im:
        im = im.convert("RGB")
        w, h = im.size
        sides = sides or ["left", "right"]
        dx, dy = int(w * share), int(h * share)
        left, right = (dx if "left" in sides else 0), (dx if "right" in sides else 0)
        top, bottom = (dy if "top" in sides else 0), (dy if "bottom" in sides else 0)
        W, H = w + left + right, h + top + bottom
        k = (area / (W * H)) ** 0.5 if area else 1.0
        W32, H32 = snap32(W * k, H * k, floor=True)
        band = max(32, min(w, h) // 12)
        xs, ys = np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
        keep = np.ones((H, W), dtype=np.float32)
        if left:
            keep *= np.clip((xs - left) / band, 0, 1)[None, :]
        if right:
            keep *= np.clip((left + w - 1 - xs) / band, 0, 1)[None, :]
        if top:
            keep *= np.clip((ys - top) / band, 0, 1)[:, None]
        if bottom:
            keep *= np.clip((top + h - 1 - ys) / band, 0, 1)[:, None]
        keep = Image.fromarray((keep * 255).round().astype(np.uint8), mode="L")
        mask = keep.point(lambda v: 0 if v == 255 else 255).filter(ImageFilter.MaxFilter(9))
        canvas = Image.new("RGB", (W, H), (127, 127, 127))
        canvas.paste(im, (left, top))
        # the margin starts from stretched edge colours, which helps the model continue the scene
        edge = im.resize((W, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(24))
        outside = Image.new("L", (W, H), 255)
        outside.paste(0, (left, top, left + w, top + h))
        canvas = Image.composite(edge, canvas, outside)
        size = (W32, H32)
        canvas, mask, keep = canvas.resize(size, Image.LANCZOS), mask.resize(size, Image.NEAREST), keep.resize(size, Image.BILINEAR)
    return _png(canvas), _png(mask), _png(keep), W32, H32


def blend_back(result: Path, canvas: Path, keep: Path) -> None:
    """The old image over the model's picture: pixel-exact inside, fading out across the band."""
    from PIL import Image

    with Image.open(result) as out, Image.open(canvas) as old, Image.open(keep) as alpha:
        out = out.convert("RGB")
        if out.size != old.size:
            return
        Image.composite(old.convert("RGB"), out, alpha.convert("L")).save(result, format="PNG")


def _png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def short_path(path: Path) -> str:
    """An ASCII form of a path for sd-cli, which opens files through the ANSI API of Windows: a path with
    Cyrillic (a user folder like C:\\Users\\Имя) fails, so the DOS 8.3 name is used when there is one."""
    s = str(path)
    if s.isascii() or os.name != "nt":
        return s
    import ctypes

    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(s, buf, 1024)
    return buf.value if n and buf.value.isascii() else s


def ascii_workdir(base: str = "") -> Path:
    """Where sd-cli reads its inputs and writes the picture: a folder with an ASCII path."""
    for candidate in (base, tempfile.gettempdir()):
        if candidate:
            d = Path(candidate) / "rag-imagegen"
            d.mkdir(parents=True, exist_ok=True)
            if short_path(d).isascii():
                return d
    d = Path(os.environ.get("SystemDrive", "C:") + "\\") / "rag-imagegen"
    d.mkdir(parents=True, exist_ok=True)
    return d


def snap32(w: float, h: float, floor: bool = False) -> tuple[int, int]:
    """Multiples of 32 near w x h (the model's patch grid), scaled down to MAX_SIDE and MAX_AREA."""
    if w <= 0 or h <= 0:
        raise ImageGenError("размер должен быть положительным")
    scale = min(1.0, MAX_SIDE / max(w, h), (MAX_AREA / (w * h)) ** 0.5)
    to32 = (lambda v: math.floor(v / 32) * 32) if floor else (lambda v: round(v / 32) * 32)
    return max(256, to32(w * scale)), max(256, to32(h * scale))


def parse_size(value: str | None) -> tuple[int, int] | str | None:
    """The size the user picked: None (auto: the chat model picks a format near a megapixel), "source" (the
    size of the picture being changed), an official ratio ("16:9") or width x height ("1920x1080")."""
    if not value or value == "auto":
        return None
    if value == "source":
        return "source"
    if value in OFFICIAL_SIZES:
        return OFFICIAL_SIZES[value]
    m = _SIZE.match(value)
    if not m:
        raise ImageGenError(f"непонятный размер «{value}»: нужно ширина×высота, например 1920x1080")
    return snap32(int(m.group(1)), int(m.group(2)))


def fit_source(src: tuple[int, int], choice: tuple[int, int] | str | None) -> tuple[int, int]:
    """The size of a change that keeps the picture's proportions (redraw, inpaint, auto edit): auto -> near a
    megapixel, "source" -> the picture's own size, a chosen size -> its pixel count in these proportions."""
    w, h = src
    if choice is None:
        return fit32(src)
    if choice == "source":
        return snap32(w, h)
    k = (choice[0] * choice[1] / (w * h)) ** 0.5
    return snap32(w * k, h * k)


def step_cost(width: int, height: int, mode: str = "generate") -> float:
    mp = width * height / 1e6 * (2 if mode == "edit" else 1)  # an edit also attends to the reference picture
    return mp * (1 + mp / 4.4)


# a crash rather than an error: access violation on Windows, SIGSEGV / SIGABRT elsewhere
_CRASH = {3221225477, -1073741819, 3221226505, -1073740791, 139, -11, 134, -6}


def sd_error(code: int, stdout: str, stderr: str) -> str:
    """What went wrong, from sd-cli's own log: sd.cpp writes it to stdout ([INFO] / [ERROR] lines), while
    stderr holds the device list of ggml — which is all the user saw before."""
    lines = (stdout + "\n" + stderr).splitlines()
    errors = [ln.strip() for ln in lines if "[ERROR" in ln]
    if errors:
        detail = "\n".join(errors[-3:])
    else:
        info = [ln.strip() for ln in stdout.splitlines() if ln.startswith("[INFO")]
        detail = f"последний шаг: {info[-1]}" if info else (stdout or stderr)[-400:].strip()
    what = "движок аварийно завершился" if code in _CRASH else "движок завершился с ошибкой"
    return f"{what} (код {code}): {detail}"


def fit32(size: tuple[int, int], limit: int = 1344, least: int = 1024) -> tuple[int, int]:
    """A generation size with the source's proportions, within the model's range and divisible by 32: a small
    photo is drawn larger (the model gives little detail far below a megapixel), a large one smaller."""
    w, h = size
    scale = min(1.0, limit / max(w, h))
    if max(w, h) * scale < least:
        scale = least / max(w, h)
    return max(256, int(w * scale) // 32 * 32), max(256, int(h * scale) // 32 * 32)


class ImageGenerator:
    def __init__(self, settings):
        self.settings, self.cfg = settings, settings.imagegen
        self.root = settings.data_dir / "generated"
        self._device: dict | None = None

    @property
    def available(self) -> bool:
        c = self.cfg
        return bool(c.enabled and c.sd_cli and c.diffusion_model and c.vae and c.llm) and all(
            Path(p).exists() for p in (c.sd_cli, c.diffusion_model, c.vae, c.llm))

    def missing(self) -> list[str]:
        c = self.cfg
        if not c.enabled:
            return ["imagegen.enabled"]
        return [k for k in ("sd_cli", "diffusion_model", "vae", "llm") if not getattr(c, k) or not Path(getattr(c, k)).exists()]

    def _model_files(self) -> dict[str, Path]:
        c = self.cfg
        if not c.diffusion_model:
            return {}
        default = Path(c.diffusion_model)
        skip = {Path(p).name.lower() for p in (c.vae, c.llm, *(x for m in c.models for x in (m.vae, m.llm))) if p}
        found = {default.name: default, **{Path(m.path).name: Path(m.path) for m in c.models if m.path}}
        for d in dict.fromkeys([default.parent, *([Path(c.models_dir)] if c.models_dir else [])]):
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if (f.suffix.lower() in (".gguf", ".safetensors") and f.name.lower() not in skip
                            and not _NOT_DIT.search(f.name)):
                        found.setdefault(f.name, f)
        return found

    def profile(self, name: str) -> dict:
        """How a model runs: its VAE, text encoder and sampling — from its entry in ``imagegen.models`` if it
        has one, the rest from the imagegen section (the default model and builds of its family)."""
        c = self.cfg
        prof = next((m for m in c.models if Path(m.path).name == name), None)
        return {"vae": (prof and prof.vae) or c.vae, "llm": (prof and prof.llm) or c.llm,
                "steps": (prof and prof.steps) or c.steps,
                "cfg_scale": prof.cfg_scale if prof and prof.cfg_scale is not None else c.cfg_scale,
                "sampler": (prof and prof.sampler) or c.sampler, "scheduler": (prof and prof.scheduler) or c.scheduler,
                "text_encoder_on_cpu": (prof.text_encoder_on_cpu if prof and prof.text_encoder_on_cpu is not None
                                        else c.text_encoder_on_cpu)}

    def models(self) -> list[dict]:
        """The diffusion models to choose from: the configured one, the models of ``imagegen.models`` (each with
        its own VAE, encoder and sampling) and the other weights of the default model's folder and of
        ``imagegen.models_dir`` — builds of the default model's family, which share its VAE and text encoder."""
        default = Path(self.cfg.diffusion_model) if self.cfg.diffusion_model else None
        out = []
        for name, p in self._model_files().items():
            m = _QUANT.search(p.stem)
            prof = self.profile(name)
            out.append({"name": name, "size": p.stat().st_size if p.exists() else 0,
                        "quant": m.group(1).upper() if m else "", "default": p == default,
                        "steps": prof["steps"], "cfg_scale": prof["cfg_scale"]})
        return out

    def model_path(self, name: str | None) -> str:
        if not name:
            return self.cfg.diffusion_model
        found = self._model_files().get(name)
        if found is None or not found.exists():
            raise ImageGenError(f"модель картинок «{name}» не найдена в папке моделей")
        return str(found)

    def device(self) -> dict:
        """Where sd-cli computes: its first GPU device, asked from sd-cli itself (``--list-devices``), e.g.
        {"backend": "Vulkan", "name": "RX 7900 XTX"}; the answer is cached."""
        if self._device is None:
            self._device = {}
            if self.cfg.sd_cli and Path(self.cfg.sd_cli).exists():
                try:
                    out = subprocess.run([self.cfg.sd_cli, "--list-devices"], capture_output=True, text=True,
                                         encoding="utf-8", errors="replace", timeout=60,
                                         cwd=str(Path(self.cfg.sd_cli).parent)).stdout
                except (OSError, subprocess.TimeoutExpired):
                    out = ""
                devices = [m.groups() for m in re.finditer(r"^([A-Za-z]+)(\d*)\t(.+)$", out, re.MULTILINE)]
                gpu = next((d for d in devices if d[0].upper() != "CPU"), devices[0] if devices else None)
                if gpu:
                    self._device = {"backend": gpu[0], "name": short_name(gpu[2])}
        return self._device

    def speed(self) -> float:
        """Seconds per step per unit of cost: the median over the last pictures made on this machine."""
        metas = sorted(self.root.glob("*/*.json"), key=lambda p: p.stat().st_mtime)[-30:] if self.root.exists() else []
        ks = []
        for meta in metas:
            try:
                g = GeneratedImage.model_validate_json(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            steps = g.steps * (g.strength if g.mode == "redraw" and g.strength else 1.0)
            cost = steps * step_cost(g.width, g.height, g.mode)
            if cost > 0 and g.elapsed_s > OVERHEAD_S:
                ks.append((g.elapsed_s - OVERHEAD_S) / cost)
        return statistics.median(ks) if ks else DEFAULT_K

    def estimate(self, mode: str, width: int, height: int, steps: int | None = None) -> float:
        """Expected seconds for a picture of this size on this machine."""
        steps = (steps or self.cfg.steps) * (self.cfg.strength if mode == "redraw" else 1.0)
        return OVERHEAD_S + steps * self.speed() * step_cost(width, height, mode)

    def _dir(self, dialog: str) -> Path:
        if not _SESSION.match(dialog or ""):
            raise ImageGenError("неверный идентификатор диалога")
        return self.root / dialog

    def path(self, dialog: str, image_id: str) -> Path:
        if not _ID.match(image_id or ""):
            raise ImageGenError("неверный идентификатор картинки")
        return self._dir(dialog) / f"{image_id}.png"

    def info(self, dialog: str, image_id: str) -> GeneratedImage:
        meta = self.path(dialog, image_id).with_suffix(".json")
        if not meta.exists():
            raise ImageGenError("картинка не найдена")
        return GeneratedImage.model_validate_json(meta.read_text(encoding="utf-8"))

    def last(self, dialog: str) -> GeneratedImage | None:
        d = self._dir(dialog)
        metas = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime) if d.exists() else []
        return GeneratedImage.model_validate_json(metas[-1].read_text(encoding="utf-8")) if metas else None

    def run(self, dialog: str, mode: str, prompt: str, *, request: str = "", width: int | None = None,
            height: int | None = None, source: bytes | None = None, source_ref: str | None = None,
            mask: bytes | None = None, sides: list[str] | None = None, strength: float | None = None,
            seed: int | None = None, steps: int | None = None, size: tuple[int, int] | str | None = None,
            model: str | None = None) -> GeneratedImage:
        """``size``: the user's choice (see parse_size); ``model``: a file name from models()."""
        if mode not in MODES:
            raise ImageGenError(f"неизвестный режим {mode}")
        if not self.available:
            raise ImageGenError("генерация картинок не настроена: " + ", ".join(self.missing()))
        if mode != "generate" and source is None:
            raise ImageGenError("для этого режима нужна картинка-источник")
        c = self.cfg
        dit = self.model_path(model)
        prof = self.profile(Path(dit).name)
        for part in ("vae", "llm"):
            if not prof[part] or not Path(prof[part]).exists():
                raise ImageGenError(f"у модели {Path(dit).name} не найден файл {part}: {prof[part] or 'не задан'}")
        out_dir = self._dir(dialog)
        out_dir.mkdir(parents=True, exist_ok=True)
        image_id = uuid.uuid4().hex[:12]
        work = ascii_workdir(c.work_dir) / image_id  # sd-cli cannot open non-ASCII paths (the user folder)
        work.mkdir()
        w = short_path(work)
        seed = int(seed if seed is not None else uuid.uuid4().int % 2**31)
        steps = int(steps or prof["steps"])
        args = [c.sd_cli, "--diffusion-model", dit, "--vae", prof["vae"], "--llm", prof["llm"],
                "--cfg-scale", str(prof["cfg_scale"]), "--sampling-method", prof["sampler"], "--steps", str(steps),
                "-s", str(seed), "-p", prompt, "-o", os.path.join(w, "out.png")]
        if prof["scheduler"]:
            args += ["--scheduler", prof["scheduler"]]
        if c.negative_prompt:
            args += ["-n", c.negative_prompt]
        if prof["text_encoder_on_cpu"]:
            # an edit feeds the picture itself to the vision-language encoder: hundreds of image tokens, which
            # the CPU encodes for minutes; then the weights stay in RAM and the GPU computes (181 s vs 312 s)
            args += ["--params-backend", "te=cpu"] if mode == "edit" else ["--backend", "te=cpu"]
        if c.offload_to_cpu:
            args.append("--offload-to-cpu")
        if c.flash_attention:
            args.append("--fa")
        if c.vae_tiling:
            args.append("--vae-tiling")
        try:
            if source is not None:
                from PIL import Image

                with Image.open(io.BytesIO(source)) as im:
                    src_size = im.size
                    (work / "source.png").write_bytes(_png(im.convert("RGB")))
            if mode == "inpaint":
                strength = strength if strength is not None else 1.0
            elif mode == "redraw":
                strength = strength if strength is not None else c.strength
            else:
                strength = None
            if mode == "outpaint":
                area = size[0] * size[1] if isinstance(size, tuple) else None
                canvas, mask_png, keep, width, height = outpaint_canvas(source, sides or [], area=area)
                (work / "init.png").write_bytes(canvas)
                (work / "mask.png").write_bytes(mask_png)
                (work / "keep.png").write_bytes(keep)
                args += ["-i", os.path.join(w, "init.png"), "--mask", os.path.join(w, "mask.png"), "--strength", "1.0"]
            elif mode == "inpaint":
                if mask is None:
                    raise ImageGenError("для дорисовки области нужна маска")
                width, height = fit_source(src_size, size)
                self._resized(work / "source.png", work / "init.png", (width, height))
                (work / "mask_in.png").write_bytes(mask)
                self._resized(work / "mask_in.png", work / "mask.png", (width, height), mask=True)
                args += ["-i", os.path.join(w, "init.png"), "--mask", os.path.join(w, "mask.png"),
                         "--strength", str(strength)]
            elif mode == "redraw":
                width, height = fit_source(src_size, size)
                self._resized(work / "source.png", work / "init.png", (width, height))
                args += ["-i", os.path.join(w, "init.png"), "--strength", str(strength)]
            elif mode == "edit":
                # the model recomposes an edit, so any chosen size goes; auto keeps the picture's proportions
                width, height = size if isinstance(size, tuple) else fit_source(src_size, size)
                args += ["-r", os.path.join(w, "source.png")]
            elif isinstance(size, tuple):
                width, height = size
            width, height = int(width or c.width), int(height or c.height)
            args += ["-W", str(width), "-H", str(height)]
            if c.unload_llm:
                unload_ollama(self.settings.llm.base_url)
            # a 2K picture takes ten times longer than a 1K one: the limit grows with the expected time
            limit = max(c.timeout_s, 3 * self.estimate(mode, width, height, steps))
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      timeout=limit, cwd=str(Path(c.sd_cli).parent))
            except subprocess.TimeoutExpired as exc:
                raise ImageGenError(f"генерация не уложилась в {limit:.0f} с") from exc
            elapsed = round(time.perf_counter() - t0, 1)
            produced = work / "out.png"
            if proc.returncode != 0 or not produced.exists():
                raise ImageGenError(sd_error(proc.returncode, proc.stdout or "", proc.stderr or ""))
            if mode == "outpaint":
                blend_back(produced, work / "init.png", work / "keep.png")
            out = out_dir / f"{image_id}.png"
            shutil.move(str(produced), out)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        info = GeneratedImage(id=image_id, dialog=dialog, mode=mode, prompt=prompt, request=request, width=width,
                              height=height, seed=seed, steps=steps, source=source_ref, model=Path(dit).name,
                              strength=strength, elapsed_s=elapsed,
                              created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        out.with_suffix(".json").write_text(info.model_dump_json(indent=1), encoding="utf-8")
        log.info("image %s (%s) in %.1f s", image_id, mode, elapsed)
        return info

    @staticmethod
    def _resized(src: Path, dst: Path, size: tuple[int, int], mask: bool = False) -> None:
        from PIL import Image

        with Image.open(src) as im:
            im = im.convert("L" if mask else "RGB").resize(size, Image.NEAREST if mask else Image.LANCZOS)
            if mask:  # anything painted counts: the brush may be semi-transparent or coloured
                im = im.point(lambda v: 255 if v > 16 else 0)
            im.save(dst, format="PNG")

    def delete_dialog(self, dialog: str) -> None:
        d = self._dir(dialog)
        if d.exists():
            for f in d.glob("*"):
                f.unlink(missing_ok=True)
            d.rmdir()


def describe_json(info: GeneratedImage) -> str:
    return json.dumps(info.model_dump(), ensure_ascii=False)
