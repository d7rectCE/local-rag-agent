"""Image generation and editing in the chat: Qwen-Image-2.1 through stable-diffusion.cpp.

Modes (the model is one for generation and editing):

    generate — text -> image;
    edit     — an image and an instruction -> the edited image (the image is a reference, ``-r``);
    redraw   — img2img: the image is the starting point, ``strength`` sets how much changes;
    inpaint  — only the area of a mask is redrawn (white = redraw);
    outpaint — the canvas is extended on the chosen sides and the new area is drawn (inpaint of the margin).

The user's request is rewritten by the chat LLM into a prompt for the generator (English is what the
model follows best; text to be drawn stays as written) and checked: sexual content involving minors is
refused before anything runs. The engine is a subprocess of ``sd-cli`` per image: the ~22 GB of weights
live in RAM only while an image is made, and the chat LLM is unloaded from the GPU first (they do not
fit together in 24 GB). Results are stored per dialog under ``data_dir/generated/<dialog>/``.

The feature is off by default (``imagegen.enabled``): the weights are the user's local choice and are not
part of the project (the Qwen-Image-2.1 builds come under the Qwen Research License).
"""

from __future__ import annotations

import io
import json
import logging
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import BaseModel

from rag_agent.llm import BaseLLM, LLMError

log = logging.getLogger(__name__)

MODES = ("generate", "edit", "redraw", "inpaint", "outpaint")
ASPECTS = {"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832), "wide": (1344, 768),
           "tall": (768, 1344)}
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
- action: "none" — картинку рисовать не нужно (вопрос, объяснение, просьба о коде); "generate" — нарисовать новую; "edit" — изменить имеющуюся картинку по инструкции (заменить фон, стиль, добавить или убрать объект, перерисовать в другом стиле); "redraw" — перерисовать картинку целиком, сохранив композицию; "outpaint" — дорисовать картинку за её краями (расширить);
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
    elapsed_s: float = 0.0
    created_at: str = ""

    @property
    def url(self) -> str:
        return f"/images/{self.dialog}/{self.id}.png"


class ImageGenError(RuntimeError):
    pass


def wants_image(question: str) -> bool:
    return bool(IMAGE_HINT.search(question or ""))


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


def outpaint_canvas(image: bytes, sides: list[str], share: float = 0.25) -> tuple[bytes, bytes, int, int]:
    """The image on a larger canvas and the mask of the new margin (white = draw). Sizes are multiples of 32."""
    from PIL import Image, ImageFilter

    with Image.open(io.BytesIO(image)) as im:
        im = im.convert("RGB")
        w, h = im.size
        sides = sides or ["left", "right"]
        dx, dy = int(w * share), int(h * share)
        left, right = (dx if "left" in sides else 0), (dx if "right" in sides else 0)
        top, bottom = (dy if "top" in sides else 0), (dy if "bottom" in sides else 0)
        W, H = w + left + right, h + top + bottom
        W32, H32 = max(32, W // 32 * 32), max(32, H // 32 * 32)
        canvas = Image.new("RGB", (W, H), (127, 127, 127))
        canvas.paste(im, (left, top))
        # the margin starts from stretched edge colours, which helps the model continue the scene
        edge = im.resize((W, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(24))
        mask = Image.new("L", (W, H), 255)
        mask.paste(0, (left, top, left + w, top + h))
        canvas = Image.composite(edge, canvas, mask)
        canvas, mask = canvas.resize((W32, H32), Image.LANCZOS), mask.resize((W32, H32), Image.NEAREST)
        # a soft seam: the mask reaches a few pixels into the old image
        mask = mask.filter(ImageFilter.MaxFilter(9))
    return _png(canvas), _png(mask), W32, H32


def _png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def fit32(size: tuple[int, int], limit: int = 1344) -> tuple[int, int]:
    """A generation size near the source's, within the model's range and divisible by 32."""
    w, h = size
    scale = min(1.0, limit / max(w, h))
    if max(w, h) * scale < 512:
        scale = 512 / max(w, h)
    return max(256, int(w * scale) // 32 * 32), max(256, int(h * scale) // 32 * 32)


class ImageGenerator:
    def __init__(self, settings):
        self.settings, self.cfg = settings, settings.imagegen
        self.root = settings.data_dir / "generated"

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
            seed: int | None = None, steps: int | None = None) -> GeneratedImage:
        if mode not in MODES:
            raise ImageGenError(f"неизвестный режим {mode}")
        if not self.available:
            raise ImageGenError("генерация картинок не настроена: " + ", ".join(self.missing()))
        if mode != "generate" and source is None:
            raise ImageGenError("для этого режима нужна картинка-источник")
        c = self.cfg
        out_dir = self._dir(dialog)
        out_dir.mkdir(parents=True, exist_ok=True)
        image_id = uuid.uuid4().hex[:12]
        work = out_dir / f".{image_id}"
        work.mkdir()
        seed = int(seed if seed is not None else uuid.uuid4().int % 2**31)
        steps = int(steps or c.steps)
        args = [c.sd_cli, "--diffusion-model", c.diffusion_model, "--vae", c.vae, "--llm", c.llm,
                "--cfg-scale", str(c.cfg_scale), "--sampling-method", c.sampler, "--steps", str(steps),
                "-s", str(seed), "-p", prompt, "-o", str(out_dir / f"{image_id}.png")]
        if c.negative_prompt:
            args += ["-n", c.negative_prompt]
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
            if mode == "outpaint":
                canvas, mask_png, width, height = outpaint_canvas(source, sides or [])
                (work / "init.png").write_bytes(canvas)
                (work / "mask.png").write_bytes(mask_png)
                args += ["-i", str(work / "init.png"), "--mask", str(work / "mask.png"), "--strength", "1.0"]
            elif mode == "inpaint":
                if mask is None:
                    raise ImageGenError("для дорисовки области нужна маска")
                width, height = fit32(src_size)
                self._resized(work / "source.png", work / "init.png", (width, height))
                (work / "mask_in.png").write_bytes(mask)
                self._resized(work / "mask_in.png", work / "mask.png", (width, height), mask=True)
                args += ["-i", str(work / "init.png"), "--mask", str(work / "mask.png"),
                         "--strength", str(strength if strength is not None else 1.0)]
            elif mode == "redraw":
                width, height = fit32(src_size)
                self._resized(work / "source.png", work / "init.png", (width, height))
                args += ["-i", str(work / "init.png"), "--strength", str(strength if strength is not None else c.strength)]
            elif mode == "edit":
                width, height = width or fit32(src_size)[0], height or fit32(src_size)[1]
                args += ["-r", str(work / "source.png")]
            width, height = int(width or c.width), int(height or c.height)
            args += ["-W", str(width), "-H", str(height)]
            if c.unload_llm:
                unload_ollama(self.settings.llm.base_url)
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      timeout=c.timeout_s, cwd=str(Path(c.sd_cli).parent))
            except subprocess.TimeoutExpired as exc:
                raise ImageGenError(f"генерация не уложилась в {c.timeout_s:.0f} с") from exc
            elapsed = round(time.perf_counter() - t0, 1)
            out = out_dir / f"{image_id}.png"
            if proc.returncode != 0 or not out.exists():
                tail = (proc.stderr or proc.stdout or "")[-600:]
                raise ImageGenError(f"движок завершился с ошибкой ({proc.returncode}): {tail}")
        finally:
            for f in work.glob("*"):
                f.unlink(missing_ok=True)
            work.rmdir()
        info = GeneratedImage(id=image_id, dialog=dialog, mode=mode, prompt=prompt, request=request, width=width,
                              height=height, seed=seed, steps=steps, source=source_ref, elapsed_s=elapsed,
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
