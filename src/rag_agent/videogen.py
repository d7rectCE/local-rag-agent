"""Video generation in the chat: LTX-2 (video with sound) through stable-diffusion.cpp's sd-cli (-M vid_gen).

Text to video, or a picture to video (an attached photo or a picture drawn in the dialog becomes the first
frame). As for pictures, the chat model writes the generator's prompt (English, what moves and what is heard)
and the request is refused before anything runs if it is sexual content involving minors. Models live in
models/video, a folder per model: the diffusion model, the video VAE, the audio VAE, the text encoder
(Gemma) and the embeddings connectors (LTX-2.3), found by modelfiles.scan; settings: the videogen section,
the family preset, the ones made in the UI. Videos are stored per dialog under data_dir/videos/<dialog>/.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from rag_agent import modelfiles
from rag_agent.imagegen import (MINOR_TERMS, SEXUAL_TERMS, _ID, _SESSION, ascii_workdir, cli_path, sd_error, short_path,
                                snap32, unload_ollama)
from rag_agent.llm import BaseLLM, LLMError
from rag_agent.model_settings import ModelSettings

log = logging.getLogger(__name__)

VIDEO_HINT = re.compile(r"видео|ролик|клип\b|анимир|анимаци|оживи|оживить|\b(video|animate|animation|clip)\b",
                        re.IGNORECASE)
ASPECTS = {"landscape": (16, 9), "portrait": (9, 16), "square": (1, 1)}

PLAN_PROMPT = """Ты готовишь запрос к генератору видео {label} (ролик со звуком на несколько секунд).
По просьбе пользователя (и истории диалога) реши:
- action: "none" — видео не нужно (вопрос, объяснение, картинка без движения); "text" — снять ролик по описанию; "image" — оживить картинку-источник (прикреплённую или последнюю нарисованную): она станет первым кадром — выбирай это, если просят анимировать, оживить, «как она едет» и т. п. про уже имеющуюся картинку;
- prompt: подробный промпт на английском: что в кадре, как движутся объекты и камера, свет, стиль; в конце — что слышно (звук мотора, ветер, музыка, речь с текстом в кавычках);
- seconds: длительность в секундах, 1–10; 0 — если пользователь не сказал;
- aspect: "landscape", "portrait" или "square";
- minor_sexual: true, если просьба предполагает сексуальный или эротический контент с участием несовершеннолетних или людей, выглядящих несовершеннолетними; иначе false.
Картинка-источник: {source}.
Верни JSON: {{"action": "...", "prompt": "...", "seconds": 0, "aspect": "...", "minor_sexual": false}}"""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["none", "text", "image"]},
        "prompt": {"type": "string"},
        "seconds": {"type": "number"},
        "aspect": {"type": "string", "enum": list(ASPECTS)},
        "minor_sexual": {"type": "boolean"},
    },
    "required": ["action", "prompt", "seconds", "aspect", "minor_sexual"],
}


class VideoPlan(BaseModel):
    action: str = "none"
    prompt: str = ""
    seconds: float = 0.0
    aspect: str = "landscape"
    refused: str | None = None


class GeneratedVideo(BaseModel):
    id: str
    dialog: str
    mode: str  # text | image (the first frame given)
    prompt: str
    request: str = ""
    width: int
    height: int
    frames: int
    fps: int
    seed: int
    steps: int
    model: str = ""
    source: str | None = None  # "upload:<id>" or "generated:<id>" for image-to-video
    elapsed_s: float = 0.0
    created_at: str = ""

    @property
    def url(self) -> str:
        return f"/videos/{self.dialog}/{self.id}.webm"

    @property
    def seconds(self) -> float:
        return round(self.frames / max(1, self.fps), 1)


class VideoGenError(RuntimeError):
    pass


def wants_video(question: str) -> bool:
    return bool(VIDEO_HINT.search(question or ""))


def plan_video(question: str, history: list[dict], llm: BaseLLM, has_source: bool, label: str) -> VideoPlan:
    """The chat LLM decides whether to make a video and writes the generator's prompt."""
    dialogue = "\n".join(f"{'Пользователь' if t['role'] == 'user' else 'Ассистент'}: {t['content'][:600]}"
                         for t in history[-6:])
    system = PLAN_PROMPT.format(label=label, source="есть" if has_source else "нет")
    user = f"История:\n{dialogue or 'пусто'}\n\nПросьба: {question}"
    try:
        data = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                        json_schema=PLAN_SCHEMA, max_tokens=700, purpose="video_plan").json()
    except LLMError:
        return VideoPlan()
    try:
        seconds = float(data.get("seconds") or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    plan = VideoPlan(action=str(data.get("action") or "none"), prompt=str(data.get("prompt") or "").strip(),
                     seconds=max(0.0, min(seconds, 10.0)), aspect=str(data.get("aspect") or "landscape"))
    if plan.action == "image" and not has_source:
        plan.action = "text"
    text = f"{question}\n{plan.prompt}"
    if data.get("minor_sexual") or (MINOR_TERMS.search(text) and SEXUAL_TERMS.search(text)):
        plan.refused = "Не могу создать такое видео: сексуальный контент с участием несовершеннолетних запрещён."
    return plan


def frames_for(seconds: float, fps: int, default: int) -> int:
    """Frames of a clip: LTX takes 8k + 1 of them (33, 65, 97, 121...)."""
    if seconds <= 0:
        return default
    return max(9, min(241, round((seconds * fps - 1) / 8) * 8 + 1))


class VideoGenerator:
    SETTING_KEYS = ("steps", "cfg_scale", "sampler", "scheduler", "negative_prompt", "frames", "fps", "width",
                    "height", "text_encoder_on_cpu", "flash_attention", "vae_tiling", "offload_to_cpu")

    def __init__(self, settings):
        self.settings, self.cfg = settings, settings.videogen
        self.root = settings.data_dir / "videos"
        self.store = ModelSettings(settings.data_dir / "model_settings.json")

    @property
    def sd_cli(self) -> str:
        return self.cfg.sd_cli or self.settings.imagegen.sd_cli

    def catalog(self) -> dict[str, dict]:
        """name -> {path, vae, llm, audio_vae, connectors, source}: videogen.diffusion_model and models/video."""
        c = self.cfg
        out: dict[str, dict] = {}
        if c.diffusion_model:
            out[Path(c.diffusion_model).name] = {"path": c.diffusion_model, "vae": c.vae, "llm": c.llm,
                                                 "audio_vae": c.audio_vae, "connectors": c.connectors, "source": "config"}
        for m in modelfiles.scan(self.settings.models_dir, "video"):
            out.setdefault(m["name"], {k: m[k] for k in ("path", "vae", "llm", "audio_vae", "connectors")}
                           | {"source": "models"})
        return out

    def default_name(self) -> str:
        return Path(self.cfg.diffusion_model).name if self.cfg.diffusion_model else next(iter(self.catalog()), "")

    @staticmethod
    def _complete(entry: dict) -> bool:
        return all(entry.get(k) and Path(entry[k]).exists() for k in ("path", "vae", "llm"))

    @property
    def available(self) -> bool:
        entry = self.catalog().get(self.default_name())
        return bool(self.cfg.enabled and self.sd_cli and Path(self.sd_cli).exists() and entry and self._complete(entry))

    def missing(self) -> list[str]:
        if not self.cfg.enabled:
            return ["videogen.enabled"]
        out = [] if self.sd_cli and Path(self.sd_cli).exists() else ["sd_cli"]
        entry = self.catalog().get(self.default_name())
        if not entry:
            return out + ["модель в models/video"]
        return out + [k for k in ("path", "vae", "llm") if not entry.get(k) or not Path(entry[k]).exists()]

    def recommended(self, name: str) -> dict:
        entry = self.catalog().get(name) or {}
        out = {k: getattr(self.cfg, k) for k in self.SETTING_KEYS}
        if entry:
            out.update({k: v for k, v in modelfiles.preset(entry["path"], entry.get("llm", "")).items()
                        if k in self.SETTING_KEYS})
        return out

    def profile(self, name: str) -> dict:
        entry = self.catalog().get(name) or {}
        return {**self.recommended(name), **self.store.get("video", name),
                **{k: entry.get(k, "") for k in ("vae", "llm", "audio_vae", "connectors")}}

    def models(self) -> list[dict]:
        default = self.default_name()
        out = []
        for name, e in self.catalog().items():
            p = Path(e["path"])
            prof = self.profile(name)
            out.append({"name": name, "size": p.stat().st_size if p.exists() else 0, "default": name == default,
                        "label": modelfiles.label(p), "family": modelfiles.family(p), "source": e["source"],
                        "complete": self._complete(e), "audio": bool(e.get("audio_vae")),
                        "steps": prof["steps"], "cfg_scale": prof["cfg_scale"]})
        return out

    def model_path(self, name: str | None) -> str:
        entry = self.catalog().get(name or self.default_name())
        if entry is None or not Path(entry["path"]).exists():
            raise VideoGenError(f"видеомодель «{name}» не найдена в папке моделей")
        return entry["path"]

    # --- storage ---------------------------------------------------------------------------------
    def _dir(self, dialog: str) -> Path:
        if not _SESSION.match(dialog or ""):
            raise VideoGenError("неверный идентификатор диалога")
        return self.root / dialog

    def path(self, dialog: str, video_id: str) -> Path:
        if not _ID.match(video_id or ""):
            raise VideoGenError("неверный идентификатор видео")
        return self._dir(dialog) / f"{video_id}.webm"

    def last(self, dialog: str) -> GeneratedVideo | None:
        d = self._dir(dialog)
        metas = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime) if d.exists() else []
        return GeneratedVideo.model_validate_json(metas[-1].read_text(encoding="utf-8")) if metas else None

    def delete_dialog(self, dialog: str) -> None:
        d = self._dir(dialog)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    # --- running ---------------------------------------------------------------------------------
    def size(self, prof: dict, aspect: str, source_size: tuple[int, int] | None) -> tuple[int, int]:
        """The frame: the settings' pixel count in the plan's proportions, or in the source picture's."""
        area = int(prof["width"]) * int(prof["height"])
        if source_size:
            w, h = source_size
        else:
            w, h = ASPECTS.get(aspect, ASPECTS["landscape"])
        k = (area / (w * h)) ** 0.5
        return snap32(w * k, h * k)

    def run(self, dialog: str, prompt: str, *, request: str = "", source: bytes | None = None,
            source_ref: str | None = None, seconds: float = 0.0, aspect: str = "landscape", seed: int | None = None,
            model: str | None = None) -> GeneratedVideo:
        if not self.available:
            raise VideoGenError("генерация видео не настроена: " + ", ".join(self.missing()))
        dit = self.model_path(model)
        prof = self.profile(Path(dit).name)
        for part in ("vae", "llm"):
            if not prof[part] or not Path(prof[part]).exists():
                raise VideoGenError(f"у модели {Path(dit).name} не найден файл {part}")
        out_dir = self._dir(dialog)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_id = uuid.uuid4().hex[:12]
        work = ascii_workdir(self.settings.imagegen.work_dir) / video_id
        work.mkdir()
        w = short_path(work)
        seed = int(seed if seed is not None else uuid.uuid4().int % 2**31)
        fps = int(prof["fps"])
        frames = frames_for(seconds, fps, int(prof["frames"]))
        src_size = None
        try:
            if source is not None:
                from PIL import Image

                with Image.open(io.BytesIO(source)) as im:
                    src_size = im.size
            width, height = self.size(prof, aspect, src_size)
            args = [self.sd_cli, "-M", "vid_gen", "--diffusion-model", cli_path(dit), "--vae", cli_path(prof["vae"]),
                    "--llm", cli_path(prof["llm"]),
                    "--cfg-scale", str(prof["cfg_scale"]), "--sampling-method", prof["sampler"],
                    "--steps", str(int(prof["steps"])), "-W", str(width), "-H", str(height),
                    "--video-frames", str(frames), "--fps", str(fps), "-s", str(seed), "-p", prompt,
                    "-o", os.path.join(w, "out.webm")]
            if prof.get("audio_vae"):
                args += ["--audio-vae", cli_path(prof["audio_vae"])]
            if prof.get("connectors"):
                args += ["--embeddings-connectors", cli_path(prof["connectors"])]
            if prof["scheduler"]:
                args += ["--scheduler", prof["scheduler"]]
            if prof["negative_prompt"]:
                args += ["-n", prof["negative_prompt"]]
            if prof["text_encoder_on_cpu"]:
                args += ["--backend", "te=cpu"]
            if prof["offload_to_cpu"]:
                args.append("--offload-to-cpu")
            if prof["flash_attention"]:
                args.append("--fa")
            if prof["vae_tiling"]:
                args += ["--vae-tiling", "--temporal-tiling"]
            if source is not None:
                from PIL import Image

                with Image.open(io.BytesIO(source)) as im:
                    im.convert("RGB").resize((width, height), Image.LANCZOS).save(work / "first.png", format="PNG")
                args += ["-i", os.path.join(w, "first.png")]
            if self.settings.imagegen.unload_llm:
                unload_ollama(self.settings.llm.base_url)
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      timeout=self.cfg.timeout_s, cwd=str(Path(self.sd_cli).parent))
            except subprocess.TimeoutExpired as exc:
                raise VideoGenError(f"видео не уложилось в {self.cfg.timeout_s:.0f} с") from exc
            elapsed = round(time.perf_counter() - t0, 1)
            produced = work / "out.webm"
            if proc.returncode != 0 or not produced.exists():
                raise VideoGenError(sd_error(proc.returncode, proc.stdout or "", proc.stderr or ""))
            out = out_dir / f"{video_id}.webm"
            shutil.move(str(produced), out)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        info = GeneratedVideo(id=video_id, dialog=dialog, mode="image" if source is not None else "text", prompt=prompt,
                              request=request, width=width, height=height, frames=frames, fps=fps, seed=seed,
                              steps=int(prof["steps"]), model=Path(dit).name, source=source_ref, elapsed_s=elapsed,
                              created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        out.with_suffix(".json").write_text(info.model_dump_json(indent=1), encoding="utf-8")
        log.info("video %s (%s) in %.1f s", video_id, info.mode, elapsed)
        return info


__all__ = ["GeneratedVideo", "VideoGenError", "VideoGenerator", "VideoPlan", "frames_for", "plan_video", "wants_video"]
