"""Settings of each model made in the UI ("Настройки" panel): sampling of the chat models, of the picture and
video generators. Kept in data_dir/model_settings.json, over the config and the family presets."""

from __future__ import annotations

import json
import threading
from pathlib import Path

# what the UI may set, by kind of model, with the type of each value
KEYS: dict[str, dict[str, type]] = {
    "text": {"temperature": float, "top_p": float, "top_k": int, "repeat_penalty": float, "num_ctx": int,
             "max_tokens": int},
    "image": {"steps": int, "cfg_scale": float, "sampler": str, "scheduler": str, "negative_prompt": str,
              "strength": float, "text_encoder_on_cpu": bool, "flash_attention": bool, "vae_tiling": bool},
    "video": {"steps": int, "cfg_scale": float, "sampler": str, "scheduler": str, "negative_prompt": str,
              "frames": int, "fps": int, "width": int, "height": int, "text_encoder_on_cpu": bool,
              "flash_attention": bool, "vae_tiling": bool, "offload_to_cpu": bool},
}
_LOCK = threading.Lock()


class ModelSettingsError(ValueError):
    pass


class ModelSettings:
    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def get(self, kind: str, name: str | None) -> dict:
        return dict(self._read().get(kind, {}).get(name or "", {})) if name else {}

    def set(self, kind: str, name: str, values: dict) -> dict:
        """Merge values in; None (or "") drops a key back to the recommended one. Returns what is stored."""
        if kind not in KEYS or not name:
            raise ModelSettingsError(f"неизвестный вид модели {kind}")
        clean = {}
        for key, value in values.items():
            typ = KEYS[kind].get(key)
            if typ is None:
                raise ModelSettingsError(f"настройка {key} не поддерживается для {kind}")
            if value is None or value == "":
                clean[key] = None
                continue
            try:
                clean[key] = typ(value) if typ is not bool else bool(value)
            except (TypeError, ValueError) as exc:
                raise ModelSettingsError(f"{key}: неверное значение {value!r}") from exc
        with _LOCK:
            data = self._read()
            cur = data.setdefault(kind, {}).setdefault(name, {})
            for key, value in clean.items():
                if value is None:
                    cur.pop(key, None)
                else:
                    cur[key] = value
            if not cur:
                data[kind].pop(name, None)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            return dict(cur)

    def reset(self, kind: str, name: str) -> None:
        with _LOCK:
            data = self._read()
            if data.get(kind, {}).pop(name, None) is not None:
                self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
