# Модели

Сюда складываются веса моделей: приложение находит их само, в git они не попадают (`.gitignore`).

```
models/
  text/    чат-модели GGUF: появляются в списке моделей чата, импорт в Ollama — кнопкой
  image/   генераторы картинок: папка на модель с её VAE и текстовым энкодером
  video/   генераторы видео: папка на модель (для LTX-2 — ещё аудио-VAE и коннекторы эмбеддингов)
```

Пример:

```
models/image/krea2-turbo/
  Krea2_turbo_uncensored_edit_v1.1-Q4_K_M.gguf   диффузионная модель
  qwen_image_vae.safetensors                     VAE
  qwen3vl_4b_fp8_scaled.safetensors              текстовый энкодер
models/video/ltx-2.3/
  ltx-2.3-22b-dev-Q4_K_M.gguf                    диффузионная модель
  ltx-2.3-22b-dev_video_vae.safetensors          VAE видео
  ltx-2.3-22b-dev_audio_vae.safetensors          VAE звука
  gemma-3-12b-it-Q4_K_M.gguf                     текстовый энкодер
  ltx-2.3-22b-dev_embeddings_connectors.safetensors
```

Как файлы узнаются (`src/rag_agent/modelfiles.py`): семейство — по метаданным GGUF (`general.architecture`: `qwen_image21`, `krea2`, `ltx2`…) или по имени; `vae` в имени — VAE, `audio_vae` — VAE звука, `connector` — коннекторы, `qwen…vl`, `gemma`, `t5`, `clip` или текстовая архитектура GGUF — энкодер. Если в папке несколько кандидатов, берётся тот, чьё имя ближе к модели.

Настройки по умолчанию — по семейству: turbo, lightning или distilled сборки — 8 шагов, CFG 1, scheduler `simple`; энкодер в fp8 считается на видеокарте. Их меняют в панели «Настройки» справа, для каждой модели отдельно.

Папку можно перенести: `storage.models_dir` в `configs/local.yaml`.
