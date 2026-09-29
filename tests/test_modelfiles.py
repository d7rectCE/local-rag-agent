"""models/: what a weights file is (GGUF metadata, names), the models of a folder with their companions, the
recommended settings of a family, the settings made in the UI."""

import struct
from pathlib import Path

import pytest

from rag_agent import modelfiles
from rag_agent.model_settings import ModelSettings, ModelSettingsError


def gguf(path: Path, arch: str) -> Path:
    """A GGUF file with only a header: general.architecture."""
    key, val = b"general.architecture", arch.encode()
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", len(key)) + key
                     + struct.pack("<I", 8) + struct.pack("<Q", len(val)) + val)
    return path


def touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"x")


def test_the_family_comes_from_the_gguf_header_or_the_name(tmp_path: Path):
    assert modelfiles.family(gguf(tmp_path / "anything.gguf", "krea2")) == "krea2"
    assert modelfiles.label(tmp_path / "anything.gguf") == "Krea 2 (anything)"
    touch(tmp_path, "qwen-image-2.1-Q6_K.safetensors")
    assert modelfiles.family(tmp_path / "qwen-image-2.1-Q6_K.safetensors") == "qwen_image21"
    # a model whose architecture is known is a diffusion model whatever its name says
    assert modelfiles.role(gguf(tmp_path / "qwen3vl-looking-name.gguf", "qwen_image21")) == "dit"
    assert modelfiles.role(gguf(tmp_path / "some-encoder.gguf", "gemma3")) == "llm"


@pytest.mark.parametrize("name, role", [
    ("qwen_image_vae.safetensors", "vae"), ("ltx-2.3-22b-dev_audio_vae.safetensors", "audio_vae"),
    ("ltx-2.3-22b-dev_embeddings_connectors.safetensors", "connectors"),
    ("qwen3vl_4b_fp8_scaled.safetensors", "llm"), ("gemma-3-12b-it-qat-UD-Q4_K_XL.gguf", "llm"),
    ("Qwen2.5-VL-7B-Instruct-Q8_0.gguf", "llm"), ("t5xxl_fp16.safetensors", "llm"),
    ("mmproj-model-f16.gguf", "vision"), ("Krea2_turbo_uncensored_edit_v1.1-Q4_K_M.gguf", "dit"),
])
def test_roles_by_name(tmp_path: Path, name: str, role: str):
    touch(tmp_path, name)
    assert modelfiles.role(tmp_path / name) == role


def test_a_folder_per_model_with_its_companions(tmp_path: Path):
    root = tmp_path / "models"
    touch(root / "image" / "krea2-turbo", "Krea2_turbo-Q4_K_M.gguf", "qwen_image_vae.safetensors",
          "qwen3vl_4b_fp8_scaled.safetensors")
    touch(root / "image" / "qwen", "qwen-image-2.1-Q6_K.gguf", "qwen_image_vae.safetensors",
          "qwen_image_2.1_vae_bf16.safetensors", "qwen3vl_8b_bf16.safetensors")
    touch(root / "video" / "ltx", "ltx-2.3-22b-dev-Q4_K_M.gguf", "ltx-2.3-22b-dev_video_vae.safetensors",
          "ltx-2.3-22b-dev_audio_vae.safetensors", "gemma-3-12b-it-Q4_K_M.gguf",
          "ltx-2.3-22b-dev_embeddings_connectors.safetensors")
    touch(root / "text", "my-chat-8b-Q4_K_M.gguf", "mmproj-f16.gguf")
    images = {m["name"]: m for m in modelfiles.scan(root, "image")}
    assert set(images) == {"Krea2_turbo-Q4_K_M.gguf", "qwen-image-2.1-Q6_K.gguf"}
    krea = images["Krea2_turbo-Q4_K_M.gguf"]
    assert Path(krea["vae"]).name == "qwen_image_vae.safetensors" and Path(krea["llm"]).name.startswith("qwen3vl_4b")
    # two VAEs in a folder: the one whose name is closer to the model's
    assert Path(images["qwen-image-2.1-Q6_K.gguf"]["vae"]).name == "qwen_image_2.1_vae_bf16.safetensors"
    [ltx] = modelfiles.scan(root, "video")
    assert Path(ltx["audio_vae"]).name.endswith("audio_vae.safetensors")
    assert Path(ltx["vae"]).name.endswith("video_vae.safetensors") and Path(ltx["llm"]).name.startswith("gemma")
    assert Path(ltx["connectors"]).name.endswith("connectors.safetensors") and ltx["family"] == "ltx2"
    assert [m["name"] for m in modelfiles.scan(root, "text")] == ["my-chat-8b-Q4_K_M.gguf"]
    assert modelfiles.scan(tmp_path / "nowhere", "image") == []


def test_recommended_settings_by_family(tmp_path: Path):
    touch(tmp_path, "Krea2_turbo-Q4_K_M.gguf", "krea2-base-Q8_0.gguf", "ltx-2.5-22b-Q8_0.gguf")
    turbo = modelfiles.preset(tmp_path / "Krea2_turbo-Q4_K_M.gguf", llm="qwen3vl_4b_fp8_scaled.safetensors")
    assert (turbo["steps"], turbo["cfg_scale"], turbo["scheduler"]) == (8, 1.0, "simple")
    assert turbo["text_encoder_on_cpu"] is False  # the CPU backend of sd.cpp crashes on fp8 weights
    base = modelfiles.preset(tmp_path / "krea2-base-Q8_0.gguf", llm="qwen3vl_4b_bf16.safetensors")
    assert base["steps"] > 8 and base["cfg_scale"] > 1 and "text_encoder_on_cpu" not in base
    ltx = modelfiles.preset(tmp_path / "ltx-2.5-22b-Q8_0.gguf")
    assert (ltx["cfg_scale"], ltx["frames"], ltx["fps"]) == (3.0, 121, 24)


def test_settings_made_in_the_ui(tmp_path: Path):
    store = ModelSettings(tmp_path / "model_settings.json")
    assert store.get("image", "m.gguf") == {}
    assert store.set("image", "m.gguf", {"steps": "12", "cfg_scale": 2, "text_encoder_on_cpu": False}) == {
        "steps": 12, "cfg_scale": 2.0, "text_encoder_on_cpu": False}
    assert store.set("image", "m.gguf", {"steps": None}) == {"cfg_scale": 2.0, "text_encoder_on_cpu": False}
    store.set("text", "qwen3.5:9b", {"temperature": 0.7})
    assert ModelSettings(tmp_path / "model_settings.json").get("text", "qwen3.5:9b") == {"temperature": 0.7}
    with pytest.raises(ModelSettingsError):
        store.set("image", "m.gguf", {"temperature": 1})  # not a setting of a picture model
    with pytest.raises(ModelSettingsError):
        store.set("image", "m.gguf", {"steps": "много"})
    store.reset("image", "m.gguf")
    assert store.get("image", "m.gguf") == {}
