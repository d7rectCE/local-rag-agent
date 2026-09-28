"""Pictures in the chat: the plan of the chat LLM, the sd-cli call for every mode (stubbed), the source of an
edit (an attached image or the last generated one), outpaint canvases, the refusal, image uploads, the API."""

import base64
import io
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rag_agent import imagegen
from rag_agent.api import create_app
from rag_agent.dialogs import DialogStore
from rag_agent.engine import Engine
from rag_agent.imagegen import outpaint_canvas, wants_edit, wants_image
from tests.conftest import FakeLLM


def png(size=(640, 480), color=(40, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeSd:
    """Stands in for sd-cli: records the arguments and writes a picture of the requested size."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.inputs: dict[str, bytes] = {}
        self.real = subprocess.run

    def __call__(self, args, **kwargs):
        if not str(args[0]).endswith("sd-cli.exe"):  # docker checks and the like
            return self.real(args, **kwargs)
        self.calls.append(list(args))
        opt = {args[i]: args[i + 1] for i in range(1, len(args) - 1) if args[i].startswith("-")}
        for flag in ("-i", "--mask", "-r"):
            if flag in opt:
                self.inputs[flag] = Path(opt[flag]).read_bytes()
        Image.new("RGB", (int(opt["-W"]), int(opt["-H"])), (200, 60, 60)).save(opt["-o"])

        class Done:
            returncode, stdout, stderr = 0, "", ""
        return Done()


@pytest.fixture
def gen_engine(settings, fake_embedder, corpus, tmp_path, monkeypatch):
    for name in ("sd-cli.exe", "dit.gguf", "vae.safetensors", "te.safetensors"):
        (tmp_path / name).write_bytes(b"x")
    g = settings.imagegen
    g.enabled, g.sd_cli, g.diffusion_model = True, str(tmp_path / "sd-cli.exe"), str(tmp_path / "dit.gguf")
    g.vae, g.llm, g.unload_llm = str(tmp_path / "vae.safetensors"), str(tmp_path / "te.safetensors"), False
    fake = FakeSd()
    monkeypatch.setattr(imagegen.subprocess, "run", fake)
    llm = FakeLLM(settings, image_plan={"action": "generate", "prompt": "a red fox in the snow, photo",
                                        "aspect": "landscape", "sides": [], "minor_sexual": False})
    eng = Engine(settings, embedder=fake_embedder, llm=llm)
    eng.index_folder(corpus)
    eng.fake_sd = fake
    yield eng
    eng.close()


def test_request_words():
    assert wants_image("Нарисуй лису в снегу") and wants_image("перерисуй это в стиле акварели")
    assert wants_image("Сгенерируй картинку с котом") and wants_image("draw a cat")
    assert not wants_image("Какой learning rate в эксперименте?") and not wants_image("Напиши функцию сортировки")
    # words of a change: they count only when the dialog has a picture (see below)
    assert wants_edit("Сделай её в стиле акварели") and wants_edit("убери фон") and wants_edit("make it brighter")
    assert not wants_edit("Какой фонд выбрать?") and not wants_edit("Напиши функцию сортировки")


def test_a_change_counts_only_with_a_picture_in_the_dialog(gen_engine):
    gen_engine.llm.image_plan = {**gen_engine.llm.image_plan, "action": "edit", "prompt": "watercolor painting"}
    ans = gen_engine.ask("Сделай её в стиле акварели", session="d5")
    assert ans.route != "image" and "image_plan" not in gen_engine.llm.kinds
    gen_engine.llm.image_plan = {**gen_engine.llm.image_plan, "action": "generate"}
    first = gen_engine.ask("Нарисуй лису", session="d5").image
    gen_engine.llm.image_plan = {**gen_engine.llm.image_plan, "action": "edit"}
    ans = gen_engine.ask("Сделай её в стиле акварели", session="d5")
    assert ans.route == "image" and ans.image["mode"] == "edit" and ans.image["source"] == f"generated:{first['id']}"


def test_a_picture_in_the_history_is_a_note(tmp_path):
    """The chat model sees what the generator did, not a reply to copy without drawing anything."""
    store = DialogStore(tmp_path / "d.sqlite")
    d = store.create()
    store.add_turn(d["id"], "assistant", "answer", "Перерисовал картинку: 512×320, 27 с.", payload={
        "answer": "Перерисовал картинку: 512×320, 27 с.",
        "image": {"mode": "redraw", "width": 512, "height": 320, "prompt": "a cat, watercolor"}})
    note = store.history(d["id"])[-1]["content"]
    assert note.startswith("[Генератор картинок перерисовал картинку 512×320") and "a cat, watercolor" in note
    assert "Перерисовал картинку:" not in note


def test_generate_from_the_chat(gen_engine, settings):
    ans = gen_engine.ask("Нарисуй лису в снегу", session="d1")
    assert ans.route == "image" and ans.image and ans.image["mode"] == "generate"
    assert (ans.image["width"], ans.image["height"]) == (1216, 832)  # "landscape"
    assert ans.image["url"] == f"/images/d1/{ans.image['id']}.png"
    args = gen_engine.fake_sd.calls[-1]
    assert args[args.index("-p") + 1] == "a red fox in the snow, photo" and "--vae-tiling" in args and "-r" not in args
    assert [s.name for s in ans.trace][-2:] == ["image_plan", "image_gen"]
    assert gen_engine.imagegen.path("d1", ans.image["id"]).exists()


def test_edit_uses_the_last_picture_or_an_attached_one(gen_engine):
    first = gen_engine.ask("Нарисуй лису", session="d1").image
    gen_engine.llm.image_plan = {**gen_engine.llm.image_plan, "action": "edit", "prompt": "make the background a night sky"}
    ans = gen_engine.ask("Перерисуй фон ночным", session="d1")
    assert ans.image["mode"] == "edit" and ans.image["source"] == f"generated:{first['id']}"
    assert "-r" in gen_engine.fake_sd.calls[-1] and "--params-backend" in gen_engine.fake_sd.calls[-1]
    # an attached photo wins over the last generated picture
    info = gen_engine.upload("d1", "photo.png", png(color=(1, 2, 3)))
    assert info.file_type == ".png" and info.n_fragments == 1
    ans = gen_engine.ask("Перерисуй в стиле акварели", uploads=[info.id], session="d1", upload_fallback=True)
    assert ans.image["source"] == f"upload:{info.id}"
    with Image.open(io.BytesIO(gen_engine.fake_sd.inputs["-r"])) as im:
        assert im.getpixel((5, 5)) == (1, 2, 3)


def test_inpaint_with_a_mask_and_outpaint(gen_engine):
    src = gen_engine.ask("Нарисуй лису", session="d2").image
    mask = Image.new("L", (src["width"], src["height"]), 0)
    mask.paste(180, (10, 10, 200, 200))  # a soft brush stroke still counts
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    request = {"mode": "inpaint", "source": f"generated:{src['id']}",
               "mask": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}
    ans = gen_engine.ask("Нарисуй здесь красный зонт", session="d2", image=request)
    assert ans.image["mode"] == "inpaint" and "--mask" in gen_engine.fake_sd.calls[-1]
    with Image.open(io.BytesIO(gen_engine.fake_sd.inputs["--mask"])) as m:
        assert m.getpixel((50, 50)) == 255 and m.getpixel((m.size[0] - 5, m.size[1] - 5)) == 0
    ans = gen_engine.ask("Расширь", session="d2", image={"mode": "outpaint", "source": f"generated:{src['id']}",
                                                        "sides": ["left", "right"]})
    assert ans.image["mode"] == "outpaint" and ans.image["width"] > src["width"] and ans.image["width"] % 32 == 0
    assert ans.image["height"] == src["height"] // 32 * 32


def test_outpaint_canvas_and_mask():
    canvas, mask, keep, w, h = outpaint_canvas(png((400, 300)), ["top"])
    assert (w, h) == (384, 352)  # 400 x 375 -> multiples of 32
    with Image.open(io.BytesIO(mask)) as m, Image.open(io.BytesIO(keep)) as k:
        assert m.size == k.size == (w, h)
        assert m.getpixel((w // 2, 2)) == 255 and m.getpixel((w // 2, h - 3)) == 0
        # the old image: gone in the new margin, kept at the far edge, fading in across the band at the seam
        assert k.getpixel((w // 2, 2)) == 0 and k.getpixel((w // 2, h - 3)) == 255
        seam = 75 * h // 375
        assert m.getpixel((w // 2, seam + 10)) == 255 and 0 < k.getpixel((w // 2, seam + 10)) < 255


def test_refusal_and_not_a_picture(gen_engine):
    gen_engine.llm.image_plan = {**gen_engine.llm.image_plan, "minor_sexual": True}
    ans = gen_engine.ask("нарисуй …", session="d3")
    assert ans.route == "image" and not ans.answerable and not ans.image and gen_engine.fake_sd.calls == []
    gen_engine.llm.image_plan = {**gen_engine.llm.image_plan, "minor_sexual": False, "action": "none"}
    ans = gen_engine.ask("Нарисуй мне план статьи списком", session="d3")  # the plan says it is not a picture
    assert ans.route != "image" and gen_engine.fake_sd.calls == []


def test_not_configured(settings, fake_embedder, corpus):
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    eng.index_folder(corpus)
    assert not eng.imagegen.available and eng.imagegen.missing() == ["imagegen.enabled"]
    assert eng.ask("Нарисуй лису", session="d4").route != "image"  # off: the chat answers as before
    eng.close()


def test_api_serves_and_deletes_pictures(gen_engine):
    with TestClient(create_app(gen_engine)) as client:
        d = client.post("/dialogs", json={}).json()
        a = client.post("/ask", json={"question": "Нарисуй лису", "dialog_id": d["id"], "session": d["id"]}).json()
        r = client.get(a["image"]["url"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert client.get(f"/images/{d['id']}/nothere.png").status_code == 400
        up = client.post(f"/uploads?session={d['id']}", files={"file": ("cat.jpg", png(), "image/png")}).json()
        assert up["file_type"] == ".png"
        assert client.get(f"/uploads/{up['id']}/file?session={d['id']}").status_code == 200
        assert client.get("/policy").json()["imagegen"] is True
        client.delete(f"/dialogs/{d['id']}")
        assert client.get(a["image"]["url"]).status_code == 404
