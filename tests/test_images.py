"""Э5: image nodes at parse time, their pixels read back, the description stage with its cache, and
the visual channel fused with the text hits by RRF."""

import base64
import io
import json
from pathlib import Path

import nbformat
from PIL import Image, ImageDraw

from rag_agent.engine import Engine
from rag_agent.images.pipeline import process_images
from rag_agent.images.sources import image_bytes
from tests.conftest import FakeLLM


def png(color=(30, 120, 200), size=(160, 120), text="loss") -> bytes:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.line([(10, 100), (60, 60), (150, 20)], fill=color, width=3)
    d.text((10, 5), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "figs").mkdir(parents=True)
    (root / "figs" / "arch.png").write_bytes(png(text="Encoder"))
    (root / "figs" / "tiny.png").write_bytes(png(size=(20, 20)))  # an icon: not described
    nb = nbformat.v4.new_notebook()
    code = nbformat.v4.new_code_cell("plt.plot(losses)\nplt.title('train loss')")
    code.outputs = [nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png()).decode(),
                                                                  "text/plain": "<Figure>"})]
    nb.cells = [nbformat.v4.new_markdown_cell("# Обучение"), code]
    (root / "train.ipynb").write_text(nbformat.writes(nb), encoding="utf-8")
    return root


class StubClassifier:
    version = "stub"

    def _load(self):
        return True

    def predict(self, data):
        return ("chart", 0.9)


class Describer(FakeLLM):
    """A VLM stand-in: describes by the prompt and records that the image was attached."""

    def _chat(self, messages, json_schema, tools, temperature, max_tokens, think):
        if messages and messages[-1].get("images"):
            self.kinds.append("describe")
            assert base64.b64decode(messages[-1]["images"][0])[:4] == b"\x89PNG"
            from rag_agent.llm import LLMResponse

            return LLMResponse(content="График loss по эпохам: падает с 1.0 до 0.2.", usage={})
        return super()._chat(messages, json_schema, tools, temperature, max_tokens, think)


def test_images_become_nodes_and_are_described_once(settings, fake_embedder, tmp_path: Path):
    settings.corpus.include_ext = [".ipynb", ".png"]
    root = image_corpus(tmp_path)
    llm = Describer(settings)
    eng = Engine(settings, embedder=fake_embedder, llm=llm)
    eng.index_folder(root)
    idx = eng.index
    images = {n.id: n for n in idx.catalog.image_nodes()}
    assert set(images) == {"figs/arch.png#image", "figs/tiny.png#image", "train.ipynb#cell2/img1"}
    nb_img = images["train.ipynb#cell2/img1"]
    assert nb_img.location.cell == 2 and "plt.plot(losses)" in nb_img.context and not nb_img.embed
    assert image_bytes(idx.root, nb_img)[:4] == b"\x89PNG"  # read back from the notebook, not stored

    stats = process_images(idx, settings, eng.llm.base, fake_embedder, classifier=StubClassifier())
    assert stats["described"] == 2 and stats["skipped"] == 1 and stats["types"] == {"chart": 2}
    node = idx.catalog.get_node("figs/arch.png#image")
    assert node.text.startswith("[Изображение: график]") and "падает с 1.0" in node.text and node.embed
    assert node.metadata["image_type"] == "chart" and node.metadata["image_sha256"]
    hits = eng.search("график loss по эпохам", 5)
    assert any(h.node.node_type == "image" for h in hits)  # the description is in the text index

    llm.kinds.clear()
    again = process_images(idx, settings, eng.llm.base, fake_embedder, classifier=StubClassifier())
    assert again["described"] == 0 and "describe" not in llm.kinds  # nothing new to describe
    for n in idx.catalog.image_nodes():  # a re-index would describe from the cache by the content hash
        n.metadata.pop("described", None)
        idx.catalog.update_node(n)
    third = process_images(idx, settings, eng.llm.base, fake_embedder, classifier=StubClassifier())
    assert third["cached"] == 2 and "describe" not in llm.kinds
    eng.close()


def test_image_stage_runs_after_indexing(settings, fake_embedder, tmp_path: Path, monkeypatch):
    settings.corpus.include_ext = [".ipynb", ".png"]
    settings.images.enabled = True
    settings.images.classify = False  # no exported classifier: the VLM names the type itself
    eng = Engine(settings, embedder=fake_embedder, llm=Describer(settings))
    progress = eng.index_folder(image_corpus(tmp_path))
    assert progress.state == "done" and progress.catalog["images"]["described"] == 2
    assert "[Изображение: изображение]" in eng.index.catalog.get_node("figs/arch.png#image").text
    eng.close()


def test_visual_channel_is_fused_by_rrf(settings, fake_embedder, tmp_path: Path, monkeypatch):
    from rag_agent.images import visual

    settings.corpus.include_ext = [".ipynb", ".png"]
    eng = Engine(settings, embedder=fake_embedder, llm=Describer(settings))
    eng.index_folder(image_corpus(tmp_path))
    process_images(eng.index, settings, eng.llm.base, fake_embedder, classifier=StubClassifier())
    monkeypatch.setattr(visual.VisualIndex, "available", property(lambda self: True))
    monkeypatch.setattr(visual.VisualIndex, "search",
                        lambda self, q, top_k=10: [({"node_id": "figs/arch.png#image"}, 9.0)])
    settings.images.channel = "fusion"
    fused = eng.search("plt.plot losses", 3)
    assert "figs/arch.png#image" in [h.node.id for h in fused]  # found by its pixels only
    settings.images.channel = "visual"
    only = eng.search("график loss по эпохам", 5)
    images = [h.node.id for h in only if h.node.node_type == "image"]
    assert images == ["figs/arch.png#image"]  # descriptions are out of the text ranking, pixels decide
    eng.close()


def test_classifier_script_generators_make_images():
    import random
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train_image_classifier as t

    rng = random.Random(0)
    for make in (t.make_chart, t.make_diagram, t.make_screenshot):
        img = make(rng)
        assert img.mode == "RGB" and min(img.size) > 100
    assert json.dumps(t.CLASSES) and len(t.PHOTOS_TRAIN) > len(t.PHOTOS_TEST)
