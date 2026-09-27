"""Image type classifier at indexing time (ТЗ S4 stage 1): the CNN trained by
scripts/train_image_classifier.py and exported to ``<data_dir>/models/image_classifier``. Without the
exported model (or without timm) the type is left to the VLM, which names it in its description."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

TYPES_RU = {"chart": "график", "diagram": "схема", "screenshot": "скриншот", "scan": "скан документа", "photo": "фото"}


class ImageClassifier:
    def __init__(self, model_dir: Path, device: str = "auto"):
        self.dir = model_dir
        self.device = device
        self._model = None
        self.classes: list[str] = []
        self.version = ""

    @property
    def available(self) -> bool:
        return (self.dir / "model.pt").exists() and (self.dir / "train.json").exists()

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if not self.available:
            return False
        try:
            import timm
            import torch
        except ImportError:
            log.warning("timm is not installed: the image type is left to the VLM")
            return False
        meta = json.loads((self.dir / "train.json").read_text(encoding="utf-8"))
        backbone = meta["backbone"].removeprefix("hf-hub:").replace("timm/", "").split(".")[0]
        model = timm.create_model(backbone, pretrained=False, num_classes=len(meta["classes"]))
        model.load_state_dict(torch.load(self.dir / "model.pt", map_location="cpu"))
        dev = "cuda" if self.device == "auto" and torch.cuda.is_available() else ("cpu" if self.device == "auto" else self.device)
        self._model, self._dev = model.eval().to(dev), dev
        self.classes = meta["classes"]
        self.version = f"{backbone}:{meta.get('best_val_acc', 0):.4f}"
        return True

    def predict(self, data: bytes) -> tuple[str, float] | None:
        if not self._load():
            return None
        import torch
        from PIL import Image
        from torchvision import transforms as T

        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with Image.open(io.BytesIO(data)) as im:
            x = tf(im.convert("RGB")).unsqueeze(0).to(self._dev)
        with torch.no_grad():
            p = torch.softmax(self._model(x), 1)[0].cpu()
        k = int(p.argmax())
        return self.classes[k], round(float(p[k]), 3)
