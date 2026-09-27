"""Э5, ТЗ S4 stage 1: the image type classifier (chart / diagram / screenshot / scan / photo).

A small timm backbone (MobileNetV3-Small, ImageNet weights) is fine-tuned on a generated dataset:

* chart      — matplotlib: lines, bars, scatter, histograms, box plots, pies, heatmaps (confusion matrices)
               with random styles, sizes, labels in Russian and English;
* diagram    — box-and-arrow drawings: pipelines, layered architectures, flowcharts with decisions, trees;
* screenshot — rendered code editors, terminals, notebooks and forms in light and dark themes;
* scan       — DocLayNet pages (runs/layout/doclaynet640_full, prepared for the layout detector) with
               scanner artefacts: grey paper, noise, blur, skew, JPEG;
* photo      — natural photos shipped with scikit-image and scikit-learn, as random crops with colour
               jitter. The photo class is small and not diverse: the weakest part of the data (report).

Splits are disjoint by seed, by the DocLayNet split and by the source photo. ``real`` evaluates on images
that were not generated: the plots in the demo notebooks' outputs, DocLayNet test pages as they are,
and the held-out photos.

    python scripts/train_image_classifier.py prepare --out runs/images/data
    python scripts/train_image_classifier.py train --data runs/images/data --out runs/images/mnv3s
    python scripts/train_image_classifier.py evaluate --model runs/images/mnv3s --data runs/images/data
    python scripts/train_image_classifier.py real --model runs/images/mnv3s
    python scripts/train_image_classifier.py export --model runs/images/mnv3s   # -> <data_dir>/models/image_classifier
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import shutil
import string
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASSES = ["chart", "diagram", "screenshot", "scan", "photo"]
BACKBONE = "hf-hub:timm/mobilenetv3_small_100.lamb_in1k"
DOCLAYNET = ROOT / "runs" / "layout" / "doclaynet640_full"
# the photos bundled with scikit-image (no pooch download) and scikit-learn; "cat" is "chelsea" again
PHOTOS_TRAIN = ["astronaut", "chelsea", "rocket", "hubble_deep_field", "immunohistochemistry", "retina", "moon", "coins",
                "grass", "gravel", "brick", "sk:flower"]
PHOTOS_TEST = ["coffee", "camera", "clock", "sk:china"]  # held out: other photos entirely

WORDS_RU = ("модель данные обучение точность потери эпоха валидация тест признак выборка метрика график "
            "кластер регрессия классификация градиент слой сеть бустинг лес дерево шаг скорость отчёт").split()
WORDS_EN = ("model data train accuracy loss epoch validation test feature sample metric score cluster "
            "regression classifier gradient layer network boosting forest tree step rate report baseline").split()


def words(rng: random.Random, n: int) -> str:
    pool = WORDS_RU if rng.random() < 0.5 else WORDS_EN
    return " ".join(rng.choice(pool) for _ in range(n))


def fig_to_image(fig) -> Image.Image:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight" if random.random() < 0.7 else None)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# --------------------------------------------------------------------------- charts

def make_chart(rng: random.Random) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = rng.choice(["default", "ggplot", "seaborn-v0_8", "bmh", "fivethirtyeight", "dark_background",
                        "seaborn-v0_8-whitegrid", "classic"])
    npr = np.random.default_rng(rng.randrange(1 << 30))
    with plt.style.context(style):
        n_axes = rng.choice([1, 1, 1, 2, 3, 4])
        rows = 1 if n_axes < 3 else 2
        cols = math.ceil(n_axes / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(rng.uniform(3.5, 9), rng.uniform(2.5, 6.5)),
                                 dpi=rng.choice([60, 72, 80, 100]), squeeze=False)
        for ax in axes.flat[:n_axes]:
            kind = rng.choice(["line", "line", "bar", "barh", "scatter", "hist", "box", "pie", "heatmap", "roc",
                               "errorbar", "area"])
            x = np.arange(rng.randint(5, 60))
            if kind == "line":
                for _ in range(rng.randint(1, 4)):
                    y = np.cumsum(npr.normal(0, 1, len(x))) if rng.random() < 0.5 else np.exp(-x / rng.uniform(5, 30)) + npr.normal(0, 0.02, len(x))
                    ax.plot(x, y, marker=rng.choice(["", "", "o", "."]), label=words(rng, 1))
            elif kind in ("bar", "barh"):
                k = rng.randint(3, 10)
                vals = npr.uniform(0.2, 1.0, k)
                labels = [words(rng, 1) for _ in range(k)]
                (ax.bar if kind == "bar" else ax.barh)(labels, vals, color=rng.choice([None, "tab:blue", "tab:orange", "tab:green"]))
                ax.tick_params(labelrotation=rng.choice([0, 30, 45]) if kind == "bar" else 0)
            elif kind == "scatter":
                for _ in range(rng.randint(1, 4)):
                    c = npr.normal(0, 3, 2)
                    pts = npr.normal(c, rng.uniform(0.3, 1.5), (rng.randint(20, 200), 2))
                    ax.scatter(pts[:, 0], pts[:, 1], s=rng.choice([5, 10, 20]), alpha=0.7, label=words(rng, 1))
            elif kind == "hist":
                for _ in range(rng.randint(1, 2)):
                    ax.hist(npr.normal(rng.uniform(-2, 2), rng.uniform(0.5, 2), 500), bins=rng.randint(10, 40),
                            alpha=0.6, label=words(rng, 1))
            elif kind == "box":
                ax.boxplot([npr.normal(i, 1, 100) for i in range(rng.randint(2, 6))])
            elif kind == "pie":
                k = rng.randint(3, 6)
                ax.pie(npr.uniform(1, 5, k), labels=[words(rng, 1) for _ in range(k)], autopct="%1.0f%%")
            elif kind == "heatmap":
                k = rng.randint(2, 10)
                m = npr.integers(0, 100, (k, k))
                im = ax.imshow(m, cmap=rng.choice(["Blues", "viridis", "magma", "coolwarm"]))
                if k <= 6:
                    for (i, j), v in np.ndenumerate(m):
                        ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
                fig.colorbar(im, ax=ax)
            elif kind == "roc":
                for _ in range(rng.randint(1, 3)):
                    fpr = np.sort(npr.uniform(0, 1, 30))
                    tpr = np.clip(fpr ** rng.uniform(0.1, 0.6), 0, 1)
                    ax.plot(np.r_[0, fpr, 1], np.r_[0, tpr, 1], label=f"AUC={rng.uniform(0.7, 0.99):.2f}")
                ax.plot([0, 1], [0, 1], "k--", lw=1)
            elif kind == "errorbar":
                ax.errorbar(x[:15], npr.normal(0, 1, len(x[:15])), yerr=npr.uniform(0.1, 0.5, len(x[:15])), fmt="o-", capsize=3)
            else:
                ax.fill_between(x, np.abs(np.cumsum(npr.normal(0, 1, len(x)))), alpha=0.5)
            if rng.random() < 0.8:
                ax.set_title(words(rng, rng.randint(1, 4)))
            if rng.random() < 0.7 and kind != "pie":
                ax.set_xlabel(words(rng, 1))
                ax.set_ylabel(words(rng, 1))
            if rng.random() < 0.5 and kind in ("line", "scatter", "hist", "roc"):
                ax.legend(fontsize=8)
            if rng.random() < 0.4 and kind != "pie":
                ax.grid(True, alpha=0.3)
        for ax in axes.flat[n_axes:]:
            ax.axis("off")
        fig.tight_layout()
        return fig_to_image(fig)


# --------------------------------------------------------------------------- diagrams

def make_diagram(rng: random.Random) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

    fig, ax = plt.subplots(figsize=(rng.uniform(4, 9), rng.uniform(3, 7)), dpi=rng.choice([60, 72, 80, 100]))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    bg = rng.choice(["white", "white", "white", "#f7f7f2", "#1e1e1e", "#eef3fb"])
    fig.patch.set_facecolor(bg)
    dark = bg == "#1e1e1e"
    palette = rng.choice([["#cfe2ff", "#d1e7dd", "#fff3cd", "#f8d7da"], ["#ffffff"], ["#e0e0e0", "#bdbdbd"],
                          ["#8ecae6", "#ffb703", "#fb8500", "#219ebc"]])
    layout = rng.choice(["pipeline", "stack", "flow", "tree", "graph"])
    boxes = []
    if layout == "pipeline":
        n = rng.randint(3, 6)
        y = rng.uniform(3, 7)
        boxes = [(0.5 + i * 9 / n, y) for i in range(n)]
    elif layout == "stack":
        n = rng.randint(3, 7)
        x = rng.uniform(2, 5)
        boxes = [(x, 0.8 + i * 8.5 / n) for i in range(n)]
    elif layout == "tree":
        boxes = [(4.5, 8.5)] + [(1 + i * 3, 5) for i in range(3)] + [(0.5 + i * 2, 1.5) for i in range(rng.randint(2, 5))]
    else:
        boxes = [(rng.uniform(0.3, 8), rng.uniform(0.5, 8.5)) for _ in range(rng.randint(4, 8))]
    w, h = rng.uniform(1.2, 2.2), rng.uniform(0.7, 1.2)
    centers = []
    for i, (x, y) in enumerate(boxes):
        color = palette[i % len(palette)]
        if layout == "flow" and rng.random() < 0.3:
            ax.add_patch(Polygon([[x + w / 2, y + h], [x + w, y + h / 2], [x + w / 2, y], [x, y + h / 2]],
                                 closed=True, fc=color, ec="#888" if dark else "black"))
        else:
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=rng.choice(["round,pad=0.1", "square,pad=0.05",
                                                                            "round4,pad=0.1"]),
                                        fc=color, ec="#aaa" if dark else "black", lw=rng.uniform(0.8, 2)))
        label = rng.choice(["Encoder", "Decoder", "Attention", "FFN", "Embedding", "Softmax", "Input", "Output",
                            "Retriever", "Reranker", "LLM", "Index", "Parser", "Router", "Agent", "Сеть", "Данные",
                            "Модель", "Ответ", "Запрос", "Поиск", "Слой", "Выход", "Вход", "Norm", "Conv", "Pool"])
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=rng.randint(7, 12),
                color="black")
        centers.append((x + w / 2, y + h / 2))
    edges = list(zip(range(len(centers) - 1), range(1, len(centers))))
    if layout == "tree":
        edges = [(0, i) for i in (1, 2, 3)] + [(1 + (i % 3), 4 + i) for i in range(len(centers) - 4)]
    elif layout == "graph":
        edges += [(rng.randrange(len(centers)), rng.randrange(len(centers))) for _ in range(rng.randint(1, 4))]
    for a, b in edges:
        if a == b:
            continue
        ax.add_patch(FancyArrowPatch(centers[a], centers[b], arrowstyle="-|>", mutation_scale=rng.randint(10, 18),
                                     color="#ccc" if dark else "black", lw=rng.uniform(0.8, 1.8),
                                     shrinkA=18, shrinkB=18,
                                     connectionstyle=rng.choice(["arc3", "arc3,rad=0.2", "angle3"])))
    if rng.random() < 0.5:
        ax.set_title(words(rng, rng.randint(2, 4)), color="white" if dark else "black")
    return fig_to_image(fig)


# --------------------------------------------------------------------------- screenshots

def _font(size: int, mono: bool) -> ImageFont.ImageFont:
    names = (["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"] if mono else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


CODE_LINES = ["import numpy as np", "import pandas as pd", "from sklearn.model_selection import train_test_split",
              "def train(model, data, epochs=10):", "    for epoch in range(epochs):", "        loss = step(model, batch)",
              "        print(f'epoch {epoch}: loss={loss:.4f}')", "    return model", "", "class Trainer:",
              "    def __init__(self, lr=1e-3):", "        self.lr = lr", "X_train, X_test = train_test_split(X)",
              "df = pd.read_csv('data.csv')", "acc = (pred == y).mean()", "# TODO: tune hyperparameters",
              "$ python train.py --lr 0.01", "Epoch 3/10 - loss: 0.2345 - acc: 0.9120", "(rag) C:\\Users> pip list",
              "Traceback (most recent call last):", "  File \"train.py\", line 42, in <module>",
              "ValueError: shapes (3,4) and (5,) not aligned", "Successfully installed torch-2.9.1",
              "SELECT name, value FROM metrics WHERE split = 'test';", "git commit -m \"fix loader\""]


def make_screenshot(rng: random.Random) -> Image.Image:
    W, H = rng.choice([(1280, 800), (1024, 768), (1440, 900), (800, 600), (1920, 1080), (900, 1200)])
    dark = rng.random() < 0.5
    bg = rng.choice([(30, 30, 30), (40, 44, 52), (13, 17, 23)]) if dark else rng.choice([(255, 255, 255), (246, 246, 246), (250, 250, 245)])
    fg = (220, 220, 220) if dark else (30, 30, 30)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    kind = rng.choice(["editor", "terminal", "notebook", "form", "browser"])
    bar = rng.randint(28, 40)
    d.rectangle([0, 0, W, bar], fill=(60, 60, 60) if dark else (225, 225, 225))  # window title bar
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        if rng.random() < 0.6:
            d.ellipse([10 + i * 20, bar // 2 - 6, 22 + i * 20, bar // 2 + 6], fill=c)
    d.text((W // 2 - 60, bar // 2 - 8), words(rng, 2), fill=fg, font=_font(14, False))
    y0 = bar + 6
    if kind in ("editor", "notebook", "browser") and rng.random() < 0.7:
        side = rng.randint(160, 280)
        d.rectangle([0, y0 - 6, side, H], fill=tuple(max(0, c - 12) for c in bg))
        for i in range(rng.randint(5, 20)):
            d.text((14, y0 + 8 + i * 24), rng.choice(["▸ ", "  ", "• "]) + words(rng, 1) + rng.choice([".py", ".ipynb", "", "/"]),
                   fill=fg, font=_font(13, False))
        x0 = side + 16
    else:
        x0 = 16
    size = rng.randint(12, 18)
    mono = _font(size, True)
    colors = [(197, 134, 192), (86, 156, 214), (206, 145, 120), (106, 153, 85), fg] if dark else \
        [(175, 0, 219), (0, 0, 255), (163, 21, 21), (0, 128, 0), fg]
    y = y0 + 10
    if kind == "form":
        for i in range(rng.randint(3, 7)):
            d.text((x0 + 20, y), words(rng, 2), fill=fg, font=_font(14, False))
            d.rounded_rectangle([x0 + 20, y + 22, x0 + 20 + rng.randint(200, 500), y + 52], radius=6,
                                outline=(120, 120, 120), fill=bg)
            y += 70
        d.rounded_rectangle([x0 + 20, y + 10, x0 + 140, y + 46], radius=8, fill=(37, 99, 235))
        d.text((x0 + 45, y + 18), rng.choice(["Отправить", "Submit", "Save", "OK"]), fill=(255, 255, 255), font=_font(14, False))
    else:
        cell = 0
        while y < H - size * 2:
            if kind == "notebook" and rng.random() < 0.15:
                d.rectangle([x0, y, W - 20, y + size * 3], outline=(120, 120, 120))
                d.text((x0 - 2, y + 4), f"[{cell}]", fill=(120, 120, 180), font=mono)
                cell += 1
            line = rng.choice(CODE_LINES)
            if kind == "editor" and rng.random() < 0.8:
                d.text((x0, y), f"{(y - y0) // (size + 6) + 1:>3}", fill=(128, 128, 128), font=mono)
                d.text((x0 + size * 3, y), line, fill=rng.choice(colors), font=mono)
            elif kind == "browser":
                d.text((x0, y), words(rng, rng.randint(4, 12)), fill=fg, font=_font(size, False))
            else:
                d.text((x0, y), line, fill=rng.choice(colors) if kind != "terminal" else fg, font=mono)
            y += size + rng.randint(4, 10)
    if rng.random() < 0.3:  # scaled like a pasted screenshot
        img = img.resize((int(W * 0.6), int(H * 0.6)), Image.BILINEAR)
    return img


# --------------------------------------------------------------------------- scans and photos

def make_scan(rng: random.Random, page: Path) -> Image.Image:
    img = Image.open(page).convert("L" if rng.random() < 0.7 else "RGB")
    if rng.random() < 0.8:
        img = img.rotate(rng.uniform(-3, 3), expand=True, fillcolor=255 if img.mode == "L" else (255, 255, 255))
    arr = np.asarray(img).astype(np.float32)
    arr = arr * rng.uniform(0.8, 1.0) + rng.uniform(0, 30)  # grey paper, faded ink
    arr += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 12), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if img.mode == "L" and rng.random() < 0.4:  # yellowish paper
        rgb = Image.merge("RGB", [img, img, img.point(lambda v: int(v * 0.9))])
        img = rgb
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=rng.randint(35, 90))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def load_photo(name: str) -> Image.Image:
    if name.startswith("sk:"):
        from sklearn.datasets import load_sample_image

        return Image.fromarray(load_sample_image(f"{name[3:]}.jpg"))
    import skimage.data

    arr = getattr(skimage.data, name)()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = (255 * (arr - arr.min()) / (arr.ptp() or 1)).astype(np.uint8)
    return Image.fromarray(arr[..., :3])


def make_photo(rng: random.Random, photo: Image.Image) -> Image.Image:
    W, H = photo.size
    s = rng.uniform(0.35, 1.0)
    w, h = int(W * s), int(H * s * rng.uniform(0.75, 1.33))
    w, h = min(w, W), min(h, H)
    x, y = rng.randint(0, W - w), rng.randint(0, H - h)
    img = photo.crop((x, y, x + w, y + h))
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.asarray(img).astype(np.float32)
    arr = arr * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)
    arr = arr * np.array([rng.uniform(0.85, 1.15) for _ in range(3)])
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-15, 15))
    return img


# --------------------------------------------------------------------------- prepare

def cmd_prepare(args) -> None:
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    counts = {"train": args.per_class, "val": max(40, args.per_class // 6), "test": max(60, args.per_class // 4)}
    seeds = {"train": 1, "val": 2, "test": 3}
    pages = {split: sorted((DOCLAYNET / d).glob("*.jpg")) for split, d in
             (("train", "train"), ("val", "validation"), ("test", "test"))}
    photos = {"train": [load_photo(n) for n in PHOTOS_TRAIN], "test": [load_photo(n) for n in PHOTOS_TEST]}
    photos["val"] = photos["train"]  # validation crops of the training photos (the test set holds other photos)
    t0 = time.time()
    for split, n in counts.items():
        for cls in CLASSES:
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            rng = random.Random(f"{seeds[split]}-{cls}")
            for i in range(n):
                if cls == "chart":
                    img = make_chart(rng)
                elif cls == "diagram":
                    img = make_diagram(rng)
                elif cls == "screenshot":
                    img = make_screenshot(rng)
                elif cls == "scan":
                    img = make_scan(rng, rng.choice(pages[split]))
                else:
                    img = make_photo(rng, rng.choice(photos[split]))
                img.thumbnail((512, 512))
                img.save(d / f"{i:05d}.png")
        print(f"{split}: {n} per class, {time.time() - t0:.0f} s", flush=True)
    (out / "meta.json").write_text(json.dumps({"classes": CLASSES, "counts": counts, "photos_train": PHOTOS_TRAIN,
                                               "photos_test": PHOTOS_TEST}, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- train / evaluate

def _setup_offline() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def transforms(train: bool):
    from torchvision import transforms as T

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.6, 1.6)), T.RandomHorizontalFlip(0.3),
                          T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.RandomGrayscale(0.1), T.ToTensor(), norm])
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])


def make_model(num_classes: int, pretrained: bool = True):
    import timm

    _setup_offline()
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=num_classes)


def cmd_train(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    torch.manual_seed(0)
    data = Path(args.data)
    train_ds = ImageFolder(data / "train", transform=transforms(True))
    val_ds = ImageFolder(data / "val", transform=transforms(False))
    assert train_ds.classes == sorted(CLASSES), train_ds.classes
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(len(train_ds.classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * math.ceil(len(train_ds) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0)
    vl = DataLoader(val_ds, batch_size=128, num_workers=args.workers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            loss = torch.nn.functional.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        acc = accuracy(model, vl, dev)
        history.append({"epoch": epoch, "loss": round(total / n, 4), "val_acc": round(acc, 4), "s": round(time.time() - t0, 1)})
        print(history[-1], flush=True)
        if acc > best:  # early stopping on the validation split: the best epoch is kept
            best = acc
            torch.save(model.state_dict(), out / "model.pt")
    (out / "train.json").write_text(json.dumps({"classes": train_ds.classes, "backbone": BACKBONE, "best_val_acc": best,
                                                "history": history, "args": vars(args)}, indent=2), encoding="utf-8")


def accuracy(model, loader, dev) -> float:
    import torch

    model.eval()
    ok = n = 0
    with torch.no_grad():
        for x, y in loader:
            ok += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
            n += len(y)
    return ok / max(n, 1)


def load_trained(path: Path):
    import torch

    meta = json.loads((path / "train.json").read_text(encoding="utf-8"))
    model = make_model(len(meta["classes"]), pretrained=False)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    return model.eval(), meta["classes"]


def report(y_true: list[int], y_pred: list[int], classes: list[str], out: Path, name: str) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    res = {"n": len(y_true), "accuracy": round(accuracy_score(y_true, y_pred), 4),
           "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true))), 4),
           "classes": classes, "confusion": cm.tolist(),
           "per_class_recall": {c: round(cm[i, i] / cm[i].sum(), 4) for i, c in enumerate(classes) if cm[i].sum()}}
    (out / f"{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "confusion"}, ensure_ascii=False))
    print(cm)
    return res


def cmd_evaluate(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    model, classes = load_trained(Path(args.model))
    ds = ImageFolder(Path(args.data) / args.split, transform=transforms(False))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=128):
            y_pred += model(x.to(dev)).argmax(1).cpu().tolist()
            y_true += y.tolist()
    report(y_true, y_pred, classes, Path(args.model), f"eval_{args.split}")


def real_set() -> list[tuple[Image.Image, str, str]]:
    """Images that were not generated: plots in the demo notebooks' outputs (charts), DocLayNet test pages
    without artefacts (scans / page images) and the held-out photos."""
    items = []
    for nb in sorted((ROOT / "demo_corpus").rglob("*.ipynb")):
        for ci, cell in enumerate(json.loads(nb.read_text(encoding="utf-8")).get("cells", []), start=1):
            for out in cell.get("outputs", []):
                png = (out.get("data") or {}).get("image/png")
                if png:
                    img = Image.open(io.BytesIO(base64.b64decode("".join(png) if isinstance(png, list) else png)))
                    items.append((img.convert("RGB"), "chart", f"{nb.name}#cell{ci}"))
    for p in sorted((DOCLAYNET / "test").glob("*.jpg"))[:40:2]:
        items.append((Image.open(p).convert("RGB"), "scan", p.name))
    for name in PHOTOS_TEST:
        items.append((load_photo(name), "photo", name))
    return items


def cmd_real(args) -> None:
    import torch

    model, classes = load_trained(Path(args.model))
    tf = transforms(False)
    y_true, y_pred, rows = [], [], []
    with torch.no_grad():
        for img, label, src in real_set():
            p = torch.softmax(model(tf(img).unsqueeze(0)), 1)[0]
            k = int(p.argmax())
            y_true.append(classes.index(label))
            y_pred.append(k)
            rows.append({"source": src, "label": label, "pred": classes[k], "p": round(float(p[k]), 3)})
    res = report(y_true, y_pred, classes, Path(args.model), "eval_real")
    res["items"] = rows
    (Path(args.model) / "eval_real.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_export(args) -> None:
    from rag_agent.config import load_settings

    target = Path(args.target) if args.target else load_settings().data_dir / "models" / "image_classifier"
    target.mkdir(parents=True, exist_ok=True)
    for f in ("model.pt", "train.json"):
        shutil.copy2(Path(args.model) / f, target / f)
    print(f"exported to {target}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", default="runs/images/data")
    p.add_argument("--per-class", type=int, default=800)
    p = sub.add_parser("train")
    p.add_argument("--data", default="runs/images/data")
    p.add_argument("--out", default="runs/images/mnv3s")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("evaluate")
    p.add_argument("--model", default="runs/images/mnv3s")
    p.add_argument("--data", default="runs/images/data")
    p.add_argument("--split", default="test")
    p = sub.add_parser("real")
    p.add_argument("--model", default="runs/images/mnv3s")
    p = sub.add_parser("export")
    p.add_argument("--model", default="runs/images/mnv3s")
    p.add_argument("--target", default="")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "train": cmd_train, "evaluate": cmd_evaluate, "real": cmd_real, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
