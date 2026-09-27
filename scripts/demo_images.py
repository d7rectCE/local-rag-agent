"""Images of the demo corpus (Э5): a diagram, a terminal screenshot, a scanned note, two charts.

Their content is consistent with the rest of the corpus (the training log of NumpyMLP, scripts/train.py,
the breast_cancer experiments) and is visible only in the pixels: no caption or file around them
states it, so a question about them is answerable only through the image pipeline (Q4, H4).

    python scripts/demo_images.py demo_corpus        # also called by build_demo_corpus.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def _font(size: int, mono: bool = False, bold: bool = False) -> ImageFont.ImageFont:
    names = (["consola.ttf", "DejaVuSansMono.ttf"] if mono else
             (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]))
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def architecture(path: Path) -> None:
    """NumpyMLP(64-128-10) as a block diagram."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    blocks = [("Вход\n64 пикселя\n(8×8)", "#e8eef9"), ("Dense 128\nReLU", "#cfe2ff"), ("Dense 10\nSoftmax", "#d1e7dd"),
              ("Класс цифры\n0–9", "#f8d7da")]
    xs = [0.6 + i * 2.4 for i in range(len(blocks))]
    for x, (label, color) in zip(xs, blocks):
        ax.add_patch(FancyBboxPatch((x, 1.0), 1.5, 1.2, boxstyle="round,pad=0.08", fc=color, ec="black", lw=1.2))
        ax.text(x + 0.75, 1.6, label, ha="center", va="center", fontsize=10)
    for a, b in zip(xs, xs[1:]):
        ax.add_patch(FancyArrowPatch((a + 1.62, 1.6), (b - 0.1, 1.6), arrowstyle="-|>", mutation_scale=14, lw=1.3))
    ax.text(5, 2.85, "NumpyMLP: прямой проход", ha="center", fontsize=12, weight="bold")
    ax.text(5, 0.45, "потери: cross-entropy + weight decay 1e-3; оптимизатор: SGD с моментумом 0.9", ha="center",
            fontsize=9, color="#444")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def terminal(path: Path, corpus: Path) -> None:
    """A screenshot of a terminal with a real run of scripts/train.py."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run([sys.executable, str(corpus / "scripts" / "train.py"), "--model", "hist_gb", "--lr", "0.05",
                              "--log", str(Path(tmp) / "runs.jsonl")], capture_output=True, text=True, cwd=tmp,
                             env={**os.environ, "PYTHONPATH": str(corpus), "PYTHONIOENCODING": "utf-8"},
                             encoding="utf-8")
    if run.returncode != 0:
        raise RuntimeError(f"scripts/train.py failed: {run.stderr[-500:]}")
    out = run.stdout.strip().splitlines()[-8:] or ["(нет вывода)"]
    lines = ["(ml) ~/demo_corpus$ python scripts/train.py --model hist_gb --lr 0.05", *out,
             "(ml) ~/demo_corpus$ git add runs && git commit -m \"hist_gb lr=0.05\"", "(ml) ~/demo_corpus$ █"]
    W, H = 980, 70 + 26 * len(lines)
    img = Image.new("RGB", (W, H), (24, 26, 33))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 34], fill=(54, 57, 66))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + i * 22, 11, 26 + i * 22, 23], fill=c)
    d.text((W // 2 - 90, 8), "Terminal — bash", fill=(210, 210, 210), font=_font(15))
    mono = _font(16, mono=True)
    for i, line in enumerate(lines):
        color = (120, 220, 140) if line.startswith("(ml)") else (220, 220, 220)
        d.text((16, 48 + i * 26), line, fill=color, font=mono)
    img.save(path)


def scanned_note(path: Path) -> None:
    """A photographed handwritten-style note: the decision of a meeting, on grey paper with noise."""
    W, H = 900, 640
    img = Image.new("L", (W, H), 236)
    d = ImageDraw.Draw(img)
    title, body = _font(30, bold=True), _font(22)
    y = 50
    d.text((60, y), "Протокол обсуждения экспериментов — 18.03.2026", fill=30, font=title)
    y += 70
    for line in ["1. Бустинг (03_gradient_boosting): оставить learning_rate = 0.05,",
                 "   при 0.2 и выше ROC-AUC на валидации падает.",
                 "2. MLP на digits: ранняя остановка по val_acc, patience = 10.",
                 "3. Аугментацию сдвигами проверить ещё на 5 сидах,",
                 "   прирост 1–2 п.п. пока может быть случайным.",
                 "4. Следующий созвон — 25.03, подготовить отчёт за март.",
                 "", "Решение: learning_rate = 0.05 для hist_gb утверждён."]:
        d.text((60, y), line, fill=35, font=body)
        y += 42
    img = img.rotate(-1.2, expand=True, fillcolor=236)
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.default_rng(7).normal(0, 9, arr.shape)
    arr *= np.linspace(0.92, 1.0, arr.shape[1])[None, :]  # uneven lighting of a phone photo
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    img.convert("RGB").save(path, quality=70) if path.suffix == ".jpg" else img.save(path)


def learning_curves(path: Path, corpus: Path) -> None:
    """Loss curves of the NumpyMLP training log with the restored best epoch marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [dict(re.findall(r"(\w+)=([\d.]+)", ln)) for ln in
            (corpus / "logs" / "train_numpy_mlp.log").read_text(encoding="utf-8").splitlines() if "epoch=" in ln]
    ep = [int(r["epoch"]) for r in rows]
    best = max(rows, key=lambda r: float(r["val_acc"]))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.plot(ep, [float(r["train_loss"]) for r in rows], label="train")
    ax.plot(ep, [float(r["val_loss"]) for r in rows], label="validation")
    b = int(best["epoch"])
    ax.scatter([b], [float(best["val_loss"])], color="red", zorder=3, s=50)
    ax.annotate(f"восстановлена эпоха {b}", (b, float(best["val_loss"])), xytext=(b - 16, 0.9),
                arrowprops={"arrowstyle": "->", "color": "red"}, color="red")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def feature_importance(path: Path) -> None:
    """Top features of a random forest on breast_cancer (impurity importance), a horizontal bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    data = load_breast_cancer()
    rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(data.data, data.target)
    order = np.argsort(rf.feature_importances_)[::-1][:8]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.barh([data.feature_names[i] for i in order][::-1], rf.feature_importances_[order][::-1], color="#4c72b0")
    for y, v in enumerate(rf.feature_importances_[order][::-1]):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_title("Random forest, breast_cancer: важность признаков")
    ax.set_xlabel("impurity importance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_images(corpus: Path) -> None:
    corpus = corpus.resolve()
    out = corpus / "images"
    out.mkdir(parents=True, exist_ok=True)
    architecture(out / "numpy_mlp_architecture.png")
    terminal(out / "terminal_train_hist_gb.png", corpus)
    scanned_note(out / "meeting_note_photo.png")
    learning_curves(out / "numpy_mlp_curves.png", corpus)
    feature_importance(out / "feature_importance_rf.png")


if __name__ == "__main__":
    build_images(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "demo_corpus"))
