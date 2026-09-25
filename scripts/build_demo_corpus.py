"""Build the public demo corpus (NFR3, NFR7).

The corpus imitates a student's research archive: a small ML package, scripts
and executed Jupyter notebooks with real outputs (metrics, tables, plots) on
scikit-learn's bundled datasets. Everything is authored here, so the corpus is
license-clean and its ground truth is known — it also backs the evaluation set
(Э2). Requires nbclient + ipykernel + scikit-learn + matplotlib + pandas.

    python scripts/build_demo_corpus.py [--no-execute]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "demo_corpus"


def dedent(s: str) -> str:
    return textwrap.dedent(s).strip("\n") + "\n"


# --------------------------------------------------------------------------
# Python package and scripts
# --------------------------------------------------------------------------

FILES: dict[str, str] = {}

FILES["ml_toolkit/__init__.py"] = dedent('''
    """ml_toolkit — вспомогательный код для учебных экспериментов по ML."""

    __version__ = "0.3.1"
''')

FILES["ml_toolkit/data.py"] = dedent('''
    """Загрузка учебных датасетов и разбиение на train/val/test."""

    from __future__ import annotations

    from dataclasses import dataclass

    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    DATASETS = {
        "breast_cancer": datasets.load_breast_cancer,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
        "diabetes": datasets.load_diabetes,
    }


    @dataclass
    class Split:
        X_train: np.ndarray
        X_val: np.ndarray
        X_test: np.ndarray
        y_train: np.ndarray
        y_val: np.ndarray
        y_test: np.ndarray
        feature_names: list[str]

        def sizes(self) -> dict[str, int]:
            return {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)}


    def load_dataset(name: str):
        """Return (X, y, feature_names) for one of the bundled datasets."""
        if name not in DATASETS:
            raise KeyError(f"unknown dataset {name!r}, available: {sorted(DATASETS)}")
        bunch = DATASETS[name]()
        names = getattr(bunch, "feature_names", None)
        if names is None:
            names = [f"f{i}" for i in range(bunch.data.shape[1])]
        return bunch.data, bunch.target, list(names)


    def make_split(name: str, val_size: float = 0.2, test_size: float = 0.2, seed: int = 42,
                   stratify: bool = True) -> Split:
        """Стратифицированное разбиение 60/20/20 с фиксированным seed."""
        X, y, names = load_dataset(name)
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y if stratify else None
        )
        rel_val = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=rel_val, random_state=seed, stratify=y_tmp if stratify else None
        )
        return Split(X_train, X_val, X_test, y_train, y_val, y_test, names)
''')

FILES["ml_toolkit/features.py"] = dedent('''
    """Препроцессинг признаков и аугментации для изображений digits (8x8)."""

    from __future__ import annotations

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    DEFAULT_SHIFTS = ((1, 0), (-1, 0), (0, 1), (0, -1))


    def make_preprocessor(poly_degree: int = 1) -> Pipeline:
        """StandardScaler, опционально с полиномиальными признаками."""
        steps = []
        if poly_degree > 1:
            steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
        steps.append(("scaler", StandardScaler()))
        return Pipeline(steps)


    def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Сдвиг изображения на (dx, dy) пикселей с заполнением нулями."""
        out = np.zeros_like(img)
        h, w = img.shape
        src_y = slice(max(0, -dy), h - max(0, dy))
        dst_y = slice(max(0, dy), h - max(0, -dy))
        src_x = slice(max(0, -dx), w - max(0, dx))
        dst_x = slice(max(0, dx), w - max(0, -dx))
        out[dst_y, dst_x] = img[src_y, src_x]
        return out


    def augment_digits(X: np.ndarray, y: np.ndarray, shifts=DEFAULT_SHIFTS) -> tuple[np.ndarray, np.ndarray]:
        """Аугментация сдвигами на 1 пиксель: выборка увеличивается в (1 + len(shifts)) раз."""
        images = X.reshape(-1, 8, 8)
        xs, ys = [X], [y]
        for dx, dy in shifts:
            shifted = np.stack([shift_image(im, dx, dy) for im in images])
            xs.append(shifted.reshape(len(X), -1))
            ys.append(y)
        return np.concatenate(xs), np.concatenate(ys)
''')

FILES["ml_toolkit/models.py"] = dedent('''
    """Фабрика моделей для экспериментов."""

    from __future__ import annotations

    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier

    MODEL_NAMES = ("logreg", "random_forest", "hist_gb", "mlp", "ridge")


    def build_model(name: str, **params):
        """Создать модель по короткому имени; params передаются в конструктор."""
        if name == "logreg":
            return LogisticRegression(max_iter=params.pop("max_iter", 5000), **params)
        if name == "random_forest":
            return RandomForestClassifier(n_estimators=params.pop("n_estimators", 300), **params)
        if name == "hist_gb":
            return HistGradientBoostingClassifier(**params)
        if name == "mlp":
            return MLPClassifier(**params)
        if name == "ridge":
            return Ridge(**params)
        raise ValueError(f"unknown model {name!r}, expected one of {MODEL_NAMES}")
''')

FILES["ml_toolkit/metrics.py"] = dedent('''
    """Метрики качества.

    calc_metrics — единая точка подсчёта accuracy, F1 и ROC-AUC для всех
    экспериментов; regression_metrics — RMSE и R^2 для регрессии.
    """

    from __future__ import annotations

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score


    def calc_metrics(y_true, y_pred, y_proba=None, average: str = "macro") -> dict[str, float]:
        """Accuracy, F1 (macro по умолчанию) и ROC-AUC, если переданы вероятности."""
        res = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average=average)),
        }
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            if y_proba.ndim == 2:
                res["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
            else:
                res["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        return res


    def regression_metrics(y_true, y_pred) -> dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {"rmse": rmse, "r2": float(r2_score(y_true, y_pred))}


    def bootstrap_ci(metric_fn, y_true, y_pred, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
        """Перцентильный бутстреп-интервал для произвольной метрики."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        rng = np.random.default_rng(seed)
        n = len(y_true)
        values = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            values.append(metric_fn(y_true[idx], y_pred[idx]))
        lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)
''')

FILES["ml_toolkit/viz.py"] = dedent('''
    """Графики для отчётов: ROC-кривая, матрица ошибок, кривая подбора гиперпараметра."""

    from __future__ import annotations

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


    def plot_roc(y_true, y_score, label: str = "model", ax=None):
        ax = ax or plt.subplots(figsize=(4, 4))[1]
        RocCurveDisplay.from_predictions(y_true, y_score, name=label, ax=ax)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title("ROC")
        return ax


    def plot_confusion(y_true, y_pred, labels=None, ax=None):
        ax = ax or plt.subplots(figsize=(4.5, 4.5))[1]
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax, colorbar=False)
        ax.set_title("Confusion matrix")
        return ax


    def plot_param_curve(values, scores, param: str, metric: str, ax=None, logx: bool = True):
        ax = ax or plt.subplots(figsize=(5, 3))[1]
        ax.plot(values, scores, marker="o")
        if logx:
            ax.set_xscale("log")
        best = int(np.argmax(scores))
        ax.scatter([values[best]], [scores[best]], color="red", zorder=3, label=f"best {param}={values[best]}")
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.legend()
        return ax
''')

FILES["ml_toolkit/utils.py"] = dedent('''
    """Мелкие утилиты: фиксация seed и журнал запусков."""

    from __future__ import annotations

    import json
    import random
    import time
    from pathlib import Path

    import numpy as np


    def set_seed(seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)


    def log_run(name: str, params: dict, metrics: dict, path: str = "runs/runs.jsonl") -> None:
        """Дописать запуск в JSONL-журнал."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        record = {"name": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "params": params, "metrics": metrics}
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\\n")
''')

FILES["scripts/train.py"] = dedent('''
    """Обучение модели из командной строки.

    Пример:
        python scripts/train.py --dataset breast_cancer --model hist_gb --lr 0.05 --max-iter 300
    """

    from __future__ import annotations

    import argparse
    import json
    import sys
    from pathlib import Path

    from sklearn.pipeline import make_pipeline

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from ml_toolkit.data import make_split  # noqa: E402
    from ml_toolkit.features import make_preprocessor  # noqa: E402
    from ml_toolkit.metrics import calc_metrics  # noqa: E402
    from ml_toolkit.models import build_model  # noqa: E402
    from ml_toolkit.utils import log_run, set_seed  # noqa: E402


    def parse_args(argv=None) -> argparse.Namespace:
        p = argparse.ArgumentParser(description="Train a classifier on a bundled dataset")
        p.add_argument("--dataset", default="breast_cancer")
        p.add_argument("--model", default="hist_gb", choices=["logreg", "random_forest", "hist_gb", "mlp"])
        p.add_argument("--lr", type=float, default=0.1, help="learning rate (hist_gb, mlp)")
        p.add_argument("--max-iter", type=int, default=200)
        p.add_argument("--C", type=float, default=1.0, help="inverse regularisation (logreg)")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--log", default="runs/runs.jsonl")
        return p.parse_args(argv)


    def model_params(args: argparse.Namespace) -> dict:
        if args.model == "hist_gb":
            return {"learning_rate": args.lr, "max_iter": args.max_iter, "random_state": args.seed}
        if args.model == "mlp":
            return {"learning_rate_init": args.lr, "max_iter": args.max_iter, "random_state": args.seed}
        if args.model == "logreg":
            return {"C": args.C}
        return {"random_state": args.seed}


    def main(argv=None) -> dict:
        args = parse_args(argv)
        set_seed(args.seed)
        split = make_split(args.dataset, seed=args.seed)
        model = make_pipeline(make_preprocessor(), build_model(args.model, **model_params(args)))
        model.fit(split.X_train, split.y_train)
        metrics = calc_metrics(split.y_val, model.predict(split.X_val), model.predict_proba(split.X_val))
        log_run(f"{args.dataset}-{args.model}", vars(args), metrics, args.log)
        print(json.dumps(metrics, indent=2))
        return metrics


    if __name__ == "__main__":
        main()
''')

FILES["scripts/legacy_metrics.py"] = dedent('''
    # Старый ручной подсчёт метрик из первой версии проекта.
    # УСТАРЕЛО: используйте ml_toolkit.metrics.calc_metrics.


    def confusion_counts(y_true, y_pred, positive=1):
        tp = fp = fn = tn = 0
        for t, p in zip(y_true, y_pred):
            if p == positive and t == positive:
                tp += 1
            elif p == positive:
                fp += 1
            elif t == positive:
                fn += 1
            else:
                tn += 1
        return tp, fp, fn, tn


    def f1_manual(y_true, y_pred, positive=1):
        """F1 для бинарной задачи без sklearn."""
        tp, fp, fn, _ = confusion_counts(y_true, y_pred, positive)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


    if __name__ == "__main__":
        print(f1_manual([1, 0, 1, 1, 0], [1, 0, 0, 1, 1]))
''')

# Longer modules (Э3): functions and classes that fixed line windows cut in the middle.

FILES["ml_toolkit/trainer.py"] = dedent(r'''
    """Обучение простого MLP на numpy: расписания learning rate, SGD с моментом,
    ранняя остановка и чекпоинты.

    Используется в experiments/08_numpy_mlp.ipynb как «ручная» альтернатива
    sklearn.neural_network.MLPClassifier из 04_digits_augmentation.
    """

    from __future__ import annotations

    import json
    import math
    from dataclasses import dataclass, field
    from pathlib import Path

    import numpy as np

    # ------------------------------------------------------------------ activations and loss


    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)


    def relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(x.dtype)


    def softmax(z: np.ndarray) -> np.ndarray:
        """Численно устойчивый softmax по строкам (вычитается максимум строки)."""
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


    def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
        """Средняя кросс-энтропия; y — целочисленные метки классов."""
        eps = 1e-12
        return float(-np.log(probs[np.arange(len(y)), y] + eps).mean())


    # ------------------------------------------------------------------ learning-rate schedules


    def constant_lr(base_lr: float, epoch: int, total_epochs: int) -> float:
        return base_lr


    def step_lr(base_lr: float, epoch: int, total_epochs: int, step_size: int = 10, gamma: float = 0.5) -> float:
        """Ступенчатое затухание: LR умножается на gamma каждые step_size эпох."""
        return base_lr * gamma ** (epoch // step_size)


    def cosine_lr(base_lr: float, epoch: int, total_epochs: int, min_lr: float = 1e-5, warmup_epochs: int = 3) -> float:
        """Косинусное затухание с линейным прогревом.

        Первые warmup_epochs эпох LR растёт линейно до base_lr, затем
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * t)),
        где t — доля пройденных эпох после прогрева.
        """
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


    SCHEDULES = {"constant": constant_lr, "step": step_lr, "cosine": cosine_lr}

    # ------------------------------------------------------------------ model


    class NumpyMLP:
        """Перцептрон с одним скрытым слоем: Linear -> ReLU -> Linear -> softmax.

        Веса инициализируются по He (нормальное распределение с дисперсией 2 / fan_in),
        смещения — нулями.
        """

        def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 0):
            rng = np.random.default_rng(seed)
            self.W1 = rng.normal(0.0, math.sqrt(2.0 / n_in), (n_in, n_hidden))
            self.b1 = np.zeros(n_hidden)
            self.W2 = rng.normal(0.0, math.sqrt(2.0 / n_hidden), (n_hidden, n_out))
            self.b2 = np.zeros(n_out)
            self._cache = None

        def forward(self, X: np.ndarray) -> np.ndarray:
            h_pre = X @ self.W1 + self.b1
            h = relu(h_pre)
            probs = softmax(h @ self.W2 + self.b2)
            self._cache = (X, h_pre, h, probs)
            return probs

        def backward(self, y: np.ndarray, weight_decay: float = 0.0) -> dict[str, np.ndarray]:
            """Градиенты кросс-энтропии по параметрам для последнего forward; L2 добавляется к весам, не к смещениям."""
            X, h_pre, h, probs = self._cache
            n = len(y)
            d_logits = probs.copy()
            d_logits[np.arange(n), y] -= 1.0
            d_logits /= n
            grads = {"W2": h.T @ d_logits + weight_decay * self.W2, "b2": d_logits.sum(axis=0)}
            d_h = (d_logits @ self.W2.T) * relu_grad(h_pre)
            grads["W1"] = X.T @ d_h + weight_decay * self.W1
            grads["b1"] = d_h.sum(axis=0)
            return grads

        def params(self) -> dict[str, np.ndarray]:
            return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

        def set_params(self, params: dict[str, np.ndarray]) -> None:
            for name, value in params.items():
                setattr(self, name, value.copy())

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return self.forward(X)

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self.forward(X).argmax(axis=1)


    # ------------------------------------------------------------------ optimiser


    class SGDMomentum:
        """SGD с моментом (heavy ball): v = momentum * v - lr * g; p = p + v."""

        def __init__(self, params: dict[str, np.ndarray], momentum: float = 0.9):
            self.momentum = momentum
            self.velocity = {k: np.zeros_like(v) for k, v in params.items()}

        def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float) -> None:
            for k in params:
                self.velocity[k] = self.momentum * self.velocity[k] - lr * grads[k]
                params[k] += self.velocity[k]  # in place: the arrays belong to the model


    # ------------------------------------------------------------------ early stopping and history


    @dataclass
    class EarlyStopping:
        """Останавливает обучение, если метрика на валидации не улучшалась patience эпох подряд.

        Улучшением считается рост больше чем на min_delta. По умолчанию patience = 5.
        """

        patience: int = 5
        min_delta: float = 1e-4
        best: float = -math.inf
        best_epoch: int = -1
        bad_epochs: int = 0

        def update(self, value: float, epoch: int) -> bool:
            """Учесть метрику эпохи; вернуть True, если пора остановиться."""
            if value > self.best + self.min_delta:
                self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
                return False
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience


    @dataclass
    class History:
        train_loss: list = field(default_factory=list)
        val_loss: list = field(default_factory=list)
        val_accuracy: list = field(default_factory=list)
        lr: list = field(default_factory=list)

        def best_epoch(self) -> int:
            return int(np.argmax(self.val_accuracy))


    # ------------------------------------------------------------------ checkpoints


    def save_checkpoint(model: NumpyMLP, path: str | Path, **meta) -> Path:
        """Сохранить веса в .npz, а метаданные (эпоха, метрика) — в JSON рядом с тем же именем."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **model.params())
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


    def load_checkpoint(model: NumpyMLP, path: str | Path) -> dict:
        """Загрузить веса из .npz в модель и вернуть метаданные из JSON (если есть)."""
        path = Path(path).with_suffix(".npz")
        data = np.load(path)
        model.set_params({k: data[k] for k in ("W1", "b1", "W2", "b2")})
        meta_path = path.with_suffix(".json")
        return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


    # ------------------------------------------------------------------ training loop


    class Trainer:
        """Обучение NumpyMLP мини-батчами с расписанием LR, weight decay и ранней остановкой.

        После обучения в модель возвращаются веса лучшей по val accuracy эпохи.
        """

        def __init__(
            self,
            model: NumpyMLP,
            lr: float = 0.05,
            epochs: int = 60,
            batch_size: int = 32,
            schedule: str = "cosine",
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            patience: int = 8,
            seed: int = 0,
            checkpoint_path: str | None = None,
            verbose: bool = False,
        ):
            if schedule not in SCHEDULES:
                raise ValueError(f"unknown schedule {schedule!r}, expected one of {sorted(SCHEDULES)}")
            self.model = model
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.schedule = schedule
            self.momentum = momentum
            self.weight_decay = weight_decay
            self.patience = patience
            self.seed = seed
            self.checkpoint_path = checkpoint_path
            self.verbose = verbose
            self.stopped_epoch: int | None = None

        def _run_epoch(self, X: np.ndarray, y: np.ndarray, opt: SGDMomentum, lr: float, rng) -> float:
            order = rng.permutation(len(X))
            losses = []
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                probs = self.model.forward(X[idx])
                losses.append(cross_entropy(probs, y[idx]))
                grads = self.model.backward(y[idx], self.weight_decay)
                opt.step(self.model.params(), grads, lr)
            return float(np.mean(losses))

        def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> History:
            rng = np.random.default_rng(self.seed)
            schedule = SCHEDULES[self.schedule]
            opt = SGDMomentum(self.model.params(), self.momentum)
            stopper = EarlyStopping(patience=self.patience)
            history = History()
            best_params = None
            for epoch in range(self.epochs):
                lr = schedule(self.lr, epoch, self.epochs)
                train_loss = self._run_epoch(X_train, y_train, opt, lr, rng)
                val_probs = self.model.predict_proba(X_val)
                val_acc = float((val_probs.argmax(axis=1) == y_val).mean())
                history.train_loss.append(train_loss)
                history.val_loss.append(cross_entropy(val_probs, y_val))
                history.val_accuracy.append(val_acc)
                history.lr.append(lr)
                stop = stopper.update(val_acc, epoch)
                if stopper.best_epoch == epoch:
                    best_params = {k: v.copy() for k, v in self.model.params().items()}
                    if self.checkpoint_path:
                        save_checkpoint(self.model, self.checkpoint_path, epoch=epoch, val_accuracy=val_acc)
                if self.verbose and (epoch % 10 == 0 or stop):
                    print(f"epoch {epoch:3d}  lr={lr:.4f}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
                if stop:
                    break
            self.stopped_epoch = epoch
            if best_params is not None:
                self.model.set_params(best_params)
            return history
''')

FILES["ml_toolkit/evaluation.py"] = dedent(r'''
    """Оценка моделей: кросс-валидация, подбор порога, калибровка, сравнение моделей."""

    from __future__ import annotations

    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    SCORINGS = ("roc_auc", "f1", "accuracy")


    def _score(model, X, y, scoring: str) -> float:
        if scoring == "roc_auc":
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                return float(roc_auc_score(y, proba[:, 1]))
            return float(roc_auc_score(y, proba, multi_class="ovr"))
        pred = model.predict(X)
        if scoring == "f1":
            return float(f1_score(y, pred, average="macro"))
        if scoring == "accuracy":
            return float((pred == y).mean())
        raise ValueError(f"unknown scoring {scoring!r}, expected one of {SCORINGS}")


    def cross_validate_model(model, X, y, n_splits: int = 5, seed: int = 42, scoring: str = "roc_auc") -> dict:
        """Стратифицированная k-fold кросс-валидация (по умолчанию 5 фолдов, shuffle, seed=42).

        Модель клонируется на каждом фолде. Возвращает среднее, стандартное отклонение
        (ddof=1) и значения по фолдам.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            fitted = clone(model).fit(X[train_idx], y[train_idx])
            scores.append(_score(fitted, X[val_idx], y[val_idx], scoring))
        scores = np.asarray(scores)
        return {"mean": float(scores.mean()), "std": float(scores.std(ddof=1)), "folds": scores.round(4).tolist()}


    def compare_models(models: dict, X, y, n_splits: int = 5, seed: int = 42, scoring: str = "roc_auc") -> pd.DataFrame:
        """Кросс-валидация нескольких моделей на одинаковых фолдах; таблица отсортирована по среднему скору."""
        rows = []
        for name, model in models.items():
            res = cross_validate_model(model, X, y, n_splits=n_splits, seed=seed, scoring=scoring)
            rows.append({"model": name, "mean": res["mean"], "std": res["std"]})
        return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True).round(4)


    def threshold_search(y_true, y_score, metric: str = "f1", grid=None) -> tuple[float, float]:
        """Подбор порога бинарного классификатора, максимизирующего F1 (или accuracy).

        По умолчанию перебираются 99 порогов от 0.01 до 0.99 с шагом 0.01.
        Возвращает (лучший порог, значение метрики).
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        grid = np.linspace(0.01, 0.99, 99) if grid is None else np.asarray(grid)
        best_t, best_v = 0.5, -1.0
        for t in grid:
            pred = (y_score >= t).astype(int)
            value = f1_score(y_true, pred) if metric == "f1" else float((pred == y_true).mean())
            if value > best_v:
                best_t, best_v = float(t), float(value)
        return best_t, best_v


    def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
        """Expected Calibration Error.

        Объекты раскладываются по n_bins бинам равной ширины по уверенности (максимальной
        вероятности); ECE — средневзвешенное по доле объектов в бине |accuracy - confidence|.
        Для бинарной задачи y_prob может быть вектором вероятностей положительного класса.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        if y_prob.ndim == 2:
            conf, pred = y_prob.max(axis=1), y_prob.argmax(axis=1)
        else:
            pred = (y_prob >= 0.5).astype(int)
            conf = np.where(pred == 1, y_prob, 1.0 - y_prob)
        correct = (pred == y_true).astype(float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
        return float(ece)


    def classification_report_df(y_true, y_pred, labels=None) -> pd.DataFrame:
        """Precision / recall / F1 / support по классам в виде DataFrame."""
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        index = labels if labels is not None else np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
        return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=index).round(4)


    def bootstrap_compare(y_true, pred_a, pred_b, metric, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
        """Парный бутстреп разницы метрики двух моделей на одной выборке.

        Возвращает (наблюдаемая разница metric(a) - metric(b), одностороннее p-значение
        гипотезы «a не лучше b» — доля бутстреп-выборок, где разница <= 0).
        """
        y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
        rng = np.random.default_rng(seed)
        observed = metric(y_true, pred_a) - metric(y_true, pred_b)
        n = len(y_true)
        worse = 0
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx]) <= 0:
                worse += 1
        return float(observed), worse / n_boot


    def learning_curve_table(model, X, y, fractions=(0.1, 0.25, 0.5, 1.0), n_splits: int = 5, seed: int = 42,
                             scoring: str = "roc_auc") -> pd.DataFrame:
        """Качество на кросс-валидации при обучении на доле train (стратифицированная подвыборка)."""
        rng = np.random.default_rng(seed)
        rows = []
        for frac in fractions:
            idx = np.concatenate([
                rng.choice(np.flatnonzero(y == c), max(2, int(frac * np.sum(y == c))), replace=False)
                for c in np.unique(y)
            ])
            res = cross_validate_model(model, X[idx], y[idx], n_splits=n_splits, seed=seed, scoring=scoring)
            rows.append({"fraction": frac, "n": len(idx), "mean": res["mean"], "std": res["std"]})
        return pd.DataFrame(rows).round(4)
''')

FILES["ml_toolkit/preprocessing.py"] = dedent(r'''
    """Трансформеры признаков в стиле scikit-learn и сборка табличного пайплайна."""

    from __future__ import annotations

    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler


    class OutlierClipper(BaseEstimator, TransformerMixin):
        """Обрезает каждый признак по квантилям, посчитанным на train.

        По умолчанию нижний квантиль 0.01 и верхний 0.99 (то есть 1% и 99%).
        """

        def __init__(self, lower: float = 0.01, upper: float = 0.99):
            self.lower = lower
            self.upper = upper

        def fit(self, X, y=None):
            if not 0.0 <= self.lower < self.upper <= 1.0:
                raise ValueError("expected 0 <= lower < upper <= 1")
            X = np.asarray(X, dtype=float)
            self.lo_ = np.quantile(X, self.lower, axis=0)
            self.hi_ = np.quantile(X, self.upper, axis=0)
            return self

        def transform(self, X):
            return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


    class LogTransformer(BaseEstimator, TransformerMixin):
        """log1p для признаков с тяжёлым хвостом.

        Если на train встречаются отрицательные значения, признак сначала сдвигается на
        -min (сдвиг shift_ запоминается и применяется к новым данным).
        """

        def __init__(self, columns=None):
            self.columns = columns

        def fit(self, X, y=None):
            X = np.asarray(X, dtype=float)
            cols = range(X.shape[1]) if self.columns is None else self.columns
            self.columns_ = list(cols)
            self.shift_ = np.maximum(0.0, -X[:, self.columns_].min(axis=0))
            return self

        def transform(self, X):
            X = np.array(X, dtype=float, copy=True)
            shifted = np.maximum(X[:, self.columns_] + self.shift_, 0.0)
            X[:, self.columns_] = np.log1p(shifted)
            return X


    class FrequencyEncoder(BaseEstimator, TransformerMixin):
        """Кодирует категориальный признак частотой категории в train.

        Неизвестные на train категории получают unknown_value (по умолчанию 0.0).
        Работает с одномерным массивом или с каждой колонкой двумерного.
        """

        def __init__(self, unknown_value: float = 0.0, normalize: bool = True):
            self.unknown_value = unknown_value
            self.normalize = normalize

        def fit(self, X, y=None):
            X = self._as_2d(X)
            self.maps_ = []
            for j in range(X.shape[1]):
                values, counts = np.unique(X[:, j], return_counts=True)
                freq = counts / counts.sum() if self.normalize else counts.astype(float)
                self.maps_.append(dict(zip(values.tolist(), freq.tolist())))
            return self

        def transform(self, X):
            X = self._as_2d(X)
            out = np.empty(X.shape, dtype=float)
            for j, mapping in enumerate(self.maps_):
                out[:, j] = [mapping.get(v, self.unknown_value) for v in X[:, j].tolist()]
            return out

        @staticmethod
        def _as_2d(X):
            X = np.asarray(X, dtype=object)
            return X.reshape(-1, 1) if X.ndim == 1 else X


    class ColumnDropper(BaseEstimator, TransformerMixin):
        """Удаляет колонки с почти нулевой дисперсией (меньше threshold на train)."""

        def __init__(self, threshold: float = 1e-8):
            self.threshold = threshold

        def fit(self, X, y=None):
            X = np.asarray(X, dtype=float)
            self.keep_ = np.flatnonzero(X.var(axis=0) > self.threshold)
            return self

        def transform(self, X):
            return np.asarray(X, dtype=float)[:, self.keep_]


    def build_tabular_pipeline(clip: bool = True, log_columns=None, poly_degree: int = 1, drop_constant: bool = True) -> Pipeline:
        """Пайплайн для числовых табличных данных.

        Порядок шагов: ColumnDropper -> OutlierClipper -> LogTransformer -> PolynomialFeatures -> StandardScaler.
        Шаги, выключенные аргументами, пропускаются.
        """
        steps = []
        if drop_constant:
            steps.append(("drop", ColumnDropper()))
        if clip:
            steps.append(("clip", OutlierClipper()))
        if log_columns is not None:
            steps.append(("log", LogTransformer(columns=log_columns)))
        if poly_degree > 1:
            steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
        steps.append(("scale", StandardScaler()))
        return Pipeline(steps)
''')

FILES["scripts/hparam_search.py"] = dedent(r'''
    """Случайный поиск гиперпараметров HistGradientBoosting с кросс-валидацией.

    Пример:
        python scripts/hparam_search.py --dataset breast_cancer --n-iter 30 --seed 0
    Результаты сохраняются в runs/hparam_search_<dataset>.csv.
    """

    from __future__ import annotations

    import argparse
    import math
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.pipeline import make_pipeline

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from ml_toolkit.data import load_dataset  # noqa: E402
    from ml_toolkit.evaluation import cross_validate_model  # noqa: E402
    from ml_toolkit.models import build_model  # noqa: E402
    from ml_toolkit.preprocessing import build_tabular_pipeline  # noqa: E402

    # (distribution, arguments): log_uniform(low, high), int(low, high) inclusive, choice(options)
    SEARCH_SPACE = {
        "learning_rate": ("log_uniform", 0.01, 0.3),
        "max_leaf_nodes": ("int", 8, 64),
        "min_samples_leaf": ("int", 5, 50),
        "l2_regularization": ("log_uniform", 1e-3, 10.0),
        "max_iter": ("choice", [100, 200, 300]),
    }


    def sample_params(space: dict, rng: np.random.Generator) -> dict:
        params = {}
        for name, (kind, *args) in space.items():
            if kind == "log_uniform":
                low, high = args
                params[name] = float(math.exp(rng.uniform(math.log(low), math.log(high))))
            elif kind == "int":
                low, high = args
                params[name] = int(rng.integers(low, high + 1))
            elif kind == "choice":
                (options,) = args
                params[name] = options[int(rng.integers(len(options)))]
            else:
                raise ValueError(f"unknown distribution {kind!r} for {name}")
        return params


    def random_search(X, y, n_iter: int = 20, seed: int = 0, n_splits: int = 5, scoring: str = "roc_auc") -> pd.DataFrame:
        """n_iter случайных конфигураций из SEARCH_SPACE, каждая оценивается k-fold кросс-валидацией."""
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_iter):
            params = sample_params(SEARCH_SPACE, rng)
            model = make_pipeline(build_tabular_pipeline(), build_model("hist_gb", random_state=seed, **params))
            res = cross_validate_model(model, X, y, n_splits=n_splits, seed=seed, scoring=scoring)
            rows.append({"trial": i, **params, "cv_mean": res["mean"], "cv_std": res["std"]})
        return pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)


    def parse_args(argv=None) -> argparse.Namespace:
        p = argparse.ArgumentParser(description="Random search for HistGradientBoosting")
        p.add_argument("--dataset", default="breast_cancer")
        p.add_argument("--n-iter", type=int, default=20)
        p.add_argument("--n-splits", type=int, default=5)
        p.add_argument("--scoring", default="roc_auc", choices=["roc_auc", "f1", "accuracy"])
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default=None, help="CSV path (default: runs/hparam_search_<dataset>.csv)")
        return p.parse_args(argv)


    def main(argv=None) -> pd.DataFrame:
        args = parse_args(argv)
        X, y, _ = load_dataset(args.dataset)
        table = random_search(X, y, n_iter=args.n_iter, seed=args.seed, n_splits=args.n_splits, scoring=args.scoring)
        out = Path(args.out or f"runs/hparam_search_{args.dataset}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(table.head(5).round(4).to_string(index=False))
        print(f"saved {len(table)} trials to {out}")
        return table


    if __name__ == "__main__":
        main()
''')

# --------------------------------------------------------------------------
# Notebooks: list of ("md" | "code", source)
# --------------------------------------------------------------------------

SETUP = dedent("""
    import sys
    sys.path.append("..")
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    %config InlineBackend.figure_format = "png"
    plt.rcParams["figure.dpi"] = 72
""")

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {}

NOTEBOOKS["experiments/01_eda_breast_cancer.ipynb"] = [
    ("md", "# EDA: Breast Cancer Wisconsin\n\nРазведочный анализ датасета перед экспериментами с классификацией "
           "(см. `02_logreg_baseline` и `03_gradient_boosting`)."),
    ("code", SETUP + "from ml_toolkit.data import load_dataset"),
    ("code", 'X, y, names = load_dataset("breast_cancer")\ndf = pd.DataFrame(X, columns=names)\ndf["target"] = y\nprint(df.shape)'),
    ("md", "## Баланс классов\n\n`target = 0` — злокачественная опухоль, `target = 1` — доброкачественная."),
    ("code", 'df["target"].value_counts()'),
    ("md", "## Связь признаков с таргетом"),
    ("code", 'corr = df.corr()["target"].drop("target").sort_values()\ncorr.head(5)'),
    ("code", dedent("""
        fig, ax = plt.subplots(figsize=(5, 3))
        for cls, label in [(0, "malignant"), (1, "benign")]:
            ax.hist(df.loc[df.target == cls, "worst concave points"], bins=30, alpha=0.6, label=label)
        ax.set_xlabel("worst concave points")
        ax.legend()
        plt.show()
    """)),
    ("md", "**Вывод:** классы умеренно несбалансированы, поэтому дальше используем стратифицированное разбиение "
           "и смотрим не только на accuracy, но и на F1 и ROC-AUC. Признаки формы опухоли (concave points, perimeter, "
           "radius) сильнее всего отрицательно коррелируют с таргетом."),
]

NOTEBOOKS["experiments/02_logreg_baseline.ipynb"] = [
    ("md", "# Бейзлайн: логистическая регрессия\n\nБинарная классификация `breast_cancer`, "
           "разбиение 60/20/20 со стратификацией, метрики на валидации."),
    ("code", SETUP + dedent("""
        from sklearn.pipeline import make_pipeline
        from ml_toolkit.data import make_split
        from ml_toolkit.features import make_preprocessor
        from ml_toolkit.models import build_model
        from ml_toolkit.metrics import calc_metrics, bootstrap_ci
        from ml_toolkit.viz import plot_roc
    """)),
    ("code", 'split = make_split("breast_cancer", seed=42)\nsplit.sizes()'),
    ("md", "## Обучение\n\nСтандартизация признаков + `LogisticRegression(C=1.0)`."),
    ("code", 'model = make_pipeline(make_preprocessor(), build_model("logreg", C=1.0))\nmodel.fit(split.X_train, split.y_train)'),
    ("code", dedent("""
        proba = model.predict_proba(split.X_val)
        pred = model.predict(split.X_val)
        metrics = calc_metrics(split.y_val, pred, proba)
        print({k: round(v, 4) for k, v in metrics.items()})
    """)),
    ("code", dedent("""
        from sklearn.metrics import f1_score
        lo, hi = bootstrap_ci(lambda t, p: f1_score(t, p, average="macro"), split.y_val, pred)
        print(f"F1 95% CI: [{lo:.3f}, {hi:.3f}]")
    """)),
    ("code", 'plot_roc(split.y_val, proba[:, 1], label="logreg C=1.0")\nplt.show()'),
    ("md", "Логистическая регрессия даёт сильный бейзлайн. Следующий шаг — градиентный бустинг "
           "(`03_gradient_boosting.ipynb`)."),
]

NOTEBOOKS["experiments/03_gradient_boosting.ipynb"] = [
    ("md", "# Градиентный бустинг (HistGradientBoosting)\n\nСравниваем с бейзлайном из `02_logreg_baseline`, "
           "та же задача и то же разбиение (seed=42)."),
    ("code", SETUP + dedent("""
        from sklearn.pipeline import make_pipeline
        from ml_toolkit.data import make_split
        from ml_toolkit.features import make_preprocessor
        from ml_toolkit.models import build_model
        from ml_toolkit.metrics import calc_metrics
        from ml_toolkit.viz import plot_param_curve
    """)),
    ("code", 'split = make_split("breast_cancer", seed=42)'),
    ("md", "## Подбор learning rate"),
    ("code", dedent("""
        rows = []
        for lr in [0.01, 0.03, 0.05, 0.1, 0.3]:
            clf = make_pipeline(make_preprocessor(), build_model(
                "hist_gb", learning_rate=lr, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, random_state=42))
            clf.fit(split.X_train, split.y_train)
            m = calc_metrics(split.y_val, clf.predict(split.X_val), clf.predict_proba(split.X_val))
            rows.append({"learning_rate": lr, **m})
        lr_table = pd.DataFrame(rows).round(4)
        lr_table
    """)),
    ("code", 'plot_param_curve(lr_table.learning_rate.tolist(), lr_table.roc_auc.tolist(), "learning_rate", "ROC-AUC (val)")\nplt.show()'),
    ("md", "## Финальная модель\n\nБерём learning rate с лучшим ROC-AUC на валидации и оцениваем на тесте один раз."),
    ("code", dedent("""
        best_lr = float(lr_table.sort_values(["roc_auc", "f1"], ascending=False).iloc[0].learning_rate)
        final = make_pipeline(make_preprocessor(), build_model(
            "hist_gb", learning_rate=best_lr, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, random_state=42))
        final.fit(split.X_train, split.y_train)
        test_metrics = calc_metrics(split.y_test, final.predict(split.X_test), final.predict_proba(split.X_test))
        print(f"best learning_rate = {best_lr}")
        print("TEST:", {k: round(v, 4) for k, v in test_metrics.items()})
    """)),
]

NOTEBOOKS["experiments/04_digits_augmentation.ipynb"] = [
    ("md", "# Digits: MLP с аугментацией сдвигами\n\nПроверяем, помогает ли аугментация сдвигами на 1 пиксель "
           "(`ml_toolkit.features.augment_digits`) многослойному перцептрону на датасете digits (8x8)."),
    ("code", SETUP + dedent("""
        from sklearn.pipeline import make_pipeline
        from ml_toolkit.data import make_split
        from ml_toolkit.features import make_preprocessor, augment_digits
        from ml_toolkit.models import build_model
        from ml_toolkit.metrics import calc_metrics
        from ml_toolkit.viz import plot_confusion
    """)),
    ("code", 'split = make_split("digits", seed=0)\nsplit.sizes()'),
    ("md", "## Гиперпараметры\n\nОдинаковые для обоих вариантов, меняется только обучающая выборка."),
    ("code", 'HPARAMS = dict(hidden_layer_sizes=(128,), learning_rate_init=3e-3, alpha=1e-4,\n               batch_size=64, max_iter=200, random_state=0)'),
    ("code", dedent("""
        base = make_pipeline(make_preprocessor(), build_model("mlp", **HPARAMS))
        base.fit(split.X_train, split.y_train)
        m_base = calc_metrics(split.y_val, base.predict(split.X_val), base.predict_proba(split.X_val))
        print("no augmentation:", {k: round(v, 4) for k, v in m_base.items()})
    """)),
    ("code", dedent("""
        X_aug, y_aug = augment_digits(split.X_train, split.y_train)
        print("train size:", len(split.y_train), "->", len(y_aug))
        aug = make_pipeline(make_preprocessor(), build_model("mlp", **HPARAMS))
        aug.fit(X_aug, y_aug)
        m_aug = calc_metrics(split.y_val, aug.predict(split.X_val), aug.predict_proba(split.X_val))
        print("with augmentation:", {k: round(v, 4) for k, v in m_aug.items()})
    """)),
    ("code", 'pd.DataFrame({"no_aug": m_base, "shift_aug": m_aug}).T.round(4)'),
    ("code", 'plot_confusion(split.y_val, aug.predict(split.X_val))\nplt.show()'),
    ("md", "Итог по аугментации — в таблице выше. Матрица ошибок показывает, какие цифры путаются чаще всего."),
]

NOTEBOOKS["experiments/05_wine_clustering.ipynb"] = [
    ("md", "# Wine clustering\n\nUnsupervised sanity check: do KMeans clusters recover the three wine cultivars?"),
    ("code", SETUP + dedent("""
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        from ml_toolkit.data import load_dataset
        from ml_toolkit.features import make_preprocessor
    """)),
    ("code", 'X, y, names = load_dataset("wine")\nXs = make_preprocessor().fit_transform(X)\nprint(Xs.shape)'),
    ("md", "## Choosing k by silhouette"),
    ("code", dedent("""
        sil = {}
        for k in range(2, 7):
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
            sil[k] = round(silhouette_score(Xs, labels), 4)
        pd.Series(sil, name="silhouette")
    """)),
    ("code", dedent("""
        km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xs)
        print("ARI vs cultivar labels:", round(adjusted_rand_score(y, km.labels_), 4))
    """)),
    ("code", dedent("""
        pcs = PCA(n_components=2, random_state=0).fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.scatter(pcs[:, 0], pcs[:, 1], c=km.labels_, s=12, cmap="viridis")
        ax.set_title("KMeans (k=3) on PCA projection")
        plt.show()
    """)),
    ("md", "Clusters align well with the true cultivars; standardisation is essential here because feature scales differ "
           "by orders of magnitude (e.g. proline vs hue)."),
]

NOTEBOOKS["experiments/06_diabetes_regression.ipynb"] = [
    ("md", "# Diabetes progression: Ridge regression\n\nRegression baseline on the `diabetes` dataset; "
           "metrics are RMSE and R² (`ml_toolkit.metrics.regression_metrics`)."),
    ("code", SETUP + dedent("""
        from sklearn.pipeline import make_pipeline
        from ml_toolkit.data import make_split
        from ml_toolkit.features import make_preprocessor
        from ml_toolkit.models import build_model
        from ml_toolkit.metrics import regression_metrics
    """)),
    ("code", 'split = make_split("diabetes", seed=7, stratify=False)\nsplit.sizes()'),
    ("md", "## Regularisation sweep"),
    ("code", dedent("""
        rows = []
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            reg = make_pipeline(make_preprocessor(), build_model("ridge", alpha=alpha)).fit(split.X_train, split.y_train)
            rows.append({"alpha": alpha, **regression_metrics(split.y_val, reg.predict(split.X_val))})
        alpha_table = pd.DataFrame(rows).round(3)
        alpha_table
    """)),
    ("code", dedent("""
        best_alpha = float(alpha_table.sort_values("rmse").iloc[0].alpha)
        final = make_pipeline(make_preprocessor(), build_model("ridge", alpha=best_alpha)).fit(split.X_train, split.y_train)
        print(f"best alpha = {best_alpha}")
        print("TEST:", {k: round(v, 3) for k, v in regression_metrics(split.y_test, final.predict(split.X_test)).items()})
    """)),
    ("md", "Linear model is a reasonable baseline; next step would be gradient boosting on the same split."),
]

NOTEBOOKS["experiments/07_scratch.ipynb"] = [
    ("md", "# Черновик\n\nБыстрые проверки, ячейки запускались не по порядку."),
    ("code", SETUP),
    ("code", "from ml_toolkit.metrics import calc_metrics\ncalc_metrics([0, 1, 1, 0], [0, 1, 0, 0])"),
    ("code", "# TODO: перенести в ml_toolkit.metrics\nresult = f1_manual([1, 0, 1], [1, 1, 1])"),
    ("md", "Не забыть: `f1_manual` лежит в `scripts/legacy_metrics.py`, в ноутбук не импортирован."),
]

NOTEBOOKS["experiments/08_numpy_mlp.ipynb"] = [
    ("md", "# MLP на numpy: косинусное расписание и ранняя остановка\n\nРучная реализация из `ml_toolkit.trainer` "
           "на тех же данных digits и том же разбиении (seed=0), что и `04_digits_augmentation`."),
    ("code", SETUP + dedent("""
        from ml_toolkit.data import make_split
        from ml_toolkit.features import make_preprocessor
        from ml_toolkit.metrics import calc_metrics
        from ml_toolkit.trainer import NumpyMLP, Trainer
    """)),
    ("code", dedent("""
        split = make_split("digits", seed=0)
        pre = make_preprocessor().fit(split.X_train)
        X_tr, X_va, X_te = (pre.transform(x) for x in (split.X_train, split.X_val, split.X_test))
        print(X_tr.shape, X_va.shape, X_te.shape)
    """)),
    ("md", "## Обучение\n\nlr = 0.005, до 80 эпох, батч 64, косинусное расписание с прогревом, SGD с моментом 0.9, "
           "weight decay 1e-3, ранняя остановка с patience = 10 по accuracy на валидации."),
    ("code", dedent("""
        model = NumpyMLP(n_in=64, n_hidden=128, n_out=10, seed=0)
        trainer = Trainer(model, lr=0.005, epochs=80, batch_size=64, schedule="cosine", momentum=0.9,
                          weight_decay=1e-3, patience=10, seed=0, verbose=True)
        history = trainer.fit(X_tr, split.y_train, X_va, split.y_val)
        print(f"stopped at epoch {trainer.stopped_epoch}, best epoch {history.best_epoch()}, "
              f"best val accuracy {max(history.val_accuracy):.4f}")
    """)),
    ("code", dedent("""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
        ax1.plot(history.train_loss, label="train")
        ax1.plot(history.val_loss, label="val")
        ax1.set_title("cross-entropy")
        ax1.legend()
        ax2.plot(history.val_accuracy, color="tab:green")
        ax2.set_title("val accuracy")
        ax3 = ax2.twinx()
        ax3.plot(history.lr, color="tab:gray", ls="--")
        ax3.set_ylabel("lr")
        plt.tight_layout()
        plt.show()
    """)),
    ("md", "## Сравнение расписаний learning rate\n\nОдинаковые гиперпараметры, до 40 эпох."),
    ("code", dedent("""
        rows = []
        for schedule in ["constant", "step", "cosine"]:
            m = NumpyMLP(64, 128, 10, seed=0)
            t = Trainer(m, lr=0.005, epochs=40, batch_size=64, weight_decay=1e-3, schedule=schedule, patience=10, seed=0)
            h = t.fit(X_tr, split.y_train, X_va, split.y_val)
            rows.append({"schedule": schedule, "best_val_acc": max(h.val_accuracy), "best_epoch": h.best_epoch(),
                         "stopped_epoch": t.stopped_epoch})
        schedule_table = pd.DataFrame(rows).round(4)
        schedule_table
    """)),
    ("md", "## Тест"),
    ("code", dedent("""
        test_metrics = calc_metrics(split.y_test, model.predict(X_te), model.predict_proba(X_te))
        print("TEST:", {k: round(v, 4) for k, v in test_metrics.items()})
    """)),
    ("md", "Ручной MLP сопоставим с `MLPClassifier` из `04_digits_augmentation`; лучшее расписание — в таблице выше."),
]

SCRATCH_EXEC_COUNTS = {0: 1, 1: 4, 2: 2}  # code cell index -> execution count (run out of order)


def build_notebook(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for i, (kind, src) in enumerate(cells, start=1):
        src = src.rstrip("\n")
        cell = new_markdown_cell(src) if kind == "md" else new_code_cell(src)
        cell.id = f"cell-{i:02d}"  # stable ids: rebuilding the corpus must not produce spurious diffs
        nb.cells.append(cell)
    return nb


def strip_volatile_metadata(nb: nbformat.NotebookNode) -> None:
    """Drop execution timestamps so that only real output changes show up in git."""
    for cell in nb.cells:
        cell.metadata.pop("execution", None)


def use_current_interpreter_for_kernel() -> None:
    """The stock python3 kernelspec launches whatever ``python`` is first on PATH;
    put this interpreter's environment first so notebooks run where the script runs."""
    env_dir = Path(sys.executable).parent
    dirs = [env_dir, env_dir / "Library" / "bin", env_dir / "Scripts", env_dir / "bin"]
    os.environ["PATH"] = os.pathsep.join([str(d) for d in dirs if d.exists()] + [os.environ.get("PATH", "")])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # keep __pycache__ out of the corpus


def execute(nb: nbformat.NotebookNode, cwd: Path, allow_errors: bool) -> None:
    from nbclient import NotebookClient

    NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=allow_errors,
                   resources={"metadata": {"path": str(cwd)}}).execute()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-execute", action="store_true", help="write notebooks without running them")
    args = ap.parse_args()
    use_current_interpreter_for_kernel()

    if OUT.exists():
        shutil.rmtree(OUT)
    for rel, content in FILES.items():
        path = OUT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")

    for rel, cells in NOTEBOOKS.items():
        path = OUT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        nb = build_notebook(cells)
        if not args.no_execute:
            print(f"executing {rel} …", flush=True)
            execute(nb, path.parent, allow_errors=rel.endswith("07_scratch.ipynb"))
        if rel.endswith("07_scratch.ipynb"):
            code_cells = [c for c in nb.cells if c.cell_type == "code"]
            for i, count in SCRATCH_EXEC_COUNTS.items():
                code_cells[i].execution_count = count
                for out in code_cells[i].get("outputs", []):
                    if "execution_count" in out:
                        out["execution_count"] = count
        nb.metadata.pop("widgets", None)
        strip_volatile_metadata(nb)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            nbformat.write(nb, f)

    (OUT / "README.md").write_text(dedent("""
        # Demo corpus

        Synthetic research archive for the public demo of local-rag-agent. All files are generated by
        `scripts/build_demo_corpus.py`; notebooks are executed on scikit-learn's bundled datasets
        (Breast Cancer Wisconsin, Wine, Digits, Diabetes). Do not edit by hand — regenerate instead.
    """), encoding="utf-8", newline="\n")
    print(f"demo corpus written to {OUT}")


if __name__ == "__main__":
    main()
