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

SCRATCH_EXEC_COUNTS = {0: 1, 1: 4, 2: 2}  # code cell index -> execution count (run out of order)


def build_notebook(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, src in cells:
        src = src.rstrip("\n")
        nb.cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))
    return nb


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
        nbformat.write(nb, str(path))

    (OUT / "README.md").write_text(dedent("""
        # Demo corpus

        Synthetic research archive for the public demo of local-rag-agent. All files are generated by
        `scripts/build_demo_corpus.py`; notebooks are executed on scikit-learn's bundled datasets
        (Breast Cancer Wisconsin, Wine, Digits, Diabetes). Do not edit by hand — regenerate instead.
    """), encoding="utf-8", newline="\n")
    print(f"demo corpus written to {OUT}")


if __name__ == "__main__":
    main()
