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
