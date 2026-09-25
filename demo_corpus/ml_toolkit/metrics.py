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
