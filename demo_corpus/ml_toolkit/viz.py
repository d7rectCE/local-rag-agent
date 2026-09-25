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
