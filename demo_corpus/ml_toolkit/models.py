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
