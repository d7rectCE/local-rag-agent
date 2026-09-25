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
