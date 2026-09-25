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
