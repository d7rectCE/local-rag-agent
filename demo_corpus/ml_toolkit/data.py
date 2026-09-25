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
