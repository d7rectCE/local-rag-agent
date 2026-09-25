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
