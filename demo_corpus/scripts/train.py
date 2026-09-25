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
