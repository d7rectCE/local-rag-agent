"""Metrics (ТЗ 5.2): retrieval (Recall@k, MRR, nDCG@10), answers, refusals, routing,
latency; percentile bootstrap confidence intervals over questions."""

from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Sequence

import numpy as np

from rag_agent.evaluation.dataset import SourceRef
from rag_agent.schema import Node

KS = (1, 3, 5, 10)


def first_match_ranks(retrieved: Sequence[Node], refs: Sequence[SourceRef]) -> tuple[dict[int, int], list[int]]:
    """For each reference: the rank (1-based) of the first retrieved node matching it.
    ``gains[i]`` is 1 when node i covers a reference not covered by an earlier
    node, so overlapping chunks of the same span are not counted twice."""
    first: dict[int, int] = {}
    gains: list[int] = []
    for rank, node in enumerate(retrieved, start=1):
        new = [j for j, ref in enumerate(refs) if j not in first and ref.matches(node)]
        for j in new:
            first[j] = rank
        gains.append(1 if new else 0)
    return first, gains


def retrieval_metrics(retrieved: Sequence[Node], refs: Sequence[SourceRef], ks: Iterable[int] = KS) -> dict[str, float]:
    first, gains = first_match_ranks(retrieved, refs)
    n = len(refs)
    out: dict[str, float] = {}
    for k in ks:
        covered = sum(1 for r in first.values() if r <= k)
        out[f"recall@{k}"] = covered / n
        out[f"hit@{k}"] = float(covered > 0)
    out["mrr"] = 1.0 / min(first.values()) if first else 0.0
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:10]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n, 10)))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def must_include_ok(text: str, patterns: Sequence[str]) -> bool | None:
    if not patterns:
        return None
    return all(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def citation_precision(cited: Sequence[Node], refs: Sequence[SourceRef]) -> float | None:
    """Share of cited fragments that belong to a reference span (annotation-based proxy)."""
    if not cited:
        return None
    return sum(1 for node in cited if any(ref.matches(node) for ref in refs)) / len(cited)


def bootstrap_ci(
    values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float] | None:
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(seed)
    stats = [stat(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n_boot)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mean_ci(values: Sequence[float | None], n_boot: int = 1000) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "n": 0, "ci": None}
    return {"mean": float(np.mean(vals)), "n": len(vals), "ci": bootstrap_ci(vals, n_boot=n_boot)}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired comparison of two configurations on the same questions: mean of
    (a - b), its percentile CI and the one-sided p-value of "a is not better than b"
    (share of resamples with mean difference <= 0)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) < 2:
        return {"diff": float(d.mean()) if len(d) else None, "ci": None, "p": None, "n": len(d)}
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(d.mean()), "ci": (float(lo), float(hi)), "p": float((means <= 0).mean()), "n": len(d)}


def precision_recall(predicted: Sequence[bool], actual: Sequence[bool]) -> dict:
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(a and not p for p, a in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def percentiles(values: Sequence[float], qs: Iterable[int] = (50, 95)) -> dict[str, float | None]:
    vals = [v for v in values if v is not None]
    return {f"p{q}": (float(np.percentile(vals, q)) if vals else None) for q in qs}
