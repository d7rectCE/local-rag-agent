"""Calibrates the CRAG thresholds (agent.crag_lower / crag_upper) on an eval set.

For every question answered from the files, the fragments retrieved by the default
pipeline are scored by the cross-encoder, as the evaluator does at run time. The
best score separates questions whose answer is in the archive from unanswerable
ones (class Q6): the table shows, per candidate threshold, how many Q6 questions a
refusal below it would catch and how many answerable questions it would wrongly
refuse. No LLM calls: embedder and reranker only.

    python scripts/calibrate_crag.py evalsets/demo_v4.yaml --out reports/e7/crag_calibration.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rag_agent.config import load_settings
from rag_agent.engine import Engine
from rag_agent.evaluation.dataset import load_evalset

THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evalset", nargs="?", default="evalsets/demo_v4.yaml")
    ap.add_argument("--out", default=None, help="write the Markdown table here (and .json next to it)")
    args = ap.parse_args()

    settings = load_settings()
    settings.catalog.extract = False  # no LLM
    es = load_evalset(args.evalset)
    engine = Engine(settings)
    engine.index_folder(es.corpus_root())
    header = settings.chunking.context_header
    rows = []
    for item in es.items:
        if item.expected_route != "corpus":
            continue
        hits = engine.search(item.retrieval_query)
        scores = engine.reranker.score(item.retrieval_query, [h.node.embedding_text(header) for h in hits]) if hits else []
        rows.append({"id": item.id, "cls": item.cls, "best": float(max(scores, default=0.0))})
    engine.close()

    q6 = np.array([r["best"] for r in rows if r["cls"] == "Q6"])
    ok = np.array([r["best"] for r in rows if r["cls"] != "Q6"])
    lines = [f"# Калибровка порогов CRAG: {es.name}", "",
             f"Лучшая оценка кросс-энкодера среди найденных фрагментов: вопросов с ответом в файлах — {len(ok)}, "
             f"неотвечаемых (Q6) — {len(q6)}.", "",
             "| | min | p10 | p25 | медиана | p75 | max |", "|---|---|---|---|---|---|---|"]
    for name, arr in (("с ответом", ok), ("Q6", q6)):
        if len(arr):
            p = np.percentile(arr, [10, 25, 50, 75])
            lines.append(f"| {name} | {arr.min():.3f} | {p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {p[3]:.3f} | {arr.max():.3f} |")
    lines += ["", "Отказ при оценке ниже порога:", "",
              "| Порог | Пойманных Q6 (recall) | Ложных отказов | Precision отказов |", "|---|---|---|---|"]
    for t in THRESHOLDS:
        caught, false = int((q6 < t).sum()), int((ok < t).sum())
        prec = caught / (caught + false) if caught + false else float("nan")
        lines.append(f"| {t:g} | {caught}/{len(q6)} ({caught / max(1, len(q6)):.2f}) | {false} | {prec:.2f} |")
    lines += ["", "По вопросам:", "", "| id | класс | лучшая оценка |", "|---|---|---|"]
    lines += [f"| {r['id']} | {r['cls']} | {r['best']:.3f} |" for r in sorted(rows, key=lambda r: r["best"])]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
