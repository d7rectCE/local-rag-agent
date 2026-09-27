"""H4 (Э5): does the visual index beat "text from the image" on visual questions (Q4)?

Four ways to find image evidence, on the Q4 questions of an eval set (retrieval only, no generation):

    no_images — the system before Э5: image nodes are not searchable (descriptions removed from the hits);
    text      — the image descriptions of the VLM in the text index (images.channel = text, the default);
    visual    — ColQwen2 over the pixels instead of the descriptions (images.channel = visual);
    fusion    — both, merged by RRF (images.channel = fusion).

Metrics: Recall@5, hit@5 and MRR over the reference sources, with bootstrap CIs, and paired bootstrap
differences between the variants. The visual index is built first (the VLM is unloaded from Ollama to
make room for ColQwen2). The server must be stopped: the local Qdrant index is single-process.

    python scripts/h4_visual.py --eval evalsets/demo_v6.yaml --out runs/h4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_agent.config import load_settings  # noqa: E402
from rag_agent.engine import Engine  # noqa: E402
from rag_agent.evaluation.dataset import load_evalset  # noqa: E402
from rag_agent.evaluation.metrics import mean_ci, paired_bootstrap, retrieval_metrics  # noqa: E402

VARIANTS = ["no_images", "text", "visual", "fusion"]


def unload_ollama(base_url: str) -> None:
    try:
        for m in httpx.get(f"{base_url}/api/ps", timeout=5).json().get("models", []):
            httpx.post(f"{base_url}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
    except httpx.HTTPError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="evalsets/demo_v6.yaml")
    ap.add_argument("--out", default="runs/h4")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--classes", default="Q4")
    args = ap.parse_args()

    es = load_evalset(ROOT / args.eval)
    items = [it for it in es.items if it.cls in args.classes.split(",") and it.sources]
    settings = load_settings()
    settings.images.visual_index = True
    engine = Engine(settings)
    corpus = (ROOT / args.eval).parent / es.corpus
    index = engine.open_corpus(corpus)
    unload_ollama(settings.llm.base_url)
    from rag_agent.images.visual import VisualIndex

    t0 = time.perf_counter()
    build = VisualIndex(index, settings).build()
    print("visual index:", build, flush=True)

    rows = {v: [] for v in VARIANTS}
    for variant in VARIANTS:
        settings.images.channel = "text" if variant in ("no_images", "text") else variant
        for it in items:
            hits = engine.search(it.retrieval_query, args.k)
            nodes = [h.node for h in hits]
            if variant == "no_images":
                nodes = [n for n in nodes if n.node_type not in ("image", "figure")]
            m = retrieval_metrics(nodes, it.sources, ks=(1, 3, 5, 10))
            rows[variant].append({"id": it.id, "question": it.question, **m,
                                  "top": [f"{n.file_path} {n.location.describe()}".strip() for n in nodes[:5]]})
        print(variant, round(sum(r["recall@5"] for r in rows[variant]) / len(items), 3), flush=True)

    summary = {"eval": es.name, "n": len(items), "k": args.k, "visual_index": build,
               "llm_free": True, "elapsed_s": round(time.perf_counter() - t0, 1), "variants": {}, "paired": {}}
    for v in VARIANTS:
        summary["variants"][v] = {m: mean_ci([r[m] for r in rows[v]]) for m in ("recall@5", "hit@5", "recall@10", "mrr")}
    for a, b in [("fusion", "text"), ("visual", "text"), ("text", "no_images"), ("fusion", "no_images")]:
        summary["paired"][f"{a}-{b}"] = {m: paired_bootstrap([r[m] for r in rows[a]], [r[m] for r in rows[b]])
                                        for m in ("recall@5", "mrr")}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "items.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["variants"], ensure_ascii=False, indent=1))
    print(json.dumps(summary["paired"], ensure_ascii=False, indent=1))
    engine.close()


if __name__ == "__main__":
    main()
