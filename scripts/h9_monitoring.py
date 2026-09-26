"""H9 bench (ТЗ ч.1: S12, Э9): calibrated monitoring with a correction for the number of
series finds degradation with fewer false alarms than detectors with default thresholds.

1. Canary questions are generated from the catalog of the demo corpus (no LLM).
2. Every canary is answered by the retrieval component (dense top-5) and by the
   reranker component (top-5 after re-ranking the dense top-10) in three states of the
   system: normal; a weaker embedder (vectors truncated to the first 128 of 1024
   dimensions, as if a small model replaced BGE-M3); files of a new domain added to the
   index (source code of web libraries installed locally — no downloads).
3. The monitoring fleet is "component x segment": 2 components x 7 segments = 14 series.
   Each step every series answers one random canary of its segment; the state switches
   at a known moment. Signals: 1 - reciprocal rank (continuous) and a top-5 miss (0/1).
4. Detectors: Page-Hinkley and DDM with their default thresholds (current practice) vs
   the same Page-Hinkley calibrated by driftfdr, without a correction and with a
   correction for the number of series (BH within a window, LORD++ online).

Fragment vectors are read from the built index; only canaries, distractors and the
reranker run on the CPU (the GPU is busy): texts up to 256 tokens, the reranker in
dynamic int8. Nothing is downloaded.

Needs driftfdr (pip install -e ".[monitoring]").

    python scripts/h9_monitoring.py --out reports/e9
"""

from __future__ import annotations

import argparse
import ast
import json
import site
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_agent.config import load_settings  # noqa: E402
from rag_agent.index.catalog import Catalog  # noqa: E402
from rag_agent.index.embedder import Embedder  # noqa: E402
from rag_agent.index.indexer import corpus_key, index_signature  # noqa: E402
from rag_agent.index.reranker import Reranker  # noqa: E402
from rag_agent.monitoring.canaries import build_canaries  # noqa: E402
from rag_agent.schema import Node  # noqa: E402

STATES = ("normal", "weak_embedder", "new_domain")
WEAK_DIMS = 128
POOL = 10
TOP = 5
TOLERANCE = 0.02  # the null of a test: the miss rate of the segment rose by at most 2 points


# --------------------------------------------------------------------------- canary answers per state


def distractor_texts(limit: int = 300, chars: int = 800) -> list[str]:
    """Source of locally installed web libraries: a domain the archive does not have."""
    roots = [Path(p) for p in site.getsitepackages()]
    texts = []
    for pkg in ("starlette", "httpx", "fastapi", "uvicorn", "httpcore"):
        for root in roots:
            for f in sorted((root / pkg).rglob("*.py")) if (root / pkg).exists() else []:
                src = f.read_text(encoding="utf-8", errors="replace")
                try:
                    tree = ast.parse(src)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                        seg = ast.get_source_segment(src, node) or ""
                        if len(seg) > 200:
                            texts.append(f"# {pkg}/{f.name}\n{seg[:chars]}")
                if len(texts) >= limit:
                    return texts[:limit]
    return texts[:limit]


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def index_vectors(index_dir: Path, ids: list[str]) -> np.ndarray:
    """Dense vectors of the fragments as the index stores them (computed on the GPU at indexing time);
    a copy of the local Qdrant folder is opened, so a running process that holds the index is not disturbed."""
    import shutil
    import tempfile

    from qdrant_client import QdrantClient

    tmp = Path(tempfile.mkdtemp(prefix="rag-h9-"))
    shutil.copytree(index_dir / "qdrant", tmp / "qdrant", ignore=shutil.ignore_patterns(".lock"))  # held by a running engine
    client = QdrantClient(path=str(tmp / "qdrant"))
    vecs, offset = {}, None
    while True:
        points, offset = client.scroll("nodes", limit=256, offset=offset, with_vectors=["dense"],
                                       with_payload=["node_id"])
        for pt in points:
            vecs[pt.payload["node_id"]] = np.asarray(pt.vector["dense"], dtype=np.float32)
        if offset is None:
            break
    client.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return np.stack([vecs[i] for i in ids])


def answers(canaries, nodes: list[Node], texts: list[str], doc: np.ndarray, embedder, reranker, n_distractors: int,
            log, cache_dir: Path):
    """rank of the first fragment answering each canary, per state and component (0 = not in the pool).
    Query and distractor vectors are cached; a (canary, fragment) pair is re-ranked once for all states."""
    t0 = time.perf_counter()
    dis_texts = distractor_texts(n_distractors) if n_distractors else []
    emb_file = cache_dir / "h9_vectors.npz"
    if emb_file.exists():
        z = np.load(emb_file)
        q, dis = z["q"], (z["dis"] if len(z["dis"]) else None)
        log(f"canary and distractor vectors loaded from {emb_file.name}")
    else:
        q = np.asarray(embedder.encode([c.question for c in canaries]).dense, dtype=np.float32)
        dis = np.asarray(embedder.encode(dis_texts).dense, dtype=np.float32) if dis_texts else None
        np.savez(emb_file, q=q, dis=dis if dis is not None else np.zeros((0, q.shape[1]), np.float32))
        log(f"embedded {len(canaries)} canaries, {0 if dis is None else len(dis)} distractors "
            f"in {time.perf_counter() - t0:.0f} s")
    pools = {}
    for state in STATES:
        D, Q = doc, q
        if state == "weak_embedder":
            D, Q = norm(doc[:, :WEAK_DIMS]), norm(q[:, :WEAK_DIMS])
        if state == "new_domain" and dis is not None:
            D = np.concatenate([doc, dis])
        pools[state] = np.argsort(-(norm(Q) @ norm(D).T), axis=1)[:, :POOL]
    need = {i: sorted({int(j) for top in pools.values() for j in top[i]}) for i in range(len(canaries))}
    n_pairs = sum(len(v) for v in need.values())
    log(f"re-ranking {n_pairs} unique pairs (of {len(canaries) * POOL * len(STATES)})")
    score: dict[tuple[int, int], float] = {}
    for k, (i, js) in enumerate(need.items(), start=1):
        cand = [texts[j] if j < len(texts) else dis_texts[j - len(texts)] for j in js]
        for j, sc in zip(js, reranker.score(canaries[i].question, cand)):
            score[(i, j)] = float(sc)
        if k % 20 == 0:
            log(f"  {k}/{len(canaries)} canaries re-ranked ({time.perf_counter() - t0:.0f} s)")
    out = {}
    for state in STATES:
        dense_rank, rerank_rank = [], []
        for i, c in enumerate(canaries):
            top = [int(j) for j in pools[state][i]]
            hits = [j < len(nodes) and c.answered_by(nodes[j]) for j in top]
            dense_rank.append(next((k + 1 for k, h in enumerate(hits) if h), 0))
            order = sorted(range(len(top)), key=lambda k: -score[(i, top[k])])
            rerank_rank.append(next((r + 1 for r, k in enumerate(order) if hits[k]), 0))
        out[state] = {"retrieval": np.array(dense_rank), "rerank": np.array(rerank_rank)}
        log(f"  {state}: retrieval miss@{TOP} {np.mean([r == 0 or r > TOP for r in dense_rank]):.3f}, "
            f"rerank miss@{TOP} {np.mean([r == 0 or r > TOP for r in rerank_rank]):.3f}")
    return out


# --------------------------------------------------------------------------- scenarios and detectors


def signals(rank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.where(rank > 0, 1.0 - 1.0 / np.maximum(rank, 1), 1.0)
    error = ((rank == 0) | (rank > TOP)).astype(float)
    return value, error


def make_scenario(ans, canaries, streams, before: str, after: str | None, onset: int, n_steps: int, seed: int):
    from driftfdr.streams import NO_CHANGE, Scenario, ScenarioConfig

    rng = np.random.default_rng(seed)
    idx = {seg: [i for i, c in enumerate(canaries) if c.segment == seg] for _, seg in streams}
    values = np.empty((len(streams), n_steps))
    errors = np.empty((len(streams), n_steps))
    truth = np.empty((len(streams), n_steps))
    for k, (comp, seg) in enumerate(streams):
        pool = np.array(idx[seg])
        draw = rng.choice(pool, size=n_steps)
        for state, sl in ((before, slice(0, onset if after else n_steps)), (after, slice(onset, n_steps))):
            if state is None:
                continue
            v, e = signals(ans[state][comp][draw[sl]])
            values[k, sl], errors[k, sl] = v, e
            truth[k, sl] = signals(ans[state][comp][pool])[1].mean()  # true error rate of the segment
    change = np.full(len(streams), NO_CHANGE, dtype=np.int64)
    if after:  # a change only where the true miss rate of the segment rose by more than the tolerance
        change[(truth[:, -1] - truth[:, 0]) > TOLERANCE] = onset
    sc = Scenario(config=ScenarioConfig(n_streams=len(streams), n_steps=n_steps), values=values, errors=errors,
                  change_start=change.copy(), change_end=change.copy(),
                  drift_kind=np.where(change != NO_CHANGE, "abrupt", "none"),
                  event=np.where(change != NO_CHANGE, 0, -1).astype(np.int64))
    return replace(sc, truth=truth, tolerance=TOLERANCE)


def methods(n_ref: int, window: int):
    from driftfdr import DDM, PageHinkley
    from driftfdr.online_fdr import RawThreshold, make_procedure

    ph = PageHinkley()
    ddm = DDM()
    return [
        ("Page-Hinkley, порог по умолчанию", ph, lambda: RawThreshold(ph.default_threshold)),
        ("DDM, порог по умолчанию", ddm, lambda: RawThreshold(ddm.default_threshold)),
        ("PH, калибровка без поправки (α=0.05)", ph, lambda: make_procedure("uncorrected", 0.05)),
        ("PH, калибровка + BH в окне (FDR 0.05)", ph, lambda: make_procedure("bh_window", 0.05)),
        ("PH, калибровка + LORD++ (FDR 0.05)", ph, lambda: make_procedure("LORD++", 0.05)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(ROOT / "demo_corpus"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "e9"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--distractors", type=int, default=200)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--int8", action=argparse.BooleanOptionalAction, default=True,
                    help="int8 dynamic quantization of the reranker on the CPU")
    args = ap.parse_args()
    import torch
    from driftfdr import MonitorConfig, run_monitor, summarize

    torch.set_num_threads(args.threads)
    log = lambda m: print(m, flush=True)  # noqa: E731
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    corpus = Path(args.corpus).resolve()
    index_dir = settings.data_dir / "indexes" / corpus_key(corpus) / index_signature(settings)
    cat = Catalog(index_dir / "catalog.sqlite")
    canaries = build_canaries(cat)
    ids = [r["id"] for r in cat.query("SELECT id FROM nodes WHERE embed = 1 ORDER BY id")]
    by_id = cat.get_nodes(ids)
    nodes = [by_id[i] for i in ids]
    texts = [n.embedding_text(settings.chunking.context_header) for n in nodes]
    emb_cfg = settings.embedding.model_copy(update={"device": "cpu", "fp16": False, "max_length": 256})
    embedder = Embedder(emb_cfg)
    reranker = Reranker(settings.retrieval.model_copy(update={"rerank_max_length": 256}), device="cpu", fp16=False,
                        local_files_only=True)
    if args.int8:  # dynamic int8 of the linear layers: 2-3x faster on the CPU, the ranking barely changes
        reranker.load()
        reranker._model = torch.ao.quantization.quantize_dynamic(reranker._model, {torch.nn.Linear}, dtype=torch.qint8)
    cache = out / "canary_ranks.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        ans = {s: {c: np.array(v) for c, v in comps.items()} for s, comps in raw["ranks"].items()}
        log(f"canary ranks loaded from {cache.name}")
    else:
        doc = index_vectors(index_dir, ids)
        log(f"{len(ids)} fragment vectors read from the index")
        ans = answers(canaries, nodes, texts, doc, embedder, reranker, args.distractors, log, out)
        cache.write_text(json.dumps({"canaries": [c.id for c in canaries], "segments": [c.segment for c in canaries],
                                     "ranks": {s: {c: v.tolist() for c, v in comps.items()} for s, comps in ans.items()}},
                                    ensure_ascii=False), encoding="utf-8")

    segments = sorted({c.segment for c in canaries})
    streams = [(comp, seg) for comp in ("retrieval", "rerank") for seg in segments]
    base = {f"{comp}|{seg}": float(np.mean(signals(ans["normal"][comp][[i for i, c in enumerate(canaries)
                                                                         if c.segment == seg]])[1]))
            for comp, seg in streams}
    scenarios = [("без изменений", "normal", None), ("слабый эмбеддер", "normal", "weak_embedder"),
                 ("файлы нового домена", "normal", "new_domain")]
    config = MonitorConfig(n_ref=300, window=100, horizon=5)
    rows = []
    for name, before, after in scenarios:
        for seed in range(args.seeds):
            sc = make_scenario(ans, canaries, streams, before, after, onset=args.steps // 2, n_steps=args.steps, seed=seed)
            caches: dict = {}
            for mname, det, proc in methods(config.n_ref, config.window):
                key = det.name
                res = run_monitor(sc, det, proc(), config, seed=seed, cache=caches.setdefault(key, {}))
                s = summarize(res, max_delay=1000)
                rows.append({"scenario": name, "seed": seed, "method": mname, **{k: (None if isinstance(v, float) and
                             not np.isfinite(v) else v) for k, v in s.items()}})
            log(f"{name}, seed {seed}: done")
    (out / "h9_runs.json").write_text(json.dumps({"rows": rows, "baseline_error": base, "streams": streams},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"saved {len(rows)} runs to {out / 'h9_runs.json'}")


if __name__ == "__main__":
    main()
