"""H3 (Э8, ТЗ S5, S9): is BGE-M3 fine-tuned on synthetic queries from the corpus better than the base model?

Promptagator-style [35]: the local LLM writes a question to every fragment of the training files, a
round-trip filter keeps questions for which the base retriever finds their own fragment in the top 20,
and the model is fine-tuned contrastively (MultipleNegativesRankingLoss: in-batch negatives plus a hard
negative — the most similar other fragment of the same file). Files are split into train and held-out
(ТЗ: "на отложенных файлах"): the evaluation uses only questions about held-out files, retrieving over
the whole corpus, both on synthetic questions and on the hand-written eval set.

    python scripts/train_embedder.py synth --corpus demo_corpus --out runs/embedder
    python scripts/train_embedder.py train --out runs/embedder
    python scripts/train_embedder.py evaluate --out runs/embedder --eval evalsets/demo_v6.yaml
    python scripts/train_embedder.py export --out runs/embedder     # -> <data_dir>/models/bge-m3-ft

The server must not hold the GPU (and the LLM is unloaded before training).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QUERY_PROMPT = """Ты составляешь вопросы, которые исследователь мог бы задать своему архиву файлов (код, ноутбуки, документы, логи). По фрагменту ниже напиши ДВА разных естественных вопроса, ответ на которые содержится в этом фрагменте: первый на русском, второй на английском. Вопрос — как его задал бы человек, не видящий фрагмента: без номеров строк и ячеек, без слов «в этом фрагменте». Не копируй фрагмент дословно.

Примеры:
Фрагмент: def calc_metrics(y_true, y_pred, y_proba=None, average="macro"): ... roc_auc_score ...
Вопрос: Где считаются метрики классификации и считается ли там ROC-AUC?
Фрагмент: epoch=27 val_loss=0.1823 val_acc=0.9639 ... restored best epoch 27
Вопрос: Which epoch had the best validation accuracy for the MLP?

Верни JSON: {"questions": ["...", "..."]}"""
QUERY_SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
                "required": ["questions"]}


def held_out(path: str, share: float = 0.3) -> bool:
    return int(hashlib.sha1(path.encode()).hexdigest(), 16) % 1000 < share * 1000


def corpus_nodes(settings, corpus: Path):
    from rag_agent.index.embedder import Embedder
    from rag_agent.index.indexer import CorpusIndex

    index = CorpusIndex(corpus.resolve(), settings, Embedder(settings.embedding))
    rows = index.catalog.query("SELECT id FROM nodes WHERE embed = 1 AND length(text) >= 120")
    nodes = list(index.catalog.get_nodes([r["id"] for r in rows]).values())
    return index, sorted(nodes, key=lambda n: n.id)


def encode(model_name: str, texts: list[str], batch: int = 16) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cuda")
    model.max_seq_length = 512
    emb = model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False,
                       convert_to_numpy=True)
    del model
    import torch

    torch.cuda.empty_cache()
    return emb.astype(np.float32)


def cmd_synth(args) -> None:
    from rag_agent.config import load_settings
    from rag_agent.llm import LLMError, make_llm

    settings = load_settings()
    index, nodes = corpus_nodes(settings, ROOT / args.corpus)
    llm = make_llm(settings.llm.model_copy(update={"model": args.model or settings.llm.model}))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for k, n in enumerate(nodes, start=1):
        text = n.text[:1500]
        try:
            qs = llm.chat([{"role": "system", "content": QUERY_PROMPT},
                           {"role": "user", "content": f"Файл: {n.file_path}\nФрагмент:\n{text}"}],
                          json_schema=QUERY_SCHEMA, max_tokens=200, purpose="synth_query").json()["questions"]
        except (LLMError, KeyError):
            continue
        for q in [str(x).strip() for x in qs[:2]]:
            if len(q) >= 10:
                rows.append({"query": q, "node_id": n.id, "file": n.file_path,
                             "split": "test" if held_out(n.file_path) else "train"})
        if k % 20 == 0:
            print(f"{k}/{len(nodes)} {time.time() - t0:.0f}s", flush=True)
    llm.unload()
    # round-trip filter (Promptagator): the base retriever must find the question's own fragment in the top 20
    texts = [n.embedding_text(settings.chunking.context_header) for n in nodes]
    ids = [n.id for n in nodes]
    doc = encode(settings.embedding.model, texts)
    qv = encode(settings.embedding.model, [r["query"] for r in rows])
    top = np.argsort(-(qv @ doc.T), axis=1)[:, :20]
    kept = []
    for r, t in zip(rows, top):
        r["roundtrip_rank"] = next((i + 1 for i, j in enumerate(t) if ids[j] == r["node_id"]), None)
        if r["roundtrip_rank"] is not None:
            kept.append(r)
    (out / "synthetic.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (out / "synthetic_kept.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept), encoding="utf-8")
    stats = {"nodes": len(nodes), "generated": len(rows), "kept": len(kept),
             "train": sum(r["split"] == "train" for r in kept), "test": sum(r["split"] == "test" for r in kept),
             "held_out_files": sorted({n.file_path for n in nodes if held_out(n.file_path)}), "model": llm.name}
    (out / "synth_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "held_out_files"}))


def cmd_train(args) -> None:
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    from rag_agent.config import load_settings

    settings = load_settings()
    out = ROOT / args.out
    index, nodes = corpus_nodes(settings, ROOT / args.corpus)
    by_id = {n.id: n for n in nodes}
    rows = [json.loads(line) for line in (out / "synthetic_kept.jsonl").read_text(encoding="utf-8").splitlines()]
    train = [r for r in rows if r["split"] == "train" and r["node_id"] in by_id]
    # hard negative: the most similar other fragment of the same file (by the base model)
    texts = {n.id: n.embedding_text(settings.chunking.context_header) for n in nodes}
    ids = list(texts)
    base = encode(settings.embedding.model, [texts[i] for i in ids])
    pos = {i: k for k, i in enumerate(ids)}
    anchors, positives, negatives = [], [], []
    rng = random.Random(0)
    for r in train:
        same = [j for j in ids if by_id[j].file_path == r["file"] and j != r["node_id"]]
        if same:
            sims = base[[pos[j] for j in same]] @ base[pos[r["node_id"]]]
            neg = same[int(np.argmax(sims))]
        else:
            neg = rng.choice([j for j in ids if j != r["node_id"]])
        anchors.append(r["query"])
        positives.append(texts[r["node_id"]])
        negatives.append(texts[neg])
    # a plain loop: the Hugging Face Trainer imports torch.distributed to build AdamW, and the ROCm build of
    # torch for Windows has no distributed package
    model = SentenceTransformer(settings.embedding.model, device="cuda")
    model.max_seq_length = 512
    loss_fn = MultipleNegativesRankingLoss(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    order = list(range(len(anchors)))
    steps = args.epochs * math.ceil(len(order) / args.batch)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / warm) * max(0.0, (steps - k) / steps))
    scaler = torch.amp.GradScaler("cuda")
    t0, step, history = time.time(), 0, []
    model.train()
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(order)
        for k in range(0, len(order), args.batch):
            idx = order[k:k + args.batch]
            feats = []
            for column in (anchors, positives, negatives):
                f = model.tokenize([column[i] for i in idx])
                feats.append({key: v.to("cuda") if hasattr(v, "to") else v for key, v in f.items()})
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(feats, None)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            history.append({"step": step, "epoch": epoch + 1, "loss": round(loss.item(), 4)})
            print(history[-1], flush=True)
    model.save(str(out / "bge-m3-ft"))
    head = Path(settings.embedding.model)
    from huggingface_hub import hf_hub_download

    sparse = head / "sparse_linear.pt" if head.is_dir() else Path(hf_hub_download(settings.embedding.model, "sparse_linear.pt",
                                                                                  local_files_only=True))
    shutil.copy2(sparse, out / "bge-m3-ft" / "sparse_linear.pt")  # the sparse head is not trained (dense only)
    (out / "train.json").write_text(json.dumps({"pairs": len(anchors), "epochs": args.epochs, "batch": args.batch,
                                                "lr": args.lr, "elapsed_s": round(time.time() - t0, 1),
                                                "history": history}, indent=2), encoding="utf-8")
    print("saved", out / "bge-m3-ft")


def cmd_evaluate(args) -> None:
    from rag_agent.config import load_settings
    from rag_agent.evaluation.dataset import load_evalset
    from rag_agent.evaluation.metrics import mean_ci, paired_bootstrap, retrieval_metrics

    settings = load_settings()
    out = ROOT / args.out
    index, nodes = corpus_nodes(settings, ROOT / args.corpus)
    texts = [n.embedding_text(settings.chunking.context_header) for n in nodes]
    ids = [n.id for n in nodes]
    rows = [json.loads(line) for line in (out / "synthetic_kept.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [r for r in rows if r["split"] == "test" and r["node_id"] in ids]
    es = load_evalset(ROOT / args.eval)
    items = [it for it in es.items if it.sources and it.cls != "G"
             and all(held_out(s.file) for s in it.sources)]  # hand-written questions about held-out files only
    result = {"synthetic_test": len(test), "evalset_items": len(items), "models": {}}
    per = {}
    for name, model in (("base", settings.embedding.model), ("fine-tuned", str(out / "bge-m3-ft"))):
        doc = encode(model, texts)
        syn_q = encode(model, [r["query"] for r in test]) if test else np.zeros((0, doc.shape[1]))
        ev_q = encode(model, [it.retrieval_query for it in items]) if items else np.zeros((0, doc.shape[1]))
        syn = []
        for q, r in zip(syn_q, test):
            order = np.argsort(-(doc @ q))[:10]
            rank = next((i + 1 for i, j in enumerate(order) if ids[j] == r["node_id"]), None)
            syn.append({"hit@5": float(rank is not None and rank <= 5), "mrr": 1.0 / rank if rank else 0.0})
        ev = []
        for q, it in zip(ev_q, items):
            order = np.argsort(-(doc @ q))[:10]
            m = retrieval_metrics([nodes[j] for j in order], it.sources, ks=(5, 10))
            ev.append({"id": it.id, **m})
        per[name] = {"syn": syn, "ev": ev}
        result["models"][name] = {
            "synthetic": {m: mean_ci([x[m] for x in syn]) for m in ("hit@5", "mrr")},
            "evalset": {m: mean_ci([x[m] for x in ev]) for m in ("recall@5", "mrr")},
        }
        print(name, json.dumps(result["models"][name], ensure_ascii=False), flush=True)
    result["paired"] = {
        "synthetic_mrr": paired_bootstrap([x["mrr"] for x in per["fine-tuned"]["syn"]], [x["mrr"] for x in per["base"]["syn"]]),
        "evalset_recall@5": paired_bootstrap([x["recall@5"] for x in per["fine-tuned"]["ev"]],
                                             [x["recall@5"] for x in per["base"]["ev"]]),
        "evalset_mrr": paired_bootstrap([x["mrr"] for x in per["fine-tuned"]["ev"]], [x["mrr"] for x in per["base"]["ev"]]),
    }
    (out / "evaluate.json").write_text(json.dumps({**result, "items": per}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["paired"], indent=1))


def cmd_export(args) -> None:
    from rag_agent.config import load_settings

    target = load_settings().data_dir / "models" / "bge-m3-ft"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ROOT / args.out / "bge-m3-ft", target)
    print(f"exported to {target}; use it with embedding.model: {target.as_posix()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("synth", "train", "evaluate", "export"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", default="demo_corpus")
        p.add_argument("--out", default="runs/embedder")
        if name == "synth":
            p.add_argument("--model", default="")
        if name == "train":
            p.add_argument("--epochs", type=int, default=2)
            p.add_argument("--batch", type=int, default=16)
            p.add_argument("--lr", type=float, default=1e-5)
        if name == "evaluate":
            p.add_argument("--eval", default="evalsets/demo_v6.yaml")
    args = ap.parse_args()
    {"synth": cmd_synth, "train": cmd_train, "evaluate": cmd_evaluate, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
