"""Ablation runner (ТЗ 5.3): several configurations on the same eval set, one
component changed at a time; one summary table with paired comparisons against
the first (reference) run.

Spec (YAML):

    name: e3
    evalset: ../../evalsets/demo_v2.yaml      # relative to the spec file
    retrieval_only: true
    runs:
      - name: lines + header (Э1)
        overrides: {chunking: {python: lines}}
      - name: ast + header
        overrides: {chunking: {python: ast}}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml
from pydantic import BaseModel, Field

from rag_agent.config import REPO_ROOT, Settings, _deep_merge
from rag_agent.engine import Engine
from rag_agent.evaluation.dataset import CLASS_NAMES, EvalSet, load_evalset, validate_evalset
from rag_agent.evaluation.metrics import paired_bootstrap
from rag_agent.evaluation.report import render_report
from rag_agent.evaluation.runner import ItemResult, run_config, run_eval, run_judge, summarize
from rag_agent.index.embedder import Embedder
from rag_agent.index.reranker import Reranker
from rag_agent.llm import make_llm

COMPARE_METRIC = "recall@5"


class AblationRun(BaseModel):
    name: str
    overrides: dict = Field(default_factory=dict)
    note: str | None = None


class AblationSpec(BaseModel):
    name: str
    evalset: str
    retrieval_only: bool = True
    judge: bool = False
    runs: list[AblationRun]
    path: Path | None = None

    def evalset_path(self) -> Path:
        p = Path(self.evalset)
        return p if p.is_absolute() or self.path is None else (self.path.parent / p).resolve()


def load_spec(path: str | Path) -> AblationSpec:
    path = Path(path)
    spec = AblationSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    spec.path = path.resolve()
    return spec


def apply_overrides(base: Settings, overrides: dict) -> Settings:
    return Settings.model_validate(_deep_merge(base.model_dump(), json.loads(json.dumps(overrides))))


def _save_run(out_dir: Path, results: list[ItemResult], summary: dict, config: dict, es: EvalSet) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    (out_dir / "report.md").write_text(render_report(summary, results, es, config), encoding="utf-8")


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() else "-" for c in text.lower())
    return "-".join(p for p in keep.split("-") if p)[:60]


def run_ablation(
    spec: AblationSpec, base: Settings, out_root: Path, log: Callable[[str], None] = print
) -> tuple[list[dict], Path]:
    es = load_evalset(spec.evalset_path())
    corpus = es.corpus_root()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = out_root / f"{stamp}-{spec.name}"
    embedders: dict[tuple, Embedder] = {}
    reranker = Reranker(base.retrieval, device=base.embedding.device, fp16=base.embedding.fp16,
                        local_files_only=base.embedding.local_files_only)
    rows = []
    per_item: list[dict[str, list[ItemResult]]] = []
    for i, run in enumerate(spec.runs, start=1):
        settings = apply_overrides(base, run.overrides)
        emb_key = (settings.embedding.model, settings.embedding.pooling, settings.embedding.max_length)
        embedder = embedders.setdefault(emb_key, Embedder(settings.embedding))
        engine = Engine(settings, embedder=embedder, reranker=reranker)
        log(f"[{i}/{len(spec.runs)}] {run.name}")
        progress = engine.index_folder(corpus)
        if progress.state != "done":
            raise RuntimeError(f"indexing failed for {run.name}: {progress.message}")
        problems = validate_evalset(es, engine.index.catalog)
        if problems:
            raise RuntimeError(f"eval set does not match the index for {run.name}: {problems[:5]}")
        results = run_eval(engine, es, retrieval_k=settings.evaluation.retrieval_k, generate=not spec.retrieval_only)
        judge_name = None
        if spec.judge and not spec.retrieval_only:
            judge_name = settings.evaluation.judge_model
            engine.llm.unload()
            judge = make_llm(settings.llm.model_copy(update={"model": judge_name, "think": settings.evaluation.judge_think}),
                             engine.tracer)
            run_judge(results, es, judge)
            judge.unload()
            judge.close()
        summary = summarize(results, es, n_boot=settings.evaluation.bootstrap)
        config = run_config(engine, es, top_k=settings.retrieval.top_k, mode=settings.retrieval.mode, route="auto",
                            judge=judge_name, retrieval_k=settings.evaluation.retrieval_k)
        config["name"] = run.name
        stats = engine.index.catalog.stats()
        summary["index"] = {"nodes": stats["nodes"], "vectors": engine.index.store.count()}
        _save_run(out_dir / f"{i:02d}-{_slug(run.name)}", results, summary, config, es)
        rows.append({"run": run, "summary": summary, "config": config})
        per_item.append({r.id: r for r in results})
        engine.close()
        rec = summary["retrieval"][COMPARE_METRIC]["mean"]
        log(f"    {COMPARE_METRIC} = {rec:.3f}")

    report = render_ablation(spec, es, rows, per_item)
    (out_dir / "ablation.md").write_text(report, encoding="utf-8")
    (out_dir / "ablation.json").write_text(
        json.dumps([{"name": r["run"].name, "overrides": r["run"].overrides, "summary": r["summary"]} for r in rows],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows, out_dir


def _fmt(x, d: int = 3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def _metric_values(items: dict[str, ItemResult], ids: list[str], metric: str) -> list[float]:
    return [items[i].retrieval[metric] for i in ids]


def render_ablation(spec: AblationSpec, es: EvalSet, rows: list[dict], per_item: list[dict[str, ItemResult]]) -> str:
    ref_items = per_item[0]
    ids = [i for i, r in ref_items.items() if r.retrieval]
    out = [
        f"# Абляции: {spec.name}",
        "",
        f"Набор `{es.name}` v{es.version}: вопросов с эталонными источниками — {len(ids)}. "
        f"Сравнение каждой конфигурации с первой (референсной) — парный бутстреп по вопросам для {COMPARE_METRIC}: "
        "Δ, 95% интервал Δ и одностороннее p для гипотезы «конфигурация не лучше референса».",
        "",
        "| # | Конфигурация | Recall@5 [95% CI] | Recall@10 | MRR@10 | nDCG@10 | Δ Recall@5 [CI], p | .py R@5 | .ipynb R@5 | Поиск p50/p95, с |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, (row, items) in enumerate(zip(rows, per_item), start=1):
        s = row["summary"]
        ret = s["retrieval"]
        r5 = ret["recall@5"]
        ci = r5["ci"]
        cmp = paired_bootstrap(_metric_values(items, ids, COMPARE_METRIC), _metric_values(ref_items, ids, COMPARE_METRIC))
        delta = "—" if k == 1 else (
            f"{cmp['diff']:+.3f} [{cmp['ci'][0]:+.2f}; {cmp['ci'][1]:+.2f}], p={cmp['p']:.3f}" if cmp["ci"] else "—"
        )
        ft = s["retrieval_by_file_type"]
        lat = s.get("retrieval_latency") or {}
        out.append(
            f"| {k} | {row['run'].name} | {r5['mean']:.3f} [{ci[0]:.2f}; {ci[1]:.2f}] | {_fmt(ret['recall@10']['mean'])} "
            f"| {_fmt(ret['mrr']['mean'])} | {_fmt(ret['ndcg@10']['mean'])} | {delta} "
            f"| {_fmt(ft.get('py', {}).get('recall@5'))} | {_fmt(ft.get('ipynb', {}).get('recall@5'))} "
            f"| {_fmt(lat.get('p50'), 2)} / {_fmt(lat.get('p95'), 2)} |"
        )
    classes = sorted({c for row in rows for c in row["summary"]["retrieval_by_class"]})
    out += ["", "Recall@5 по классам вопросов:", "",
            "| # | Конфигурация | " + " | ".join(f"{c} {CLASS_NAMES[c]}" for c in classes) + " |",
            "|---|---|" + "---|" * len(classes)]
    for k, row in enumerate(rows, start=1):
        by = row["summary"]["retrieval_by_class"]
        out.append(f"| {k} | {row['run'].name} | " + " | ".join(_fmt(by.get(c, {}).get("recall@5")) for c in classes) + " |")
    if any("judge" in row["summary"] for row in rows):
        out += ["", "| # | Конфигурация | Корректность (судья) | Верность контексту | Ложные отказы | Латентность p95, с |",
                "|---|---|---|---|---|---|"]
        for k, row in enumerate(rows, start=1):
            s = row["summary"]
            j = s.get("judge", {})
            out.append(
                f"| {k} | {row['run'].name} | {_fmt((j.get('correctness') or {}).get('mean'))} "
                f"| {_fmt((j.get('faithfulness') or {}).get('mean'))} | {s.get('refusals', {}).get('false_refusals', '—')} "
                f"| {_fmt(s.get('latency', {}).get('all', {}).get('p95'), 1)} |"
            )
    out += ["", "Переопределения настроек:", ""]
    out += [f"{k}. **{row['run'].name}** — `{json.dumps(row['run'].overrides, ensure_ascii=False)}`"
            + (f" — {row['run'].note}" if row["run"].note else "") for k, row in enumerate(rows, start=1)]
    return "\n".join(out) + "\n"
