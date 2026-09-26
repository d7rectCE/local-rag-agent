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
    classes: list[str] | None = None  # only these question classes (e.g. Q3 for H5)
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
    if spec.classes:
        es.items = [i for i in es.items if i.cls in spec.classes]
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
        if problems:  # e.g. a baseline parser that drops comments: those references count as misses
            log(f"    {len(problems)} reference(s) match no fragment in this configuration: {problems[:3]}")
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
        rec = summary["retrieval"][COMPARE_METRIC]["mean"]  # None for sets without reference fragments (Q7, Q10)
        log(f"    {COMPARE_METRIC} = {_fmt(rec)}")

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


def _score(r: ItemResult) -> float | None:
    """Judge score when the judge ran, otherwise the must-include check."""
    if r.judge is not None and r.judge.error is None:
        return r.judge.score
    return None if r.must_include_ok is None else float(r.must_include_ok)


def _reasoning_section(rows: list[dict], per_item: list[dict[str, ItemResult]]) -> list[str]:
    """H11 (ТЗ ч.2): accuracy against generated tokens and latency for each reasoning setting,
    per question class, and the share of extra tokens that bought no accuracy [Chen et al. 2025]."""
    ref = per_item[0]
    out = ["", "## Рассуждения: точность против затрат", "",
           "Точность — оценка судьи (1 / 0.5 / 0), без судьи — доля ответов с обязательными фрагментами. "
           "Δ — парный бутстреп по вопросам относительно первой конфигурации. Токены — все сгенерированные "
           "(рассуждение + ответ) на вопрос.", "",
           "| # | Конфигурация | Точность | Δ [95% CI], p | Токенов (рассужд.) | С рассуждением | Обрезано | Эскалаций "
           "| Латентность p50 / p95, с | p95 с рассужд., с |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k, (row, items) in enumerate(zip(rows, per_item), start=1):
        rs = row["summary"].get("reasoning") or {}
        ids = [i for i in items if i in ref and _score(items[i]) is not None and _score(ref[i]) is not None]
        delta = "—"
        if k > 1 and ids:
            cmp = paired_bootstrap([_score(items[i]) for i in ids], [_score(ref[i]) for i in ids])
            if cmp["ci"]:
                delta = f"{cmp['diff']:+.3f} [{cmp['ci'][0]:+.2f}; {cmp['ci'][1]:+.2f}], p={cmp['p']:.3f}"
        scores = [s for r in items.values() if (s := _score(r)) is not None]
        lat = rs.get("latency") or {}
        out.append(
            f"| {k} | {row['run'].name} | {_fmt(sum(scores) / len(scores) if scores else None)} | {delta} "
            f"| {_fmt(rs.get('generated_tokens'), 0)} ({_fmt(rs.get('reasoning_tokens'), 0)}) "
            f"| {_fmt(rs.get('reasoning_share'), 2)} | {_fmt(rs.get('truncated_share'), 2)} | {rs.get('escalated', 0)} "
            f"| {_fmt(lat.get('p50'), 1)} / {_fmt(lat.get('p95'), 1)} | {_fmt((rs.get('latency_reasoning') or {}).get('p95'), 1)} |"
        )

    classes = sorted({c for row in rows for c in (row["summary"].get("reasoning") or {}).get("by_class", {})})
    out += ["", "По классам вопросов: точность / токенов на вопрос / доля с рассуждением.", "",
            "| Класс | " + " | ".join(f"{k}. {row['run'].name}" for k, row in enumerate(rows, start=1)) + " |",
            "|---|" + "---|" * len(rows)]
    for c in classes:
        cells = []
        for row, items in zip(rows, per_item):
            by = ((row["summary"].get("reasoning") or {}).get("by_class") or {}).get(c, {})
            scores = [s for r in items.values() if r.cls == c and (s := _score(r)) is not None]
            acc = sum(scores) / len(scores) if scores else None
            cells.append(f"{_fmt(acc, 2)} / {_fmt(by.get('generated_tokens'), 0)} / {_fmt(by.get('reasoning_share'), 2)}")
        out.append(f"| {c} {CLASS_NAMES.get(c, '')} (n={len([r for r in ref.values() if r.cls == c])}) | " + " | ".join(cells) + " |")

    # "overthinking": extra tokens over the no-reasoning run spent on questions whose score did not improve
    base = next((k for k, row in enumerate(rows) if not (row["summary"].get("reasoning") or {}).get("reasoning_share")), None)
    if base is not None:
        out += ["", f"Токены сверх конфигурации «{rows[base]['run'].name}», не давшие прироста точности на вопросе:", ""]
        for k, (row, items) in enumerate(zip(rows, per_item), start=1):
            if k - 1 == base:
                continue
            extra = wasted = 0
            for i, r in items.items():
                b = per_item[base].get(i)
                if b is None or _score(r) is None or _score(b) is None:
                    continue
                more = max(0, r.generated_tokens - b.generated_tokens)
                extra += more
                if _score(r) <= _score(b):
                    wasted += more
            share = f"{wasted / extra:.2f}" if extra else "—"
            out.append(f"- {k}. {row['run'].name}: {share} (лишних токенов всего {extra}, из них без прироста {wasted})")
    return out


def _sql_section(rows: list[dict], per_item: list[dict[str, ItemResult]]) -> list[str]:
    """H5: the SQL tool answers aggregate questions (Q3) that retrieval alone cannot."""
    ref = per_item[0]
    out = ["", "## SQL по каталогу экспериментов (H5)", "",
           "Точность — судья (без судьи — обязательные фрагменты) на вопросах Q3; Δ — парный бутстреп относительно "
           "первой конфигурации. EX — точность исполнения SQL по эталонным запросам.", "",
           "| # | Конфигурация | Точность Q3 | Δ Q3 [95% CI], p | SQL на Q3 | SQL на прочих | Ошибок SQL | EX |",
           "|---|---|---|---|---|---|---|---|"]
    for k, (row, items) in enumerate(zip(rows, per_item), start=1):
        s = row["summary"].get("sql") or {}
        ids = [i for i, r in items.items() if r.cls == "Q3" and i in ref and _score(r) is not None and _score(ref[i]) is not None]
        scores = [_score(items[i]) for i in ids]
        delta = "—"
        if k > 1 and ids:
            cmp = paired_bootstrap(scores, [_score(ref[i]) for i in ids])
            if cmp["ci"]:
                delta = f"{cmp['diff']:+.3f} [{cmp['ci'][0]:+.2f}; {cmp['ci'][1]:+.2f}], p={cmp['p']:.3f}"
        ex = s.get("execution_accuracy") or {}
        out.append(
            f"| {k} | {row['run'].name} | {_fmt(sum(scores) / len(scores) if scores else None)} | {delta} "
            f"| {_fmt(s.get('used_q3'), 2)} | {_fmt(s.get('used_other'), 2)} | {s.get('errors', 0)} "
            f"| {_fmt(ex.get('mean'))} (n={s.get('n_gold', 0)}) |"
        )
    return out


def _web_section(rows: list[dict], per_item: list[dict[str, ItemResult]]) -> list[str]:
    """H15: adaptive web access improves Q10 and does not hurt questions about the files."""
    out = ["", "## Интернет (H15)", "",
           "Точность — судья (без судьи — обязательные фрагменты) отдельно на Q10 и на остальных вопросах. "
           "Неподтверждённые — доля предложений веб-ответов без ссылки или с числами, которых нет в цитируемых фрагментах.",
           "", "| # | Конфигурация | Точность Q10 | Точность прочих | В интернет: Q10 / прочие | Неподтверждённые | p95 веб-ответа, с |",
           "|---|---|---|---|---|---|---|"]
    for k, (row, items) in enumerate(zip(rows, per_item), start=1):
        w = row["summary"].get("web") or {}
        q10 = [s for r in items.values() if r.cls == "Q10" and (s := _score(r)) is not None]
        other = [s for r in items.values() if r.cls != "Q10" and (s := _score(r)) is not None]
        out.append(f"| {k} | {row['run'].name} | {_fmt(sum(q10) / len(q10) if q10 else None)} "
                   f"| {_fmt(sum(other) / len(other) if other else None)} | {_fmt(w.get('used_q10'), 2)} / "
                   f"{_fmt(w.get('used_other'), 2)} | {_fmt(w.get('unsupported_share'), 2)} "
                   f"| {_fmt((w.get('latency_web') or {}).get('p95'), 1)} |")
    return out


def _uploads_section(rows: list[dict]) -> list[str]:
    """H12: Self-Route vs the whole file vs retrieval only, per file size: accuracy and cost."""
    files = sorted({f for row in rows for f in row["summary"].get("uploads", {})})
    out = ["", "## Загруженные файлы (H12)", "",
           "В ячейке: точность (судья; без судьи — обязательные фрагменты) / токенов на вопрос (промпт + ответ) / "
           "p95 латентности, с; ниже — какими путями шли ответы.", "",
           "| # | Конфигурация | " + " | ".join(files) + " |", "|---|---|" + "---|" * len(files)]
    for k, row in enumerate(rows, start=1):
        up = row["summary"].get("uploads", {})
        cells = []
        for f in files:
            u = up.get(f, {})
            acc = u.get("correctness") if u.get("correctness") is not None else u.get("must_include")
            routes = ", ".join(f"{r}: {n}" for r, n in sorted((u.get("routes") or {}).items(), key=lambda kv: str(kv[0])))
            cells.append(f"{_fmt(acc, 2)} / {_fmt(u.get('tokens'), 0)} / {_fmt((u.get('latency') or {}).get('p95'), 1)}"
                         f"<br>{routes}")
        out.append(f"| {k} | {row['run'].name} | " + " | ".join(cells) + " |")
    return out


def _agent_section(rows: list[dict]) -> list[str]:
    """H6 (CRAG lowers the share of wrong answers, keeps correct refusals on Q6) and NFR2 latency."""
    out = ["", "## Агент и проверка релевантности (H6, NFR2)", "",
           "Неверные — доля ответов с оценкой судьи «incorrect» среди вопросов, ответ на которые есть в файлах. "
           "Отказы на Q6 — precision / recall отказов. Латентность — p95, с.", "",
           "| # | Конфигурация | Корректность | Неверные | Отказы Q6: P / R | Ложные отказы | Отказов CRAG | Через агента "
           "| Шагов | p95 Q1 | p95 прямой | p95 агент |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, row in enumerate(rows, start=1):
        s = row["summary"]
        a, rf, j = s.get("agent") or {}, s.get("refusals") or {}, s.get("judge") or {}
        out.append(
            f"| {k} | {row['run'].name} | {_fmt((j.get('correctness') or {}).get('mean'))} | {_fmt(a.get('wrong_rate'))} "
            f"| {_fmt(rf.get('precision'), 2)} / {_fmt(rf.get('recall'), 2)} | {rf.get('false_refusals', '—')} "
            f"| {a.get('crag_refusals', 0)} ({a.get('crag_refusals_q6', 0)} на Q6) | {_fmt(a.get('used'), 2)} "
            f"| {_fmt(a.get('mean_steps'), 1)} | {_fmt((a.get('latency_q1') or {}).get('p95'), 1)} "
            f"| {_fmt((a.get('latency_direct') or {}).get('p95'), 1)} | {_fmt((a.get('latency_agent') or {}).get('p95'), 1)} |"
        )
    return out


def render_ablation(spec: AblationSpec, es: EvalSet, rows: list[dict], per_item: list[dict[str, ItemResult]]) -> str:
    ref_items = per_item[0]
    ids = [i for i, r in ref_items.items() if r.retrieval]
    types = sorted({ft for row in rows for ft in row["summary"]["retrieval_by_file_type"]})
    out = [f"# Абляции: {spec.name}", ""]
    if not ids:  # uploads (Q7), web (Q10): no fragments of the corpus to find, only answers to judge
        out.append(f"Набор `{es.name}` v{es.version}: у вопросов нет эталонных фрагментов корпуса, метрики поиска "
                   "не считаются — только качество и стоимость ответов.")
    else:
        out += [
            f"Набор `{es.name}` v{es.version}: вопросов с эталонными источниками — {len(ids)}. "
            f"Сравнение каждой конфигурации с первой (референсной) — парный бутстреп по вопросам для {COMPARE_METRIC}: "
            "Δ, 95% интервал Δ и одностороннее p для гипотезы «конфигурация не лучше референса».",
            "",
            "| # | Конфигурация | Recall@5 [95% CI] | Recall@10 | MRR@10 | nDCG@10 | Δ Recall@5 [CI], p | "
            + "".join(f".{ft} R@5 | " for ft in types) + "Поиск p50/p95, с |",
            "|---|---|---|---|---|---|---|" + "---|" * len(types) + "---|",
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
                f"| {''.join(_fmt(ft.get(t, {}).get('recall@5')) + ' | ' for t in types)}"
                f"{_fmt(lat.get('p50'), 2)} / {_fmt(lat.get('p95'), 2)} |"
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
    if any((row["summary"].get("reasoning") or {}).get("reasoning_share") for row in rows):
        out += _reasoning_section(rows, per_item)
    if any((row["summary"].get("sql") or {}).get("used_q3") for row in rows):
        out += _sql_section(rows, per_item)
    if any((row["summary"].get("web") or {}).get("used_q10") is not None
           or (row["summary"].get("web") or {}).get("used_other") for row in rows):
        out += _web_section(rows, per_item)
    if any(row["summary"].get("uploads") for row in rows):
        out += _uploads_section(rows)
    if any((row["summary"].get("agent") or {}).get("used") or (row["summary"].get("agent") or {}).get("crag_refusals")
           for row in rows):
        out += _agent_section(rows)
    out += ["", "Переопределения настроек:", ""]
    out += [f"{k}. **{row['run'].name}** — `{json.dumps(row['run'].overrides, ensure_ascii=False)}`"
            + (f" — {row['run'].note}" if row["run"].note else "") for k, row in enumerate(rows, start=1)]
    return "\n".join(out) + "\n"
