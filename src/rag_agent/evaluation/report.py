"""Markdown report of an evaluation run."""

from __future__ import annotations

from rag_agent.evaluation.dataset import CLASS_NAMES, EvalSet
from rag_agent.evaluation.runner import ItemResult

METRIC_LABELS = {
    "recall@1": "Recall@1",
    "recall@3": "Recall@3",
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "hit@5": "Hit@5",
    "mrr": "MRR@10",
    "ndcg@10": "nDCG@10",
}


def _f(x, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _ci(stat: dict | None) -> str:
    if not stat or stat.get("mean") is None:
        return "—"
    ci = stat.get("ci")
    ci_txt = f" [{ci[0]:.2f}; {ci[1]:.2f}]" if ci else ""
    return f"{stat['mean']:.3f}{ci_txt} (n={stat['n']})"


def _short(text: str | None, n: int = 160) -> str:
    text = (text or "").replace("\n", " ").replace("|", "\\|").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def render_report(summary: dict, results: list[ItemResult], es: EvalSet, config: dict) -> str:
    items = {i.id: i for i in es.items}
    out = [f"# Оценка: {config.get('name') or es.name}", ""]
    out += [
        f"- Набор: `{es.name}` v{es.version}, вопросов {summary['n_items']} (ошибок выполнения {summary['n_errors']})",
        f"- Корпус: `{config['corpus']}`, индекс `{config['index_signature']}`",
        f"- Эмбеддер: `{config['embedder']}`; чанкинг .py: `{config['chunking']['python']}`",
        f"- Поиск: `{config['retrieval']['mode']}`"
        f"{', реранкер' if config['retrieval'].get('rerank') else ''}"
        f"{', точный поиск имён' if config['retrieval'].get('symbols') else ''}; "
        f"фрагментов в контексте {config['retrieval']['top_k']}, метрики поиска по top-{config['retrieval']['eval_k']}",
        f"- Заголовки фрагментов: {'да' if config['chunking'].get('context_header', True) else 'нет'}; "
        f"контекст ноутбука: {'да' if config['chunking'].get('notebook_context') else 'нет'}",
        f"- LLM: `{config['llm']}`; маршрут: `{config['route']}`; судья: `{config.get('judge') or 'не запускался'}`",
        f"- Git: `{config.get('git')}`; дата: {config['date']}",
        "",
        "Интервалы — 95% перцентильный бутстреп по вопросам.",
        "",
        "## Поиск",
        "",
        "| Метрика | Значение |",
        "|---|---|",
    ]
    out += [f"| {METRIC_LABELS[m]} | {_ci(v)} |" for m, v in summary["retrieval"].items()]
    lat = summary.get("retrieval_latency") or {}
    if lat.get("p50") is not None:
        out.append(f"| Латентность поиска p50 / p95, с | {lat['p50']:.3f} / {lat['p95']:.3f} |")
    out += ["", "| Класс | n | Recall@5 | MRR@10 | nDCG@10 |", "|---|---|---|---|---|"]
    for c, v in summary["retrieval_by_class"].items():
        out.append(f"| {c} {CLASS_NAMES[c]} | {v['n']} | {_f(v['recall@5'])} | {_f(v['mrr'])} | {_f(v['ndcg@10'])} |")
    out += ["", "| Тип файла | эталонных фрагментов | Recall@1 | Recall@5 | Recall@10 |", "|---|---|---|---|---|"]
    for ft, v in summary["retrieval_by_file_type"].items():
        out.append(f"| .{ft} | {v['spans']} | {_f(v['recall@1'])} | {_f(v['recall@5'])} | {_f(v['recall@10'])} |")

    if "routing" in summary:
        a = summary["answers"]
        out += [
            "",
            "## Ответы",
            "",
            "| Метрика | Значение |",
            "|---|---|",
            f"| Точность маршрутизатора | {_ci(summary['routing']['accuracy'])} |",
            f"| Обязательные элементы ответа (must_include) | {_ci(a['must_include'])} |",
            f"| Точность цитирования по разметке | {_ci(a['citation_precision_ref'])} |",
            f"| Доля ответов со ссылками | {_f(a['grounded_rate'])} |",
        ]
        j = summary.get("judge")
        if j:
            out += [
                f"| Корректность, Q1–Q6 (судья) | {_ci(j['correctness'])} |",
                f"| Корректность, общие вопросы (судья) | {_ci(j['correctness_general'])} |",
                f"| Верность контексту (faithfulness) | {_ci(j['faithfulness'])} |",
                f"| Точность цитирования (судья) | {_f(j['citation_precision'])} |",
                f"| Релевантность ответа | {_ci(j['relevance'])} |",
            ]
        rf = summary["refusals"]
        out += [
            f"| Отказы на Q6: precision / recall | {_f(rf['precision'])} / {_f(rf['recall'])} |",
            f"| Ложные отказы на Q1–Q5 | {rf['false_refusals']} |",
            "",
            f"Маршрутизация (ожидалось→получено): "
            + ", ".join(f"{k}: {v}" for k, v in summary["routing"]["confusion"].items()),
        ]
        if summary.get("judge_by_class"):
            out += ["", "| Класс | n | Корректность (судья) |", "|---|---|---|"]
            for c, v in summary["judge_by_class"].items():
                out.append(f"| {c} {CLASS_NAMES[c]} | {v['n']} | {_f(v['correctness'])} |")
        lat = summary["latency"]
        out += [
            "",
            "## Латентность",
            "",
            "| Маршрут | p50, с | p95, с |",
            "|---|---|---|",
            *(f"| {k} | {_f(v['p50'], 1)} | {_f(v['p95'], 1)} |" for k, v in lat.items() if isinstance(v, dict)),
            "",
            f"Токенов на вопрос в среднем: {_f(lat['mean_tokens'], 0)}",
        ]

    out += ["", "## Разбор ошибок", ""]
    rows = []
    for r in results:
        item = items[r.id]
        problems = []
        if r.error:
            problems.append(f"ошибка: {r.error}")
        if r.retrieval is not None and r.retrieval["recall@5"] == 0:
            problems.append("эталон не найден в top-5")
        if r.route_ok is False:
            problems.append(f"маршрут {r.route} вместо {r.expected_route}")
        if r.refused and item.cls != "Q6":
            problems.append("ложный отказ")
        if item.cls == "Q6" and r.refused is False:
            problems.append("нет отказа")
        if r.judge and r.judge.correctness in ("incorrect", "partial") and item.cls != "Q6":
            problems.append(f"судья: {r.judge.correctness} — {r.judge.rationale}")
        if r.must_include_ok is False:
            problems.append("нет обязательных элементов")
        if problems:
            rows.append(f"| {r.id} | {_short(r.question, 80)} | {_short('; '.join(problems), 200)} | {_short(r.answer)} |")
    if rows:
        out += ["| id | Вопрос | Проблемы | Ответ |", "|---|---|---|---|", *rows]
    else:
        out.append("Проблемных вопросов нет.")
    return "\n".join(out) + "\n"
