"""LLM extraction of experiments from notebooks into the catalog (ТЗ S2, S6).

One structured-output call per notebook returns its experiments with metric values
and hyperparameters, each with the number of the cell it comes from. Every value is
then checked against that cell: a metric must be printed in the cell's output or
source (up to rounding), a hyperparameter must be written in its code. Values not
found there are looked up in the other cells of the notebook and otherwise dropped,
so a hallucinated number cannot reach the catalog. Names of metrics and splits are
normalised (ROC-AUC -> roc_auc, validation -> val), which keeps SQL filters simple.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict

from rag_agent.config import CatalogConfig
from rag_agent.llm import BaseLLM, LLMError
from rag_agent.schema import Node, NodeType

log = logging.getLogger(__name__)

EXTRACT_PROMPT = """Ты извлекаешь из Jupyter-ноутбука сведения об экспериментах для реляционного каталога. Ячейки даны с номерами.

Эксперимент — обучение и оценка модели (или разведочный анализ данных, task="eda") на одном датасете. В одном ноутбуке может быть несколько экспериментов; перебор гиперпараметра одной модели — это один эксперимент с несколькими вариантами.

Для каждого эксперимента верни:
- title — короткое название; task — classification, regression, clustering или eda;
- dataset — имя датасета так, как оно написано в коде (например breast_cancer); model — класс или название модели (например LogisticRegression);
- cell — номер ячейки, где эксперимент описан или начинается;
- metrics — каждое напечатанное значение метрики: name (accuracy, f1, roc_auc, rmse, r2, ari, silhouette, loss и т.п.), value (число ровно как в выводе), split (val, test, train, cv или none), variant (чем этот запуск отличается внутри эксперимента, например "learning_rate=0.3", "shift_aug", "schedule=step"; пустая строка, если запуск один), cell — номер ячейки, в выводе которой напечатано значение;
- hyperparameters — явно заданные в коде значения: name (как в коде), value (как в коде), variant, cell.

Правила: бери только то, что есть в ячейках; не придумывай значений; если метрика напечатана в таблице по вариантам, дай одну запись на каждую строку таблицы. Не повторяй одно и то же значение дважды."""

_METRIC = {
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "value": {"type": "number"}, "split": {"type": "string"},
        "variant": {"type": "string"}, "cell": {"type": "integer"},
    },
    "required": ["name", "value", "split", "variant", "cell"],
}
_HPARAM = {
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "value": {"type": "string"}, "variant": {"type": "string"},
        "cell": {"type": "integer"},
    },
    "required": ["name", "value", "variant", "cell"],
}
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "experiments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"}, "task": {"type": "string"}, "dataset": {"type": "string"},
                    "model": {"type": "string"}, "cell": {"type": "integer"},
                    "metrics": {"type": "array", "items": _METRIC},
                    "hyperparameters": {"type": "array", "items": _HPARAM},
                },
                "required": ["title", "task", "dataset", "model", "cell", "metrics", "hyperparameters"],
            },
        }
    },
    "required": ["experiments"],
}

METRIC_SYNONYMS = {
    "roc_auc_score": "roc_auc", "rocauc": "roc_auc", "auc": "roc_auc", "roc": "roc_auc", "auc_roc": "roc_auc",
    "acc": "accuracy", "accuracy_score": "accuracy", "best_accuracy": "accuracy",
    "f1_macro": "f1", "macro_f1": "f1", "f1_score": "f1", "f_1": "f1",
    "r²": "r2", "r^2": "r2", "r2_score": "r2",
    "adjusted_rand_score": "ari", "adjusted_rand_index": "ari", "silhouette_score": "silhouette",
    "cross_entropy": "loss", "log_loss": "loss",
}
SPLITS = {
    "validation": "val", "valid": "val", "val": "val", "dev": "val",
    "test": "test", "holdout": "test", "train": "train", "training": "train",
    "cv": "cv", "cross_validation": "cv", "crossval": "cv", "": "none", "none": "none", "null": "none", "n/a": "none",
}
TASKS = {"binary_classification": "classification", "multiclass_classification": "classification",
         "classification": "classification", "regression": "regression", "clustering": "clustering",
         "eda": "eda", "exploratory_data_analysis": "eda", "analysis": "eda"}


def _snake(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(text).strip().lower())


def normalize_metric(name: str, split: str) -> tuple[str, str]:
    name, split = _snake(name), SPLITS.get(_snake(split), _snake(split) or "none")
    for prefix, sp in (("val_", "val"), ("valid_", "val"), ("test_", "test"), ("train_", "train"), ("best_val_", "val")):
        if name.startswith(prefix) and len(name) > len(prefix):
            name, split = name[len(prefix):], sp if split == "none" else split
    return METRIC_SYNONYMS.get(name, name), split


# --------------------------------------------------------------------------- grounding

_NUM = re.compile(r"[-+]?(?:\d+[.,]\d+|\d+|[.,]\d+)(?:[eE][-+]?\d+)?")


def _decimals(token: str) -> int:
    mant = re.split(r"[eE]", token)[0]
    frac = re.split(r"[.,]", mant)
    d = len(frac[1]) if len(frac) > 1 else 0
    exp = re.split(r"[eE]", token)
    return d - int(exp[1]) if len(exp) > 1 else d


def numbers(text: str) -> list[tuple[float, int]]:
    out = []
    for tok in _NUM.findall(text or ""):
        try:
            out.append((float(tok.replace(",", ".")), _decimals(tok)))
        except ValueError:
            continue
    return out


def value_in(text: str, value: float) -> bool:
    """The value is written in the text: equal at the printed precision, or the printed
    number rounded by the model to at least 4 decimals (0.97777 -> 0.9778); a percentage
    (97.78) counts too. A coarser value (0.99 for a printed 0.9925) does not."""
    if value != value:
        return False
    own = _decimals(repr(float(value)))
    for x, dec in numbers(text):
        for printed, d in ((x, dec), (x / 100, dec + 2)):
            if abs(printed - value) <= 1e-9 * max(1.0, abs(printed)):
                return True
            if 4 <= own < d and abs(round(printed, own) - value) <= 1e-12:
                return True
    return False


def _hparam_in(text: str, value: str) -> bool:
    v = str(value).strip().strip("'\"")
    if not v:
        return False
    if v in (text or ""):
        return True
    try:
        return value_in(text, float(v))
    except ValueError:
        return False


def _num(value: str) -> float | None:
    try:
        return float(str(value).strip().strip("'\""))
    except ValueError:
        return None


# --------------------------------------------------------------------------- notebook text


def cell_texts(nodes: list[Node]) -> dict[int, dict[str, str]]:
    """cell -> {"type": code|markdown, "source": ..., "output": ...} from the notebook's fragments."""
    cells: dict[int, dict[str, str]] = defaultdict(lambda: {"type": "", "source": "", "output": ""})
    for n in sorted(nodes, key=lambda n: (n.location.cell or 0, n.id)):
        c = n.location.cell
        if c is None:
            continue
        if n.node_type == NodeType.CELL_OUTPUT:
            cells[c]["output"] += n.text
        elif n.node_type in (NodeType.CODE_CELL, NodeType.MARKDOWN_CELL):
            cells[c]["type"] = "code" if n.node_type == NodeType.CODE_CELL else "markdown"
            # a long cell is split into overlapping parts: keep lines not already present
            if cells[c]["source"]:
                known = set(cells[c]["source"].split("\n"))
                cells[c]["source"] += "\n" + "\n".join(l for l in n.text.split("\n") if l not in known)
            else:
                cells[c]["source"] = n.text
    return dict(cells)


def notebook_prompt(cells: dict[int, dict[str, str]], max_chars: int) -> str:
    per_cell = max(400, max_chars // max(1, len(cells)))
    parts = []
    for c in sorted(cells):
        cell = cells[c]
        src = cell["source"][:per_cell]
        parts.append(f"[ячейка {c} · {cell['type'] or 'output'}]\n{src}")
        if cell["output"]:
            parts.append(f"[вывод ячейки {c}]\n{cell['output'][:per_cell]}")
    return "\n\n".join(parts)[:max_chars]


# --------------------------------------------------------------------------- extraction


def _ground(items: list[dict], cells: dict[int, dict[str, str]], check) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for it in items:
        cited = it.get("cell")
        order = ([cited] if cited in cells else []) + [c for c in sorted(cells) if c != cited]
        home = next((c for c in order if check(cells[c]["source"] + "\n" + cells[c]["output"], it)), None)
        if home is None:
            dropped += 1
            continue
        kept.append({**it, "cell": home})
    return kept, dropped


def extract_notebook(nodes: list[Node], llm: BaseLLM, cfg: CatalogConfig) -> tuple[list[dict], int]:
    """Experiments of one notebook, grounded in its cells; returns (experiments, dropped values)."""
    cells = cell_texts(nodes)
    if not cells:
        return [], 0
    resp = llm.chat(
        [{"role": "system", "content": EXTRACT_PROMPT},
         {"role": "user", "content": notebook_prompt(cells, cfg.notebook_chars)}],
        json_schema=EXTRACT_SCHEMA, max_tokens=4096, purpose="extract",
    )
    data = resp.json()
    experiments, dropped = [], 0
    for e in data.get("experiments") or []:
        metrics = []
        for m in e.get("metrics") or []:
            try:
                value = float(m["value"])
            except (KeyError, TypeError, ValueError):
                dropped += 1
                continue
            name, split = normalize_metric(m.get("name", ""), m.get("split", ""))
            if name:
                metrics.append({"name": name, "value": value, "split": split,
                                "variant": str(m.get("variant") or "").strip(), "cell": m.get("cell")})
        metrics, d1 = _ground(metrics, cells, lambda text, it: value_in(text, it["value"]))
        seen, unique = set(), []
        for m in metrics:  # the same printed value extracted twice
            key = (m["name"], m["split"], m["variant"], round(m["value"], 9))
            if key not in seen:
                seen.add(key)
                unique.append(m)
        hparams = [{"name": str(h.get("name", "")).strip(), "value": str(h.get("value", "")).strip().strip("'\""),
                    "variant": str(h.get("variant") or "").strip(), "cell": h.get("cell")}
                   for h in e.get("hyperparameters") or [] if str(h.get("name", "")).strip()]
        hparams, d2 = _ground(hparams, cells, lambda text, it: _hparam_in(text, it["value"]))
        for h in hparams:
            h["value_num"] = _num(h["value"])
        dropped += d1 + d2
        cell = e.get("cell") if e.get("cell") in cells else min(cells)
        experiments.append({
            "title": str(e.get("title") or "").strip(),
            "task": TASKS.get(_snake(e.get("task", "")), _snake(e.get("task", ""))),
            "dataset": _snake(e.get("dataset", "")),
            "model": str(e.get("model") or "").strip(),
            "cell": cell, "metrics": unique, "hyperparameters": hparams,
        })
    return experiments, dropped


def update_catalog(index, llm: BaseLLM, cfg: CatalogConfig, progress=None,
                   cancel: threading.Event | None = None) -> dict:
    """Extract every notebook whose content or extraction model changed since the last run."""
    catalog = index.catalog
    files = catalog.file_states()
    done = catalog.extraction_states()
    todo = [p for p, (h, status) in sorted(files.items())
            if p.endswith(".ipynb") and status == "ok" and done.get(p) != (h, llm.name)]
    stats = {"notebooks": len(todo), "experiments": 0, "values": 0, "dropped": 0, "errors": 0, "seconds": 0.0}
    t0 = time.perf_counter()
    for k, path in enumerate(todo, start=1):
        if cancel is not None and cancel.is_set():
            break
        if progress is not None:
            progress.current = f"каталог экспериментов: {path} ({k}/{len(todo)})"
        try:
            experiments, dropped = extract_notebook(catalog.file_nodes(path), llm, cfg)
            catalog.replace_extraction(path, files[path][0], llm.name, experiments, dropped)
            stats["experiments"] += len(experiments)
            stats["values"] += sum(len(e["metrics"]) + len(e["hyperparameters"]) for e in experiments)
            stats["dropped"] += dropped
        except LLMError as exc:  # keep going: the notebook is retried on the next run
            log.warning("extraction failed for %s: %s", path, exc)
            stats["errors"] += 1
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return stats
