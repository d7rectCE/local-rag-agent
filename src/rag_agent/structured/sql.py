"""SQL tool over the analytics database (ТЗ S6), CHESS-style:

1. the schema is narrowed to the tables closest to the question (embeddings of the
   table and column descriptions) plus the tables they join through;
2. values of categorical columns (datasets, models, metric names) that match the
   question are passed as hints, so filters use the stored spelling;
3. a few similar "question -> SQL" examples come from a built-in, extendable base;
4. the model writes one SELECT; it is validated (sqlglot when installed), forced
   under a LIMIT and run on a read-only connection whose authorizer allows reads
   only, with a timeout; on an error or an empty result the model gets feedback and
   retries, at most ``attempts`` times.
"""

from __future__ import annotations

import difflib
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field

from rag_agent.config import CatalogConfig
from rag_agent.llm import BaseLLM, LLMError
from rag_agent.structured.schema import TABLE_BY_NAME, TABLES

# joins the model needs when it picks one side of them
JOINS = {"metrics": ("experiments",), "hyperparameters": ("experiments",), "cells": ("notebooks",)}
N_TABLES = 4
N_EXAMPLES = 3

EXAMPLES = [
    {"question": "Сколько экспериментов проведено на каждом датасете?",
     "sql": "SELECT dataset, COUNT(*) AS n_experiments FROM experiments GROUP BY dataset ORDER BY n_experiments DESC"},
    {"question": "В каком эксперименте самый низкий RMSE на валидации?",
     "sql": "SELECT e.path, e.model, m.value AS rmse, m.variant, m.cell FROM metrics m JOIN experiments e "
            "ON e.id = m.experiment_id WHERE m.name = 'rmse' AND m.split = 'val' ORDER BY m.value ASC LIMIT 1"},
    {"question": "Какие модели обучались на датасете wine?",
     "sql": "SELECT DISTINCT model, path, cell FROM experiments WHERE dataset = 'wine'"},
    {"question": "Средний F1 на тесте по всем экспериментам",
     "sql": "SELECT AVG(value) AS mean_f1, COUNT(*) AS n FROM metrics WHERE name = 'f1' AND split = 'test'"},
    {"question": "Какие значения max_depth перебирались и какой accuracy на валидации дал каждый?",
     "sql": "SELECT variant, value AS accuracy, path, cell FROM metrics WHERE name = 'accuracy' AND split = 'val' "
            "AND variant LIKE 'max_depth=%' ORDER BY value DESC"},
    {"question": "Какие функции вызываются чаще всего?",
     "sql": "SELECT name, COUNT(*) AS n_calls, COUNT(DISTINCT path) AS n_files FROM calls GROUP BY name "
            "ORDER BY n_calls DESC LIMIT 10"},
    {"question": "В каких ноутбуках есть ячейки, упавшие с ошибкой?",
     "sql": "SELECT path, cell, error_name FROM cells WHERE has_error = 1 ORDER BY path, cell"},
    {"question": "Где определён класс Pipeline и какие у него методы?",
     "sql": "SELECT qualname, kind, path, line FROM functions WHERE name = 'Pipeline' OR qualname LIKE 'Pipeline.%'"},
    {"question": "Какой learning rate задан в экспериментах с градиентным бустингом?",
     "sql": "SELECT h.value, h.variant, e.path, h.cell FROM hyperparameters h JOIN experiments e ON e.id = h.experiment_id "
            "WHERE h.name LIKE '%learning_rate%' AND e.model LIKE '%Boost%'"},
    {"question": "Сколько ноутбуков в архиве и сколько в них ячеек?",
     "sql": "SELECT COUNT(*) AS n_notebooks, SUM(n_cells) AS n_cells FROM notebooks"},
]

SQL_PROMPT = """Ты пишешь один SQL-запрос SQLite к каталогу исследовательского архива пользователя, чтобы ответить на вопрос.

Схема (только эти таблицы и колонки):
{schema}

Правила:
- только SELECT (можно WITH); никаких изменений данных;
- строковые значения в условиях бери из подсказок значений, если они даны: в каталоге они записаны именно так;
- у метрик указывай и name, и split, если вопрос о конкретной выборке;
- если в результате есть значения из ноутбуков, верни и колонки path и cell строк-источников — по ним ставятся ссылки;
- не больше {max_rows} строк;
- если по каталогу ответить нельзя, верни пустой sql и объясни почему в reason.

Верни JSON: {{"sql": "...", "reason": "..."}}"""

SQL_SCHEMA = {
    "type": "object",
    "properties": {"sql": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["sql", "reason"],
}

SPLIT_WORDS = {"валидац": "val", "validation": "val", "valid": "val", "тест": "test", "test": "test",
               "обучающ": "train", "train": "train", "кросс-валид": "cv", "cross-valid": "cv"}

# sqlite3 authorizer action codes that only read (select, read a column, call a function, recursive CTE)
_ALLOWED = {getattr(sqlite3, "SQLITE_SELECT", 21), getattr(sqlite3, "SQLITE_READ", 20),
            getattr(sqlite3, "SQLITE_FUNCTION", 31), getattr(sqlite3, "SQLITE_RECURSIVE", 33)}


class SQLValidationError(ValueError):
    pass


class SQLResult(BaseModel):
    question: str
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list] = Field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    reason: str = ""
    attempts: int = 0
    tables: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.sql)

    def markdown(self, max_rows: int = 30) -> str:
        if not self.columns:
            return "(пустой результат)"
        head = "| " + " | ".join(self.columns) + " |\n|" + "---|" * len(self.columns)
        body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in self.rows[:max_rows]]
        more = [f"(ещё {len(self.rows) - max_rows} строк)"] if len(self.rows) > max_rows else []
        cut = ["(результат обрезан по LIMIT)"] if self.truncated else []
        return "\n".join([head, *body, *more, *cut]) if body else head + "\n(0 строк)"

    def provenance(self) -> list[tuple[str, int | None]]:
        """(path, cell) of the rows, when the query returned them."""
        cols = [c.lower() for c in self.columns]
        pi = next((cols.index(c) for c in ("path", "file", "file_path", "notebook") if c in cols), None)
        if pi is None:
            return []
        ci = cols.index("cell") if "cell" in cols else None
        out = []
        for row in self.rows:
            key = (row[pi], row[ci] if ci is not None else None)
            if isinstance(key[0], str) and key not in out:
                out.append(key)
        return out


def _fmt(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v).replace("|", "\\|").replace("\n", " ")[:200]


def validate_sql(sql: str, max_rows: int, tables: set[str] = frozenset(t.name for t in TABLES)) -> str:
    """One read-only SELECT over known tables, returned with a LIMIT of max_rows + 1
    (one extra row tells that the result was cut). Raises SQLValidationError."""
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        raise SQLValidationError("пустой запрос")
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:  # a coarser check; the read-only connection and the authorizer still hold
        if ";" in sql or not re.match(r"(?is)^\s*(select|with)\b", sql):
            raise SQLValidationError("нужен ровно один запрос SELECT")
        ctes = {m.lower() for m in re.findall(r"(?i)(?:\bwith(?:\s+recursive)?|,)\s*([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s+as\s*\(", sql)}
        used = {m.lower() for m in re.findall(r"(?i)\b(?:from|join)\s+([A-Za-z_]\w*)", sql)} - ctes
        unknown = sorted(used - {t.lower() for t in tables})
        if unknown:
            raise SQLValidationError(f"неизвестные таблицы: {', '.join(unknown)}; доступны: {', '.join(sorted(tables))}")
        return f"SELECT * FROM ({sql}) LIMIT {max_rows + 1}"
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise SQLValidationError(f"синтаксическая ошибка: {str(exc).splitlines()[0][:300]}") from exc
    if len(statements) != 1:
        raise SQLValidationError("нужен ровно один запрос")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SQLValidationError(f"разрешён только SELECT, получено {tree.key.upper()}")
    forbidden = tuple(getattr(exp, n) for n in ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command",
                                                  "Pragma", "Attach", "Detach", "Merge") if hasattr(exp, n))
    bad = next((n for n in tree.walk() if isinstance(n, forbidden)), None)
    if bad is not None:
        raise SQLValidationError(f"запрещённая операция {bad.key.upper()}")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    used = {t.name.lower() for t in tree.find_all(exp.Table)} - ctes
    unknown = sorted(used - {t.lower() for t in tables})
    if unknown:
        raise SQLValidationError(f"неизвестные таблицы: {', '.join(unknown)}; доступны: {', '.join(sorted(tables))}")
    return f"SELECT * FROM ({tree.sql(dialect='sqlite')}) LIMIT {max_rows + 1}"


def _authorizer(action, *_args) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


def run_readonly(db: Path, sql: str, timeout_s: float) -> tuple[list[str], list[list]]:
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    try:
        conn.set_authorizer(_authorizer)
        deadline = time.perf_counter() + timeout_s
        conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 2000)
        try:
            cur = conn.execute(sql)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc):
                raise sqlite3.OperationalError(f"превышен таймаут {timeout_s:g} с") from exc
            raise
        return [d[0] for d in cur.description or []], [list(r) for r in cur.fetchall()]
    finally:
        conn.close()


class SQLTool:
    def __init__(self, db: Path, llm: BaseLLM, cfg: CatalogConfig, embedder=None):
        self.db, self.llm, self.cfg, self.embedder = db, llm, cfg, embedder
        self.examples = list(EXAMPLES)
        if cfg.examples and Path(cfg.examples).expanduser().exists():
            self.examples += yaml.safe_load(Path(cfg.examples).expanduser().read_text(encoding="utf-8")) or []
        self._vectors: dict[str, np.ndarray] = {}

    # --- schema narrowing, values, examples -------------------------------------
    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.embedder.encode(texts).dense, dtype=np.float32)

    def _ranked(self, key: str, question: str, docs: list[str]) -> list[int]:
        if self.embedder is None:
            q = set(_words(question))
            scores = [len(q & set(_words(d))) for d in docs]
        else:
            if key not in self._vectors:
                self._vectors[key] = self._embed(docs)
            scores = list(self._vectors[key] @ self._embed([question])[0])
        return sorted(range(len(docs)), key=lambda i: -scores[i])

    def narrow(self, question: str, hinted: set[str]) -> list[str]:
        order = self._ranked("tables", question, [t.search_text() for t in TABLES])
        picked = [TABLES[i].name for i in order[:N_TABLES]]
        for name in [*picked, *sorted(hinted)]:
            for extra in (name, *JOINS.get(name, ())):
                if extra not in picked:
                    picked.append(extra)
        return picked

    def value_hints(self, question: str) -> list[str]:
        q = question.lower()
        tokens = set(_words(q))
        hints = []
        for word, split in SPLIT_WORDS.items():
            if word in q:
                hint = f"metrics.split = '{split}'"
                if hint not in hints:
                    hints.append(hint)
        try:
            for t in TABLES:
                for col in t.value_columns:
                    _, rows = run_readonly(self.db, f"SELECT DISTINCT {col} FROM {t.name} WHERE {col} IS NOT NULL "
                                                    f"AND {col} != '' LIMIT 500", self.cfg.timeout_s)
                    for (value,) in rows:
                        if _matches(str(value), q, tokens):
                            hints.append(f"{t.name}.{col} = '{value}'")
        except sqlite3.Error:
            pass
        return hints[:25]

    def similar_examples(self, question: str) -> list[dict]:
        order = self._ranked("examples", question, [e["question"] for e in self.examples])
        return [self.examples[i] for i in order[:N_EXAMPLES]]

    # --- generation loop --------------------------------------------------------
    def run(self, question: str) -> SQLResult:
        t0 = time.perf_counter()
        res = SQLResult(question=question)
        if not self.db.exists():
            res.error = "каталог ещё не построен"
            return res
        res.hints = self.value_hints(question)
        res.tables = self.narrow(question, {h.split(".")[0] for h in res.hints})
        schema = "\n\n".join(TABLE_BY_NAME[t].describe() for t in res.tables)
        shots = "\n\n".join(f"Вопрос: {e['question']}\nSQL: {e['sql']}" for e in self.similar_examples(question))
        user = (f"Похожие примеры:\n{shots}\n\n"
                + (f"Подсказки значений:\n" + "\n".join(res.hints) + "\n\n" if res.hints else "")
                + f"Вопрос: {question}")
        messages = [{"role": "system", "content": SQL_PROMPT.format(schema=schema, max_rows=self.cfg.max_rows)},
                    {"role": "user", "content": user}]
        for attempt in range(1, self.cfg.attempts + 1):
            res.attempts = attempt
            try:
                data = self.llm.chat(messages, json_schema=SQL_SCHEMA, max_tokens=600, purpose="sql").json()
            except LLMError as exc:
                res.error = f"LLM: {exc}"
                break
            sql, res.reason = str(data.get("sql") or "").strip(), str(data.get("reason") or "")
            messages.append({"role": "assistant", "content": f'{{"sql": {sql!r}, "reason": {res.reason!r}}}'})
            if not sql:
                res.error = f"модель не нашла запроса: {res.reason}" if res.reason else "модель не нашла запроса"
                res.sql = None
                break
            res.sql = sql
            try:
                final = validate_sql(sql, self.cfg.max_rows)
                res.columns, rows = run_readonly(self.db, final, self.cfg.timeout_s)
            except (SQLValidationError, sqlite3.Error) as exc:
                res.error = str(exc)
                messages.append({"role": "user", "content": f"Запрос не выполнен: {exc}. Исправь запрос."})
                continue
            res.truncated = len(rows) > self.cfg.max_rows
            res.rows, res.error = rows[: self.cfg.max_rows], None
            if rows or attempt == self.cfg.attempts:
                break
            messages.append({"role": "user", "content": "Запрос вернул 0 строк. Проверь значения в условиях "
                                                        "(строки пишутся как в подсказках) и исправь запрос."})
        res.latency_s = round(time.perf_counter() - t0, 3)
        return res


def _words(text: str) -> list[str]:
    return re.findall(r"[\w\-]+", text.lower())


def _matches(value: str, question: str, tokens: set[str]) -> bool:
    """A stored value is mentioned in the question: literally (breast_cancer ~ "breast cancer",
    roc_auc ~ "ROC-AUC") or as a close spelling of one of its words."""
    v = value.lower()
    if len(v) < 2:
        return False
    flat = re.sub(r"[\s_\-]+", "", v)
    q_flat = re.sub(r"[\s_\-]+", "", question)
    if len(flat) >= 3 and flat in q_flat:
        return True
    parts = [p for p in re.split(r"[\s_\-=]+", v) if len(p) >= 4]
    return any(difflib.get_close_matches(p, tokens, n=1, cutoff=0.85) for p in parts)
