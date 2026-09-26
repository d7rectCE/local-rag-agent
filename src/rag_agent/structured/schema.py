"""Schema of the analytics database the SQL tool queries (ТЗ S6).

The database is a separate SQLite file, rebuilt from the index catalog after every
indexing run and opened read-only by the SQL tool, so queries never see the
internal tables (fragment texts of the vector index, meta) and cannot change them.
Table and column descriptions are what the tool narrows the schema by (CHESS).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    doc: str


@dataclass(frozen=True)
class Table:
    name: str
    doc: str
    columns: tuple[Column, ...] = field(default_factory=tuple)
    # columns whose distinct values are matched against the question (value retrieval)
    value_columns: tuple[str, ...] = ()

    def ddl(self) -> str:
        cols = ",\n    ".join(f"{c.name} {c.type}" for c in self.columns)
        return f"CREATE TABLE {self.name} (\n    {cols}\n)"

    def describe(self) -> str:
        """The table as the SQL prompt shows it: DDL with a comment per column."""
        cols = ",\n".join(f"    {c.name} {c.type}  -- {c.doc}" for c in self.columns)
        return f"-- {self.doc}\nCREATE TABLE {self.name} (\n{cols}\n);"

    def search_text(self) -> str:
        return f"{self.name}: {self.doc}. " + "; ".join(f"{c.name} — {c.doc}" for c in self.columns)


C = Column
TABLES: tuple[Table, ...] = (
    Table("files", "все проиндексированные файлы архива (all indexed files)", (
        C("path", "TEXT", "путь относительно корня архива, например experiments/02_logreg_baseline.ipynb"),
        C("file_type", "TEXT", "тип файла: py, ipynb, pdf, docx, txt, md, log"),
        C("size_bytes", "INTEGER", "размер файла в байтах"),
        C("n_fragments", "INTEGER", "число проиндексированных фрагментов"),
        C("status", "TEXT", "ok или error — удалось ли разобрать файл"),
    ), value_columns=("file_type",)),
    Table("notebooks", "Jupyter-ноутбуки (notebooks)", (
        C("path", "TEXT", "путь к ноутбуку"),
        C("title", "TEXT", "заголовок из первой markdown-ячейки"),
        C("n_cells", "INTEGER", "число ячеек"),
        C("n_code_cells", "INTEGER", "число ячеек с кодом"),
        C("out_of_order", "INTEGER", "1, если ячейки выполнялись не по порядку"),
    )),
    Table("cells", "ячейки ноутбуков: исходный код или markdown и текст вывода (notebook cells)", (
        C("path", "TEXT", "путь к ноутбуку"),
        C("cell", "INTEGER", "номер ячейки, с 1"),
        C("cell_type", "TEXT", "code или markdown"),
        C("source", "TEXT", "исходный текст ячейки"),
        C("output", "TEXT", "текст вывода ячейки (может быть NULL)"),
        C("has_error", "INTEGER", "1, если выполнение ячейки закончилось исключением"),
        C("error_name", "TEXT", "имя исключения, например NameError"),
    ), value_columns=("error_name",)),
    Table("functions", "функции, классы и методы, определённые в коде (definitions)", (
        C("name", "TEXT", "имя"),
        C("qualname", "TEXT", "полное имя, например Trainer.fit"),
        C("kind", "TEXT", "function, class или method"),
        C("signature", "TEXT", "сигнатура"),
        C("path", "TEXT", "файл, где определено"),
        C("line", "INTEGER", "строка начала определения"),
        C("cell", "INTEGER", "номер ячейки, если определено в ноутбуке"),
    ), value_columns=("kind",)),
    Table("calls", "вызовы функций в коде и ноутбуках (call sites)", (
        C("name", "TEXT", "имя вызываемой функции, например calc_metrics"),
        C("caller", "TEXT", "функция, из которой вызов сделан (NULL — верхний уровень)"),
        C("path", "TEXT", "файл с вызовом"),
        C("line", "INTEGER", "строка вызова"),
        C("cell", "INTEGER", "номер ячейки, если вызов в ноутбуке"),
    )),
    Table("experiments", "эксперименты, извлечённые из ноутбуков: задача, датасет, модель (experiments)", (
        C("id", "INTEGER", "идентификатор эксперимента"),
        C("path", "TEXT", "ноутбук эксперимента"),
        C("title", "TEXT", "краткое название"),
        C("task", "TEXT", "тип задачи: classification, regression, clustering, eda"),
        C("dataset", "TEXT", "датасет, как он назван в коде, например breast_cancer"),
        C("model", "TEXT", "модель, например LogisticRegression, HistGradientBoosting"),
        C("cell", "INTEGER", "ячейка, где эксперимент описан или начат"),
    ), value_columns=("task", "dataset", "model")),
    Table("metrics", "значения метрик экспериментов, по одной строке на значение (metric values)", (
        C("experiment_id", "INTEGER", "ссылка на experiments.id"),
        C("path", "TEXT", "ноутбук"),
        C("name", "TEXT", "метрика в нижнем регистре: accuracy, f1, roc_auc, rmse, r2, ari, silhouette"),
        C("value", "REAL", "значение"),
        C("split", "TEXT", "выборка: val, test, train, cv или none"),
        C("variant", "TEXT", "вариант внутри эксперимента, например learning_rate=0.3 или shift_aug; пусто для единственного"),
        C("cell", "INTEGER", "ячейка, в выводе которой напечатано значение"),
    ), value_columns=("name", "split", "variant")),
    Table("hyperparameters", "гиперпараметры экспериментов, заданные в коде (hyperparameters)", (
        C("experiment_id", "INTEGER", "ссылка на experiments.id"),
        C("path", "TEXT", "ноутбук"),
        C("name", "TEXT", "имя как в коде, например learning_rate, alpha, C"),
        C("value", "TEXT", "значение как в коде"),
        C("value_num", "REAL", "числовое значение или NULL"),
        C("variant", "TEXT", "вариант внутри эксперимента или пусто"),
        C("cell", "INTEGER", "ячейка, где задано значение"),
    ), value_columns=("name",)),
    Table("relations", "связи между файлами и фрагментами: вызовы, импорты, код→вывод, иллюстрации (relations)", (
        C("src_path", "TEXT", "файл-источник связи"),
        C("src", "TEXT", "фрагмент-источник"),
        C("dst", "TEXT", "фрагмент или имя, на которое указывает связь"),
        C("type", "TEXT", "тип: calls, imports, produces, illustrates, uses_var"),
    ), value_columns=("type",)),
)
TABLE_BY_NAME = {t.name: t for t in TABLES}
