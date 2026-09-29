"""The code agent writes a new program of several modules (the "project" pipeline): a data survey, a plan,
the files one by one, the checks (compile, tests, the plan's command), the fix loop; the user's database is
read by the code but never ends up in the diff or back in the user's folder."""

import sqlite3
from pathlib import Path

import pytest

from rag_agent.code.agent import CodeAgent, interfaces
from rag_agent.code.workspace import Workspace
from rag_agent.engine import Engine
from rag_agent.router import CODE_HINT
from tests.conftest import FakeLLM
from tests.test_code_agent import HostSandbox

PLAN = {
    "name": "shop_reports", "title": "Отчёты по продажам", "summary": "Суммы продаж по товарам из data/shop.db.",
    "architecture": "Sale (dataclass), SalesRepository — чтение SQLite, ReportService — суммы, CLI в __main__",
    "files": [
        {"path": "models.py", "purpose": "модели", "api": "Sale(product, amount)"},
        {"path": "repository.py", "purpose": "доступ к данным", "api": "SalesRepository(path).all() -> list[Sale]"},
        {"path": "service.py", "purpose": "бизнес-логика", "api": "ReportService(repo).totals() -> dict"},
        {"path": "tests/test_service.py", "purpose": "тесты", "api": ""},
        {"path": "README.md", "purpose": "описание", "api": ""},
        {"path": "__main__.py", "purpose": "CLI", "api": "report [--db]"},
    ],
    "run": "python -m shop_reports report", "check": "python -m shop_reports report", "requirements": [],
}

FILES = {
    "shop_reports/__init__.py": '"""Отчёты по продажам магазина."""',
    "shop_reports/models.py": '''"""Модели данных."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Sale:
    product: str
    amount: float''',
    "shop_reports/repository.py": '''"""Чтение продаж из SQLite (только чтение)."""
import sqlite3
from pathlib import Path

from shop_reports.models import Sale

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "shop.db"


class SalesRepository:
    def __init__(self, path: Path = DEFAULT_DB):
        self.path = Path(path)

    def all(self) -> list[Sale]:
        con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        try:
            return [Sale(p, a) for p, a in con.execute("SELECT product, amount FROM sales")]
        finally:
            con.close()''',
    "shop_reports/service.py": '''"""Отчёты по продажам."""
from collections import defaultdict

from shop_reports.repository import SalesRepository


class ReportService:
    def __init__(self, repo: SalesRepository):
        self.repo = repo

    def totals(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for sale in self.repo.all():
            out[sale.product] += sale.amount
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))''',
    "shop_reports/__main__.py": '''"""Точка входа: python -m shop_reports report [--db путь]."""
import argparse

from shop_reports.repository import DEFAULT_DB, SalesRepository
from shop_reports.service import ReportService


def main() -> None:
    parser = argparse.ArgumentParser(prog="shop_reports")
    parser.add_argument("command", choices=["report"])
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()
    for product, total in ReportService(SalesRepository(args.db)).totals().items():
        print(f"{product}: {total:.2f}")


if __name__ == "__main__":
    main()''',
    "shop_reports/tests/test_service.py": '''import sqlite3

from shop_reports.repository import SalesRepository
from shop_reports.service import ReportService


def test_totals(tmp_path):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE sales (product TEXT, amount REAL)")
    con.executemany("INSERT INTO sales VALUES (?, ?)", [("a", 1.0), ("b", 5.0), ("a", 2.5)])
    con.commit()
    con.close()
    assert ReportService(SalesRepository(db)).totals() == {"b": 5.0, "a": 3.5}''',
    "shop_reports/README.md": "# Отчёты по продажам\n\nЗапуск из корня папки:\n\n```bash\npython -m shop_reports report\n```",
}


@pytest.fixture
def shop(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    (root / "data").mkdir(parents=True)
    con = sqlite3.connect(root / "data" / "shop.db")
    con.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT, amount REAL)")
    con.executemany("INSERT INTO sales (product, amount) VALUES (?, ?)", [("чай", 10.0), ("кофе", 30.0), ("чай", 5.0)])
    con.commit()
    con.close()
    (root / "notes.md").write_text("Магазин: продажи в data/shop.db\n", encoding="utf-8")
    return root


def project_llm(settings, files=FILES, agent=None):
    return FakeLLM(settings, pipeline={"pipeline": "project", "target": "data/shop.db"}, project_plan=PLAN,
                   project_files=files, agent=agent)


def test_the_database_is_read_but_never_applied(tmp_path: Path, shop: Path):
    ws = Workspace.create(tmp_path / "ws", shop)
    assert "data/shop.db" in ws.files()  # the code can read it
    (ws.root / "data" / "shop.db").write_bytes(b"changed by a run")
    (ws.root / "report.md").write_text("ok", encoding="utf-8")
    assert ws.changed() == [("A", "report.md")]  # ignored by git: not in the diff
    assert ws.apply_to(shop) == ["report.md"]
    assert (shop / "data" / "shop.db").read_bytes().startswith(b"SQLite format 3")  # the user's copy is intact


def test_a_program_is_planned_written_and_checked(settings, tmp_path: Path, shop: Path):
    ws = Workspace.create(tmp_path / "ws", shop)
    llm = project_llm(settings)
    res = CodeAgent(settings, llm, HostSandbox(), ws, "Напиши по базе shop.db сервис отчётов о продажах", "p1").run()
    assert res.pipeline == "project" and res.status == "done", res.summary
    # the survey read the database for the plan: its schema and rows are in the planner's prompt
    plan_prompt = llm.calls[llm.kinds.index("project_plan")][-1]["content"]
    assert "CREATE TABLE sales" in plan_prompt and "3 строк" in plan_prompt
    assert res.project["root"] == "shop_reports" and "shop_reports/__main__.py" in res.project["files"]
    added = {path for st, path in res.changed if st == "A"}
    assert {"shop_reports/service.py", "shop_reports/tests/test_service.py", "shop_reports/README.md"} <= added
    assert not any(p.endswith(".db") for p in added)
    # the files are written in order: code before the entry point, then tests, then the README
    kinds = [st["args"]["path"] for st in res.steps if st["action"] == "create_file"]
    assert kinds[0] == "shop_reports/__init__.py" and kinds[-1] == "shop_reports/README.md"
    assert kinds.index("shop_reports/__main__.py") < kinds.index("shop_reports/tests/test_service.py")
    # a file written later sees the interfaces of the earlier ones
    service_prompt = next(c[-1]["content"] for c in llm.calls if "Файл: shop_reports/service.py" in c[-1]["content"])
    assert "class SalesRepository:" in service_prompt and "def all(self) -> list[Sale]" in service_prompt
    # the checks ran: compile, the tests, the program itself on the user's data
    assert "Проверка пройдена" in res.summary and "кофе: 30.00" in res.summary and "чай: 15.00" in res.summary
    assert "agent" not in llm.kinds  # nothing to fix
    readme = (ws.root / "shop_reports" / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Отчёты") and "```bash" in readme  # the outer fence is gone, the inner one stays


def test_a_failing_check_goes_to_the_fix_loop(settings, tmp_path: Path, shop: Path):
    broken = {**FILES, "shop_reports/service.py": FILES["shop_reports/service.py"].replace(
        "out[sale.product] += sale.amount", "out[sale.product] = sale.amount")}
    fix = [
        {"thought": "суммы перезаписываются", "action": "edit_file", "args": {
            "path": "shop_reports/service.py", "old": "out[sale.product] = sale.amount",
            "new": "out[sale.product] += sale.amount"}},
        {"thought": "", "action": "run", "args": {"command": "pytest -q shop_reports/tests"}},
        {"thought": "", "action": "finish", "args": {"summary": "исправлено накопление сумм"}},
    ]
    ws = Workspace.create(tmp_path / "ws", shop)
    llm = project_llm(settings, files=broken, agent=fix)
    res = CodeAgent(settings, llm, HostSandbox(), ws, "Напиши по базе shop.db сервис отчётов о продажах", "p2").run()
    assert "agent" in llm.kinds and res.status == "done", res.summary
    assert "Исправления: исправлено накопление сумм" in res.summary and "Проверка пройдена" in res.summary
    # the loop was told what failed and what the plan is
    assert "Проверка не прошла" not in res.summary
    seed = llm.calls[llm.kinds.index("agent")][1]["content"]
    assert "Создан проект shop_reports/" in seed and "assert" in seed


def test_a_taken_folder_name_gets_a_suffix(settings, tmp_path: Path, shop: Path):
    (shop / "shop_reports").mkdir()
    (shop / "shop_reports" / "old.txt").write_text("мой файл", encoding="utf-8")
    ws = Workspace.create(tmp_path / "ws", shop)
    files = {k.replace("shop_reports/", "shop_reports_2/", 1): v.replace("shop_reports", "shop_reports_2")
             for k, v in FILES.items()}
    res = CodeAgent(settings, project_llm(settings, files=files), HostSandbox(), ws,
                    "Напиши по базе shop.db сервис отчётов о продажах", "p3").run()
    assert res.project["root"] == "shop_reports_2" and res.project["run"] == "python -m shop_reports_2 report"
    assert res.status == "done", res.summary
    assert (ws.root / "shop_reports" / "old.txt").read_text(encoding="utf-8") == "мой файл"


def test_interfaces_list_classes_and_signatures(tmp_path: Path):
    ws = Workspace.create(tmp_path / "ws", None)
    ws.create_file("pkg/models.py", FILES["shop_reports/models.py"])
    ws.create_file("pkg/repository.py", FILES["shop_reports/repository.py"])
    text = interfaces(ws, "pkg")
    assert "class Sale:\n    product: str\n    amount: float" in text
    assert "def __init__(self, path: Path=DEFAULT_DB)" in text and "def all(self) -> list[Sale]" in text


def test_the_chat_hands_a_program_to_the_code_agent():
    assert CODE_HINT.search("Напиши по базе shop.db сервис отчётов о продажах")
    assert CODE_HINT.search("Разработай небольшое приложение для учёта задач")
    assert not CODE_HINT.search("Напиши функцию бинарного поиска")


def test_the_report_gets_the_project_and_its_readme(settings, fake_embedder, shop: Path):
    settings.corpus.include_ext = [".md"]
    llm = project_llm(settings)
    eng = Engine(settings, embedder=fake_embedder, llm=llm)
    eng.index_folder(shop)
    res = eng.code_task("Напиши по базе shop.db сервис отчётов о продажах", sandbox=HostSandbox())
    assert res.status == "done" and res.project["root"] == "shop_reports"
    eng._code_report("Напиши сервис", res.task, res, network=False)
    body = llm.calls[-1][-1]["content"]
    assert "Проект: папка shop_reports/" in body and "README проекта:\n# Отчёты по продажам" in body
    assert "Запуск: python -m shop_reports report" in body
    applied = eng.apply_code(res.task_id).applied
    assert "shop_reports/__main__.py" in applied and not any(a.endswith(".db") for a in applied)
    assert (shop / "shop_reports" / "service.py").exists()
    eng._thread.join(30)
    eng.close()
