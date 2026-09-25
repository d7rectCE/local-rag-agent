"""Every module must at least compile — covers files the other tests never import (ui.py, cli.py)."""

import pytest

from rag_agent.config import REPO_ROOT

SOURCES = sorted((REPO_ROOT / "src").rglob("*.py")) + sorted((REPO_ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_compiles(path):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
