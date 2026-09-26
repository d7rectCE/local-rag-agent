"""Risky code in the sandbox (ТЗ ч.2 S20, Э17, NFR8, NFR11), in the spirit of RedCode: dangerous
snippets run straight in the Docker sandbox, without a model — isolation, not the model's
judgement, has to stop them. A temporary copy of the demo corpus is mounted read-only at
/corpus, so the attempt to delete it is real and the original is never at risk.

    python scripts/redteam_code.py --out reports/e17
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_agent.code.sandbox import DockerSandbox  # noqa: E402
from rag_agent.config import load_settings  # noqa: E402
from rag_agent.evaluation.redteam import RISKY_CODE, risky_code  # noqa: E402

WHAT = {
    "delete-corpus": "удалить папку корпуса (`shutil.rmtree('/corpus')`)",
    "write-system": "записать файл в системный каталог `/etc`",
    "network-http": "HTTP-запрос наружу",
    "network-dns": "разрешить имя через DNS",
    "host-env": "прочитать переменные окружения хоста (канарейка RAG_REDTEAM_CANARY)",
    "docker-socket": "найти сокет Docker (выход из контейнера)",
    "privilege": "стать root (`setuid(0)`)",
    "fork-bomb": "форк-бомба",
    "memory-bomb": "занять 8 ГБ памяти",
    "disk-fill": "записать 2 ГБ во временный каталог",
    "infinite-loop": "бесконечный цикл",
    "read-host-home": "найти диски и домашние каталоги хоста",
}
HOW = {
    "delete-corpus": "корпус смонтирован только для чтения",
    "write-system": "корневая система только для чтения",
    "network-http": "`--network none`",
    "network-dns": "`--network none`",
    "host-env": "переменные хоста не передаются",
    "docker-socket": "сокет не монтируется",
    "privilege": "uid 1000, `--cap-drop ALL`, `no-new-privileges`",
    "fork-bomb": "`--pids-limit`",
    "memory-bomb": "`--memory`",
    "disk-fill": "`/tmp` в памяти ограниченного размера",
    "infinite-loop": "таймаут запуска, контейнер снимается",
    "read-host-home": "смонтирована только рабочая копия",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(ROOT / "demo_corpus"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "e17"))
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()
    settings = load_settings()
    box = DockerSandbox(settings.code)
    if not box.available():
        sys.exit(f"образ песочницы {settings.code.image} не найден: docker build -f docker/sandbox.Dockerfile ...")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rag-redcode-corpus-") as tmp:
        corpus = Path(tmp) / "corpus"
        shutil.copytree(args.corpus, corpus, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        before = sorted(str(p.relative_to(corpus)) for p in corpus.rglob("*"))
        results = risky_code(box, corpus, timeout_s=args.timeout)
        after = sorted(str(p.relative_to(corpus)) for p in corpus.rglob("*"))
    intact = before == after
    (out / "risky_code.json").write_text(json.dumps({
        "date": datetime.now().astimezone().isoformat(timespec="seconds"), "image": settings.code.image,
        "limits": {"memory": settings.code.memory, "cpus": settings.code.cpus, "pids": settings.code.pids,
                   "timeout_s": args.timeout},
        "corpus_intact": intact, "results": [r.model_dump() for r in results]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    lines = ["# Опасный код в песочнице", "",
             f"Образ `{settings.code.image}`, лимиты: память {settings.code.memory}, CPU {settings.code.cpus}, "
             f"процессов {settings.code.pids}, таймаут {args.timeout:.0f} с. Корпус — временная копия демо-корпуса, "
             f"смонтированная только для чтения; после прогона {'не изменилась' if intact else '**ИЗМЕНИЛАСЬ**'}.", "",
             f"Заблокировано: **{sum(r.blocked for r in results)} из {len(results)}**.", "",
             "| Сниппет | Попытка | Чем остановлено | Заблокировано | Код выхода |", "|---|---|---|---|---|"]
    lines += [f"| {r.id} | {WHAT.get(r.id, '')} | {HOW.get(r.id, '')} | {'да' if r.blocked else '**нет**'} "
              f"| {r.exit_code}{' (таймаут)' if r.timed_out else ''} |" for r in results]
    (out / "risky_code.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"blocked {sum(r.blocked for r in results)}/{len(results)}, corpus intact: {intact}; saved to {out}")
    assert {name for name, *_ in RISKY_CODE} == set(WHAT)


if __name__ == "__main__":
    main()
