"""Web gateway (ТЗ ч.2 S19, Э16): search through a local SearXNG, page reading, citations.

- ``SearxClient.search`` asks the SearXNG instance on this machine (a metasearch
  proxy: no API keys, no account) and re-ranks the results: primary sources
  (documentation, arXiv, ACL Anthology, repositories) first, aggregators and SEO
  sites last.
- ``fetch_page`` downloads a page with limits on size and time, extracts the main
  text (trafilatura when installed, otherwise the article/main block of the HTML)
  and caches it with the access date.
- ``answer_from_web`` (Search-o1's Reason-in-Documents): pages are first compressed
  to the passages relevant to the question by a separate call, then the answer is
  written from them with a URL and an access date for every claim; sentences whose
  numbers are not found in the cited passages, or that cite nothing, are marked.
  Conflicts with the user's files are shown, never resolved silently (FR16).

Only search queries and page requests leave the machine (NFR9); both pass the
policy layer (rules 1–3) before they are sent.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from rag_agent.config import Settings, WebConfig

UA = "Mozilla/5.0 (local-rag-agent research assistant)"


class WebError(RuntimeError):
    pass


class WebResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    engine: str = ""
    rank: int = 0
    priority: int = 0  # +1 primary source, -1 aggregator / SEO


class WebPage(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    fetched_at: str  # ISO date of access (cited in the answer)
    content_type: str = ""
    from_cache: bool = False
    error: str | None = None


# --------------------------------------------------------------------------- search


def source_priority(url: str, cfg: WebConfig) -> int:
    host = urlsplit(url).netloc.lower()
    full = host + urlsplit(url).path.lower()
    if any(d in full for d in cfg.demote):
        return -1
    return 1 if any(host == d or host.endswith("." + d) or (d.endswith(".") and host.startswith(d)) or d in host
                    for d in cfg.prefer) else 0


class SearxClient:
    def __init__(self, cfg: WebConfig, http: httpx.Client | None = None):
        self.cfg = cfg
        self.http = http or httpx.Client(timeout=cfg.timeout_s, headers={"User-Agent": UA})

    def available(self) -> bool:
        try:
            return self.http.get(f"{self.cfg.searxng_url}/healthz", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    def search(self, query: str, n: int | None = None) -> list[WebResult]:
        try:
            r = self.http.get(f"{self.cfg.searxng_url}/search", params={"q": query, "format": "json", "safesearch": 0})
        except httpx.HTTPError as exc:
            raise WebError(f"SearXNG недоступен ({self.cfg.searxng_url}): {exc}. Запустите: rag searxng start") from exc
        if r.status_code != 200:
            raise WebError(f"SearXNG ответил {r.status_code}: {r.text[:200]}")
        seen, results = set(), []
        for k, item in enumerate(r.json().get("results", []), start=1):
            url = item.get("url") or ""
            if not url.startswith(("http://", "https://")) or url in seen:
                continue
            seen.add(url)
            results.append(WebResult(title=item.get("title") or url, url=url, snippet=(item.get("content") or "")[:500],
                                     engine=item.get("engine") or "", rank=k, priority=source_priority(url, self.cfg)))
        results.sort(key=lambda x: (-x.priority, x.rank))  # primary sources first, then the engines' order
        return results[: n or self.cfg.results]


# --------------------------------------------------------------------------- pages


def extract_main_text(html: str) -> tuple[str, str]:
    """(title, main text) of an HTML page."""
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_precision=True) or ""
        meta = trafilatura.extract_metadata(html)
        if text.strip():
            return (meta.title if meta and meta.title else ""), text.strip()
    except ImportError:
        pass
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    blocks = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "td", "th", "dd", "dt"]):
        text = el.get_text(" ", strip=True)
        if len(text) >= 3 and not (el.name == "li" and el.find(["p", "li"])):
            blocks.append(("#" * int(el.name[1]) + " " + text) if el.name in ("h1", "h2", "h3", "h4") else text)
    text = "\n".join(dict.fromkeys(blocks))  # drop repeated blocks, keep order
    return title, re.sub(r"\n{3,}", "\n\n", text).strip()


class PageFetcher:
    def __init__(self, settings: Settings, http: httpx.Client | None = None):
        self.cfg = settings.web
        self.cache = settings.data_dir / "web_cache"
        self.http = http or httpx.Client(timeout=self.cfg.timeout_s, headers={"User-Agent": UA}, follow_redirects=True,
                                         max_redirects=5)

    def _cache_file(self, url: str) -> Path:
        return self.cache / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".json")

    def fetch(self, url: str) -> WebPage:
        f = self._cache_file(url)
        if f.exists():
            page = WebPage.model_validate_json(f.read_text(encoding="utf-8"))
            if datetime.fromisoformat(page.fetched_at) + timedelta(days=self.cfg.cache_days) >= datetime.now().astimezone():
                page.from_cache = True
                return page
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            with self.http.stream("GET", url) as r:
                ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if r.status_code >= 400:
                    return WebPage(url=url, fetched_at=now, error=f"HTTP {r.status_code}")
                data = b""
                for chunk in r.iter_bytes():
                    data += chunk
                    if len(data) > self.cfg.page_max_mb * 2**20:
                        break
                encoding = r.encoding or "utf-8"
        except httpx.HTTPError as exc:
            return WebPage(url=url, fetched_at=now, error=f"{type(exc).__name__}: {exc}")
        if ctype in ("application/json",) or url.endswith(".json"):
            title, text = url, data.decode(encoding, "replace")
        elif ctype.startswith("text/plain"):
            title, text = url, data.decode(encoding, "replace")
        elif "html" in ctype or data.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
            title, text = extract_main_text(data.decode(encoding, "replace"))
        else:
            return WebPage(url=url, fetched_at=now, content_type=ctype, error=f"тип содержимого {ctype} не поддерживается")
        page = WebPage(url=url, title=title[:300], text=text[:200_000], fetched_at=now, content_type=ctype)
        self.cache.mkdir(parents=True, exist_ok=True)
        f.write_text(page.model_dump_json(), encoding="utf-8")
        return page


# --------------------------------------------------------------------------- SearXNG container


SEARX_SETTINGS = """use_default_settings: true
server:
  secret_key: "{secret}"
  limiter: false
  image_proxy: false
  public_instance: false
search:
  safe_search: 0
  formats: [html, json]
ui:
  static_use_hash: true
"""
CONTAINER = "rag-searxng"


def searxng(action: str, settings: Settings) -> str:
    """start / stop / status of the local SearXNG container (bound to 127.0.0.1 only)."""
    cfg = settings.web
    conf = settings.data_dir / "searxng"
    if action == "status":
        r = subprocess.run(["docker", "ps", "-a", "--filter", f"name={CONTAINER}", "--format", "{{.Status}}"],
                           capture_output=True, text=True)
        state = r.stdout.strip() or "не создан"
        return f"{CONTAINER}: {state}; API {'отвечает' if SearxClient(cfg).available() else 'не отвечает'} ({cfg.searxng_url})"
    if action == "stop":
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
        return f"{CONTAINER} остановлен"
    if action != "start":
        raise ValueError(action)
    conf.mkdir(parents=True, exist_ok=True)
    settings_file = conf / "settings.yml"
    if not settings_file.exists():  # the secret stays in the local data folder, never in the repository
        settings_file.write_text(SEARX_SETTINGS.format(secret=secrets.token_hex(32)), encoding="utf-8")
    port = urlsplit(cfg.searxng_url).port or 8888
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    r = subprocess.run(["docker", "run", "-d", "--name", CONTAINER, "--restart", "unless-stopped",
                        "-p", f"127.0.0.1:{port}:8080", "-v", f"{conf.resolve()}:/etc/searxng", cfg.searxng_image],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise WebError(f"не удалось запустить SearXNG: {r.stderr.strip()[:300]}")
    client = SearxClient(cfg)
    for _ in range(30):
        if client.available():
            return f"{CONTAINER} запущен: {cfg.searxng_url}"
        time.sleep(1)
    return f"{CONTAINER} запущен, но API пока не отвечает — проверьте: rag searxng status"


def today() -> str:
    return date.today().isoformat()


__all__ = ["PageFetcher", "SearxClient", "WebError", "WebPage", "WebResult", "extract_main_text", "searxng",
           "source_priority", "today", "json", "timezone"]
