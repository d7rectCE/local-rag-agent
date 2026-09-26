"""Э16: web gateway — SearXNG results, page extraction and cache, rules 1 and 3, citations, support check."""

import json
from pathlib import Path

import httpx

from rag_agent.engine import Engine
from rag_agent.policy import Policy, Provenance
from rag_agent.web import PageFetcher, SearxClient, extract_main_text, source_priority
from rag_agent.web_qa import answer_from_web, mark_unsupported
from tests.conftest import FakeLLM

RELEASES = "<html><head><title>Releases</title></head><body><nav>Menu Sign in</nav><article><h1>Releases</h1>" \
           "<p>scikit-learn 1.9.1 — bug-fix release published on 2026-09-10, with fixes for HistGradientBoosting.</p>" \
           "<p>scikit-learn 1.9.0 — new features in the 1.9 series for estimators and metrics.</p>" \
           + "<p>Older notes about previous releases and their many changes over the years.</p>" * 5 + \
           "</article><script>track()</script><footer>© site</footer></body></html>"
SEO = "<html><body><article>" + "".join(f"<p>Tip {i}: machine learning advice for beginners, part {i}.</p>"
                                          for i in range(10)) + "</article></body></html>"


def transport(pages: dict[str, str], requests: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requests.append(url)
        if request.url.path == "/search":
            results = [{"url": "https://medium.com/tips", "title": "Tips", "content": "tips", "engine": "brave"},
                       {"url": "https://github.com/scikit-learn/scikit-learn/releases", "title": "Releases",
                        "content": "1.9.1", "engine": "brave"},
                       {"url": "https://github.com/scikit-learn/scikit-learn/releases", "title": "dup", "engine": "google"}]
            return httpx.Response(200, json={"results": results})
        if request.url.path == "/healthz":
            return httpx.Response(200, text="OK")
        body = pages.get(url)
        return httpx.Response(200, text=body, headers={"content-type": "text/html"}) if body else httpx.Response(404)
    return httpx.MockTransport(handler)


def web_parts(settings, requests):
    pages = {"https://github.com/scikit-learn/scikit-learn/releases": RELEASES, "https://medium.com/tips": SEO}
    http = httpx.Client(transport=transport(pages, requests), follow_redirects=True)
    return SearxClient(settings.web, http), PageFetcher(settings, http)


def test_ranking_and_extraction(settings):
    assert source_priority("https://arxiv.org/abs/1", settings.web) == 1
    assert source_priority("https://medium.com/x", settings.web) == -1
    assert source_priority("https://example.org/", settings.web) == 0
    client, _ = web_parts(settings, [])
    results = client.search("scikit-learn latest version")
    assert [r.url for r in results] == ["https://github.com/scikit-learn/scikit-learn/releases", "https://medium.com/tips"]
    title, text = extract_main_text(RELEASES)
    assert title == "Releases" and "1.9.1" in text and "Sign in" not in text and "track()" not in text


def test_pages_are_cached_with_the_access_date(settings):
    requests = []
    _, fetcher = web_parts(settings, requests)
    first = fetcher.fetch("https://github.com/scikit-learn/scikit-learn/releases")
    second = fetcher.fetch("https://github.com/scikit-learn/scikit-learn/releases")
    assert not first.from_cache and second.from_cache and len(requests) == 1 and first.fetched_at
    assert first.error is None and "1.9.1" in first.text
    assert fetcher.fetch("https://example.org/missing").error == "HTTP 404"


def test_support_check_marks_unsupported_sentences():
    text, ok, bad = mark_unsupported(
        "Последняя версия — scikit-learn 1.9.1 [1]. Она вышла 12 сентября 2026 года [1]. Также вышла версия 2.0 без ссылки.",
        {1: "scikit-learn 1.9.1 released on 2026-09-10."})
    assert ok == 1 and bad == 2
    assert "12" in text.split("⚠")[1] or "числа не найдены" in text
    assert "нет ссылки" in text


def test_web_answer_cites_urls_and_dates(settings):
    requests = []
    client, fetcher = web_parts(settings, requests)
    reply = {"answerable": True, "answer": "Последняя версия — scikit-learn 1.9.1 [1].", "general": ""}
    llm = FakeLLM(settings, reply=reply)
    policy = Policy(web=True)
    ans = answer_from_web("Какая последняя версия scikit-learn?", llm, settings, policy=policy, prov=Provenance(),
                          client=client, fetcher=fetcher)
    assert ans.route == "web" and ans.answerable and "⚠" not in ans.answer
    src = ans.sources[0]
    assert src.file_type == "web" and src.file_path == "https://github.com/scikit-learn/scikit-learn/releases"
    assert "обращение" in src.location and "опубликовано 2026-09-10" in src.location
    read = next(s.detail for s in ans.trace if s.name == "web_read")
    assert [p.get("passages") for p in read["pages"]] == [1, 0]  # the SEO page had nothing relevant
    assert llm.kinds == ["web_query", "web_compress", "web_compress", "answer"]


def test_rule1_private_names_stop_the_web_search(settings):
    requests = []
    client, fetcher = web_parts(settings, requests)
    llm = FakeLLM(settings, web_query="compute_f1 returns nan")
    policy = Policy({"compute_f1"}, web=True)
    ans = answer_from_web("Почему compute_f1 возвращает nan?", llm, settings, policy=policy, prov=Provenance(),
                          client=client, fetcher=fetcher)
    [pending] = ans.pending
    assert pending["tool"] == "web_search" and pending["rule"] == "rule 1" and requests == []  # nothing was sent
    ans = answer_from_web("Почему compute_f1 возвращает nan?", llm, settings, policy=policy, prov=Provenance(),
                          client=client, fetcher=fetcher, confirmed={pending["key"]})
    assert not ans.pending and any("/search" in r for r in requests)


def test_engine_web_modes(settings, fake_embedder, corpus: Path):
    requests = []
    client, fetcher = web_parts(settings, requests)
    route = {"route": "general", "standalone_question": "Какая последняя версия scikit-learn?", "reasoning": "none",
             "aggregate": False, "web": True}
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings, route=route))
    eng._web_client, eng._page_fetcher = client, fetcher  # no real network in tests
    eng.index_folder(corpus)
    assert eng.ask("Какая последняя версия scikit-learn?").route == "general"  # web is off by default
    assert eng.ask("Какая последняя версия scikit-learn?", web="auto").route == "web"
    eng.llm.route = {**route, "web": False}
    assert eng.ask("Что такое F1?", web="auto").route == "general"  # the router did not ask for the web
    assert eng.ask("Что такое F1?", web="always").route == "web"
    eng.close()
