"""Chains of steps in one message: the parser, the run (a picture of a step is the source of the next video),
the stop on a failed step, the history of a chain."""

from rag_agent.chains import describe, parse_chain
from tests.test_imagegen import gen_engine  # noqa: F401 - the stub sd-cli and a picture model
from tests.test_videogen import vid_engine  # noqa: F401 - plus an LTX-2 folder in models/video

EXAMPLE = ("[IMG] сгенерируй мне машину мерседес в горах -> [VID] анимируй, как она едет и входит в поворот со "
           "звуками -> [TEXT] расскажи, что сделал")


def test_the_parser():
    steps = parse_chain(EXAMPLE)
    assert [(s.kind, s.text) for s in steps] == [
        ("img", "сгенерируй мне машину мерседес в горах"),
        ("vid", "анимируй, как она едет и входит в поворот со звуками"), ("text", "расскажи, что сделал")]
    assert describe(steps) == "картинка → видео → текст"
    ru = parse_chain("[КАРТИНКА] кот → [ВИДЕО] оживи → [ТЕКСТ]")
    assert [s.kind for s in ru] == ["img", "vid", "text"] and ru[-1].text == "Расскажи, что сделано."
    code = parse_chain("[CODE] напиши функцию f -> int по данным data.csv -> [TEXT] объясни")
    assert [s.kind for s in code] == ["code", "text"] and code[0].text.endswith("f -> int по данным data.csv")
    assert parse_chain("Что такое a -> b в Haskell?") is None  # no tags: a plain message
    assert [s.kind for s in parse_chain("[IMG] рыжий кот")] == ["img"]  # one tag forces the route


def test_a_picture_goes_on_to_the_video_and_the_text_knows_both(vid_engine):  # noqa: F811
    llm = vid_engine.llm
    llm.video_plan = {"action": "none", "prompt": "the car drives into a turn, engine sound", "seconds": 2,
                      "aspect": "landscape", "minor_sexual": False}  # a tag forces the step whatever the plan says
    llm.route = {"route": "general", "standalone_question": "Расскажи, что сделал"}
    ans = vid_engine.ask(EXAMPLE, session="c1")
    assert ans.route == "chain" and [p["kind"] for p in ans.chain] == ["img", "vid", "text"]
    pic, clip = ans.chain[0]["image"], ans.chain[1]["video"]
    assert clip["mode"] == "image" and clip["source"] == f"generated:{pic['id']}" and "-i" in vid_engine.fake_sd.calls[-1]
    assert ans.chain[2]["answer"] == "Общий ответ." and ans.image["id"] == pic["id"] and ans.video["id"] == clip["id"]
    general_call = llm.calls[llm.kinds.index("general")]
    history = "\n".join(m["content"] for m in general_call)
    assert "Шаг 1 [img]" in history and "Шаг 2 [vid]" in history  # the text step knows what was made
    assert [s.name for s in ans.trace].count("chain_step") == 3


def test_a_failed_step_stops_the_chain(gen_engine):  # noqa: F811 - no video model here
    gen_engine.llm.route = {"route": "general", "standalone_question": "Расскажи"}
    ans = gen_engine.ask("[IMG] кот -> [VID] оживи -> [TEXT] расскажи", session="c2")
    kinds = [p["kind"] for p in ans.chain]
    assert kinds == ["img", "vid", "stop"] and not ans.answerable
    assert "не настроен" in ans.chain[1]["answer"] and "остановлена на шаге 2" in ans.chain[2]["answer"]
    assert "general" not in gen_engine.llm.kinds  # the text step never ran


def test_a_chain_is_one_turn_of_the_history(vid_engine, tmp_path):  # noqa: F811
    from rag_agent.dialogs import DialogStore

    vid_engine.llm.route = {"route": "general", "standalone_question": "Расскажи"}
    ans = vid_engine.ask("[IMG] кот -> [TEXT] расскажи", session="c3")
    store = DialogStore(tmp_path / "d.sqlite")
    d = store.create()
    store.add_turn(d["id"], "assistant", "answer", ans.answer, payload=ans.model_dump())
    note = store.history(d["id"])[-1]["content"]
    assert note.startswith("Шаг 1 [img] «кот»: [Генератор картинок нарисовал") and "Шаг 2 [text]" in note


def test_the_canvas_graph_runs_every_node_after_its_inputs():
    import pytest

    from rag_agent.chains import GraphError, graph_steps

    nodes = [{"id": "t", "kind": "text", "text": "расскажи", "inputs": ["v", "i2"]},
             {"id": "v", "kind": "vid", "text": "оживи", "inputs": ["i1"], "options": {"seconds": 3, "junk": 1}},
             {"id": "i1", "kind": "img", "text": "кот", "options": {"size": "16:9"}},
             {"id": "i2", "kind": "img", "text": "пёс"}]
    order = graph_steps(nodes)
    ids = [s.id for s in order]
    assert ids.index("i1") < ids.index("v") < ids.index("t") and ids.index("i2") < ids.index("t")
    assert next(s for s in order if s.id == "v").options == {"seconds": 3}  # only the known options
    for bad, msg in (([], "пуст"), ([{"id": "a", "kind": "img", "inputs": ["b"]}], "несуществующим"),
                     ([{"id": "a", "kind": "img", "inputs": ["b"]}, {"id": "b", "kind": "vid", "inputs": ["a"]}], "цикл"),
                     ([{"id": "a", "kind": "audio"}], "неизвестный вид")):
        with pytest.raises(GraphError, match=msg):
            graph_steps(bad)


def test_a_canvas_run_takes_the_picture_along_its_links(vid_engine):  # noqa: F811
    vid_engine.llm.video_plan = {"action": "text", "prompt": "cat jumps", "seconds": 1, "aspect": "landscape",
                                 "minor_sexual": False}
    graph = [{"id": "a", "kind": "img", "text": "кот", "options": {"size": "1:1"}},
             {"id": "b", "kind": "img", "text": "пёс"},
             {"id": "c", "kind": "vid", "text": "оживи кота", "inputs": ["a"], "options": {"seconds": 2}}]
    ans = vid_engine.ask("Холст: картинка, картинка → видео", session="g1", graph=graph)
    parts = {p["id"]: p for p in ans.chain}
    assert (parts["a"]["image"]["width"], parts["a"]["image"]["height"]) == (2048, 2048)  # the node's size
    assert parts["c"]["video"]["source"] == f"generated:{parts['a']['image']['id']}"  # its link, not the last picture
    assert parts["c"]["video"]["frames"] == 49  # 2 s at 24 fps from the node, not the plan's 1 s
