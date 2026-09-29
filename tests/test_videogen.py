"""Videos in the chat: an LTX-2 folder in models/video, text to video and a picture to video through sd-cli
(stubbed), frames of 8k + 1, the API, deletion with the dialog."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_agent.api import create_app
from rag_agent.videogen import frames_for, wants_video
from tests.test_imagegen import gen_engine  # noqa: F401 - the fixture: a stub sd-cli and a picture model

LTX = ("ltxv23_uncensored_v1.4-Q4_K_M.gguf", "ltxv23_uncensored_v1.4_video_vae.safetensors",
       "ltxv23_uncensored_v1.4_audio_vae.safetensors", "ltxv23_uncensored_v1.4_projections.safetensors",
       "gemma-3-12b-it-ablit-norms-biproj-Q4_K_M.gguf")


@pytest.fixture
def vid_engine(gen_engine):  # noqa: F811
    folder = gen_engine.settings.models_dir / "video" / "ltx-2.3"
    folder.mkdir(parents=True)
    for name in LTX:
        (folder / name).write_bytes(b"x")
    gen_engine.video_folder = folder
    return gen_engine


def opt(args, flag):
    return args[args.index(flag) + 1]


def test_words_and_frames():
    assert wants_video("Сними видео, как машина едет по горам") and wants_video("анимируй её")
    assert wants_video("Оживи картинку") and not wants_video("Нарисуй машину")
    assert frames_for(0, 24, 33) == 33  # the settings' default when no duration is asked
    assert frames_for(3, 24, 33) == 73 and (73 - 1) % 8 == 0
    assert frames_for(100, 24, 33) == 241  # capped


def test_the_ltx_folder_is_found_with_its_audio_and_connectors(vid_engine):
    v = vid_engine.videogen
    [m] = v.models()
    assert v.available and m["name"] == LTX[0] and m["audio"] and m["complete"] and m["family"] == "ltx2"
    prof = v.profile(LTX[0])
    assert Path(prof["audio_vae"]).name == LTX[2] and Path(prof["connectors"]).name == LTX[3]
    assert Path(prof["llm"]).name.startswith("gemma") and prof["cfg_scale"] == 6.0 and prof["frames"] == 33


def test_text_to_video(vid_engine):
    vid_engine.llm.video_plan = {"action": "text", "prompt": "a red car on a mountain road, engine roar",
                                 "seconds": 3, "aspect": "landscape", "minor_sexual": False}
    ans = vid_engine.ask("Сними видео: мерседес в горах входит в поворот", session="v1")
    assert ans.route == "video" and ans.video["mode"] == "text" and ans.video["url"].endswith(".webm")
    args = vid_engine.fake_sd.calls[-1]
    assert opt(args, "-M") == "vid_gen" and opt(args, "--video-frames") == "73" and opt(args, "--fps") == "24"
    assert Path(opt(args, "--audio-vae")).name == LTX[2] and Path(opt(args, "--embeddings-connectors")).name == LTX[3]
    assert (int(opt(args, "-W")), int(opt(args, "-H"))) == (1024, 576) and "-i" not in args
    assert "--temporal-tiling" in args and opt(args, "-p") == "a red car on a mountain road, engine roar"
    assert vid_engine.videogen.path("v1", ans.video["id"]).read_bytes() == b"webm"
    assert [s.name for s in ans.trace][-2:] == ["video_plan", "video_gen"]


def test_a_picture_becomes_the_first_frame(vid_engine):
    pic = vid_engine.ask("Нарисуй машину в горах", session="v2").image  # 1216 x 832 (landscape)
    vid_engine.llm.video_plan = {"action": "image", "prompt": "the car drives into a turn", "seconds": 0,
                                 "aspect": "landscape", "minor_sexual": False}
    ans = vid_engine.ask("Анимируй, как она едет и входит в поворот", session="v2")
    assert ans.video["mode"] == "image" and ans.video["source"] == f"generated:{pic['id']}"
    args = vid_engine.fake_sd.calls[-1]
    assert "-i" in args and opt(args, "--video-frames") == "33"
    w, h = int(opt(args, "-W")), int(opt(args, "-H"))
    assert abs(w / h - 1216 / 832) < 0.05  # the frame keeps the picture's proportions


def test_not_a_video_and_the_refusal(vid_engine):
    ans = vid_engine.ask("Что такое видеокарта?", session="v3")  # the plan says "none"
    assert ans.route != "video" and not ans.video
    vid_engine.llm.video_plan = {**vid_engine.llm.video_plan, "action": "text", "minor_sexual": True}
    ans = vid_engine.ask("сними видео …", session="v3")
    assert ans.route == "video" and not ans.answerable and not ans.video


def test_api_serves_lists_and_deletes_videos(vid_engine):
    vid_engine.llm.video_plan = {"action": "text", "prompt": "waves", "seconds": 0, "aspect": "square",
                                 "minor_sexual": False}
    with TestClient(create_app(vid_engine)) as client:
        d = client.post("/dialogs", json={}).json()
        a = client.post("/ask", json={"question": "Сними видео с волнами", "dialog_id": d["id"], "session": d["id"]}).json()
        r = client.get(a["video"]["url"])
        assert r.status_code == 200 and r.headers["content-type"] == "video/webm"
        info = client.get("/videogen").json()
        assert info["available"] and info["models"][0]["audio"]
        s = client.get("/settings/video").json()
        assert s["recommended"]["fps"] == 24 and s["info"]["audio_vae"] == LTX[2]
        assert "media-src 'self'" in client.get("/").headers["content-security-policy"]
        client.delete(f"/dialogs/{d['id']}")
        assert client.get(a["video"]["url"]).status_code == 404
