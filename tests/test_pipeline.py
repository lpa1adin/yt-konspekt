from __future__ import annotations

import re

import pytest

from app.pipeline import Cancelled, LLMUnavailable, PipelineError
from app.pipeline.export import (
    check_writable,
    export_markdown,
    file_base,
    normalize_md,
    safe_filename,
    unique_path,
)
from app.pipeline.llm import LLMClient, clean_llm_output
from app.pipeline.source import (
    YouTubeSource,
    friendly_error,
    parse_json3,
    parse_vtt,
    pick_track,
    video_id,
)
from app.pipeline.summarize import (
    SummaryCache,
    insert_section,
    repair_timecodes,
    build_document,
    extract_brief,
    linkify_timecodes,
    load_prompts,
    render,
    summarize,
)
from app.pipeline.transcript import build_transcript, fmt_ts, split_chunks, ts_to_seconds
from tests.fakes import FIXTURES, FakeLLMServer, FakeYDLFactory, long_segments, make_info

# ───────────── источник ─────────────


@pytest.mark.parametrize("url", [
    "https://youtu.be/dQw4w9WgXcQ?si=abc",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s",
    "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
    "https://youtube.com/shorts/dQw4w9WgXcQ",
    "https://www.youtube.com/live/dQw4w9WgXcQ?feature=share",
    "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
    "dQw4w9WgXcQ",
])
def test_video_id(url):
    assert video_id(url) == "dQw4w9WgXcQ"


def test_video_id_bad():
    assert video_id("https://example.com/watch") is None


def test_parse_json3():
    segs = parse_json3((FIXTURES / "sample.json3").read_text(encoding="utf-8"))
    assert segs == [(0.0, "Привет, это тестовое видео"), (12.5, "про фотосинтез"),
                    (31.0, "Хлорофилл поглощает свет")]


def test_parse_vtt_removes_tags_and_dupes():
    segs = parse_vtt((FIXTURES / "sample.vtt").read_text(encoding="utf-8"))
    assert [t for _, t in segs] == ["привет это", "тестовое видео", "про фотосинтез"]
    assert segs[-1][0] == pytest.approx(62.25)


def test_pick_track_priority():
    t = [{"ext": "json3", "url": "u"}]
    info = {"language": "de", "subtitles": {"en": t, "de-DE": t, "live_chat": t},
            "automatic_captions": {"de-orig": t}}
    assert pick_track(info, ["ru", "en"]).key == "de-DE"           # ручные на языке видео
    info["subtitles"] = {"en": t, "live_chat": t}
    assert pick_track(info, ["ru", "en"]).key == "en"              # затем ru/en
    info["subtitles"] = {"live_chat": t}
    tr = pick_track(info, ["ru", "en"])
    assert (tr.key, tr.kind) == ("de-orig", "auto_subs")           # затем -orig
    info["automatic_captions"] = {"fr": t, "ru": t}
    assert pick_track(info, ["ru", "en"]).key == "ru"              # затем авто ru/en
    assert pick_track({"subtitles": {"live_chat": t}}, ["ru"]) is None


def test_fetch_subtitles_and_errors():
    f = FakeYDLFactory().add(make_info("abcdefghijk"))
    src = YouTubeSource(ydl_factory=f, retry_pause=0)
    info = src.extract_info("https://www.youtube.com/watch?v=abcdefghijk")
    segs, track = src.fetch_subtitles(info, ["ru", "en"])
    assert track.kind == "manual_subs" and len(segs) == 3

    f.add(make_info("liveliveliv", live=True))
    with pytest.raises(PipelineError, match="прямой эфир"):
        src.extract_info("https://www.youtube.com/watch?v=liveliveliv")


def test_network_retries_then_ok():
    f = FakeYDLFactory().add(make_info("abcdefghijk"))
    f.info_errors = [OSError("Read timed out"), OSError("Read timed out")]
    src = YouTubeSource(ydl_factory=f, retry_pause=0)
    assert src.extract_info("https://www.youtube.com/watch?v=abcdefghijk")["title"]
    assert f.count("info") == 3


def test_friendly_errors():
    assert "не бот" in friendly_error(Exception("Sign in to confirm you're not a bot")).message
    assert "cookies" in friendly_error(Exception("HTTP Error 429: Too Many Requests")).hint.lower()
    assert "возрастным" in friendly_error(Exception("Sign in to confirm your age")).message


# ───────────── расшифровка ─────────────

def test_timecodes():
    assert fmt_ts(65) == "01:05" and fmt_ts(3725) == "1:02:05"
    assert ts_to_seconds("1:02:05") == 3725 and ts_to_seconds("12:34") == 754


def test_build_transcript_blocks():
    segs = [(0, "а"), (10, "б"), (29, "в"), (31, "г"), (3700, "д")]
    assert build_transcript(segs) == "[00:00] а б в\n[00:31] г\n[1:01:40] д"


def test_split_chunks():
    text = "\n".join(f"[00:{i:02d}] " + "x" * 50 for i in range(20))
    chunks = split_chunks(text, 200)
    assert all(len(c) <= 200 for c in chunks)
    assert "\n".join(chunks) == text


# ───────────── LLM ─────────────

def test_clean_llm_output():
    assert clean_llm_output("<think>hmm\n</think>\n```markdown\n## A\n- b\n```") == "## A\n- b"
    assert clean_llm_output("## Plain") == "## Plain"


def test_llm_stream_and_think_retry():
    with FakeLLMServer(reject_think=True) as s:
        llm = LLMClient("ollama", "qwen3:8b", s.url)
        llm.check()
        seen = []
        res = llm.chat("sys", "Расшифровка [00:12] текст", on_text=seen.append)
        assert res.text.startswith("## Кратко") and "<think>" not in res.text
        assert len(seen) > 1 and not res.truncated
        assert "think" not in s.requests[-1]
        assert s.requests[-1]["options"]["num_ctx"] == 16384


def test_llm_length_and_openai_backend():
    with FakeLLMServer() as s:
        s.length_on = {1}
        assert LLMClient("ollama", "qwen3:8b", s.url).chat("s", "u").truncated
        res = LLMClient("openai", "qwen3:8b", s.url + "/v1").chat("s", "u [01:00]")
        assert "## Кратко" in res.text


def test_llm_unavailable_and_missing_model():
    with pytest.raises(LLMUnavailable, match="Нет связи"):
        LLMClient("ollama", "qwen3:8b", "http://127.0.0.1:9").check()
    with FakeLLMServer(models=["llama3:8b"]) as s:
        with pytest.raises(LLMUnavailable, match="нет модели"):
            LLMClient("ollama", "qwen3:8b", s.url).check()
        with pytest.raises(LLMUnavailable):
            LLMClient("ollama", "qwen3:8b", s.url).chat("s", "u")


def test_llm_cancel():
    with FakeLLMServer() as s:
        s.delay = 0.05
        flag = {"n": 0}

        def cancel():
            flag["n"] += 1
            return flag["n"] > 3
        with pytest.raises(Cancelled):
            LLMClient("ollama", "qwen3:8b", s.url).chat("s", "u", cancel=cancel)


# ───────────── конспект ─────────────

def test_linkify_keeps_existing_links():
    md = "### [01:05] Тема\n- см. [1:00:00]\n- [00:05](https://youtu.be/x?t=5)"
    out = linkify_timecodes(md, "VID")
    assert "[01:05](https://youtu.be/VID?t=65)" in out
    assert "[1:00:00](https://youtu.be/VID?t=3600)" in out
    assert out.count("https://youtu.be/x?t=5") == 1 and "VID?t=5)" not in out


def test_render_leaves_other_braces():
    assert render("{title} {json} {i}/{n}", title="T", i=1, n=2) == "T {json} 1/2"


def test_summarize_single_pass(llm_server):
    llm = LLMClient("ollama", "qwen3:8b", llm_server.url)
    transcript = build_transcript(long_segments(2))
    res = summarize(llm, transcript, "Т", "ru", "full", 18000, load_prompts())
    assert res.parts == 1 and "## Конспект" in res.markdown
    assert "русский" in llm_server.requests[0]["messages"][0]["content"]
    assert "## Ключевые термины" in llm_server.requests[0]["messages"][1]["content"]


def test_summarize_short_transcript_stops(llm_server):
    llm = LLMClient("ollama", "qwen3:8b", llm_server.url)
    with pytest.raises(PipelineError, match="почти нет речи"):
        summarize(llm, "[00:00] ля-ля", "Т", "ru", "full", 18000, load_prompts())
    assert llm_server.chat_count == 0


def test_summarize_map_reduce_and_resume(llm_server):
    llm = LLMClient("ollama", "qwen3:8b", llm_server.url)
    transcript = build_transcript(long_segments(60))
    n = len(split_chunks(transcript, 20000))
    assert n == 8
    saved = {}
    llm_server.fail_on = {5}  # падаем на части 5 из 8
    cache = SummaryCache("k")
    with pytest.raises(PipelineError):
        summarize(llm, transcript, "Т", "ru", "short", 20000, load_prompts(), cache=cache,
                  save_cache=lambda c: saved.update(raw=c.to_json()))
    assert len(SummaryCache.from_json(saved["raw"], "k").items) == 4
    before = llm_server.chat_count
    progress = []
    res = summarize(llm, transcript, "Т", "ru", "short", 20000, load_prompts(),
                    cache=SummaryCache.from_json(saved["raw"], "k"),
                    on_progress=lambda f, m: progress.append(m))
    chunk_requests = [r for r in llm_server.requests[before:] if "фрагмент" in r["messages"][1]["content"]]
    assert len(chunk_requests) == 4                       # продолжили с пятой части
    assert "часть 5 из 8" in progress and res.parts == 8
    assert "## Главное" in llm_server.requests[-1]["messages"][1]["content"]
    assert SummaryCache.from_json(saved["raw"], "other").items == {}


def test_build_document_and_brief():
    meta = {"id": "VID", "title": "Лекция", "channel": "Канал", "duration_s": 3725}
    doc = build_document("## Кратко\nО **свете** и [00:05](https://youtu.be/x).\n\n## Конспект", meta,
                         "ручные субтитры (ru)", "qwen3:8b")
    assert doc.startswith("# Лекция\n\n**Видео:** [Лекция](https://www.youtube.com/watch?v=VID)")
    assert "**Длительность:** 1:02:05" in doc and "\n\n---\n\n## Кратко" in doc
    assert extract_brief(doc) == "О свете и 00:05."


# ───────────── экспорт ─────────────

def test_normalize_md():
    assert normalize_md("текст\n## Заголовок\nабзац\n- пункт\n- пункт2") == \
        "текст\n\n## Заголовок\nабзац\n\n- пункт\n- пункт2\n"


def test_filenames(tmp_path):
    assert safe_filename('Что: такое / "ИИ"?  ') == "Что такое ИИ"
    (tmp_path / "a.md").write_text("x")
    assert unique_path(tmp_path / "a.md").name == "a (2).md"
    meta = {"id": "X", "title": "A/B", "channel": "Ch"}
    assert file_base("{channel}/{title}", meta) == "Ch/A B"


def test_export_md_and_unwritable(tmp_path):
    p1 = export_markdown("# T\n## A\n- b", tmp_path, "Имя", "md")
    p2 = export_markdown("# T", tmp_path, "Имя", "md")
    assert p1.name == "Имя.md" and p2.name == "Имя (2).md"
    assert p1.read_text(encoding="utf-8") == "# T\n\n## A\n\n- b\n"
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(PipelineError, match="Конспект остался в библиотеке"):
        export_markdown("# T", blocker / "sub", "a", "md")
    assert not check_writable(blocker / "sub")[0] and check_writable(tmp_path)[0]


def test_export_docx_builtin(tmp_path, monkeypatch):
    pytest.importorskip("docx")
    monkeypatch.setattr("app.pipeline.export.shutil.which", lambda _: None)
    p = export_markdown("# T\n## A\n- **b** [00:05](https://youtu.be/x?t=5)", tmp_path, "d", "docx")
    assert p.suffix == ".docx" and p.stat().st_size > 1000
    assert not (tmp_path / "d.md").exists()


def test_export_docx_pandoc(tmp_path):
    import shutil
    if not shutil.which("pandoc"):
        pytest.skip("pandoc не установлен")
    p = export_markdown("# T\n## A\n- b", tmp_path, "d", "docx", keep_markdown=True)
    assert p.suffix == ".docx" and (tmp_path / "d.md").exists()


# ───────────── длинные видео: части не ужимаются, таймкоды настоящие ─────────────

def test_repair_timecodes():
    valid = list(range(3600, 5400, 30))           # фрагмент 1:00:00–1:29:30
    assert repair_timecodes("### [05:30] x", valid, 3600, 5430) == "### [1:05:30] x"   # потерянный час
    assert repair_timecodes("[1:10:00] и [70:00]", valid, 3600, 5430) == "[1:10:00] и [1:10:00]"
    assert repair_timecodes("[00:20]", valid, 3600, 5430) == "[1:00:30]"    # выдумка → ближайший блок
    assert repair_timecodes("[3:00:00]", valid, 3600, 5430) == "[1:29:30]"  # за пределами → край фрагмента
    assert repair_timecodes("[00:05](https://youtu.be/x?t=5)", valid, 3600, 5430).endswith("?t=5)")


def test_insert_section():
    out = insert_section("## Кратко\nО курсе.\n\n## Конспект\n### [00:00] выдумка\n\n## Выводы\n- да",
                         "Конспект", "### [1:00:00] Часть")
    assert out == "## Кратко\nО курсе.\n\n## Конспект\n\n### [1:00:00] Часть\n\n## Выводы\n- да"


def test_long_video_keeps_every_part_with_real_timecodes(llm_server):
    llm_server.drop_hours = True
    llm = LLMClient("ollama", "qwen3:8b", llm_server.url)
    transcript = build_transcript(long_segments(150))           # лекция на 2,5 часа
    res = summarize(llm, transcript, "Курс", "ru", "full", 20000, load_prompts())
    md = res.markdown
    assert res.parts >= 7
    for i in range(1, res.parts + 1):                            # каждая часть попала в документ
        assert f"Часть {i}" in md
    assert "выдумка" not in md and "Начало" not in md            # раздел «Конспект» модели заменён частями
    assert md.index("## Кратко") < md.index("## Конспект") < md.index("## Ключевые термины")
    heads = [ts_to_seconds(t) for t in re.findall(r"^### \[([\d:]+)\]", md, flags=re.M)]
    assert len(heads) == res.parts and heads == sorted(heads)    # таймкоды идут по порядку видео
    assert heads[-1] > 2 * 3600                                  # часы восстановлены
    final_req = llm_server.requests[-1]["messages"][1]["content"]
    assert "## Ключевые термины" in final_req and "## Конспект" not in final_req
