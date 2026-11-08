from __future__ import annotations

import asyncio
import time

import pytest

from app.db import Database
from app.events import EventBus
from app.jobs import Deps, Worker
from tests.fakes import FakeTranscriber, FakeYDLFactory, long_segments, make_info

FINAL = ("done", "failed", "cancelled")


def setup(tmp_path, llm_url, ydl=None, transcriber=None):
    db = Database(tmp_path / "t.sqlite3")
    db.set_setting("llm.base_url", llm_url)
    db.set_setting("output_dir", str(tmp_path / "out"))
    db.set_setting("format", "md")
    deps = Deps(ydl_factory=ydl, retry_pause=0, wait_pause=0.2, stream_interval=0,
                transcriber_factory=(lambda w: (transcriber, "fake-whisper large-v3-turbo", "")) if transcriber
                else None)
    return db, EventBus(), deps


async def wait_job(db, jid, until=FINAL, timeout=20.0, cond=None):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        j = db.get_job(jid)
        if (cond and cond(j)) or (not cond and j["status"] in until):
            return j
        await asyncio.sleep(0.02)
    raise AssertionError(f"timeout, job={db.get_job(jid)}")


def run(coro):
    return asyncio.run(coro)


def test_full_job_with_subtitles(tmp_path, llm_server):
    ydl = FakeYDLFactory().add(make_info("abcdefghijk"), segments=long_segments(3))
    db, bus, deps = setup(tmp_path, llm_server.url, ydl)

    async def main():
        w = Worker(db, bus, tmp_path, deps)
        w.start()
        events = []
        async with bus.subscribe() as q:
            job = db.create_job("https://youtu.be/abcdefghijk", "abcdefghijk", "full",
                                {"format": "md", "out_dir": str(tmp_path / "out")})
            w.wake()
            j = await wait_job(db, job["id"])
            while not q.empty():
                events.append(q.get_nowait())
        await w.stop()
        return j, events

    j, events = run(main())
    assert j["status"] == "done", j
    stages = [e["stage"] for e in events]
    assert {"info", "subtitles", "summarizing", "exporting"} <= set(stages)
    assert any(e.get("text") for e in events)  # текст модели шёл потоком
    assert (tmp_path / "out" / "Тестовое видео.md").exists()
    s = db.latest_summary("abcdefghijk")
    assert s["version"] == 1 and "[00:00](https://youtu.be/abcdefghijk?t=0)" in s["markdown"]
    assert db.latest_transcript("abcdefghijk")["source"] == "manual_subs"
    assert llm_server.unloads >= 1  # очередь опустела → keep_alive: 0
    assert db.library("слово5")  # FTS по расшифровке


def test_full_job_with_whisper_and_resummarize(tmp_path, llm_server):
    ydl = FakeYDLFactory().add(make_info("abcdefghijk", subs=None))
    tr = FakeTranscriber(long_segments(2))
    db, bus, deps = setup(tmp_path, llm_server.url, ydl, tr)

    async def main():
        w = Worker(db, bus, tmp_path, deps)
        w.start()
        j1 = db.create_job("abcdefghijk", "abcdefghijk", "full", {"format": ""})
        w.wake()
        r1 = await wait_job(db, j1["id"])
        j2 = db.create_job("abcdefghijk", "abcdefghijk", "resummarize", {"detail": "short", "format": ""})
        w.wake()
        r2 = await wait_job(db, j2["id"])
        await w.stop()
        return r1, r2

    r1, r2 = run(main())
    assert r1["status"] == "done" and r2["status"] == "done", (r1, r2)
    assert tr.calls == 1 and tr.unloaded == 1          # перегенерация не запускает Whisper
    assert ydl.count("audio") == 1
    assert not (tmp_path / "cache" / "abcdefghijk").exists()  # аудио удалено
    versions = db.summaries_for("abcdefghijk")
    assert [v["version"] for v in versions] == [2, 1] and versions[0]["detail"] == "short"
    assert "распознавание Whisper" in db.latest_summary("abcdefghijk")["markdown"]


def test_cancel_during_whisper(tmp_path, llm_server):
    ydl = FakeYDLFactory().add(make_info("abcdefghijk", subs=None))
    tr = FakeTranscriber(long_segments(2), chunks=50, delay=0.05)
    db, bus, deps = setup(tmp_path, llm_server.url, ydl, tr)

    async def main():
        w = Worker(db, bus, tmp_path, deps)
        w.start()
        job = db.create_job("abcdefghijk", "abcdefghijk", "full", {})
        w.wake()
        await wait_job(db, job["id"], cond=lambda j: j["stage"] == "transcribing" and j["progress"] > 0)
        assert w.cancel(job["id"])
        j = await wait_job(db, job["id"])
        await w.stop()
        return j

    j = run(main())
    assert j["status"] == "cancelled" and llm_server.chat_count == 0
    assert tr.unloaded == 1


def test_cancel_during_llm_stream(tmp_path, llm_server):
    ydl = FakeYDLFactory().add(make_info("abcdefghijk"), segments=long_segments(3))
    db, bus, deps = setup(tmp_path, llm_server.url, ydl)
    llm_server.delay = 0.1

    async def main():
        w = Worker(db, bus, tmp_path, deps)
        w.start()
        job = db.create_job("abcdefghijk", "abcdefghijk", "full", {})
        w.wake()
        await wait_job(db, job["id"], cond=lambda j: j["stage"] == "summarizing" and llm_server.chat_count > 0)
        await asyncio.sleep(0.2)
        w.cancel(job["id"])
        j = await wait_job(db, job["id"], timeout=5)
        await w.stop()
        return j

    j = run(main())
    assert j["status"] == "cancelled"
    assert db.latest_transcript("abcdefghijk") is not None  # расшифровка сохранена


def test_retry_continues_from_part_5(tmp_path, llm_server):
    ydl = FakeYDLFactory().add(make_info("abcdefghijk", duration=3600), segments=long_segments(60))
    db, bus, deps = setup(tmp_path, llm_server.url, ydl)
    db.set_setting("llm.chunk_chars", 20000)
    llm_server.fail_on = {5}

    async def main():
        w = Worker(db, bus, tmp_path, deps)
        w.start()
        job = db.create_job("abcdefghijk", "abcdefghijk", "full", {})
        w.wake()
        j1 = await wait_job(db, job["id"])
        before = llm_server.chat_count
        info_calls = ydl.count("info")
        w.retry(job["id"])
        j2 = await wait_job(db, job["id"], cond=lambda j: j["status"] in FINAL and j["started_at"] and
                            llm_server.chat_count > before)
        j2 = await wait_job(db, job["id"])
        await w.stop()
        return j1, j2, before, info_calls

    j1, j2, before, info_calls = run(main())
    assert j1["status"] == "failed" and "500" in j1["error"]
    assert j2["status"] == "done", j2
    chunk_reqs = [r for r in llm_server.requests[before:] if "фрагмент" in r["messages"][1]["content"]]
    assert [r["messages"][1]["content"].split("фрагмент ")[1][:6] for r in chunk_reqs][0] == "5 из 8"
    assert ydl.count("info") == info_calls  # метаданные и расшифровка не перезапрашивались
    assert j2["partials_json"] is None


def test_llm_unavailable_waits_then_runs(tmp_path, llm_server):
    ydl = FakeYDLFactory().add(make_info("abcdefghijk"), segments=long_segments(3))
    db, bus, deps = setup(tmp_path, "http://127.0.0.1:9", ydl)

    async def main():
        w = Worker(db, bus, tmp_path, deps)
        w.start()
        job = db.create_job("abcdefghijk", "abcdefghijk", "full", {})
        w.wake()
        j1 = await wait_job(db, job["id"], cond=lambda j: j["waiting"] == 1)
        db.set_setting("llm.base_url", llm_server.url)  # «запустили Ollama»
        w.retry_waiting()
        j2 = await wait_job(db, job["id"])
        await w.stop()
        return j1, j2

    j1, j2 = run(main())
    assert j1["status"] == "queued" and "Ollama" in j1["message"]
    assert j2["status"] == "done"
    assert ydl.count("info") == 1


def test_short_transcript_fails_without_llm(tmp_path, llm_server):
    ydl = FakeYDLFactory().add(make_info("abcdefghijk"), segments=[(0, "музыка")])
    db, bus, deps = setup(tmp_path, llm_server.url, ydl)

    async def main():
        w = Worker(db, bus, tmp_path, deps)
        w.start()
        job = db.create_job("abcdefghijk", "abcdefghijk", "full", {})
        w.wake()
        j = await wait_job(db, job["id"])
        await w.stop()
        return j

    j = run(main())
    assert j["status"] == "failed" and "почти нет речи" in j["error"]
    assert llm_server.chat_count == 0


def test_export_failure_keeps_summary(tmp_path, llm_server):
    ydl = FakeYDLFactory().add(make_info("abcdefghijk"), segments=long_segments(3))
    db, bus, deps = setup(tmp_path, llm_server.url, ydl)
    blocker = tmp_path / "blocker"
    blocker.write_text("x")

    async def main():
        w = Worker(db, bus, tmp_path, deps)
        w.start()
        job = db.create_job("abcdefghijk", "abcdefghijk", "full", {"format": "md", "out_dir": str(blocker / "x")})
        w.wake()
        j = await wait_job(db, job["id"])
        await w.stop()
        return j

    j = run(main())
    assert j["status"] == "done" and "файл не сохранён" in j["message"]
    assert db.latest_summary("abcdefghijk")


def test_restart_marks_running_failed(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    j = db.create_job("u", None, "full", {})
    db.update_job(j["id"], status="running")
    db.close()
    db2 = Database(tmp_path / "t.sqlite3")
    assert db2.mark_interrupted() == 1
    assert db2.get_job(j["id"])["status"] == "failed"
    assert "перезапуском" in db2.get_job(j["id"])["error"]


def test_db_versions_search_delete(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    db.upsert_video({"id": "v1", "title": "Фотосинтез", "channel": "Био", "duration_s": 60, "language": "ru",
                     "thumbnail_url": ""})
    t = db.add_transcript("v1", "manual_subs", "ru", [(0.0, "Хлорофилл поглощает свет")])
    db.add_summary("v1", t["id"], "# Фотосинтез\n## Кратко\nО растениях", detail="full", language="ru", model="m")
    s2 = db.add_summary("v1", t["id"], "# Фотосинтез\n## Кратко\nПравка", detail="full", language="ru",
                        model="m", is_edited=True)
    assert s2["version"] == 2 and s2["is_edited"] == 1
    res = db.library("хлорофил")
    assert res and "<mark>" in res[0]["snippet_html"]
    assert db.library("", channel="Био") and not db.library("", channel="Нет")
    assert db.library('"; DROP TABLE videos; --') == []
    db.delete_video("v1")
    assert db.get_video("v1") is None and db.library("хлорофилл") == []
    assert db.q("SELECT COUNT(*) AS n FROM summaries")[0]["n"] == 0


@pytest.mark.parametrize("val", [True, "x", 3, {"a": [1]}])
def test_settings_roundtrip(tmp_path, val):
    db = Database(tmp_path / "t.sqlite3")
    db.set_setting("k", val)
    assert db.get_settings()["k"] == val
