from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.jobs import Deps  # noqa: E402
from app.main import create_app, templates  # noqa: E402
from tests.fakes import FakeYDLFactory, long_segments, make_info  # noqa: E402


@pytest.fixture
def client(tmp_path, llm_server):
    ydl = FakeYDLFactory().add(make_info("abcdefghijk"), segments=long_segments(3))
    ydl.add(make_info("bbbbbbbbbbb", title="Второе видео"), segments=long_segments(2))
    app = create_app(tmp_path / "data", Deps(ydl_factory=ydl, retry_pause=0, wait_pause=0.2, stream_interval=0))
    db = app.state.s.db
    db.set_setting("llm.base_url", llm_server.url)
    db.set_setting("output_dir", str(tmp_path / "out"))
    with TestClient(app) as c:
        c.db = db
        c.tmp = tmp_path
        yield c


def wait_done(c, jid, timeout=20):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        jobs = {j["job_id"]: j for j in c.get("/api/jobs").json()["jobs"]}
        if jobs[jid]["status"] in ("done", "failed", "cancelled"):
            return jobs[jid]
        time.sleep(0.05)
    raise AssertionError(jobs[jid])


def test_jobs_flow(client):
    r = client.post("/api/jobs", json={"urls": ["https://youtu.be/abcdefghijk", "not a url"],
                                       "format": "md", "detail": "short"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["created"]) == 1 and data["errors"][0]["url"] == "not a url"
    jid = data["created"][0]["job_id"]
    j = wait_done(client, jid)
    assert j["status"] == "done" and j["export_path"].endswith("Тестовое видео.md")

    # повтор того же видео — предупреждение, а не дубль
    r = client.post("/api/jobs", json={"urls": ["https://www.youtube.com/watch?v=abcdefghijk"]})
    assert r.json()["skipped"][0]["reason"] == "уже в библиотеке" and not r.json()["created"]

    s = client.db.latest_summary("abcdefghijk")
    r = client.put(f"/api/summaries/{s['id']}", json={"markdown": "# Правка\n## Кратко\nНовый текст"})
    assert r.json()["version"] == 2
    r = client.post(f"/api/summaries/{r.json()['id']}/export", json={"format": "md",
                                                                       "out_dir": str(client.tmp / "exp")})
    assert r.json()["ok"] and (client.tmp / "exp" / "Тестовое видео.md").exists()
    r = client.post(f"/api/summaries/{s['id']}/export", json={"format": "md", "out_dir": "/dev/null/x"})
    assert r.status_code == 400 and "библиотеке" in r.json()["hint"]

    r = client.post("/api/videos/abcdefghijk/summaries", json={"detail": "full"})
    assert r.json()["ok"]
    j2 = wait_done(client, r.json()["job"]["job_id"])
    assert j2["status"] == "done"
    assert [v["version"] for v in client.db.summaries_for("abcdefghijk")] == [3, 2, 1]

    assert client.delete("/api/videos/abcdefghijk").json()["ok"]
    assert client.db.get_video("abcdefghijk") is None


def test_unwritable_folder_rejected_immediately(client, tmp_path):
    blocker = tmp_path / "f"
    blocker.write_text("x")
    r = client.post("/api/jobs", json={"urls": ["bbbbbbbbbbb"], "format": "md", "out_dir": str(blocker / "d")})
    assert r.status_code == 400 and "Нет доступа" in r.json()["errors"][0]["error"]


def test_cancel_and_retry(client, llm_server):
    llm_server.hold.clear()  # модель «думает» бесконечно
    r = client.post("/api/jobs", json={"urls": ["bbbbbbbbbbb"], "format": ""})
    jid = r.json()["created"][0]["job_id"]
    t0 = time.monotonic()
    while client.db.get_job(jid)["stage"] != "summarizing" and time.monotonic() - t0 < 10:
        time.sleep(0.05)
    assert client.post(f"/api/jobs/{jid}/cancel").json()["ok"]
    llm_server.hold.set()
    assert wait_done(client, jid)["status"] == "cancelled"
    assert client.post(f"/api/jobs/{jid}/retry").json()["ok"]
    assert wait_done(client, jid)["status"] == "done"


def test_events_stream(client):
    """SSE: события задачи приходят в поток (читаем шину напрямую — TestClient плохо дружит с бесконечным SSE)."""
    import asyncio

    bus = client.app.state.s.bus
    got = []

    async def collect():
        async with bus.subscribe() as q:
            st = client.app.state.s
            st.db.create_job("bbbbbbbbbbb", "bbbbbbbbbbb", "full", {"format": ""})
            st.worker.wake()
            while True:
                ev = await asyncio.wait_for(q.get(), 10)
                got.append(ev)
                if ev.get("status") == "done":
                    return

    client.portal.call(collect)
    assert {"job_id", "status", "stage", "progress", "message"} <= set(got[-1])
    assert json.dumps(got[-1], ensure_ascii=False)


def test_status_and_models(client, llm_server):
    st = client.get("/api/status").json()
    keys = {c["key"]: c for c in st["checks"]}
    assert keys["llm_server"]["ok"] and keys["llm_model"]["ok"]
    assert "yt_dlp" in keys and "ffmpeg" in keys
    assert client.get("/api/models").json()["models"] == ["qwen3:8b"]
    assert client.post("/api/models/pull", json={"name": "llama3.2:3b"}).json()["ok"]
    t0 = time.monotonic()
    while client.get("/api/status").json()["pulls"].get("llama3.2:3b", {}).get("status") != "done":
        assert time.monotonic() - t0 < 5
        time.sleep(0.05)
    assert llm_server.pulled == ["llama3.2:3b"]


def test_check_folder(client, tmp_path):
    assert client.post("/api/check-folder", json={"path": str(tmp_path / "new")}).json()["ok"]
    assert not client.post("/api/check-folder", json={"path": ""}).json()["ok"]


def test_settings_save(client):
    r = client.post("/settings", data={"llm_model": "llama3:8b", "llm_chunk_chars": "12000", "format": "md",
                                       "auto_export": "on", "prompt_system": "Свой промпт {language}"},
                    follow_redirects=False)
    assert r.status_code == 303
    s = client.db.get_settings()
    assert s["llm.model"] == "llama3:8b" and s["llm.chunk_chars"] == 12000 and s["auto_export"] is True
    assert s["keep_audio"] is False and s["prompt.system"] == "Свой промпт {language}"
    assert client.post("/settings/prompts/system/reset").json()["ok"]
    assert "prompt.system" not in client.db.get_settings()


@pytest.mark.skipif(templates is None, reason="Jinja2 не установлен")
def test_html_pages(client):
    r = client.post("/api/jobs", json={"urls": ["abcdefghijk"], "format": ""})
    wait_done(client, r.json()["created"][0]["job_id"])
    for url in ["/", "/library", "/library?q=слово5", "/videos/abcdefghijk", "/settings", "/status"]:
        resp = client.get(url)
        assert resp.status_code == 200, (url, resp.text[:500])
    page = client.get("/videos/abcdefghijk").text
    assert "/static/js/player.js" in page and 'id="player"' in page and 'href="https://youtu.be/abcdefghijk?t=0"' in page
    assert "<mark>" in client.get("/library?q=слово5", headers={"HX-Request": "true",
                                                                "HX-Target": "library-grid"}).text
    r = client.post("/api/jobs", data={"urls": "https://youtu.be/abcdefghijk"}, headers={"HX-Request": "true"})
    assert "уже есть в библиотеке" in r.text
