"""SQLite (WAL) + FTS5 через стандартный sqlite3. Одно соединение, защищённое блокировкой."""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
import threading
from html import escape
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL DEFAULT '',
    duration_s INTEGER NOT NULL DEFAULT 0,
    language TEXT NOT NULL DEFAULT '',
    thumbnail_url TEXT NOT NULL DEFAULT '',
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    source TEXT NOT NULL,              -- manual_subs | auto_subs | whisper
    source_detail TEXT NOT NULL DEFAULT '',
    segments_json TEXT NOT NULL,       -- [[start_s, text], …]
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    transcript_id INTEGER REFERENCES transcripts(id) ON DELETE SET NULL,
    version INTEGER NOT NULL,
    detail TEXT NOT NULL DEFAULT 'full',
    language TEXT NOT NULL DEFAULT 'ru',
    model TEXT NOT NULL DEFAULT '',
    prompt_hash TEXT NOT NULL DEFAULT '',
    markdown TEXT NOT NULL,
    is_edited INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(video_id, version)
);
CREATE TABLE IF NOT EXISTS exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    summary_id INTEGER NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
    format TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    video_id TEXT,
    kind TEXT NOT NULL DEFAULT 'full',     -- full | resummarize | export
    params_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued', -- queued | running | done | failed | cancelled
    stage TEXT,                            -- info | subtitles | audio | transcribing | summarizing | exporting
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT,
    waiting INTEGER NOT NULL DEFAULT 0,    -- 1: ждёт Ollama / модель
    summary_id INTEGER,
    export_path TEXT,
    partials_json TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, id);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
    video_id UNINDEXED, title, channel, summary_text, transcript_text,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""

ACTIVE = ("queued", "running")


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _row(r: sqlite3.Row | None) -> dict | None:
    return dict(r) if r is not None else None


def fts_query(q: str) -> str:
    """Безопасный запрос FTS5: каждое слово в кавычках с поиском по префиксу."""
    words = re.findall(r"\w+", q, flags=re.U)
    return " ".join(f'"{w}"*' for w in words)


def _mark(snippet: str) -> str:
    return escape(snippet).replace("\x02", "<mark>").replace("\x03", "</mark>")


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def q(self, sql: str, args: tuple | list = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql: str, args: tuple | list = ()) -> dict | None:
        with self._lock:
            return _row(self.conn.execute(sql, args).fetchone())

    def x(self, sql: str, args: tuple | list = ()) -> int:
        with self._lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid or 0

    # ───────── settings ─────────

    def get_settings(self) -> dict[str, Any]:
        return {r["key"]: json.loads(r["value_json"]) for r in self.q("SELECT * FROM settings")}

    def set_setting(self, key: str, value: Any) -> None:
        if value is None:
            self.x("DELETE FROM settings WHERE key=?", (key,))
        else:
            self.x("INSERT INTO settings(key, value_json) VALUES(?, ?) "
                   "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                   (key, json.dumps(value, ensure_ascii=False)))

    # ───────── videos / transcripts ─────────

    def upsert_video(self, meta: dict) -> None:
        self.x("""INSERT INTO videos(id, title, channel, duration_s, language, thumbnail_url, added_at)
                  VALUES(:id, :title, :channel, :duration_s, :language, :thumbnail_url, :added_at)
                  ON CONFLICT(id) DO UPDATE SET title=excluded.title, channel=excluded.channel,
                  duration_s=excluded.duration_s, language=excluded.language,
                  thumbnail_url=excluded.thumbnail_url""", {**meta, "added_at": now()})
        self.reindex(meta["id"])

    def get_video(self, vid: str) -> dict | None:
        return self.one("SELECT * FROM videos WHERE id=?", (vid,))

    def delete_video(self, vid: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM search WHERE video_id=?", (vid,))
            self.conn.execute("DELETE FROM jobs WHERE video_id=? AND status NOT IN ('queued','running')", (vid,))
            self.conn.execute("DELETE FROM videos WHERE id=?", (vid,))
            self.conn.commit()

    def add_transcript(self, vid: str, source: str, detail: str, segments: list) -> dict:
        tid = self.x("INSERT INTO transcripts(video_id, source, source_detail, segments_json, created_at) "
                     "VALUES(?,?,?,?,?)",
                     (vid, source, detail, json.dumps([[round(s, 2), t] for s, t in segments],
                                                      ensure_ascii=False), now()))
        self.reindex(vid)
        return self.get_transcript(tid)

    def get_transcript(self, tid: int) -> dict | None:
        r = self.one("SELECT * FROM transcripts WHERE id=?", (tid,))
        if r:
            r["segments"] = [(float(s), t) for s, t in json.loads(r["segments_json"])]
        return r

    def latest_transcript(self, vid: str, whisper_only: bool = False) -> dict | None:
        sql = "SELECT id FROM transcripts WHERE video_id=?"
        if whisper_only:
            sql += " AND source='whisper'"
        r = self.one(sql + " ORDER BY id DESC LIMIT 1", (vid,))
        return self.get_transcript(r["id"]) if r else None

    # ───────── summaries ─────────

    def add_summary(self, vid: str, transcript_id: int | None, markdown: str, *, detail: str, language: str,
                    model: str, prompt_hash: str = "", is_edited: bool = False) -> dict:
        with self._lock:
            v = self.conn.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM summaries WHERE video_id=?",
                                  (vid,)).fetchone()[0]
            cur = self.conn.execute(
                "INSERT INTO summaries(video_id, transcript_id, version, detail, language, model, prompt_hash, "
                "markdown, is_edited, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (vid, transcript_id, v, detail, language, model, prompt_hash, markdown, int(is_edited), now()))
            self.conn.commit()
            sid = cur.lastrowid
        self.reindex(vid)
        return self.get_summary(sid)

    def get_summary(self, sid: int) -> dict | None:
        return self.one("SELECT * FROM summaries WHERE id=?", (sid,))

    def summaries_for(self, vid: str) -> list[dict]:
        return self.q("SELECT id, version, detail, language, model, is_edited, created_at FROM summaries "
                      "WHERE video_id=? ORDER BY version DESC", (vid,))

    def latest_summary(self, vid: str) -> dict | None:
        return self.one("SELECT * FROM summaries WHERE video_id=? ORDER BY version DESC LIMIT 1", (vid,))

    # ───────── exports ─────────

    def add_export(self, sid: int, fmt: str, path: str) -> dict:
        eid = self.x("INSERT INTO exports(summary_id, format, path, created_at) VALUES(?,?,?,?)",
                     (sid, fmt, path, now()))
        return self.one("SELECT * FROM exports WHERE id=?", (eid,))

    def get_export(self, eid: int) -> dict | None:
        return self.one("SELECT * FROM exports WHERE id=?", (eid,))

    def exports_for_video(self, vid: str) -> list[dict]:
        return self.q("SELECT e.*, s.version FROM exports e JOIN summaries s ON s.id=e.summary_id "
                      "WHERE s.video_id=? ORDER BY e.id DESC", (vid,))

    # ───────── поиск ─────────

    def reindex(self, vid: str) -> None:
        v = self.get_video(vid)
        if not v:
            return
        s = self.latest_summary(vid)
        t = self.one("SELECT segments_json FROM transcripts WHERE video_id=? ORDER BY id DESC LIMIT 1", (vid,))
        ttext = " ".join(x[1] for x in json.loads(t["segments_json"])) if t else ""
        with self._lock:
            self.conn.execute("DELETE FROM search WHERE video_id=?", (vid,))
            self.conn.execute("INSERT INTO search(video_id, title, channel, summary_text, transcript_text) "
                              "VALUES(?,?,?,?,?)",
                              (vid, v["title"], v["channel"], s["markdown"] if s else "", ttext))
            self.conn.commit()

    def library(self, q: str = "", channel: str = "", sort: str = "new", period: str = "") -> list[dict]:
        where, args = ["EXISTS (SELECT 1 FROM summaries s WHERE s.video_id=v.id)"], []
        if channel:
            where.append("v.channel=?")
            args.append(channel)
        days = {"week": 7, "month": 31, "year": 366}.get(period)
        if days:
            where.append("v.added_at >= ?")
            args.append((dt.datetime.now() - dt.timedelta(days=days)).isoformat(timespec="seconds"))
        order = {"new": "v.added_at DESC", "old": "v.added_at ASC", "title": "v.title COLLATE NOCASE"}.get(
            sort, "v.added_at DESC")
        fq = fts_query(q) if q else ""
        if q and not fq:
            return []
        if fq:
            sql = (f"SELECT v.*, snippet(search, -1, char(2), char(3), '…', 14) AS snip, bm25(search) AS rank "
                   f"FROM search JOIN videos v ON v.id=search.video_id "
                   f"WHERE search MATCH ? AND {' AND '.join(where)} ORDER BY rank, {order}")
            rows = self.q(sql, [fq, *args])
        else:
            rows = self.q(f"SELECT v.* FROM videos v WHERE {' AND '.join(where)} ORDER BY {order}", args)
        for r in rows:
            s = self.latest_summary(r["id"])
            r["summary_markdown"] = s["markdown"] if s else ""
            r["versions"] = self.one("SELECT COUNT(*) AS n FROM summaries WHERE video_id=?", (r["id"],))["n"]
            if r.get("snip"):
                r["snippet_html"] = _mark(r["snip"])
        return rows

    def channels(self) -> list[str]:
        return [r["channel"] for r in self.q(
            "SELECT DISTINCT v.channel FROM videos v JOIN summaries s ON s.video_id=v.id "
            "WHERE v.channel != '' ORDER BY v.channel COLLATE NOCASE")]

    # ───────── jobs ─────────

    def create_job(self, url: str, video_id: str | None, kind: str, params: dict) -> dict:
        jid = self.x("INSERT INTO jobs(url, video_id, kind, params_json, created_at, message) VALUES(?,?,?,?,?,?)",
                     (url, video_id, kind, json.dumps(params, ensure_ascii=False), now(), "в очереди"))
        return self.get_job(jid)

    def get_job(self, jid: int) -> dict | None:
        r = self.one("SELECT j.*, v.title, v.thumbnail_url, v.channel FROM jobs j "
                     "LEFT JOIN videos v ON v.id=j.video_id WHERE j.id=?", (jid,))
        if r:
            r["params"] = json.loads(r["params_json"] or "{}")
        return r

    def list_jobs(self, limit: int = 50) -> list[dict]:
        rows = self.q("SELECT j.*, v.title, v.thumbnail_url, v.channel FROM jobs j "
                      "LEFT JOIN videos v ON v.id=j.video_id ORDER BY j.id DESC LIMIT ?", (limit,))
        for r in rows:
            r["params"] = json.loads(r["params_json"] or "{}")
        return rows

    def update_job(self, jid: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.x(f"UPDATE jobs SET {cols} WHERE id=?", [*fields.values(), jid])

    def next_queued(self) -> dict | None:
        r = self.one("SELECT id FROM jobs WHERE status='queued' ORDER BY waiting ASC, id ASC LIMIT 1")
        return self.get_job(r["id"]) if r else None

    def active_job_for_video(self, vid: str) -> dict | None:
        return self.one("SELECT * FROM jobs WHERE video_id=? AND status IN ('queued','running') LIMIT 1", (vid,))

    def mark_interrupted(self) -> int:
        with self._lock:
            cur = self.conn.execute(
                "UPDATE jobs SET status='failed', error='Прервано перезапуском приложения.', "
                "message='прервано перезапуском', finished_at=? WHERE status='running'", (now(),))
            self.conn.commit()
            return cur.rowcount

    def delete_finished_jobs(self) -> None:
        self.x("DELETE FROM jobs WHERE status IN ('done','cancelled')")
