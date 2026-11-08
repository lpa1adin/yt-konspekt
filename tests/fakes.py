"""Фейки для тестов: yt-dlp, Whisper и HTTP-сервер LLM (Ollama + OpenAI-совместимый)."""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def long_segments(minutes: int, words_per_seg: int = 25) -> list[tuple[float, str]]:
    """Длинная «лекция»: сегмент каждые 5 секунд."""
    segs = []
    for i in range(minutes * 12):
        words = " ".join(f"слово{(i * 7 + j) % 97}" for j in range(words_per_seg))
        segs.append((i * 5.0, f"фраза {i} {words}"))
    return segs


def json3_from_segments(segs: list[tuple[float, str]]) -> str:
    return json.dumps({"events": [{"tStartMs": int(t * 1000), "segs": [{"utf8": s}]} for t, s in segs]},
                      ensure_ascii=False)


def make_info(vid: str = "dQw4w9WgXcQ", *, title: str = "Тестовое видео", subs: str | None = "manual",
              language: str = "ru", duration: int = 95, live: bool = False) -> dict:
    sub_url = f"https://subs.example/{vid}.json3"
    info = {
        "id": vid, "title": title, "channel": "Канал тестов", "duration": duration,
        "language": language, "thumbnail": f"https://i.ytimg.com/vi/{vid}/hq.jpg",
        "subtitles": {}, "automatic_captions": {},
    }
    track = [{"ext": "json3", "url": sub_url}, {"ext": "vtt", "url": sub_url.replace("json3", "vtt")}]
    if subs == "manual":
        info["subtitles"] = {language: track, "live_chat": [{"ext": "json", "url": "x"}]}
    elif subs == "auto":
        info["automatic_captions"] = {f"{language}-orig": track, "en": [{"ext": "json3", "url": "bad"}]}
    if live:
        info["live_status"] = "is_live"
    return info


class FakeYDL:
    """Подмена yt_dlp.YoutubeDL. Создаётся фабрикой FakeYDLFactory."""

    def __init__(self, factory: FakeYDLFactory, opts: dict):
        self.f = factory
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        self.f.calls.append(("info", url))
        if self.f.info_errors:
            raise self.f.info_errors.pop(0)
        vid = url.split("v=")[-1]
        return self.f.infos[vid]

    def urlopen(self, url):
        self.f.calls.append(("subs", url))
        data = self.f.sub_payloads.get(url)
        if data is None:
            raise OSError("HTTP Error 404")

        class R:
            def read(self_inner):
                return data.encode("utf-8")
        return R()

    def download(self, urls):
        self.f.calls.append(("audio", urls[0]))
        out = Path(self.opts["outtmpl"].replace("%(ext)s", "webm"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-audio")
        return 0


class FakeYDLFactory:
    def __init__(self):
        self.infos: dict[str, dict] = {}
        self.sub_payloads: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []
        self.info_errors: list[Exception] = []

    def add(self, info: dict, segments: list[tuple[float, str]] | None = None, raw_json3: str | None = None):
        self.infos[info["id"]] = info
        payload = raw_json3 if raw_json3 is not None else (
            json3_from_segments(segments) if segments is not None else
            (FIXTURES / "sample.json3").read_text(encoding="utf-8"))
        self.sub_payloads[f"https://subs.example/{info['id']}.json3"] = payload
        return self

    def __call__(self, opts: dict):
        return FakeYDL(self, opts)

    def count(self, kind: str) -> int:
        return sum(1 for k, _ in self.calls if k == kind)


class FakeTranscriber:
    name = "fake-whisper"

    def __init__(self, segments: list[tuple[float, str]], chunks: int = 3, delay: float = 0.0):
        self.segments = segments
        self.chunks = chunks
        self.delay = delay
        self.calls = 0
        self.unloaded = 0

    def transcribe(self, audio, on_progress=lambda f, m: None, cancel=None):
        from app.pipeline import check_cancel
        self.calls += 1
        assert Path(audio).exists()
        for i in range(self.chunks):
            check_cancel(cancel)
            time.sleep(self.delay)
            on_progress((i + 1) / self.chunks, f"кусок {i + 1} из {self.chunks}")
        return list(self.segments)

    def unload(self):
        self.unloaded += 1


# ───────────────────────── фейковый сервер LLM ─────────────────────────

def fake_answer(user: str) -> str:
    ts = re.search(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]", user)
    t = ts.group(1) if ts else "00:00"
    m = re.search(r"фрагмент (\d+) из (\d+)", user)
    if m:
        return f"### [{t}] Часть {m.group(1)}\n- мысль части {m.group(1)} " + "подробность " * 40
    if "частичные конспекты" in user:
        return f"### [{t}] Сведённая часть\n- общий пункт"
    return (f"<think>размышляю</think>```markdown\n## Кратко\nВидео о фотосинтезе и свете.\n\n"
            f"## Конспект\n### [{t}] Начало\n- хлорофилл поглощает свет\n- уже ссылка "
            f"[00:05](https://youtu.be/x?t=5)\n\n## Ключевые термины\n- **Хлорофилл** — пигмент\n\n"
            f"## Выводы\n- свет важен\n```")


class FakeLLMServer:
    def __init__(self, models=("qwen3:8b",), reject_think: bool = False):
        self.models = list(models)
        self.reject_think = reject_think
        self.requests: list[dict] = []
        self.fail_on: set[int] = set()      # номера chat-запросов (с 1), на которых вернуть 500
        self.length_on: set[int] = set()    # номера, где done_reason=length
        self.delay = 0.0                    # пауза между кусками стрима
        self.hold = threading.Event()       # если сброшен и hold_on — сервер ждёт
        self.hold.set()
        self.pulled: list[str] = []
        self.unloads = 0
        self._chat_n = 0
        self._lock = threading.Lock()
        server = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    return self._json(200, {"models": [{"name": n} for n in server.models]})
                if self.path == "/v1/models":
                    return self._json(200, {"data": [{"id": n} for n in server.models]})
                self._json(404, {"error": "nope"})

            def do_POST(self):
                ln = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(ln) or b"{}")
                if self.path == "/api/generate":
                    server.unloads += 1
                    return self._json(200, {"done": True})
                if self.path == "/api/pull":
                    server.pulled.append(payload.get("model"))
                    return self._stream([{"status": "pulling", "total": 100, "completed": c}
                                         for c in (10, 60, 100)] + [{"status": "success"}])
                if self.path in ("/api/chat", "/v1/chat/completions"):
                    if self.path == "/api/chat" and server.reject_think and "think" in payload:
                        return self._json(400, {"error": "think not supported"})
                    with server._lock:
                        server._chat_n += 1
                        n = server._chat_n
                        server.requests.append(payload)
                    if payload.get("model") not in server.models:
                        return self._json(404, {"error": f"model '{payload.get('model')}' not found"})
                    if n in server.fail_on:
                        return self._json(500, {"error": "boom"})
                    server.hold.wait(10)
                    user = payload["messages"][-1]["content"]
                    text = fake_answer(user)
                    pieces = [text[i:i + 40] for i in range(0, len(text), 40)]
                    reason = "length" if n in server.length_on else "stop"
                    if self.path == "/api/chat":
                        lines = [{"message": {"content": p}, "done": False} for p in pieces]
                        lines.append({"message": {"content": ""}, "done": True, "done_reason": reason})
                        return self._stream(lines)
                    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}" for p in pieces]
                    lines.append(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': reason}]})}")
                    lines.append("data: [DONE]")
                    return self._stream(lines, raw=True)
                self._json(404, {"error": "nope"})

            def _stream(self, items, raw=False):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for it in items:
                        line = it if raw else json.dumps(it, ensure_ascii=False)
                        self.wfile.write((line + "\n").encode())
                        self.wfile.flush()
                        if server.delay:
                            time.sleep(server.delay)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    @property
    def chat_count(self) -> int:
        return self._chat_n

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.hold.set()
        self.httpd.shutdown()
        self.httpd.server_close()
