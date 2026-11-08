"""FastAPI-приложение: HTML-страницы (Jinja2 + HTMX) и JSON API. Слушает только 127.0.0.1."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import DEFAULTS, HOST, PORT, data_dir, effective_config, get_dotted, load_toml, setup_logging
from .db import Database
from .events import EventBus
from .jobs import STAGE_TITLES, Deps, Worker, event_from_job, export_document, job_view
from .pipeline import PipelineError
from .pipeline.export import (
    available_formats,
    check_writable,
    open_path,
    pick_folder_dialog,
    reveal_path,
)
from .pipeline.llm import LLMClient
from .pipeline.source import video_id
from .pipeline.summarize import PROMPT_NAMES, PROMPT_TITLES, default_prompt, extract_brief, linkify_timecodes
from .pipeline.transcript import build_blocks, fmt_ts

log = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent

try:  # Jinja2 нужен только HTML-страницам; JSON API работает и без него
    from fastapi.templating import Jinja2Templates
    templates: Any = Jinja2Templates(directory=str(APP_DIR / "templates"))
except (ImportError, AssertionError):  # pragma: no cover
    templates = None

try:
    from markdown_it import MarkdownIt
    _md = MarkdownIt("commonmark", {"html": False, "breaks": False}).enable("table").enable("strikethrough")
except ImportError:  # pragma: no cover
    _md = None

LANGUAGES = [("ru", "Русский"), ("en", "English"), ("uk", "Українська"), ("de", "Deutsch"),
             ("fr", "Français"), ("es", "Español"), ("it", "Italiano"), ("zh", "中文")]
DETAILS = [("full", "Подробный"), ("short", "Краткий")]

# Поля формы настроек: имя поля → (ключ, тип)
SETTINGS_FIELDS: dict[str, tuple[str, type]] = {
    "output_dir": ("output_dir", str), "format": ("format", str), "auto_export": ("auto_export", bool),
    "filename_template": ("filename_template", str), "keep_markdown": ("keep_markdown", bool),
    "docx_template": ("docx_template", str), "language": ("language", str), "detail": ("detail", str),
    "keep_audio": ("keep_audio", bool), "cookies_from_browser": ("cookies_from_browser", str),
    "llm_backend": ("llm.backend", str), "llm_base_url": ("llm.base_url", str), "llm_model": ("llm.model", str),
    "llm_num_ctx": ("llm.num_ctx", int), "llm_chunk_chars": ("llm.chunk_chars", int),
    "llm_temperature": ("llm.temperature", float),
    "whisper_engine": ("whisper.engine", str), "whisper_model": ("whisper.model", str),
    "whisper_language": ("whisper.language", str), "subtitles_use": ("subtitles.use", bool),
}


def render_md(text: str) -> str:
    if _md is None:
        from html import escape
        return f"<pre>{escape(text)}</pre>"
    return _md.render(text)


def is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


async def read_body(request: Request) -> dict:
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            data = await request.json()
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}
    if "form" in ctype:
        form = await request.form()
        out: dict[str, Any] = {}
        for k in form:
            vals = form.getlist(k)
            out[k] = vals if len(vals) > 1 else vals[0]
        return out
    return {}


def truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "on", "yes", "да")


class AppState:
    def __init__(self, base: Path, deps: Deps | None):
        self.data_dir = base
        self.db = Database(base / "konspekt.sqlite3")
        self.bus = EventBus()
        self.worker = Worker(self.db, self.bus, base, deps)
        self.pulls: dict[str, dict] = {}

    def config(self, params: dict | None = None) -> dict:
        return effective_config(self.db.get_settings(), params, load_toml(base=self.data_dir))


def create_app(base_dir: Path | None = None, deps: Deps | None = None, start_worker: bool = True) -> FastAPI:
    base = base_dir or data_dir()
    base.mkdir(parents=True, exist_ok=True)
    state = AppState(base, deps)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        n = state.db.mark_interrupted()
        if n:
            log.info("Задач, прерванных перезапуском: %d", n)
        state.bus.bind(asyncio.get_running_loop())
        if start_worker:
            state.worker.start()
        yield
        await state.worker.stop()

    app = FastAPI(title="YT-Конспект", version=__version__, lifespan=lifespan)
    app.state.s = state
    app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

    # ───────────────────────── помощники ─────────────────────────

    def page(request: Request, name: str, ctx: dict | None = None, status_code: int = 200) -> Response:
        if templates is None:
            return HTMLResponse("<h1>Не установлен Jinja2</h1><p>Выполните: uv pip install jinja2</p>", 500)
        cfg = state.config()
        base_ctx = {
            "request": request, "cfg": cfg, "version": __version__, "formats": available_formats(),
            "languages": LANGUAGES, "details": DETAILS, "stage_titles": STAGE_TITLES,
            "nav": name.split("/")[-1].split(".")[0], "fmt_ts": fmt_ts,
            "waiting": state.db.one("SELECT message FROM jobs WHERE waiting=1 AND status='queued' LIMIT 1"),
        }
        base_ctx.update(ctx or {})
        return templates.TemplateResponse(request, name, base_ctx, status_code=status_code)

    def err(message: str, code: int = 400, hint: str = "") -> JSONResponse:
        return JSONResponse({"ok": False, "error": message, "hint": hint}, status_code=code)

    def job_card(request: Request, job_id: int) -> Response:
        job = state.db.get_job(job_id)
        if not job:
            return HTMLResponse("", 404)
        return page(request, "partials/job_card.html", {"job": job_view(job)})

    def model_list(cfg: dict) -> list[str]:
        try:
            names = LLMClient.from_config(cfg).list_models()
        except PipelineError:
            names = []
        cur = cfg["llm"]["model"]
        return names if cur in names else [cur, *names]

    # ───────────────────────── страницы ─────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        cfg = state.config()
        jobs = [job_view(j) for j in state.db.list_jobs(30)]
        models = await asyncio.to_thread(model_list, cfg)
        return page(request, "index.html", {"jobs": jobs, "models": models})

    @app.get("/partials/jobs", response_class=HTMLResponse)
    async def jobs_partial(request: Request):
        jobs = [job_view(j) for j in state.db.list_jobs(30)]
        return page(request, "partials/jobs.html", {"jobs": jobs})

    @app.get("/partials/jobs/{job_id}", response_class=HTMLResponse)
    async def job_partial(request: Request, job_id: int):
        return job_card(request, job_id)

    @app.get("/library", response_class=HTMLResponse)
    async def library(request: Request, q: str = "", channel: str = "", sort: str = "new", period: str = ""):
        items = state.db.library(q, channel, sort, period)
        for it in items:
            it["brief"] = extract_brief(it["summary_markdown"])
        ctx = {"items": items, "q": q, "channel": channel, "sort": sort, "period": period,
               "channels": state.db.channels()}
        if is_htmx(request) and request.headers.get("HX-Target") == "library-grid":
            return page(request, "partials/library_grid.html", ctx)
        return page(request, "library.html", ctx)

    @app.get("/videos/{vid}", response_class=HTMLResponse)
    async def video_page(request: Request, vid: str, v: int | None = None):
        video = state.db.get_video(vid)
        if not video:
            return page(request, "notfound.html", {"what": "Видео не найдено в библиотеке."}, 404)
        versions = state.db.summaries_for(vid)
        summary = None
        if versions:
            sid = next((s["id"] for s in versions if s["version"] == v), versions[0]["id"])
            summary = state.db.get_summary(sid)
        transcript = state.db.get_transcript(summary["transcript_id"]) if summary and summary["transcript_id"] \
            else state.db.latest_transcript(vid)
        blocks = [{"t": int(t), "ts": fmt_ts(t), "text": text}
                  for t, text in build_blocks(transcript["segments"])] if transcript else []
        active = state.db.active_job_for_video(vid)
        cfg = state.config()
        models = await asyncio.to_thread(model_list, cfg)
        return page(request, "video.html", {
            "video": video, "summary": summary, "versions": versions, "blocks": blocks,
            "transcript": transcript, "html": render_md(summary["markdown"]) if summary else "",
            "exports": [e | {"name": Path(e["path"]).name} for e in state.db.exports_for_video(vid)],
            "models": models,
            "active_job": job_view(state.db.get_job(active["id"])) if active else None,
        })

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request, saved: int = 0):
        cfg = state.config()
        overrides = state.db.get_settings()
        prompts = [{"name": n, "title": PROMPT_TITLES[n], "text": overrides.get(f"prompt.{n}") or default_prompt(n),
                    "custom": bool(overrides.get(f"prompt.{n}"))} for n in PROMPT_NAMES]
        models = await asyncio.to_thread(model_list, cfg)
        return page(request, "settings.html", {"prompts": prompts, "models": models, "saved": saved,
                                               "data_dir": str(state.data_dir)})

    @app.post("/settings")
    async def settings_save(request: Request):
        body = await read_body(request)
        errors = []
        for field, (key, typ) in SETTINGS_FIELDS.items():
            if typ is bool:
                state.db.set_setting(key, truthy(body.get(field, False)))
                continue
            if field not in body:
                continue
            raw = str(body[field]).strip()
            try:
                val: Any = typ(raw) if raw != "" or typ is str else get_dotted(DEFAULTS, key)
            except ValueError:
                errors.append(f"{field}: неверное значение")
                continue
            state.db.set_setting(key, val)
        langs = str(body.get("subtitles_langs", "")).strip()
        if langs:
            state.db.set_setting("subtitles.langs", [x.strip() for x in langs.split(",") if x.strip()])
        for n in PROMPT_NAMES:
            if f"prompt_{n}" in body:
                text = str(body[f"prompt_{n}"]).replace("\r\n", "\n").strip()
                state.db.set_setting(f"prompt.{n}", None if text == default_prompt(n).strip() or not text else text)
        if errors:
            return err("; ".join(errors))
        if is_htmx(request) or "application/json" not in request.headers.get("content-type", ""):
            return RedirectResponse("/settings?saved=1", status_code=303)
        return {"ok": True}

    @app.post("/settings/prompts/{name}/reset")
    async def prompt_reset(name: str):
        if name not in PROMPT_NAMES:
            return err("Нет такого промпта", 404)
        state.db.set_setting(f"prompt.{name}", None)
        return {"ok": True, "text": default_prompt(name)}

    @app.get("/status", response_class=HTMLResponse)
    async def status_page(request: Request):
        from .status import gather
        info = await asyncio.to_thread(gather, state.config(), state.data_dir)
        return page(request, "status.html", {"status": info})

    # ───────────────────────── задачи ─────────────────────────

    @app.post("/api/jobs")
    async def create_jobs(request: Request):
        body = await read_body(request)
        raw = body.get("urls") or body.get("url") or []
        if isinstance(raw, str):
            raw = raw.splitlines()
        urls = [u.strip() for u in raw if u and u.strip()]
        cfg = state.config()
        fmt = body.get("format", cfg["format"] if cfg.get("auto_export") else "")
        if fmt == "none":
            fmt = ""
        out_dir = str(body.get("out_dir") or cfg["output_dir"]).strip()
        params = {
            "detail": body.get("detail") or cfg["detail"],
            "language": body.get("language") or cfg["language"],
            "model": body.get("model") or cfg["llm"]["model"],
            "force_whisper": truthy(body.get("force_whisper", False)),
            "format": fmt, "out_dir": out_dir,
        }
        force = truthy(body.get("force", False))
        created, skipped, errors = [], [], []
        if not urls:
            errors.append({"url": "", "error": "Вставьте хотя бы одну ссылку на YouTube."})
        if fmt and urls:
            ok, msg = await asyncio.to_thread(check_writable, out_dir)
            if not ok:
                errors.append({"url": "", "error": msg + " Выберите другую папку."})
                urls = []
        seen = set()
        for url in urls:
            vid = video_id(url)
            if not vid:
                errors.append({"url": url, "error": "Не похоже на ссылку YouTube."})
                continue
            if vid in seen:
                continue
            seen.add(vid)
            if state.db.active_job_for_video(vid):
                skipped.append({"url": url, "video_id": vid, "reason": "уже в очереди"})
                continue
            if not force and state.db.latest_summary(vid):
                video = state.db.get_video(vid)
                skipped.append({"url": url, "video_id": vid, "reason": "уже в библиотеке",
                                "title": video["title"] if video else vid})
                continue
            job = state.db.create_job(url, vid, "full", params)
            created.append(event_from_job(job))
        if created:
            state.worker.wake()
        if is_htmx(request):
            resp = page(request, "partials/submit_result.html",
                        {"created": created, "skipped": skipped, "errors": errors, "body": body})
            if created:
                resp.headers["HX-Trigger"] = "jobs-changed"
            return resp
        code = 200 if created or skipped else 400
        return JSONResponse({"ok": bool(created), "created": created, "skipped": skipped, "errors": errors},
                            status_code=code)

    @app.get("/api/jobs")
    async def list_jobs():
        return {"jobs": [event_from_job(j) | {"url": j["url"], "kind": j["kind"], "summary_id": j["summary_id"],
                                              "export_path": j["export_path"], "created_at": j["created_at"]}
                         for j in state.db.list_jobs(100)]}

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(request: Request, job_id: int):
        if not state.worker.cancel(job_id):
            if is_htmx(request):
                return job_card(request, job_id)
            return err("Задачу нельзя отменить", 409)
        if is_htmx(request):
            return job_card(request, job_id)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/retry")
    async def retry_job(request: Request, job_id: int):
        ok = state.worker.retry(job_id)
        if is_htmx(request):
            return job_card(request, job_id)
        return {"ok": True} if ok else err("Задачу нельзя повторить", 409)

    @app.post("/api/jobs/clear")
    async def clear_jobs(request: Request):
        state.db.delete_finished_jobs()
        if is_htmx(request):
            return await jobs_partial(request)
        return {"ok": True}

    @app.get("/api/events")
    async def events(request: Request):
        from sse_starlette.sse import EventSourceResponse

        async def gen():
            async with state.bus.subscribe() as q:
                yield {"event": "hello", "data": json.dumps({"subscribers": state.bus.subscribers})}
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=10)
                    except TimeoutError:
                        continue
                    yield {"data": json.dumps(ev, ensure_ascii=False)}

        return EventSourceResponse(gen(), ping=15)

    # ───────────────────────── видео и конспекты ─────────────────────────

    @app.post("/api/videos/{vid}/summaries")
    async def resummarize(request: Request, vid: str):
        if not state.db.get_video(vid) or not state.db.latest_transcript(vid):
            return err("Расшифровка не найдена", 404)
        if state.db.active_job_for_video(vid):
            return err("По этому видео уже идёт задача", 409)
        body = await read_body(request)
        cfg = state.config()
        params = {"detail": body.get("detail") or cfg["detail"],
                  "language": body.get("language") or cfg["language"],
                  "model": body.get("model") or cfg["llm"]["model"],
                  "format": cfg["format"] if cfg.get("auto_export") else "", "out_dir": cfg["output_dir"]}
        if body.get("transcript_id"):
            params["transcript_id"] = int(body["transcript_id"])
        job = state.db.create_job(f"https://youtu.be/{vid}", vid, "resummarize", params)
        state.worker.wake()
        if is_htmx(request):
            return job_card(request, job["id"])
        return {"ok": True, "job": event_from_job(job)}

    @app.get("/api/summaries/{sid}")
    async def get_summary(sid: int):
        s = state.db.get_summary(sid)
        return s if s else err("Конспект не найден", 404)

    @app.put("/api/summaries/{sid}")
    async def edit_summary(request: Request, sid: int):
        s = state.db.get_summary(sid)
        if not s:
            return err("Конспект не найден", 404)
        body = await read_body(request)
        text = str(body.get("markdown", "")).replace("\r\n", "\n")
        if not text.strip():
            return err("Пустой конспект не сохраняю")
        text = linkify_timecodes(text.rstrip(), s["video_id"]) + "\n"
        new = state.db.add_summary(s["video_id"], s["transcript_id"], text, detail=s["detail"],
                                   language=s["language"], model=s["model"], prompt_hash=s["prompt_hash"],
                                   is_edited=True)
        return {"ok": True, "id": new["id"], "version": new["version"],
                "url": f"/videos/{s['video_id']}?v={new['version']}"}

    @app.post("/api/summaries/{sid}/export")
    async def export_summary(request: Request, sid: int):
        s = state.db.get_summary(sid)
        if not s:
            return err("Конспект не найден", 404)
        body = await read_body(request)
        cfg = state.config()
        fmt = body.get("format") or cfg["format"] or "md"
        if fmt not in available_formats():
            return err(f"Формат {fmt} недоступен на этой системе")
        out_dir = str(body.get("out_dir") or cfg["output_dir"])
        ok, msg = await asyncio.to_thread(check_writable, out_dir)
        if not ok:
            return err(msg, hint="Выберите другую папку — конспект остаётся в библиотеке.")
        video = state.db.get_video(s["video_id"])
        try:
            path = await asyncio.to_thread(export_document, state.db, s, video, fmt, out_dir, cfg)
        except PipelineError as e:
            return err(e.message, hint=e.hint)
        exp = state.db.one("SELECT * FROM exports WHERE path=? ORDER BY id DESC LIMIT 1", (str(path),))
        return {"ok": True, "export_id": exp["id"], "path": str(path), "name": path.name}

    @app.post("/api/exports/{eid}/{action}")
    async def export_action(eid: int, action: str):
        e = state.db.get_export(eid)
        if not e:
            return err("Экспорт не найден", 404)
        if not Path(e["path"]).exists():
            return err("Файл не найден — возможно, его переместили или удалили.", 404)
        if action == "reveal":
            reveal_path(e["path"])
        elif action == "open":
            open_path(e["path"])
        else:
            return err("Неизвестное действие", 404)
        return {"ok": True}

    @app.post("/api/open-path")
    async def open_any(request: Request):
        """«Показать в папке» для файла из карточки задачи (только пути из таблицы exports)."""
        body = await read_body(request)
        p = str(body.get("path", ""))
        if not state.db.one("SELECT id FROM exports WHERE path=?", (p,)) or not Path(p).exists():
            return err("Файл не найден", 404)
        (reveal_path if body.get("action", "reveal") == "reveal" else open_path)(p)
        return {"ok": True}

    @app.delete("/api/videos/{vid}")
    async def delete_video(request: Request, vid: str):
        if not state.db.get_video(vid):
            return err("Видео не найдено", 404)
        if state.db.active_job_for_video(vid):
            return err("Сначала отмените задачу по этому видео", 409)
        state.db.delete_video(vid)
        if is_htmx(request):
            return Response(status_code=200, headers={"HX-Redirect": "/library"})
        return {"ok": True}

    # ───────────────────────── система ─────────────────────────

    @app.get("/api/status")
    async def api_status():
        from .status import gather
        info = await asyncio.to_thread(gather, state.config(), state.data_dir)
        info["pulls"] = state.pulls
        info["queue"] = state.db.one("SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued','running')")["n"]
        return info

    @app.get("/api/models")
    async def api_models():
        cfg = state.config()
        return {"models": await asyncio.to_thread(model_list, cfg), "default": cfg["llm"]["model"]}

    @app.post("/api/models/pull")
    async def pull_model(request: Request):
        body = await read_body(request)
        cfg = state.config()
        name = str(body.get("name") or cfg["llm"]["model"]).strip()
        if cfg["llm"]["backend"] != "ollama":
            return err("Скачивание доступно только для Ollama")
        if state.pulls.get(name, {}).get("status") == "running":
            return {"ok": True, "name": name, "already": True}
        llm = LLMClient.from_config(cfg)
        state.pulls[name] = {"status": "running", "progress": 0, "message": "начинаю…"}

        def emit(**kw):
            state.pulls[name].update(kw)
            state.bus.publish({"type": "pull", "model": name, **state.pulls[name]})

        def work():
            try:
                def on(obj):
                    total, done = obj.get("total"), obj.get("completed")
                    prog = (done / total) if total and done else state.pulls[name]["progress"]
                    emit(progress=round(prog, 4), message=obj.get("status", ""))
                llm.pull(name, on)
                emit(status="done", progress=1, message="модель скачана")
            except PipelineError as e:
                emit(status="failed", message=str(e))
            except Exception as e:  # noqa: BLE001
                emit(status="failed", message=f"Ошибка: {e}")
            state.worker.retry_waiting()

        threading.Thread(target=work, daemon=True, name=f"pull-{name}").start()
        return {"ok": True, "name": name}

    @app.post("/api/pick-folder")
    async def pick_folder(request: Request):
        body = await read_body(request)
        path = await asyncio.to_thread(pick_folder_dialog, str(body.get("initial", "")))
        if not path:
            return {"ok": False, "path": None, "error": "Папка не выбрана — можно вписать путь вручную."}
        ok, msg = await asyncio.to_thread(check_writable, path)
        return {"ok": ok, "path": path.rstrip("/") or path, "error": "" if ok else msg}

    @app.post("/api/check-folder")
    async def check_folder(request: Request):
        body = await read_body(request)
        ok, msg = await asyncio.to_thread(check_writable, str(body.get("path", "")))
        return {"ok": ok, "path": msg if ok else body.get("path"), "error": "" if ok else msg}

    @app.post("/api/retry-waiting")
    async def retry_waiting():
        state.worker.retry_waiting()
        return {"ok": True}

    return app


def run() -> None:
    ap = argparse.ArgumentParser(prog="python -m app", description="YT-Конспект — локальный сервер")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    ap.add_argument("--window", action="store_true", help="открыть в отдельном окне (pywebview)")
    args = ap.parse_args()

    import uvicorn

    base = data_dir()
    log_path = setup_logging(base)
    url = f"http://{HOST}:{args.port}"
    print(f"YT-Конспект {__version__}: {url}\nДанные: {base}\nЖурнал: {log_path}\nОстановить: Ctrl+C",
          file=sys.stderr)
    app = create_app(base)

    if args.window:
        try:
            import webview  # pywebview
        except ImportError:
            print("pywebview не установлен (uv pip install pywebview) — открываю в браузере.", file=sys.stderr)
        else:
            server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=args.port, log_level="warning"))
            t = threading.Thread(target=server.run, daemon=True)
            t.start()
            webview.create_window("YT-Конспект", url, width=1280, height=860)
            webview.start()
            server.should_exit = True
            t.join(timeout=5)
            return

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    run()
