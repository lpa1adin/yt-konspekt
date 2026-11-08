"""Очередь задач: один asyncio-воркер, задачи строго по одной.

Whisper и LLM никогда не работают одновременно — главное условие для 16 ГБ памяти.
Задача выполняется в отдельном потоке (asyncio.to_thread), прогресс пишется в БД и в шину событий.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import effective_config, load_toml
from .db import Database, now
from .events import EventBus
from .pipeline import Cancelled, LLMUnavailable, PipelineError, check_cancel
from .pipeline.export import export_markdown, file_base
from .pipeline.llm import LLMClient
from .pipeline.source import YouTubeSource, video_id, video_meta, watch_url
from .pipeline.summarize import (
    SummaryCache,
    build_document,
    cache_key,
    linkify_timecodes,
    load_prompts,
    prompt_hash,
    summarize,
)
from .pipeline.transcribe import make_transcriber
from .pipeline.transcript import build_transcript

log = logging.getLogger(__name__)

STAGE_TITLES = {
    "info": "данные о видео",
    "subtitles": "субтитры",
    "audio": "скачиваю аудио",
    "transcribing": "распознаю речь",
    "summarizing": "пишу конспект",
    "exporting": "сохраняю файл",
}
STATUS_TITLES = {"queued": "в очереди", "running": "в работе", "done": "готово",
                 "failed": "ошибка", "cancelled": "отменено"}


@dataclass
class Deps:
    """Внешние зависимости — подменяются фейками в тестах."""
    ydl_factory: Callable[[dict], Any] | None = None
    transcriber_factory: Callable[[dict], tuple[Any, str, str]] | None = None
    retry_pause: float = 3.0
    wait_pause: float = 15.0   # пауза, если Ollama не запущена
    stream_interval: float = 0.25


@dataclass
class _Running:
    job_id: int
    cancel: threading.Event = field(default_factory=threading.Event)
    llm: LLMClient | None = None


def event_from_job(job: dict, **extra) -> dict:
    ev = {"job_id": job["id"], "status": job["status"], "stage": job.get("stage"),
          "progress": round(float(job.get("progress") or 0), 4), "message": job.get("message") or "",
          "waiting": bool(job.get("waiting")), "video_id": job.get("video_id"),
          "title": job.get("title"), "error": job.get("error")}
    ev.update(extra)
    return ev


class Worker:
    def __init__(self, db: Database, bus: EventBus, data_dir: Path, deps: Deps | None = None):
        self.db = db
        self.bus = bus
        self.data_dir = data_dir
        self.cache_dir = data_dir / "cache"
        self.deps = deps or Deps()
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.current: _Running | None = None
        self._llm_dirty: LLMClient | None = None  # модель, которую надо выгрузить, когда очередь опустеет
        self._last_stream = 0.0

    # ───────── жизненный цикл ─────────

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.bus.bind(self._loop)
        self._task = asyncio.create_task(self.run(), name="yt-konspekt-worker")

    async def stop(self) -> None:
        if self.current:
            self.current.cancel.set()
            if self.current.llm:
                self.current.llm.abort()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    def wake(self) -> None:
        """Потокобезопасно разбудить воркер."""
        loop = getattr(self, "_loop", None)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is None or running is loop:
            self._wake.set()
        elif not loop.is_closed():
            loop.call_soon_threadsafe(self._wake.set)

    async def _sleep(self, seconds: float | None) -> None:
        """Ждать пробуждения. Флаг сбрасывается в начале цикла run(), до проверки очереди,
        чтобы не потерять wake(), пришедший во время выгрузки модели."""
        try:
            if seconds is None:
                await self._wake.wait()
            else:
                await asyncio.wait_for(self._wake.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def run(self) -> None:
        while True:
            self._wake.clear()
            job = self.db.next_queued()
            if job is None:
                if self._llm_dirty is not None:  # очередь пуста — освобождаем память Ollama
                    llm, self._llm_dirty = self._llm_dirty, None
                    await asyncio.to_thread(llm.unload)
                    continue  # за время выгрузки могли добавить задачу
                await self._sleep(None)
                continue
            if job.get("waiting"):
                # все оставшиеся задачи ждут Ollama — пробуем снова через паузу или по кнопке
                await self._sleep(self.deps.wait_pause)
                job = self.db.next_queued()
                if job is None:
                    continue
            self.current = _Running(job["id"])
            try:
                await asyncio.to_thread(self._execute, job, self.current)
            except Exception:  # noqa: BLE001 — воркер не должен умирать
                log.exception("Сбой воркера на задаче %s", job["id"])
                self._set(job["id"], status="failed", error="Внутренняя ошибка, подробности в журнале.",
                          finished_at=now())
            finally:
                self.current = None

    # ───────── управление ─────────

    def cancel(self, job_id: int) -> bool:
        job = self.db.get_job(job_id)
        if not job or job["status"] not in ("queued", "running"):
            return False
        if self.current and self.current.job_id == job_id:
            self.current.cancel.set()
            if self.current.llm:
                self.current.llm.abort()
            self._set(job_id, message="отменяю…")
        else:
            self._set(job_id, status="cancelled", message="отменено", waiting=0, finished_at=now())
        return True

    def retry(self, job_id: int) -> bool:
        job = self.db.get_job(job_id)
        if not job or job["status"] in ("running",):
            return False
        if job["status"] == "queued":  # «Проверить снова» у ждущей задачи
            self._set(job_id, waiting=0)
        else:
            self._set(job_id, status="queued", error=None, waiting=0, progress=0,
                      message="в очереди (повтор)", finished_at=None)
        self.wake()
        return True

    def retry_waiting(self) -> None:
        """Ollama запустили или модель скачали — ждущие задачи пробуют снова сразу."""
        for j in self.db.q("SELECT id FROM jobs WHERE status='queued' AND waiting=1"):
            self._set(j["id"], waiting=0, message="в очереди")
        self.wake()

    # ───────── внутреннее ─────────

    def _set(self, job_id: int, publish: bool = True, **fields) -> dict:
        self.db.update_job(job_id, **fields)
        job = self.db.get_job(job_id)
        if publish and job:
            self.bus.publish(event_from_job(job))
        return job

    def _config(self, job: dict) -> dict:
        return effective_config(self.db.get_settings(), job["params"], load_toml(base=self.data_dir))

    def _prompts(self) -> dict[str, str]:
        s = self.db.get_settings()
        return load_prompts({k[7:]: v for k, v in s.items() if k.startswith("prompt.") and v})

    def _execute(self, job: dict, run: _Running) -> None:
        jid = job["id"]
        cancel = run.cancel.is_set
        self._set(jid, status="running", started_at=job.get("started_at") or now(), error=None, waiting=0)
        try:
            cfg = self._config(job)
            if job["kind"] == "full":
                self._run_full(job, cfg, run)
            elif job["kind"] == "resummarize":
                self._run_resummarize(job, cfg, run)
            else:
                raise PipelineError(f"Неизвестный тип задачи: {job['kind']}")
        except Cancelled:
            self._set(jid, status="cancelled", message="отменено", finished_at=now())
        except LLMUnavailable as e:
            if cancel():
                self._set(jid, status="cancelled", message="отменено", finished_at=now())
            else:
                self._set(jid, status="queued", waiting=1, stage="summarizing",
                          message=f"⏳ {e.message} {e.hint}".strip())
        except PipelineError as e:
            self._set(jid, status="failed", error=str(e), message=e.message, finished_at=now())
        except Exception as e:  # noqa: BLE001
            log.exception("Задача %s упала", jid)
            self._set(jid, status="failed", error=f"Неожиданная ошибка: {e}",
                      message="неожиданная ошибка", finished_at=now())

    def _progress(self, jid: int, stage: str):
        def fn(frac: float, msg: str) -> None:
            self._set(jid, stage=stage, progress=max(0.0, min(frac, 1.0)), message=msg)
        return fn

    def _stream_fn(self, jid: int):
        def fn(text: str) -> None:
            t = time.monotonic()
            if t - self._last_stream >= self.deps.stream_interval:
                self._last_stream = t
                job = self.db.get_job(jid)
                if job:
                    self.bus.publish(event_from_job(job, text=text[-1500:]))
        return fn

    def _transcriber(self, wcfg: dict):
        if self.deps.transcriber_factory:
            return self.deps.transcriber_factory(wcfg)
        return make_transcriber(wcfg)

    def _run_full(self, job: dict, cfg: dict, run: _Running) -> None:
        jid = job["id"]
        cancel = run.cancel.is_set
        vid = job.get("video_id") or video_id(job["url"])
        if not vid:
            raise PipelineError(f"Не похоже на ссылку YouTube: {job['url']}")
        if not job.get("video_id"):
            self._set(jid, video_id=vid)
        force_whisper = bool(job["params"].get("force_whisper"))
        transcript = self.db.latest_transcript(vid, whisper_only=force_whisper)
        video = self.db.get_video(vid)

        if transcript is None or video is None:
            source = YouTubeSource(cfg.get("cookies_from_browser", ""), self.deps.ydl_factory,
                                   retry_pause=self.deps.retry_pause)
            self._set(jid, stage="info", progress=0, message="получаю данные о видео…")
            info = source.extract_info(watch_url(vid), cancel=cancel)
            video = video_meta(info, vid)
            self.db.upsert_video(video)
            self._set(jid, message=video["title"])
            check_cancel(cancel)

        if transcript is None:
            got = None
            if cfg["subtitles"].get("use", True):
                self._set(jid, stage="subtitles", message="ищу субтитры…")
                got = source.fetch_subtitles(info, cfg["subtitles"].get("langs", ["ru", "en"]), cancel=cancel)
            check_cancel(cancel)
            if got:
                segs, track = got
                transcript = self.db.add_transcript(vid, track.kind, track.desc, segs)
            else:
                transcript = self._whisper(jid, vid, cfg, source, run)

        check_cancel(cancel)
        summary = self._summarize(job, cfg, video, transcript, run)
        self._maybe_export(jid, cfg, video, summary)

    def _whisper(self, jid: int, vid: str, cfg: dict, source: YouTubeSource, run: _Running) -> dict:
        cancel = run.cancel.is_set
        transcriber, engine_desc, hint = self._transcriber(cfg["whisper"])
        audio_dir = self.cache_dir / vid
        self._set(jid, stage="audio", progress=0, message="скачиваю аудио…" + (f" {hint}" if hint else ""))
        audio = source.download_audio(watch_url(vid), audio_dir, cancel=cancel)
        check_cancel(cancel)
        self._set(jid, stage="transcribing", progress=0, message="распознаю речь…")
        try:
            segs = transcriber.transcribe(audio, self._progress(jid, "transcribing"), cancel)
        finally:
            transcriber.unload()  # память — до запуска LLM
        if not cfg.get("keep_audio"):
            shutil.rmtree(audio_dir, ignore_errors=True)
        if not segs:
            raise PipelineError("Не удалось распознать речь — текст пустой.")
        return self.db.add_transcript(vid, "whisper", engine_desc, segs)

    def _summarize(self, job: dict, cfg: dict, video: dict, transcript: dict, run: _Running) -> dict:
        jid = job["id"]
        cancel = run.cancel.is_set
        llm = LLMClient.from_config(cfg)
        run.llm = llm
        self._set(jid, stage="summarizing", progress=0, message=f"проверяю {llm.model}…")
        llm.check()
        self._llm_dirty = llm
        prompts = self._prompts()
        phash = prompt_hash(prompts)
        chunk_chars = int(cfg["llm"].get("chunk_chars", 18000))
        key = cache_key(transcript["id"], llm.model, cfg["detail"], cfg["language"], chunk_chars, phash)
        fresh = self.db.get_job(jid)
        cache = SummaryCache.from_json(fresh.get("partials_json"), key)

        def save_cache(c: SummaryCache) -> None:
            self.db.update_job(jid, partials_json=c.to_json())

        res = summarize(llm, build_transcript(transcript["segments"]), video["title"], cfg["language"],
                        cfg["detail"], chunk_chars, prompts, cache=cache, save_cache=save_cache,
                        on_progress=self._progress(jid, "summarizing"), on_text=self._stream_fn(jid),
                        cancel=cancel)
        source_desc = ("распознавание Whisper (" + transcript["source_detail"] + ")"
                       if transcript["source"] == "whisper" else transcript["source_detail"])
        md = build_document(linkify_timecodes(res.markdown, video["id"]), video, source_desc, llm.model)
        summary = self.db.add_summary(video["id"], transcript["id"], md, detail=cfg["detail"],
                                      language=cfg["language"], model=llm.model, prompt_hash=phash)
        warn = (" ⚠ Ответ модели обрезан по контексту — уменьшите chunk_chars в настройках и нажмите «Переделать»."
                if res.truncated else "")
        self._set(jid, summary_id=summary["id"], partials_json=None, message="конспект готов" + warn)
        return summary

    def _maybe_export(self, jid: int, cfg: dict, video: dict, summary: dict) -> None:
        fmt = cfg.get("format") or ""
        job = self.db.get_job(jid)
        msg = job["message"]
        if fmt and cfg.get("output_dir") and job["params"].get("export", True):
            self._set(jid, stage="exporting", progress=1, message="сохраняю файл…")
            try:
                path = export_document(self.db, summary, video, fmt, cfg["output_dir"], cfg)
                self._set(jid, status="done", export_path=str(path), finished_at=now(),
                          message=f"{msg} · файл: {path.name}")
                return
            except PipelineError as e:
                self._set(jid, status="done", finished_at=now(), error=str(e),
                          message=f"{msg} · ⚠ файл не сохранён: {e.message}")
                return
        self._set(jid, status="done", progress=1, finished_at=now(), message=msg)

    def _run_resummarize(self, job: dict, cfg: dict, run: _Running) -> None:
        vid = job["video_id"]
        video = self.db.get_video(vid)
        tid = job["params"].get("transcript_id")
        transcript = self.db.get_transcript(tid) if tid else self.db.latest_transcript(vid)
        if not video or not transcript:
            raise PipelineError("Расшифровка не найдена — добавьте видео заново.")
        summary = self._summarize(job, cfg, video, transcript, run)
        self._maybe_export(job["id"], cfg, video, summary)


def export_document(db: Database, summary: dict, video: dict, fmt: str, out_dir: str, cfg: dict) -> Path:
    base = file_base(cfg.get("filename_template", "{title}"), video)
    path = export_markdown(summary["markdown"], out_dir, base, fmt,
                           docx_template=cfg.get("docx_template", ""),
                           keep_markdown=bool(cfg.get("keep_markdown")))
    db.add_export(summary["id"], path.suffix.lstrip("."), str(path))
    return path


def job_view(job: dict) -> dict:
    """Поля для шаблона карточки задачи."""
    j = dict(job)
    j["status_title"] = STATUS_TITLES.get(j["status"], j["status"])
    j["stage_title"] = STAGE_TITLES.get(j.get("stage") or "", "")
    j["percent"] = int(round((j.get("progress") or 0) * 100))
    j["params"] = j.get("params") or json.loads(j.get("params_json") or "{}")
    return j
