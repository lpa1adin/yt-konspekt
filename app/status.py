"""Проверки для экрана «Состояние»: что установлено и как это починить."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .pipeline import LLMUnavailable, PipelineError
from .pipeline.export import find_pages_app
from .pipeline.llm import LLMClient
from .pipeline.source import JS_HINT, has_ejs, js_runtimes, yt_dlp_version
from .pipeline.transcribe import has_module, is_apple_silicon, resolve_engine


def _check(key: str, label: str, ok: bool | None, detail: str = "", hint: str = "", action: str = "") -> dict:
    return {"key": key, "label": label, "ok": ok, "detail": detail, "hint": hint, "action": action}


def gather(cfg: dict, data_dir: Path) -> dict:
    checks: list[dict] = []
    llm = LLMClient.from_config(cfg)
    models: list[str] = []
    backend_name = llm.title
    try:
        models = llm.list_models()
        ok_detail = (f"{llm.base} отвечает" if llm.is_local
                     else "программа найдена, вход выполнен" if llm.backend == "cli" else "ключ API принят")
        checks.append(_check("llm_server", backend_name, True, ok_detail))
        has = llm.has_model(models)
        if llm.backend == "ollama":
            hint = f"Нажмите «Скачать модель» или выполните: ollama pull {llm.model}"
        elif llm.is_local:
            hint = "Загрузите модель в LM Studio."
        else:
            hint = "Выберите модель из списка в «Настройки → Модель»."
        checks.append(_check(
            "llm_model", f"Модель {llm.display_model}", has,
            ("скачана" if llm.is_local else "доступна") if has else "не найдена",
            "" if has else hint, action="pull" if (not has and llm.backend == "ollama") else ""))
    except LLMUnavailable as e:
        checks.append(_check("llm_server", backend_name, False, e.message,
                             "Установите Ollama с ollama.com и запустите приложение Ollama."
                             if llm.backend == "ollama" else e.hint))
        checks.append(_check("llm_model", f"Модель {llm.model}", None, "нельзя проверить без сервера"))
    except PipelineError as e:  # нет ключа, ключ не подошёл, не установлен пакет anthropic
        checks.append(_check("llm_server", backend_name, False, e.message, e.hint))
        checks.append(_check("llm_model", f"Модель {llm.display_model}", None,
                             "нельзя проверить, пока не исправлен пункт выше"))

    engine, hint = resolve_engine(cfg["whisper"].get("engine", "auto"))
    names = {"mlx": "mlx-whisper (Apple Silicon)", "faster": "faster-whisper", "none": "не установлен"}
    whisper_hint = hint
    if engine == "none":
        whisper_hint = ("uv pip install mlx-whisper" if is_apple_silicon() else "uv pip install faster-whisper")
    checks.append(_check("whisper", "Распознавание речи (Whisper)", engine != "none",
                         f"{names[engine]}, модель {cfg['whisper'].get('model')}", whisper_hint))

    ff = shutil.which("ffmpeg")
    checks.append(_check("ffmpeg", "ffmpeg", bool(ff), ff or "не найден",
                         "" if ff else "macOS: brew install ffmpeg · Windows: winget install ffmpeg · "
                                      "Linux: sudo apt install ffmpeg"))
    pd = shutil.which("pandoc")
    docx_ok = bool(pd) or has_module("docx")
    checks.append(_check("pandoc", "pandoc (экспорт .docx)", bool(pd) if pd else (None if docx_ok else False),
                         pd or ("не найден — используется встроенный конвертер python-docx" if docx_ok
                                else "не найден"),
                         "" if pd else "macOS: brew install pandoc (лучше оформление .docx)"))
    if sys.platform == "darwin":
        app = find_pages_app()
        checks.append(_check("pages", "Pages (экспорт .pages)", app is not None,
                             f"установлен: {app.name}" if app else "не найден",
                             "" if app else "Установите Pages из App Store или используйте .docx"))
    ver = yt_dlp_version()
    checks.append(_check("yt_dlp", "yt-dlp", bool(ver), ver or "не установлен",
                         "Держите свежим: uv pip install -U yt-dlp" if ver else "uv pip install yt-dlp"))

    rts = [n for n, c in js_runtimes().items() if c.get("path")]
    ejs = has_ejs()
    checks.append(_check("js_runtime", "JavaScript для YouTube (deno / node)", bool(rts and ejs),
                         ", ".join(rts) + ("" if ejs else ", нет пакета yt-dlp-ejs") if rts else "не найден",
                         "" if rts and ejs else JS_HINT))

    out = Path(cfg.get("output_dir") or "~").expanduser()
    probe = out if out.exists() else Path.home()
    try:
        free = shutil.disk_usage(probe).free / 1e9
        checks.append(_check("disk", "Свободное место", free > 5, f"{free:.1f} ГБ ({probe})",
                             "" if free > 5 else "Освободите место: модели и аудио занимают несколько ГБ."))
    except OSError:
        pass
    return {"checks": checks, "models": models, "model": llm.model, "backend": llm.backend,
            "local": llm.is_local,
            "data_dir": str(data_dir), "ok": all(c["ok"] is not False for c in checks)}
