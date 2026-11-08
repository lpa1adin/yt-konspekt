"""Распознавание речи: mlx-whisper (Apple Silicon) или faster-whisper (остальные).

Оба движка за одним интерфейсом: Transcriber.transcribe(path, on_progress, cancel) → segments.
После работы вызывать unload(), чтобы освободить память до запуска LLM.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import CancelFn, PipelineError, ProgressFn, check_cancel, noop_progress
from .transcript import Segment, fmt_ts

log = logging.getLogger(__name__)

CHUNK_SECONDS = 600  # куски по 10 минут: прогресс для mlx-whisper, у которого нет колбэка


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def resolve_engine(engine: str = "auto") -> tuple[str, str]:
    """Возвращает (движок, подсказка). Движок: mlx | faster | none."""
    hint = ""
    if engine in ("auto", "mlx"):
        if is_apple_silicon() and has_module("mlx_whisper"):
            if shutil.which("ffmpeg"):
                return "mlx", ""
            hint = "Для mlx-whisper нужен ffmpeg (brew install ffmpeg) — использую faster-whisper."
        elif engine == "mlx":
            hint = "mlx-whisper недоступен (нужен Mac на Apple Silicon и пакет mlx-whisper)."
    if has_module("faster_whisper"):
        return "faster", hint
    if not hint:
        hint = ("Не установлен движок распознавания: uv pip install mlx-whisper (Mac M) "
                "или uv pip install faster-whisper.")
    return "none", hint


class Transcriber:
    name = "base"

    def transcribe(self, audio: Path, on_progress: ProgressFn = noop_progress,
                   cancel: CancelFn | None = None) -> list[Segment]:
        raise NotImplementedError

    def unload(self) -> None:
        gc.collect()


def probe_duration(audio: Path) -> float:
    if not shutil.which("ffprobe"):
        return 0.0
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(audio)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def split_audio(audio: Path, out_dir: Path, seconds: int = CHUNK_SECONDS) -> list[Path]:
    """Режет аудио на куски WAV 16 кГц моно (ffmpeg)."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(audio), "-ac", "1", "-ar", "16000",
         "-f", "segment", "-segment_time", str(seconds), str(out_dir / "part%04d.wav")],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise PipelineError(f"ffmpeg не смог обработать аудио: {r.stderr.strip()[:300]}")
    return sorted(out_dir.glob("part*.wav"))


class MlxTranscriber(Transcriber):
    name = "mlx"

    def __init__(self, model: str = "large-v3-turbo", language: str = ""):
        self.repo = model if "/" in model else f"mlx-community/whisper-{model}"
        self.language = language or None

    def transcribe(self, audio, on_progress=noop_progress, cancel=None):
        import mlx_whisper

        total = probe_duration(audio)
        out: list[Segment] = []
        lang = self.language
        with tempfile.TemporaryDirectory(prefix="ytk-") as tmp:
            on_progress(0.0, "подготовка аудио…")
            parts = split_audio(audio, Path(tmp))
            if not parts:
                return []
            for i, part in enumerate(parts):
                check_cancel(cancel)
                offset = i * CHUNK_SECONDS
                result = mlx_whisper.transcribe(str(part), path_or_hf_repo=self.repo, language=lang)
                lang = lang or result.get("language")  # язык первого куска — для остальных
                for s in result.get("segments", []):
                    text = s.get("text", "").strip()
                    if text:
                        out.append((offset + float(s["start"]), text))
                done = min(offset + CHUNK_SECONDS, total) if total else offset + CHUNK_SECONDS
                frac = (i + 1) / len(parts)
                on_progress(frac, f"{fmt_ts(done)} / {fmt_ts(total)}" if total else f"кусок {i + 1} из {len(parts)}")
        return out

    def unload(self):
        try:
            from mlx_whisper import transcribe as mt
            holder = getattr(mt, "ModelHolder", None)
            if holder is not None:
                holder.model = None
                holder.model_path = None
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        try:
            import mlx.core as mx
            clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
            if clear:
                clear()
        except Exception:  # noqa: BLE001
            pass


class FasterTranscriber(Transcriber):
    name = "faster"

    def __init__(self, model: str = "large-v3-turbo", language: str = "", compute_type: str = "int8"):
        self.model_name = model
        self.language = language or None
        self.compute_type = compute_type
        self._model = None

    def transcribe(self, audio, on_progress=noop_progress, cancel=None):
        from faster_whisper import WhisperModel

        on_progress(0.0, "загрузка модели…")
        self._model = WhisperModel(self.model_name, device="auto", compute_type=self.compute_type)
        segments, info = self._model.transcribe(str(audio), language=self.language, vad_filter=True)
        total = float(getattr(info, "duration", 0) or 0)
        out: list[Segment] = []
        for s in segments:
            check_cancel(cancel)
            if s.text.strip():
                out.append((float(s.start), s.text.strip()))
            if total:
                on_progress(min(s.end / total, 1.0), f"{fmt_ts(s.end)} / {fmt_ts(total)}")
        return out

    def unload(self):
        self._model = None
        gc.collect()


def make_transcriber(wcfg: dict) -> tuple[Transcriber, str, str]:
    """Возвращает (транскрайбер, описание для документа, подсказка)."""
    engine, hint = resolve_engine(wcfg.get("engine", "auto"))
    model = wcfg.get("model", "large-v3-turbo")
    lang = wcfg.get("language", "")
    if engine == "mlx":
        return MlxTranscriber(model, lang), f"mlx-whisper {model}", hint
    if engine == "faster":
        return FasterTranscriber(model, lang, wcfg.get("compute_type", "int8")), f"faster-whisper {model}", hint
    raise PipelineError("Субтитров нет, а движок распознавания речи не установлен.", hint)
