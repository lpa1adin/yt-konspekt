"""YouTube через yt-dlp: ID видео, метаданные, субтитры, аудио.

yt-dlp создаётся через фабрику `ydl_factory(opts)`, чтобы в тестах подменить его фейком.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import CancelFn, PipelineError, check_cancel
from .transcript import Segment

log = logging.getLogger(__name__)

YdlFactory = Callable[[dict], Any]

_ID_PATTERNS = [
    r"youtu\.be/([A-Za-z0-9_-]{11})",
    r"[?&]v=([A-Za-z0-9_-]{11})",
    r"/v/([A-Za-z0-9_-]{11})",
    r"/(?:shorts|live|embed)/([A-Za-z0-9_-]{11})",
]


def video_id(url: str) -> str | None:
    url = url.strip()
    for p in _ID_PATTERNS:
        m = re.search(p, url)
        if m:
            return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    return None


def watch_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


# ───────────────────────── парсинг субтитров ─────────────────────────

def parse_json3(raw: str) -> list[Segment]:
    data = json.loads(raw)
    segs: list[Segment] = []
    for ev in data.get("events", []):
        if "segs" not in ev:
            continue
        text = "".join(s.get("utf8", "") for s in ev["segs"]).replace("\n", " ").strip()
        if text:
            segs.append((ev.get("tStartMs", 0) / 1000.0, text))
    return segs


def parse_vtt(raw: str) -> list[Segment]:
    segs: list[Segment] = []
    last_line = None
    start = None
    for line in raw.splitlines():
        line = line.strip()
        if "-->" in line:
            t = line.split("-->")[0].strip().replace(",", ".")
            try:
                nums = [float(p) for p in t.split(":")]
            except ValueError:
                continue
            start = sum(n * 60 ** (len(nums) - 1 - i) for i, n in enumerate(nums))
            continue
        if not line or start is None or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        text = re.sub(r"<[^>]+>", "", line).strip()
        # у автосубтитров строки «прокручиваются» и повторяются — убираем дубли
        if text and text != last_line:
            segs.append((start, text))
            last_line = text
    return segs


@dataclass
class Track:
    formats: list[dict]
    key: str
    kind: str  # manual_subs | auto_subs

    @property
    def desc(self) -> str:
        return f"{'ручные субтитры' if self.kind == 'manual_subs' else 'автосубтитры'} ({self.key})"


def pick_track(info: dict, langs: list[str]) -> Track | None:
    """Ручные на языке видео → ручные ru/en → автодорожка *-orig → авто на языке видео/ru/en."""
    manual = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
    auto = info.get("automatic_captions") or {}
    vlang = (info.get("language") or "").split("-")[0]
    order = ([vlang] if vlang else []) + [lang for lang in langs if lang != vlang]

    def find(tracks: dict, lang: str) -> str | None:
        if lang in tracks:
            return lang
        for k in tracks:
            if k.split("-")[0] == lang and not k.endswith("-orig"):
                return k
        return None

    for lang in order:
        k = find(manual, lang)
        if k:
            return Track(manual[k], k, "manual_subs")
    for k in auto:  # оригинальная (не переведённая) автодорожка
        if k.endswith("-orig"):
            return Track(auto[k], k, "auto_subs")
    for lang in order:
        k = find(auto, lang)
        if k:
            return Track(auto[k], k, "auto_subs")
    if manual:
        k = next(iter(manual))
        return Track(manual[k], k, "manual_subs")
    return None


# ───────────────────────── ошибки yt-dlp → понятные сообщения ─────────────────────────

_NETWORK_MARKERS = ("timed out", "timeout", "connection reset", "network is unreachable",
                    "temporary failure in name resolution", "urlopen error", "connection aborted",
                    "remote end closed", "nodename nor servname")


def _is_network_error(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _NETWORK_MARKERS)


def friendly_error(exc: Exception) -> PipelineError:
    text = str(exc)
    low = text.lower()
    if any(m in low for m in ("private video", "confirm your age", "age-restricted", "members-only")):
        return PipelineError(
            "Видео недоступно: оно закрытое или с возрастным ограничением.",
            "Укажите в настройках «Cookies из браузера», где вы вошли в YouTube.")
    if "sign in to confirm" in low or "not a bot" in low or "429" in low or "too many requests" in low:
        return PipelineError(
            "YouTube просит подтвердить, что вы не бот.",
            "Обновите yt-dlp (uv pip install -U yt-dlp) и укажите в настройках «Cookies из браузера» "
            "(Safari / Chrome / Firefox), где вы вошли в YouTube.")
    if "video unavailable" in low or "has been removed" in low or "is not available" in low:
        return PipelineError("Видео недоступно (удалено или ограничено в вашей стране).")
    if "live event will begin" in low or "is live" in low or "premieres in" in low:
        return PipelineError("Сейчас идёт прямой эфир или премьера.", "Дождитесь окончания эфира.")
    if _is_network_error(text):
        return PipelineError("Нет связи с YouTube.", "Проверьте интернет и нажмите «Повторить».")
    return PipelineError(f"yt-dlp не смог обработать видео: {text[:300]}",
                         "Попробуйте обновить yt-dlp (uv pip install -U yt-dlp).")


def _with_retries(fn: Callable[[], Any], retries: int = 2, pause: float = 3.0,
                  cancel: CancelFn | None = None) -> Any:
    """Сетевые ошибки повторяем 2 раза с паузой, остальные пробрасываем сразу."""
    attempt = 0
    while True:
        try:
            return fn()
        except PipelineError:
            raise
        except Exception as e:  # noqa: BLE001
            if attempt < retries and _is_network_error(str(e)):
                attempt += 1
                log.warning("Сетевая ошибка (%s), повтор %d/%d", e, attempt, retries)
                for _ in range(int(pause * 10)):
                    check_cancel(cancel)
                    time.sleep(0.1)
                continue
            raise friendly_error(e) from e


def _default_factory(opts: dict):
    try:
        import yt_dlp
    except ImportError as e:
        raise PipelineError("Не установлен yt-dlp.", "Установите: uv pip install yt-dlp") from e
    return yt_dlp.YoutubeDL(opts)


def yt_dlp_version() -> str | None:
    try:
        import yt_dlp.version
        return yt_dlp.version.__version__
    except Exception:  # noqa: BLE001
        return None


class YouTubeSource:
    def __init__(self, cookies_from_browser: str = "", ydl_factory: YdlFactory | None = None,
                 retry_pause: float = 3.0):
        self.cookies = (cookies_from_browser or "").strip().lower()
        self.factory = ydl_factory or _default_factory
        self.retry_pause = retry_pause

    def _opts(self, **extra) -> dict:
        opts = {"quiet": True, "no_warnings": True, "noprogress": True}
        if self.cookies:
            opts["cookiesfrombrowser"] = (self.cookies,)
        opts.update(extra)
        return opts

    def extract_info(self, url: str, cancel: CancelFn | None = None) -> dict:
        def run():
            with self.factory(self._opts(skip_download=True)) as ydl:
                return ydl.extract_info(url, download=False)

        info = _with_retries(run, pause=self.retry_pause, cancel=cancel)
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            raise PipelineError("Сейчас идёт прямой эфир.", "Дождитесь окончания эфира и добавьте ссылку снова.")
        return info

    def fetch_subtitles(self, info: dict, langs: list[str],
                        cancel: CancelFn | None = None) -> tuple[list[Segment], Track] | None:
        track = pick_track(info, langs)
        if not track:
            return None
        by_ext = {f.get("ext"): f for f in track.formats if f.get("url")}
        with self.factory(self._opts(skip_download=True)) as ydl:
            for ext, parser in (("json3", parse_json3), ("vtt", parse_vtt)):
                check_cancel(cancel)
                fmt = by_ext.get(ext)
                if not fmt:
                    continue
                try:
                    raw = _with_retries(lambda f=fmt: ydl.urlopen(f["url"]).read().decode("utf-8", "replace"),
                                        pause=self.retry_pause, cancel=cancel)
                    segs = parser(raw)
                except PipelineError as e:
                    log.warning("Не удалось загрузить субтитры (%s): %s", ext, e)
                    continue
                except Exception as e:  # noqa: BLE001
                    log.warning("Не удалось разобрать субтитры (%s): %s", ext, e)
                    continue
                if segs:
                    return segs, track
        return None

    def download_audio(self, url: str, dest_dir: Path, cancel: CancelFn | None = None) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for old in dest_dir.glob("audio.*"):
            if old.suffix in (".part", ".ytdl"):
                old.unlink(missing_ok=True)
        existing = [p for p in dest_dir.glob("audio.*") if p.suffix not in (".part", ".ytdl")]
        if existing:  # уже скачано при прошлой попытке
            return existing[0]

        def hook(d: dict) -> None:
            if cancel and cancel():
                raise PipelineError("Отменено")  # yt-dlp прервёт загрузку

        def run():
            opts = self._opts(format="bestaudio/best", outtmpl=str(dest_dir / "audio.%(ext)s"),
                              progress_hooks=[hook])
            with self.factory(opts) as ydl:
                ydl.download([url])

        try:
            _with_retries(run, pause=self.retry_pause, cancel=cancel)
        except PipelineError:
            check_cancel(cancel)
            raise
        files = [p for p in dest_dir.glob("audio.*") if p.suffix not in (".part", ".ytdl")]
        if not files:
            raise PipelineError("Не удалось скачать аудио.", "Нажмите «Повторить».")
        return files[0]


def video_meta(info: dict, vid: str) -> dict:
    thumb = info.get("thumbnail") or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
    return {
        "id": vid,
        "title": info.get("title") or vid,
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration_s": int(info.get("duration") or 0),
        "language": (info.get("language") or "").split("-")[0],
        "thumbnail_url": thumb,
    }
