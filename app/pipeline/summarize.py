"""Конспект: промпты, map-reduce для длинных расшифровок, таймкоды-ссылки, шапка документа."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import CancelFn, PipelineError, ProgressFn, check_cancel, noop_progress
from .llm import LLMClient
from .transcript import fmt_ts, split_chunks, ts_to_seconds

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
PROMPT_NAMES = ["system", "full", "short", "chunk", "merge", "final"]
PROMPT_TITLES = {
    "system": "Системный промпт",
    "full": "Структура: подробный конспект",
    "short": "Структура: краткий конспект",
    "chunk": "Конспект одной части (длинные видео)",
    "merge": "Сведение частей группами",
    "final": "Итоговый запрос",
}
MIN_TRANSCRIPT_CHARS = 200

LANG_NAMES = {
    "ru": "русский", "en": "английский", "de": "немецкий", "fr": "французский",
    "es": "испанский", "uk": "украинский", "it": "итальянский", "zh": "китайский",
}


def default_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def load_prompts(overrides: dict[str, str] | None = None) -> dict[str, str]:
    overrides = overrides or {}
    return {n: (overrides.get(n) or default_prompt(n)) for n in PROMPT_NAMES}


def prompt_hash(prompts: dict[str, str]) -> str:
    blob = json.dumps([prompts[n] for n in PROMPT_NAMES], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def render(template: str, **values) -> str:
    """Подставляет {title} {language} {transcript} {i} {n} {structure}; прочие фигурные скобки не трогает."""
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    return out


# ───────────────────────── постобработка ─────────────────────────

_TS_RE = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\](?!\()")


def linkify_timecodes(md: str, vid: str) -> str:
    """[мм:сс] → [мм:сс](https://youtu.be/ID?t=секунды); уже готовые ссылки не трогаем."""
    return _TS_RE.sub(lambda m: f"[{m.group(1)}](https://youtu.be/{vid}?t={ts_to_seconds(m.group(1))})", md)


def build_document(summary: str, meta: dict, source_desc: str, model: str,
                   date: dt.date | None = None) -> str:
    title = meta.get("title") or "Без названия"
    url = f"https://www.youtube.com/watch?v={meta['id']}"
    lines = [f"**Видео:** [{title}]({url})"]
    if meta.get("channel"):
        lines.append(f"**Канал:** {meta['channel']}")
    if meta.get("duration_s"):
        lines.append(f"**Длительность:** {fmt_ts(meta['duration_s'])}")
    lines.append(f"**Конспект:** {(date or dt.date.today()):%d.%m.%Y}, модель {model}, текст — {source_desc}")
    return f"# {title}\n\n" + "  \n".join(lines) + "\n\n---\n\n" + summary.strip() + "\n"


def extract_brief(md: str, limit: int = 220) -> str:
    """Первая строка раздела «Кратко» — для карточки в библиотеке."""
    m = re.search(r"^##\s+Кратко\s*\n+(.+?)(?:\n\n|\n#|$)", md, flags=re.M | re.S)
    text = m.group(1) if m else ""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`#>]", "", text).replace("\n", " ").strip()
    return text[:limit] + ("…" if len(text) > limit else "")


# ───────────────────────── map-reduce ─────────────────────────

@dataclass
class SummaryResult:
    markdown: str
    truncated: bool = False
    parts: int = 1


@dataclass
class SummaryCache:
    """Кэш частей map-reduce. Хранится в jobs.partials_json, чтобы «Повторить» шёл с места падения."""
    key: str = ""
    items: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"key": self.key, "items": self.items}, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str | None, key: str) -> SummaryCache:
        try:
            data = json.loads(raw or "{}")
        except ValueError:
            data = {}
        if data.get("key") == key:
            return cls(key, dict(data.get("items") or {}))
        return cls(key, {})


def cache_key(transcript_id: int | str, model: str, detail: str, language: str,
              chunk_chars: int, phash: str) -> str:
    return hashlib.sha256(f"{transcript_id}|{model}|{detail}|{language}|{chunk_chars}|{phash}".encode()).hexdigest()[:16]


def summarize(llm: LLMClient, transcript: str, title: str, language: str, detail: str,
              chunk_chars: int, prompts: dict[str, str], *,
              cache: SummaryCache | None = None,
              save_cache: Callable[[SummaryCache], None] | None = None,
              on_progress: ProgressFn = noop_progress,
              on_text: Callable[[str], None] | None = None,
              cancel: CancelFn | None = None) -> SummaryResult:
    if len(transcript.strip()) < MIN_TRANSCRIPT_CHARS:
        raise PipelineError("В видео почти нет речи — конспект не составлен.",
                            "Похоже, это музыка или видео без слов. Выдумывать содержание мы не будем.")
    cache = cache or SummaryCache()
    lang_name = LANG_NAMES.get(language, language)
    structure = prompts["full" if detail == "full" else "short"]
    system = render(prompts["system"], language=lang_name, title=title)
    truncated = False

    def ask(key: str, user: str) -> str:
        nonlocal truncated
        check_cancel(cancel)
        if key in cache.items:
            return cache.items[key]
        res = llm.chat(system, user, on_text=on_text, cancel=cancel)
        truncated = truncated or res.truncated
        cache.items[key] = res.text
        if save_cache:
            save_cache(cache)
        return res.text

    chunks = split_chunks(transcript, chunk_chars)
    n = len(chunks)
    if n == 1:
        on_progress(0.0, "пишу конспект…")
        text = ask("single", render(prompts["final"], title=title, language=lang_name,
                                    structure=structure, transcript=transcript, i=1, n=1))
        on_progress(1.0, "готово")
        return SummaryResult(text, truncated, 1)

    steps = n + 1
    partials: list[str] = []
    for i, ch in enumerate(chunks, 1):
        on_progress((i - 1) / steps, f"часть {i} из {n}")
        partials.append(ask(f"chunk:{i}", render(prompts["chunk"], title=title, language=lang_name,
                                                 transcript=ch, i=i, n=n)))

    # частичные конспекты не влезают в один запрос — сводим группами
    rnd = 0
    while sum(len(p) for p in partials) > chunk_chars and len(partials) > 1:
        groups = split_chunks("\n\n".join(partials), chunk_chars)
        if len(groups) >= len(partials):  # дальше не ужимается
            break
        rnd += 1
        merged = []
        for j, g in enumerate(groups, 1):
            on_progress(n / steps, f"свожу части: группа {j} из {len(groups)}")
            merged.append(ask(f"merge:{rnd}:{j}", render(prompts["merge"], title=title, language=lang_name,
                                                         transcript=g, i=j, n=len(groups))))
        partials = merged

    on_progress(n / steps, "собираю итоговый конспект…")
    text = ask("final", render(prompts["final"], title=title, language=lang_name, structure=structure,
                               transcript="\n\n".join(partials), i=n, n=n))
    on_progress(1.0, "готово")
    return SummaryResult(text, truncated, n)
