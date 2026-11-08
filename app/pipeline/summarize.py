"""Конспект: промпты, map-reduce для длинных расшифровок, таймкоды-ссылки, шапка документа."""

from __future__ import annotations

import bisect
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

    all_times = block_times(transcript)
    end_time = (all_times[-1] + 30) if all_times else 0
    chunks = split_chunks(transcript, chunk_chars)
    n = len(chunks)
    if n == 1:
        on_progress(0.0, "пишу конспект…")
        text = ask("single", render(prompts["final"], title=title, language=lang_name,
                                    structure=structure, transcript=transcript, i=1, n=1))
        on_progress(1.0, "готово")
        return SummaryResult(repair_timecodes(text, all_times, 0, end_time), truncated, 1)

    # 1. Конспект каждой части. Эти тексты и есть основной раздел «Конспект» итогового документа:
    #    итоговый запрос их больше не пересказывает, иначе 4 часа видео ужимаются в один короткий ответ.
    steps = n + 1
    parts: list[str] = []
    for i, ch in enumerate(chunks, 1):
        on_progress((i - 1) / steps, f"часть {i} из {n}")
        times = block_times(ch)
        lo, hi = (times[0], times[-1] + 30) if times else (0, end_time)
        raw = ask(f"chunk:{i}", render(prompts["chunk"], title=title, language=lang_name, transcript=ch,
                                       i=i, n=n, start=fmt_ts(lo), end=fmt_ts(hi)))
        parts.append(repair_timecodes(clean_part(raw), times, lo, hi))
    body = "\n\n".join(p for p in parts if p)
    section_times = block_times(body, anywhere=True) or all_times

    # 2. Для итогового запроса части при необходимости сводятся группами — только как материал
    #    для «Кратко», «Ключевых терминов» и «Выводов»; в документ сведённый текст не идёт.
    material = parts
    rnd = 0
    while sum(len(p) for p in material) > chunk_chars and len(material) > 1:
        groups = split_chunks("\n\n".join(material), chunk_chars)
        if len(groups) >= len(material):  # дальше не ужимается
            break
        rnd += 1
        merged = []
        for j, g in enumerate(groups, 1):
            on_progress(n / steps, f"свожу части для итогов: группа {j} из {len(groups)}")
            merged.append(ask(f"merge:{rnd}:{j}", render(prompts["merge"], title=title, language=lang_name,
                                                         transcript=g, i=j, n=len(groups))))
        material = merged

    # 3. Итоговый запрос. Для подробного конспекта модель пишет всё, кроме раздела «Конспект»,
    #    а его код собирает из частей; краткий конспект целиком пишет модель.
    full = detail == "full"
    final_structure = without_section(structure, "Конспект") if full else structure
    on_progress(n / steps, "собираю итоговый конспект…")
    text = ask("final", render(prompts["final"], title=title, language=lang_name, structure=final_structure,
                               transcript="\n\n".join(material), i=n, n=n))
    text = repair_timecodes(text, section_times, 0, end_time)
    if full:
        text = insert_section(text, "Конспект", body)
    on_progress(1.0, "готово")
    return SummaryResult(text, truncated, n)


# ───────────────────────── проверка ответа модели ─────────────────────────

_LINE_TS = re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]")


def block_times(text: str, anywhere: bool = False) -> list[int]:
    """Секунды таймкодов: в начале строк расшифровки или (anywhere) в любом месте текста."""
    rx = _TS_RE if anywhere else _LINE_TS
    found = (rx.finditer(text) if anywhere else (rx.match(line) for line in text.splitlines()))
    return sorted({ts_to_seconds(m.group(1)) for m in found if m})


def repair_timecodes(md: str, valid: list[int], lo: int, hi: int) -> str:
    """Чинит таймкоды, которые исказила модель, и привязывает их к реальным моментам видео.

    Типичные ошибки: потерянные часы ([05:30] вместо [1:05:30]) и выдуманные отметки вне фрагмента.
    Таймкод вне [lo, hi] сначала пробуем дополнить часами, затем берём ближайший блок расшифровки.
    """
    if not valid:
        return md
    valid = sorted(valid)

    def nearest(x: int) -> int:
        i = bisect.bisect_left(valid, x)
        options = valid[max(0, i - 1):i + 1]
        return min(options, key=lambda v: abs(v - x))

    def fix(m: re.Match) -> str:
        s = ts_to_seconds(m.group(1))
        candidates = [s + 3600 * k for k in range(10) if lo <= s + 3600 * k <= hi]
        target = candidates[0] if candidates else min(max(s, lo), hi)
        return f"[{fmt_ts(nearest(target))}]"

    return _TS_RE.sub(fix, md)


def clean_part(text: str) -> str:
    """Конспект части: убираем заголовки уровня #/## (модель иногда добавляет «## Конспект»)."""
    lines = [ln for ln in text.splitlines() if not re.match(r"^#{1,2}\s", ln)]
    return "\n".join(lines).strip()


def _section_re(name: str) -> re.Pattern:
    return re.compile(rf"^##\s+{re.escape(name)}\b.*?(?=^##\s|\Z)", flags=re.M | re.S)


def without_section(structure: str, name: str) -> str:
    return _section_re(name).sub("", structure).strip()


def insert_section(text: str, name: str, body: str) -> str:
    """Вставляет «## name» после раздела «Кратко» (раздел с тем же именем из ответа модели убирается)."""
    text = _section_re(name).sub("", text).strip()
    section = f"## {name}\n\n{body.strip()}\n\n"
    m = re.search(r"^##\s+(?!Кратко\b)", text, flags=re.M)
    if m:
        return text[:m.start()] + section + text[m.start():]
    return f"{text}\n\n{section}".strip()
